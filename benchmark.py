# %%
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Constants ──────────────────────────────────────────────────────────────
HIDDEN_MLP  = 128   # → 16,897 params  (optimal width from grid search)
HIDDEN_LSTM = 63    # → 16,696 params  (matched to MLP, within 1.2%)

SEED        = 42
TBPTT_CHUNK = 50    # backprop through 50 steps at a time, carry hidden state across chunks

CHECK_EVERY = 50
PATIENCE    = 8
MAX_EPOCHS  = 15000
MIN_DELTA   = 1e-4
LR_MLP      = 1e-2
LR_LSTM     = 1e-3   # overridden by grid search

MLP_LR_GRID    = [1e-1, 1e-2, 5e-3, 1e-3]
LSTM_LR_GRID   = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
LSTM_PATIENCE  = 32
GS_EPOCHS      = MAX_EPOCHS
AR_ORDER       = 50

SNAP_EPOCHS = frozenset({0, 200, 400, 600, 800, 1000,
                         1500, 2000, 2500, 3000,
                         6000, 9000, 12000, 15000})

# Best LRs from completed grid searches — skip GS for these signals
BEST_LRS = {
    "Decay":       {"MLP": 1e-2, "LSTM": 1e-3},
    "Chirp":       {"MLP": 1e-2, "LSTM": 1e-3},
    "Multi-Scale": {"MLP": 5e-3, "LSTM": 1e-2},
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

L = 1000

# %%
# ── Models: t_norm ∈ [0,1] → y(t) ─────────────────────────────────────────

class MLP(nn.Module):
    """Pointwise: scalar t → scalar y."""
    def __init__(self, hidden=HIDDEN_MLP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t):
        return self.net(t)   # (N, 1)


class SeqLSTM(nn.Module):
    """Sequence: t₀, t₁, … tₙ → y₀, y₁, … yₙ  (hidden state tracks phase)."""
    def __init__(self, hidden=HIDDEN_LSTM):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, t_seq, hc=None):
        # t_seq: (1, chunk, 1)
        out, hc_out = self.lstm(t_seq, hc)   # (1, chunk, hidden)
        return self.fc(out), hc_out           # (1, chunk, 1), (h, c)


def init_lstm(model):
    """Orthogonal hidden weights + forget gate bias = 1 (Jozefowicz et al. 2015)."""
    for name, p in model.named_parameters():
        if 'weight_hh' in name:
            # QR (needed for orthogonal init) is unsupported on MPS — init on CPU then copy back
            tmp = p.data.cpu()
            nn.init.orthogonal_(tmp)
            p.data.copy_(tmp)
        elif 'bias' in name:
            nn.init.zeros_(p)
            n = p.shape[0] // 4
            p.data[n:2*n].fill_(1.0)   # forget gate slice


# %%
# ── Smoke test ─────────────────────────────────────────────────────────────
print(f"MLP    params: {sum(p.numel() for p in MLP().parameters()):,}")
print(f"LSTM   params: {sum(p.numel() for p in SeqLSTM().parameters()):,}")

# %%
# ── Signals ────────────────────────────────────────────────────────────────
t_np   = np.linspace(0, 10, L)
t_norm = (t_np / t_np[-1]).astype(np.float32)

decay   = (np.exp(-0.3 * t_np) * np.sin(2 * np.pi * 1.5 * t_np)).astype(np.float32)

f0, f1  = 0.5, 2.0
chirp   = np.sin(2 * np.pi * (f0 * t_np + 0.5 * (f1 - f0) / t_np[-1] * t_np**2)).astype(np.float32)

carrier    = np.sin(2 * np.pi * 0.15 * t_np)
r0, r1     = 4.0, 12.0
ripple     = 0.6 * np.sin(2 * np.pi * (r0 * t_np + 0.5 * (r1 - r0) / t_np[-1] * t_np**2))
multiscale = (carrier + ripple).astype(np.float32)

SIGNALS = {
    "Decay":       decay,
    "Chirp":       chirp,
    "Multi-Scale": multiscale,
}

# %%
# ── Training ───────────────────────────────────────────────────────────────

def fit_ar(y_arr, order=AR_ORDER):
    """Fit AR(order) via OLS, predict by autoregressive rollout from true initial conditions."""
    n = len(y_arr)
    X = np.column_stack([y_arr[order-1-i:n-1-i] for i in range(order)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_arr[order:], rcond=None)
    pred = np.zeros(n)
    pred[:order] = y_arr[:order]
    for t in range(order, n):
        pred[t] = coeffs @ pred[t-order:t][::-1]
    return pred


def train_interp(model, T, Y, lr, is_seq=False,
                 max_epochs=MAX_EPOCHS, check_every=CHECK_EVERY, patience=PATIENCE,
                 snap_epochs=None):
    """
    Fit model to the full signal.
    T: (L, 1)  normalized time
    Y: (L, 1)  signal values
    is_seq: True for SeqLSTM (wraps T into a single batch-of-one sequence)
    """
    model   = model.to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.01)
    loss_fn = nn.MSELoss()
    best_loss, best_state, strikes = float('inf'), None, 0
    snapshots = {}

    T_in = T.unsqueeze(0) if is_seq else T   # (1, L, 1) or (L, 1)

    def _snap():
        model.eval()
        with torch.no_grad():
            if is_seq:
                out, _ = model(T.unsqueeze(0))
                s = out.squeeze(0).squeeze(1).cpu().numpy()
            else:
                s = model(T).squeeze(1).cpu().numpy()
        model.train()
        return s

    if snap_epochs and 0 in snap_epochs:
        snapshots[0] = _snap()

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        if is_seq:
            out, _ = model(T_in)             # SeqLSTM returns (out, hc)
            pred = out.squeeze(0)            # (1, L, 1) → (L, 1)
        else:
            pred = model(T_in)
        loss = loss_fn(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if epoch % check_every == 0:
            mse_val = loss.item()
            if mse_val < best_loss - MIN_DELTA:
                best_loss  = mse_val
                best_state = {k: p.clone() for k, p in model.state_dict().items()}
                strikes    = 0
                print(f"    ep {epoch:5d}  mse={mse_val:.6f}  improved", flush=True)
            else:
                strikes += 1
                print(f"    ep {epoch:5d}  mse={mse_val:.6f}  strike {strikes}/{patience}", flush=True)
            if strikes >= patience:
                break

        opt.step()
        sched.step()

        if snap_epochs and epoch in snap_epochs:
            snapshots[epoch] = _snap()

    if best_state:
        model.load_state_dict(best_state)
    return best_loss, epoch, snapshots


def grid_search_mlp(T, Y):
    """Fixed seed across LRs so only LR varies, not initialization."""
    best_mse, best_lr = float('inf'), None
    for lr in MLP_LR_GRID:
        torch.manual_seed(SEED)
        m = MLP().to(DEVICE)
        print(f"  gs lr={lr:.0e} ...", flush=True)
        mse_val, ep, _ = train_interp(m, T, Y, lr=lr, is_seq=False,
                                      max_epochs=GS_EPOCHS, patience=LSTM_PATIENCE)
        print(f"  gs lr={lr:.0e}  mse={mse_val:.6f}  ep={ep}", flush=True)
        if mse_val < best_mse:
            best_mse, best_lr = mse_val, lr
    print(f"  Best lr={best_lr:.0e}  gs_mse={best_mse:.6f}")
    return best_lr, best_mse


def grid_search_lstm(T, Y):
    """Fixed seed across LRs so only LR varies, not initialization."""
    best_mse, best_lr = float('inf'), None
    for lr in LSTM_LR_GRID:
        torch.manual_seed(SEED)
        m = SeqLSTM().to(DEVICE)
        init_lstm(m)
        print(f"  gs lr={lr:.0e} ...", flush=True)
        mse_val, ep, _ = train_interp(m, T, Y, lr=lr, is_seq=True,
                                      max_epochs=GS_EPOCHS, patience=LSTM_PATIENCE)
        print(f"  gs lr={lr:.0e}  mse={mse_val:.6f}  ep={ep}", flush=True)
        if mse_val < best_mse:
            best_mse, best_lr = mse_val, lr
    print(f"  Best lr={best_lr:.0e}  gs_mse={best_mse:.6f}")
    return best_lr, best_mse


def predict(model, T, is_seq=False):
    model.eval()
    with torch.no_grad():
        if is_seq:
            out, _ = model(T.unsqueeze(0))   # (1, L, 1)
            return out.squeeze(0).squeeze(1).cpu().numpy()
        return model(T).squeeze(1).cpu().numpy()


# %%
# ── Main loop ──────────────────────────────────────────────────────────────
MODEL_NAMES = ["AR", "MLP", "LSTM"]
mse   = {}
preds = {}

T_all = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # (L, 1)

for sig_name, y in SIGNALS.items():
    print(f"\n{'='*55}\n  {sig_name}\n{'='*55}")
    Y_all = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)   # (L, 1)
    mse[sig_name]   = {}
    preds[sig_name] = {}

    # ── AR ────────────────────────────────────────────────────────────────
    p_ar = fit_ar(y)
    preds[sig_name]['AR'] = p_ar
    mse[sig_name]['AR']   = float(np.mean((p_ar - y) ** 2))
    print(f"  AR      fit MSE = {mse[sig_name]['AR']:.6f}")

    # ── Get best LRs (cached or grid search) ─────────────────────────────
    if sig_name in BEST_LRS:
        best_mlp_lr  = BEST_LRS[sig_name]["MLP"]
        best_lstm_lr = BEST_LRS[sig_name]["LSTM"]
        print(f"  Cached LRs — MLP={best_mlp_lr:.0e}  LSTM={best_lstm_lr:.0e}")
    else:
        print(f"  MLP grid search (lr in {MLP_LR_GRID}) ...")
        best_mlp_lr, _ = grid_search_mlp(T_all, Y_all)
        print(f"  LSTM grid search (lr in {LSTM_LR_GRID}) ...")
        best_lstm_lr, _ = grid_search_lstm(T_all, Y_all)

    # ── Retrain both with 2× patience ─────────────────────────────────────
    print(f"\n  Retraining MLP  lr={best_mlp_lr:.0e}  patience={LSTM_PATIENCE*2} ...")
    torch.manual_seed(SEED)
    mlp = MLP().to(DEVICE)
    _, mlp_ep, snaps_mlp = train_interp(mlp, T_all, Y_all, lr=best_mlp_lr,
                                        is_seq=False, patience=LSTM_PATIENCE * 2,
                                        snap_epochs=SNAP_EPOCHS)
    p_mlp = predict(mlp, T_all, is_seq=False)

    print(f"\n  Retraining LSTM lr={best_lstm_lr:.0e}  patience={LSTM_PATIENCE*2} ...")
    torch.manual_seed(SEED)
    lstm = SeqLSTM().to(DEVICE)
    init_lstm(lstm)
    _, lstm_ep, snaps_lstm = train_interp(lstm, T_all, Y_all, lr=best_lstm_lr,
                                          is_seq=True, patience=LSTM_PATIENCE * 2,
                                          snap_epochs=SNAP_EPOCHS)
    p_lstm = predict(lstm, T_all, is_seq=True)

    preds[sig_name]['MLP']  = p_mlp
    preds[sig_name]['LSTM'] = p_lstm
    mse[sig_name]['MLP']    = float(np.mean((p_mlp - y) ** 2))
    mse[sig_name]['LSTM']   = float(np.mean((p_lstm - y) ** 2))
    print(f"  MLP  fit MSE = {mse[sig_name]['MLP']:.6f}  (ep {mlp_ep})")
    print(f"  LSTM fit MSE = {mse[sig_name]['LSTM']:.6f}  (ep {lstm_ep})")
    sig_key = sig_name.lower().replace(' ', '_')
    np.savez(f"snapshots_{sig_key}_mlp.npz",  **{str(k): v for k, v in snaps_mlp.items()})
    np.savez(f"snapshots_{sig_key}_lstm.npz", **{str(k): v for k, v in snaps_lstm.items()})
    print(f"  Snapshots saved → snapshots_{sig_key}_{{mlp,lstm}}.npz")


# %%
# ── Save predictions to disk ───────────────────────────────────────────────
import json
np.save("t_np.npy", t_np)
for sig_name, y in SIGNALS.items():
    np.save(f"signal_{sig_name.lower().replace(' ', '_')}.npy", y)
    for model_name in MODEL_NAMES:
        np.save(f"pred_{sig_name.lower().replace(' ', '_')}_{model_name.lower()}.npy",
                preds[sig_name][model_name])
with open("mse_results.json", "w") as f:
    json.dump(mse, f, indent=2)
print("Predictions saved.")

# %%
# ── MSE table ──────────────────────────────────────────────────────────────
col_w  = 12
header = f"{'Signal':<16}" + "".join(f"{m:>{col_w}}" for m in MODEL_NAMES)
print("\n" + "=" * len(header))
print(header)
print("-" * len(header))
for sig_name in SIGNALS:
    row = f"{sig_name:<16}" + "".join(f"{mse[sig_name][m]:>{col_w}.6f}" for m in MODEL_NAMES)
    print(row)
print("=" * len(header))


# %%
# ── Figures ────────────────────────────────────────────────────────────────
colors = {"AR": "tab:orange", "MLP": "tab:green", "LSTM": "tab:purple"}

for sig_name, y_sig in SIGNALS.items():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    fig.suptitle(f"{sig_name} — Function Approximation  t → y(t)", fontsize=12, fontweight='bold')

    for c, model_name in enumerate(MODEL_NAMES):
        ax = axes[c]
        ax.plot(t_np, y_sig, color='steelblue', lw=2.0, label='True')
        ax.plot(t_np, preds[sig_name][model_name],
                color=colors[model_name], lw=2.0, ls='--', label='Fit')
        sig_min, sig_max = float(y_sig.min()), float(y_sig.max())
        margin = (sig_max - sig_min) * 0.2
        ax.set_ylim(sig_min - margin, sig_max + margin)
        ax.set_title(f'{model_name}   MSE={mse[sig_name][model_name]:.5f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Time')
        if c == 0:
            ax.set_ylabel('Amplitude')

    legend_els = [
        Line2D([0], [0], color='steelblue', lw=1.5, label='True signal'),
        Line2D([0], [0], color='gray',      lw=1.5, ls='--', label='Model fit'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    fname = f"figure_{sig_name.lower().replace(' ', '_')}_interp.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")

# %%
