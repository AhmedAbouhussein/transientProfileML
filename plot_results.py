# %%
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Load saved results ──────────────────────────────────────────────────────
t_np        = np.load("t_np.npy")
SIGNALS     = ["Decay", "Chirp", "Multi-Scale"]
MODEL_NAMES   = ["AR", "MLP", "LSTM"]
DISPLAY_NAMES = {"AR": "AR", "MLP": "MLP-NN", "LSTM": "LSTM"}
colors        = {"AR": "tab:orange", "MLP": "tab:green", "LSTM": "tab:purple"}

with open("mse_results.json") as f:
    mse = json.load(f)

signals, preds = {}, {}
for sig_name in SIGNALS:
    key = sig_name.lower().replace(' ', '_')
    signals[sig_name] = np.load(f"signal_{key}.npy")
    preds[sig_name]   = {}
    for m in MODEL_NAMES:
        preds[sig_name][m] = np.load(f"pred_{key}_{m.lower()}.npy")

# Success/failure classification
SUCCESS = {
    ("Decay", "AR"), ("Decay", "MLP"), ("Decay", "LSTM"),
                     ("Chirp", "MLP"), ("Chirp", "LSTM"),
                                       ("Multi-Scale", "LSTM"),
}

LW_TRUE  = 1.0   # thin  — true signal
LW_MODEL = 2.5   # thick — model fit

legend_els = [
    Line2D([0], [0], color='steelblue', lw=LW_TRUE,  label='True signal'),
    Line2D([0], [0], color='gray',      lw=LW_MODEL, ls='--', label='Model fit'),
]

# ── Helper: draw one signal+fit cell ───────────────────────────────────────
def draw_cell(ax, sig_name, model_name, show_mse=True, equal_lw=False):
    y = signals[sig_name]
    p = preds[sig_name][model_name]
    sig_min, sig_max = float(y.min()), float(y.max())
    margin = (sig_max - sig_min) * 0.2
    p_plot = np.clip(p, sig_min - 10 * (sig_max - sig_min),
                         sig_max + 10 * (sig_max - sig_min))
    lw_t = 1.2 if equal_lw else LW_TRUE
    lw_m = 1.2 if equal_lw else LW_MODEL
    ax.plot(t_np, y, color='steelblue', lw=lw_t)
    ax.plot(t_np, p_plot, color=colors[model_name], lw=lw_m, ls='--')
    ax.set_ylim(sig_min - margin, sig_max + margin)
    ax.tick_params(labelsize=7)
    if show_mse:
        mse_val = mse[sig_name][model_name]
        mse_str = f"{mse_val:.5f}" if mse_val < 1e6 else f"{mse_val:.2e}"
        ax.set_title(f"MSE={mse_str}", fontsize=8)

def blank_cell(ax):
    ax.set_facecolor('#cccccc')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#999999')
        spine.set_linewidth(0.5)

# ── 1. Individual plots (one per signal, three models side by side) ─────────
for sig_name in SIGNALS:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    fig.suptitle(f"{sig_name} — Function Approximation  t → y(t)",
                 fontsize=12, fontweight='bold')
    for c, model_name in enumerate(MODEL_NAMES):
        ax = axes[c]
        draw_cell(ax, sig_name, model_name)
        ax.set_title(f"{model_name}   MSE={mse[sig_name][model_name]:.5f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Time")
        if c == 0:
            ax.set_ylabel("Amplitude")
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    fname = f"figure_{sig_name.lower().replace(' ', '_')}_interp.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)

# ── 2. 3×3 grid plots ───────────────────────────────────────────────────────
def make_grid(title, fname, include):
    """include: set of (sig_name, model_name) pairs to show."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for r, sig_name in enumerate(SIGNALS):
        axes[r, 0].set_ylabel(sig_name, fontsize=10, fontweight='bold', labelpad=8)
        for c, model_name in enumerate(MODEL_NAMES):
            ax = axes[r, c]
            if (sig_name, model_name) in include:
                equal = (sig_name == "Multi-Scale" and model_name == "LSTM")
                draw_cell(ax, sig_name, model_name, equal_lw=equal)
                ax.set_xlabel("Time", fontsize=7)
            else:
                blank_cell(ax)

    # Set column labels last so they aren't overwritten by draw_cell's MSE title
    for c, model_name in enumerate(MODEL_NAMES):
        axes[0, c].set_title(DISPLAY_NAMES[model_name], fontsize=11, fontweight='bold', pad=8)

    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)

make_grid("Success Cases — Model fits signal well",
          "figure_grid_success.png",
          include=SUCCESS)

make_grid("Failure Cases — Model fails to fit signal",
          "figure_grid_failure.png",
          include={(s, m) for s in SIGNALS for m in MODEL_NAMES} - SUCCESS)

# %%
