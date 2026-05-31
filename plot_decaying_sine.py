#%%
import numpy as np
import matplotlib.pyplot as plt

#%% --- Parameters ---
A0 = 1.0       # initial amplitude
GAMMA = 0.3    # decay rate (thermal diffusion / energy dissipation)
OMEGA = 2 * np.pi * 1.0  # angular frequency (rad/s), here 1 Hz
PHI = 0.0      # phase shift (radians)

T_START = 0.0
T_END = 10.0
N_POINTS = 1000

#%% --- Signal ---
t = np.linspace(T_START, T_END, N_POINTS)
a = A0 * np.exp(-GAMMA * t) * np.sin(OMEGA * t + PHI)
envelope = A0 * np.exp(-GAMMA * t)

#%% --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(t, a, color="steelblue", linewidth=1.5, label=r"$a(t) = A_0 e^{-\gamma t} \sin(\omega t + \phi)$")
ax.plot(t, envelope, color="tomato", linewidth=1.2, linestyle="--", label=r"$\pm A_0 e^{-\gamma t}$ (envelope)")
ax.plot(t, -envelope, color="tomato", linewidth=1.2, linestyle="--")

ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax.fill_between(t, -envelope, envelope, alpha=0.08, color="tomato")

ax.set_xlabel("Time $t$", fontsize=13)
ax.set_ylabel("Amplitude $a(t)$", fontsize=13)
ax.set_title(
    rf"Decaying Sine Wave  ($A_0={A0}$, $\gamma={GAMMA}$, $\omega={OMEGA:.2f}$ rad/s, $\phi={PHI}$)",
    fontsize=13,
)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
