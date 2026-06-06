# %%
import numpy as np
import matplotlib.pyplot as plt

L = 1000
t = np.linspace(0, 10, L)

# %%
# Signal 1: Decaying Sine
decay = np.exp(-0.3 * t) * np.sin(2 * np.pi * 1.5 * t)

# Signal 2: Chirp (linearly increasing frequency)
f0, f1 = 0.5, 2.0
phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t**2)
chirp = np.sin(phase)

# Signal 3: Multi-Scale Sine (low-freq carrier + high-freq ripple)
carrier = np.sin(2 * np.pi * 0.15 * t)
ripple = 0.6 * np.sin(2 * np.pi * 5.0 * t)
multiscale = carrier + ripple

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, decay)
axes[0].set_title("Signal 1: Decaying Sine")
axes[0].set_ylabel("Amplitude")

axes[1].plot(t, chirp, color="tab:orange")
axes[1].set_title("Signal 2: Chirp")
axes[1].set_ylabel("Amplitude")

axes[2].plot(t, multiscale, color="tab:green")
axes[2].set_title("Signal 3: Multi-Scale Sine")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time")

plt.tight_layout()
plt.show()

# %%
