"""Generate docs/img/hero.png + docs/img/demo.gif from run.csv."""
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive: render to file, no GUI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

os.makedirs("docs/img", exist_ok=True)

df = pd.read_csv("run.csv")
df["t_s"] = (df.ts_ns - df.ts_ns.iloc[0]) * 1e-9
for c in ["x", "y", "z"]:
    df[f"err_{c}"] = df[f"p{c}"] - df[f"e{c}"]
df["err_norm"] = np.linalg.norm(df[["err_x", "err_y", "err_z"]].values, axis=1)

# ---- Static hero screenshot --------------------------------------------------
fig = plt.figure(figsize=(13, 5.8), facecolor="#000a0a")
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1],
                      hspace=0.35, wspace=0.22, left=0.04, right=0.985,
                      top=0.92, bottom=0.10)
ax3 = fig.add_subplot(gs[:, 0], projection="3d")
ax3.set_facecolor("#000a0a")
ax3.plot(df.px, df.py, df.pz, color="#00ff9a", lw=1.6, label="truth (sim)")
ax3.plot(df.ex, df.ey, df.ez, color="#ffb000", lw=1.0, alpha=0.95, label="EKF estimate")
ax3.scatter([df.px.iloc[0]], [df.py.iloc[0]], [df.pz.iloc[0]], c="#00ff9a", s=60, label="launch")
ax3.scatter([df.px.iloc[-1]], [df.py.iloc[-1]], [df.pz.iloc[-1]], c="#ff2e63", s=60, label="impact")
ax3.set_xlabel("x [m]", color="#9bd1d1")
ax3.set_ylabel("y [m]", color="#9bd1d1")
ax3.set_zlabel("z [m]", color="#9bd1d1")
ax3.tick_params(colors="#9bd1d1")
ax3.set_title("AEGIS-LINK :: 3D ballistic arc - truth vs EKF estimate",
              color="#9bd1d1", pad=12)
ax3.legend(facecolor="#000a0a", edgecolor="#1f3a3a",
           labelcolor="#9bd1d1", loc="upper left")
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.pane.set_facecolor((0.0, 0.04, 0.04, 1.0))
    axis.pane.set_edgecolor("#1f3a3a")

ax_err = fig.add_subplot(gs[0, 1])
ax_err.set_facecolor("#000a0a")
ax_err.plot(df.t_s, df.err_norm * 1000, color="#ffb000", lw=0.9)
ax_err.axvspan(0, 10, color="orange", alpha=0.10, label="warm-up")
ax_err.set_yscale("log")
ax_err.set_ylabel("|err| [mm]", color="#9bd1d1")
ax_err.tick_params(colors="#9bd1d1")
ax_err.grid(alpha=0.3, color="#1f3a3a")
ax_err.legend(facecolor="#000a0a", edgecolor="#1f3a3a",
              labelcolor="#9bd1d1", fontsize=8, loc="upper right")
ax_err.set_title("Position error (truth - EKF)", color="#9bd1d1", fontsize=10)

ax_d2 = fig.add_subplot(gs[1, 1])
ax_d2.set_facecolor("#000a0a")
ax_d2.plot(df.t_s, np.maximum(df.d2, 1e-2), color="#00ff9a", lw=0.9)
ax_d2.axhline(16.81, color="#ff2e63", ls="--", lw=0.8, alpha=0.8)
ax_d2.set_yscale("log")
ax_d2.set_xlabel("t [s]", color="#9bd1d1")
ax_d2.set_ylabel(r"$d^2$", color="#9bd1d1")
ax_d2.tick_params(colors="#9bd1d1")
ax_d2.grid(alpha=0.3, color="#1f3a3a")
ax_d2.set_title(r"Mahalanobis $d^2$ + $\chi^2_{0.99}$ gate",
                color="#9bd1d1", fontsize=10)

fig.savefig("docs/img/hero.png", dpi=110, facecolor="#000a0a")
plt.close(fig)
print(f"hero.png: {os.path.getsize('docs/img/hero.png') / 1024:.0f} KB")

# ---- Animated GIF ------------------------------------------------------------
FPS = 20
DUR = 9.0
TRAIL = 220
n_frames = int(FPS * DUR)
idx = np.linspace(0, len(df) - 1, n_frames).astype(int)

fig = plt.figure(figsize=(10, 5.5), facecolor="#000a0a")
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.22,
                      left=0.05, right=0.985, top=0.93, bottom=0.10)
ax3 = fig.add_subplot(gs[0, 0], projection="3d")
ax3.set_facecolor("#000a0a")
axe = fig.add_subplot(gs[0, 1])
axe.set_facecolor("#000a0a")

ax3.set_xlim(df.px.min(), df.px.max())
ax3.set_ylim(df.py.min(), df.py.max())
ax3.set_zlim(0, df.pz.max() * 1.05)
ax3.set_xlabel("x [m]", color="#9bd1d1")
ax3.set_ylabel("y [m]", color="#9bd1d1")
ax3.set_zlabel("z [m]", color="#9bd1d1")
ax3.tick_params(colors="#9bd1d1")
ax3.set_title("Replay 3D - truth (green) / EKF (amber)", color="#9bd1d1", pad=10)
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.pane.set_facecolor((0.0, 0.04, 0.04, 1.0))
    axis.pane.set_edgecolor("#1f3a3a")

axe.set_xlim(0, df.t_s.iloc[-1])
axe.set_ylim(1, max(2, df.err_norm.iloc[200:].quantile(0.999) * 1000 * 1.2))
axe.set_yscale("log")
axe.set_xlabel("t [s]", color="#9bd1d1")
axe.set_ylabel("|err| [mm] (log)", color="#9bd1d1")
axe.tick_params(colors="#9bd1d1")
axe.grid(alpha=0.3, color="#1f3a3a")
axe.set_title("Position error", color="#9bd1d1")

l_truth, = ax3.plot([], [], [], color="#00ff9a", lw=1.5)
l_est, = ax3.plot([], [], [], color="#ffb000", lw=1.0, alpha=0.95)
h_truth = ax3.scatter([df.px.iloc[0]], [df.py.iloc[0]], [df.pz.iloc[0]], c="#00ff9a", s=45)
h_est = ax3.scatter([df.ex.iloc[0]], [df.ey.iloc[0]], [df.ez.iloc[0]], c="#ffb000", s=22)
l_err, = axe.plot([], [], color="#ffb000", lw=1.0)
hud = fig.text(0.02, 0.96, "", color="#00ff9a", fontsize=10,
               family="monospace", fontweight="bold")


def frame(k):
    i = idx[k]
    j = max(0, i - TRAIL)
    sub = df.iloc[j:i + 1]
    l_truth.set_data_3d(sub.px.values, sub.py.values, sub.pz.values)
    l_est.set_data_3d(sub.ex.values, sub.ey.values, sub.ez.values)
    h_truth._offsets3d = ([sub.px.iloc[-1]], [sub.py.iloc[-1]], [sub.pz.iloc[-1]])
    h_est._offsets3d = ([sub.ex.iloc[-1]], [sub.ey.iloc[-1]], [sub.ez.iloc[-1]])
    l_err.set_data(df.t_s.iloc[:i + 1].values,
                   np.maximum(df.err_norm.iloc[:i + 1].values * 1000, 0.5))
    hud.set_text(f"t={df.t_s.iloc[i]:5.1f}s  "
                 f"|err|={df.err_norm.iloc[i] * 1000:7.1f} mm  "
                 f"d2={df.d2.iloc[i]:7.2f}")
    return l_truth, l_est, l_err


print(f"Rendering {n_frames} frames -> docs/img/demo.gif ...")
anim = FuncAnimation(fig, frame, frames=n_frames,
                     interval=1000 / FPS, blit=False)
anim.save("docs/img/demo.gif", writer=PillowWriter(fps=FPS), dpi=80)
plt.close(fig)
print(f"demo.gif:  {os.path.getsize('docs/img/demo.gif') / 1024:.0f} KB")
