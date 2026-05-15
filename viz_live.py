"""
AEGIS-LINK :: viz_live.py
=========================

Real-time "tactical radar room" viewer.

Subscribes simultaneously to:
  * tcp://127.0.0.1:5555  -> ground truth (Julia simulator)
  * tcp://127.0.0.1:5556  -> EKF estimate (C++ tracker)

and renders, at ~30 FPS, a dark-themed scene with:
  - 3D trajectory (truth = neon green, EKF estimate = amber)
  - 3-sigma uncertainty sphere on the latest estimate
  - side telemetry panel: |err|, d^2, altitude, speed, alert flag

Designed to run on integrated GPUs (Iris Xe class): no shaders, no VTK,
just matplotlib drawing thin lines on a black canvas. CPU cost is ~5%.

Usage
-----
  # in another terminal, after `./run_demo.sh 60` is RUNNING:
  source .venv/bin/activate
  python viz_live.py

  # quit with Ctrl+C or by closing the window.
"""
from __future__ import annotations

import signal
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import os
import matplotlib
# Force a GUI backend BEFORE importing pyplot. The default in headless venvs
# is "agg" (non-interactive) which silently produces no window. Try Qt first,
# then Tk; if neither is available, give a helpful error message.
#
# WSL2 quirk: Qt's "xcb" plugin needs libxcb-cursor0 which is often missing.
# WSLg also exposes a Wayland socket -> prefer wayland there.
if "WAYLAND_DISPLAY" in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "wayland"

def _select_gui_backend():
    for be in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(be, force=True)
            return be
        except Exception:
            continue
    return None

_be = _select_gui_backend()
if _be is None or matplotlib.get_backend().lower() == "agg":
    raise SystemExit(
        "[viz_live] No interactive matplotlib backend available.\n"
        "  Install one of:  pip install PyQt5     (recommended)\n"
        "                   sudo apt install python3-tk   (system-wide)\n"
        f"  Current backend: {matplotlib.get_backend()}"
    )

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import zmq

# --------------------------------------------------------------------------
#  Wire format (mirrors shared/messages.h)
# --------------------------------------------------------------------------
_PKT_FMT  = "<IIQ6d6dHH12s"
_PKT_SIZE = struct.calcsize(_PKT_FMT)
assert _PKT_SIZE == 128

PROD_SIM     = 1
PROD_TRACKER = 2
FLAG_MANEUVER = 0x0001

ENDPOINT_TRUTH    = "tcp://127.0.0.1:5555"
ENDPOINT_ESTIMATE = "tcp://127.0.0.1:5556"


@dataclass
class Sample:
    t:        float          # seconds since first packet (per stream)
    pos:      np.ndarray     # (3,)
    vel:      np.ndarray     # (3,)
    cov_diag: np.ndarray     # (6,) variances of (px,py,pz,vx,vy,vz)
    flags:    int


def _decode(buf: bytes) -> Sample:
    f = struct.unpack(_PKT_FMT, buf)
    state = np.asarray(f[3:9],  dtype=np.float64)
    covd  = np.asarray(f[9:15], dtype=np.float64)
    return Sample(
        t=f[2] * 1e-9,
        pos=state[0:3].copy(),
        vel=state[3:6].copy(),
        cov_diag=covd,
        flags=f[16],
    )


# --------------------------------------------------------------------------
#  Background ZMQ subscribers (one thread per stream, lock-free deques)
# --------------------------------------------------------------------------
class _Stream:
    """Bounded ring of samples populated by a background thread."""
    def __init__(self, endpoint: str, maxlen: int = 6000):
        self.buf: "deque[Sample]" = deque(maxlen=maxlen)
        self._endpoint = endpoint
        self._stop = threading.Event()
        self._thr  = threading.Thread(target=self._run, daemon=True)
        self.t0: float | None = None
        self.last: Sample | None = None

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, 200)   # ms
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        while not self._stop.is_set():
            try:
                msg = sock.recv()
            except zmq.error.Again:
                continue
            except zmq.error.ZMQError:
                break
            if len(msg) != _PKT_SIZE:
                continue
            s = _decode(msg)
            if self.t0 is None:
                self.t0 = s.t
            s = Sample(s.t - self.t0, s.pos, s.vel, s.cov_diag, s.flags)
            self.buf.append(s)
            self.last = s
        sock.close(0)


# --------------------------------------------------------------------------
#  "Tactical" matplotlib styling
# --------------------------------------------------------------------------
TRUTH_COLOR    = "#00ff9a"   # neon green
ESTIMATE_COLOR = "#ffb000"   # amber
GRID_COLOR     = "#1f3a3a"
ALERT_COLOR    = "#ff2e63"
TEXT_COLOR     = "#9bd1d1"

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "#000a0a",
        "axes.facecolor":    "#000a0a",
        "savefig.facecolor": "#000a0a",
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "grid.linestyle":    ":",
        "font.family":       "monospace",
        "font.size":         9,
    })


# --------------------------------------------------------------------------
#  Main viewer
# --------------------------------------------------------------------------
class LiveViewer:
    def __init__(self, history_s: float = 30.0, fps: int = 30):
        self.history_s = history_s
        self.fps       = fps

        self.truth = _Stream(ENDPOINT_TRUTH)
        self.est   = _Stream(ENDPOINT_ESTIMATE)

        _setup_style()
        self.fig = plt.figure(figsize=(13, 7.5), num="AEGIS-LINK :: live")
        gs = GridSpec(3, 3, figure=self.fig,
                      width_ratios=[2.4, 1, 1],
                      height_ratios=[1.1, 1.1, 1.1],
                      hspace=0.35, wspace=0.30,
                      left=0.04, right=0.985, top=0.94, bottom=0.07)

        # 3D scene (left, all rows)
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self._init_3d_axes()

        # right column: 3 stacked panels
        self.ax_err  = self.fig.add_subplot(gs[0, 1:])
        self.ax_d2   = self.fig.add_subplot(gs[1, 1:])
        self.ax_alt  = self.fig.add_subplot(gs[2, 1:])
        for ax, ylabel in [(self.ax_err, "|err| [m]"),
                           (self.ax_d2,  r"$d^2$ (Mahalanobis)"),
                           (self.ax_alt, "altitude [m]")]:
            ax.set_facecolor("#000a0a")
            ax.grid(True, alpha=0.4)
            ax.set_ylabel(ylabel)
            ax.tick_params(labelsize=8)
        self.ax_d2.set_yscale("log")
        self.ax_alt.set_xlabel("t [s]")

        # threshold lines on d^2 panel
        self.ax_d2.axhline(16.81, color=ALERT_COLOR, ls="--", lw=0.8, alpha=0.7)
        self.ax_d2.axhline(22.46, color=ALERT_COLOR, ls=":",  lw=0.8, alpha=0.7)

        # HUD text (top-left of 3D)
        self.hud = self.fig.text(
            0.045, 0.955, "AEGIS-LINK / TACTICAL  ::  awaiting telemetry...",
            color=TRUTH_COLOR, fontsize=10, fontweight="bold")

        self.alert_box = self.fig.text(
            0.045, 0.02, "", color=ALERT_COLOR, fontsize=11, fontweight="bold")

        # artists we will mutate
        self._truth_line, = self.ax3d.plot([], [], [], color=TRUTH_COLOR,
                                           lw=1.4, label="truth")
        self._est_line,   = self.ax3d.plot([], [], [], color=ESTIMATE_COLOR,
                                           lw=1.0, alpha=0.95, label="EKF")
        self._truth_head  = self.ax3d.scatter([], [], [], c=TRUTH_COLOR, s=35)
        self._est_head    = self.ax3d.scatter([], [], [], c=ESTIMATE_COLOR, s=18)
        self._sigma_surf  = None  # rebuilt each frame

        self.ax3d.legend(loc="upper right", framealpha=0.0,
                         labelcolor=TEXT_COLOR, fontsize=9)

        # right-panel artists
        self._err_line, = self.ax_err.plot([], [], color=ESTIMATE_COLOR, lw=1.0)
        self._d2_line,  = self.ax_d2.plot([], [], color=ESTIMATE_COLOR, lw=1.0)
        self._alt_line_t, = self.ax_alt.plot([], [], color=TRUTH_COLOR,    lw=1.0)
        self._alt_line_e, = self.ax_alt.plot([], [], color=ESTIMATE_COLOR, lw=0.9, alpha=0.9)

        # signals
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

    # ------------------------------------------------------------------
    def _init_3d_axes(self):
        ax = self.ax3d
        ax.set_facecolor("#000a0a")
        ax.set_title("3D scene  ::  truth / estimate / 3-sigma sphere",
                     pad=12)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.0, 0.04, 0.04, 1.0))
            axis.pane.set_edgecolor(GRID_COLOR)
            axis._axinfo["grid"]["color"]     = GRID_COLOR
            axis._axinfo["grid"]["linestyle"] = ":"
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

    # ------------------------------------------------------------------
    def _autorange_3d(self, pts: np.ndarray):
        if pts.size == 0:
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ctr  = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        span = max(span, 50.0) * 0.6 + 5.0
        self.ax3d.set_xlim(ctr[0] - span, ctr[0] + span)
        self.ax3d.set_ylim(ctr[1] - span, ctr[1] + span)
        self.ax3d.set_zlim(max(0.0, ctr[2] - span), ctr[2] + span)

    # ------------------------------------------------------------------
    def _build_sigma_sphere(self, center, sigma_xyz):
        """3-sigma sphere proxy (uses max(sigma_xyz) for radius -> conservative)."""
        r = 3.0 * float(np.max(sigma_xyz))
        u = np.linspace(0, 2*np.pi, 18)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z, r

    # ------------------------------------------------------------------
    def _redraw(self, _frame):
        # ---- 3D scene ----
        if self.truth.buf:
            tp = np.array([s.pos for s in self.truth.buf])
            self._truth_line.set_data_3d(tp[:, 0], tp[:, 1], tp[:, 2])
            self._truth_head._offsets3d = ([tp[-1, 0]], [tp[-1, 1]], [tp[-1, 2]])
        else:
            tp = np.empty((0, 3))

        if self.est.buf:
            ep = np.array([s.pos for s in self.est.buf])
            self._est_line.set_data_3d(ep[:, 0], ep[:, 1], ep[:, 2])
            self._est_head._offsets3d = ([ep[-1, 0]], [ep[-1, 1]], [ep[-1, 2]])
        else:
            ep = np.empty((0, 3))

        all_pts = np.vstack([tp, ep]) if (tp.size or ep.size) else np.empty((0, 3))
        self._autorange_3d(all_pts)

        # 3-sigma sphere on latest estimate
        if self._sigma_surf is not None:
            try:
                self._sigma_surf.remove()
            except Exception:
                pass
            self._sigma_surf = None
        if self.est.last is not None:
            sig_xyz = np.sqrt(self.est.last.cov_diag[0:3])
            sx, sy, sz, r = self._build_sigma_sphere(self.est.last.pos, sig_xyz)
            self._sigma_surf = self.ax3d.plot_surface(
                sx, sy, sz, color=ESTIMATE_COLOR, alpha=0.10,
                linewidth=0, antialiased=False, shade=False)

        # ---- right panels: pair samples by closest timestamp ----
        if self.truth.buf and self.est.buf:
            tt = np.array([s.t for s in self.truth.buf])
            tp = np.array([s.pos for s in self.truth.buf])
            te = np.array([s.t for s in self.est.buf])
            ep = np.array([s.pos for s in self.est.buf])
            cv = np.array([s.cov_diag[0:3] for s in self.est.buf])

            # interpolate truth at estimate timestamps
            tx = np.interp(te, tt, tp[:, 0])
            ty = np.interp(te, tt, tp[:, 1])
            tz = np.interp(te, tt, tp[:, 2])
            err = np.sqrt((ep[:, 0]-tx)**2 + (ep[:, 1]-ty)**2 + (ep[:, 2]-tz)**2)

            # quick d^2 using only positional cov diag + R=0.05
            R_pos = 0.05**2
            S_inv_diag = 1.0 / (cv + R_pos)
            d2 = ((ep[:, 0]-tx)**2 * S_inv_diag[:, 0]
                + (ep[:, 1]-ty)**2 * S_inv_diag[:, 1]
                + (ep[:, 2]-tz)**2 * S_inv_diag[:, 2])

            # window
            t_now = te[-1]
            mask  = te >= (t_now - self.history_s)
            self._err_line.set_data(te[mask], err[mask])
            self._d2_line.set_data(te[mask],  np.maximum(d2[mask], 1e-3))
            self._alt_line_t.set_data(tt[tt >= t_now - self.history_s],
                                      tp[tt >= t_now - self.history_s, 2])
            self._alt_line_e.set_data(te[mask], ep[mask, 2])

            for ax in (self.ax_err, self.ax_d2, self.ax_alt):
                ax.set_xlim(max(0.0, t_now - self.history_s), t_now + 0.5)
                ax.relim(); ax.autoscale_view(scalex=False, scaley=True)

        # ---- HUD ----
        if self.est.last is not None and self.truth.last is not None:
            spd = float(np.linalg.norm(self.est.last.vel))
            err_now = float(np.linalg.norm(self.est.last.pos - self.truth.last.pos))
            sigma_pos = float(np.sqrt(np.sum(self.est.last.cov_diag[0:3])))
            n_t = len(self.truth.buf)
            n_e = len(self.est.buf)
            self.hud.set_text(
                f"AEGIS-LINK / TACTICAL  ::  "
                f"truth pkts {n_t:5d}   est pkts {n_e:5d}   "
                f"|v| {spd:6.1f} m/s   |err| {err_now*1000:7.1f} mm   "
                f"sigma_p {sigma_pos*1000:7.1f} mm")

            if self.est.last.flags & FLAG_MANEUVER:
                self.alert_box.set_text("** MANEUVER DETECTED **")
            else:
                self.alert_box.set_text("")
        else:
            self.hud.set_text("AEGIS-LINK / TACTICAL  ::  awaiting telemetry...")

        return ()

    # ------------------------------------------------------------------
    def _shutdown(self, *_):
        self.truth.stop()
        self.est.stop()
        plt.close(self.fig)

    def run(self):
        self.truth.start()
        self.est.start()
        interval_ms = int(1000 / self.fps)
        # cache_frame_data=False because we render off live deques
        self._anim = animation.FuncAnimation(
            self.fig, self._redraw, interval=interval_ms,
            blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self._shutdown()


def main():
    print("AEGIS-LINK live viewer")
    print(f"  truth   <- {ENDPOINT_TRUTH}")
    print(f"  estimate<- {ENDPOINT_ESTIMATE}")
    print("  close the window or Ctrl+C to quit.")
    LiveViewer(history_s=30.0, fps=30).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
