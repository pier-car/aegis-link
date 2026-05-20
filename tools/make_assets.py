"""Generate docs/img/hero.png + docs/img/demo.gif from run.csv (and
optionally engagement.csv).

Usage
-----
    python tools/make_assets.py                       # auto-detect both CSVs
    python tools/make_assets.py --no-engagement       # EKF-only assets
    python tools/make_assets.py --run path/to/run.csv \
        --engagement path/to/engagement.csv --out-dir docs/img

Schema expected
---------------
* run.csv          (from ai_orchestrator)
    ts_ns, packet_id, d2, maneuver, lock_state, sigma_p,
    px, py, pz, ex, ey, ez
* engagement.csv   (from engagement_engine, optional)
    ts_ns, state, ix, iy, iz, vx, vy, vz,
    range, closing, pred_miss, fuel_frac, a_cmd
  where `state` is one of IDLE / ARMED / ENGAGED / KILL / MISS.

The script degrades gracefully: if `engagement.csv` is missing or the
`--no-engagement` flag is passed, it produces the legacy EKF-only assets
so older runs and CI keep working.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive: render to file, no GUI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --------------------------------------------------------------------------
#  Palette (kept in sync with viz_live.py)
# --------------------------------------------------------------------------
TRUTH_COLOR       = "#00ff9a"   # neon green
ESTIMATE_COLOR    = "#ffb000"   # amber
INTERCEPTOR_COLOR = "#00d9ff"   # cyan
LOS_COLOR         = "#ff66cc"   # magenta
LOCK_COLOR        = "#ffe14a"   # yellow (lock-on / ARMED)
ENGAGE_COLOR      = "#00d9ff"   # cyan (ENGAGED window)
KILL_COLOR        = "#ff2e63"   # red
TEXT_COLOR        = "#9bd1d1"
GRID_COLOR        = "#1f3a3a"
BG_COLOR          = "#000a0a"
PANE_COLOR        = (0.0, 0.04, 0.04, 1.0)


# --------------------------------------------------------------------------
#  Engagement helpers
# --------------------------------------------------------------------------
def _load_engagement(path: str | None) -> pd.DataFrame | None:
    """Return engagement DataFrame or None if unavailable / unusable."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        eng = pd.read_csv(path)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[assets] could not parse {path}: {exc}", file=sys.stderr)
        return None
    needed = {"ts_ns", "state", "ix", "iy", "iz"}
    if not needed.issubset(eng.columns):
        print(f"[assets] {path} missing columns "
              f"{sorted(needed - set(eng.columns))}, ignoring",
              file=sys.stderr)
        return None
    return eng


def _engagement_summary(eng: pd.DataFrame) -> dict:
    """Outcome / CPA / engagement window timestamps in ns."""
    out = {"outcome": "NO_LOCK", "cpa_m": float("nan"),
           "t_locked_ns": None, "t_engaged_ns": None,
           "t_terminal_ns": None}
    if eng is None or eng.empty:
        return out
    state = eng["state"].astype(str)
    if (state == "KILL").any():
        out["outcome"] = "KILL"
        out["t_terminal_ns"] = int(eng.loc[state == "KILL", "ts_ns"].iloc[0])
    elif (state == "MISS").any():
        out["outcome"] = "MISS"
        out["t_terminal_ns"] = int(eng.loc[state == "MISS", "ts_ns"].iloc[0])
    if (state == "ARMED").any():
        out["t_locked_ns"]  = int(eng.loc[state == "ARMED", "ts_ns"].iloc[0])
    if (state == "ENGAGED").any():
        out["t_engaged_ns"] = int(eng.loc[state == "ENGAGED", "ts_ns"].iloc[0])
    if "range" in eng.columns:
        r = pd.to_numeric(eng["range"], errors="coerce")
        r = r[np.isfinite(r) & (r > 0)]
        if not r.empty:
            out["cpa_m"] = float(r.min())
    return out


def _shade_phase(ax, df_run, eng_summary, history_t0_s=0.0):
    """Shade LOCKED (yellow) and ENGAGED (cyan) windows on a time-axis panel.

    `df_run` must contain a `t_s` column (seconds since first run.csv sample)
    and the original `ts_ns`. We map engagement timestamps to the run timeline
    via linear interpolation on ts_ns -> t_s so the bands line up with the
    EKF error/d^2 traces drawn on the same axis.
    """
    if eng_summary is None or df_run.empty:
        return
    ts_run = df_run["ts_ns"].to_numpy(dtype=np.int64)
    t_run  = df_run["t_s"].to_numpy(dtype=np.float64)

    def _to_t_s(ts_ns):
        if ts_ns is None:
            return None
        ts_ns = int(ts_ns)
        if ts_ns <= ts_run[0]:
            return float(t_run[0])
        if ts_ns >= ts_run[-1]:
            return float(t_run[-1])
        return float(np.interp(ts_ns, ts_run, t_run))

    t_lock = _to_t_s(eng_summary["t_locked_ns"])
    t_eng  = _to_t_s(eng_summary["t_engaged_ns"])
    t_end  = _to_t_s(eng_summary["t_terminal_ns"]) or float(t_run[-1])
    label_used = False
    if t_lock is not None:
        t_lock_end = t_eng if t_eng is not None else t_end
        if t_lock_end > t_lock:
            ax.axvspan(t_lock, t_lock_end, color=LOCK_COLOR,
                       alpha=0.10, label="LOCKED")
            label_used = True
    if t_eng is not None and t_end > t_eng:
        ax.axvspan(t_eng, t_end, color=ENGAGE_COLOR,
                   alpha=0.12, label="ENGAGED")
        label_used = True
    if eng_summary["t_terminal_ns"] is not None:
        ax.axvline(t_end, color=KILL_COLOR, lw=0.9, alpha=0.8,
                   label=eng_summary["outcome"])
        label_used = True
    return label_used


# --------------------------------------------------------------------------
#  Hero PNG
# --------------------------------------------------------------------------
def make_hero(df: pd.DataFrame, eng: pd.DataFrame | None,
              out_path: str) -> None:
    summary = _engagement_summary(eng) if eng is not None else None

    fig = plt.figure(figsize=(13, 5.8), facecolor=BG_COLOR)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.22, left=0.04, right=0.985,
                          top=0.92, bottom=0.10)

    # ---- 3D arc -----------------------------------------------------------
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.set_facecolor(BG_COLOR)
    ax3.plot(df.px, df.py, df.pz, color=TRUTH_COLOR, lw=1.6,
             label="truth (sim)")
    ax3.plot(df.ex, df.ey, df.ez, color=ESTIMATE_COLOR, lw=1.0, alpha=0.95,
             label="EKF estimate")
    ax3.scatter([df.px.iloc[0]], [df.py.iloc[0]], [df.pz.iloc[0]],
                c=TRUTH_COLOR, s=60, label="launch")
    ax3.scatter([df.px.iloc[-1]], [df.py.iloc[-1]], [df.pz.iloc[-1]],
                c=KILL_COLOR, s=60, label="impact (target)")

    title_extra = ""
    if eng is not None and not eng.empty:
        eng_pts = eng[["ix", "iy", "iz"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if not eng_pts.empty:
            ax3.plot(eng_pts.ix, eng_pts.iy, eng_pts.iz,
                     color=INTERCEPTOR_COLOR, lw=1.1, alpha=0.95,
                     label="interceptor")
            ax3.scatter([eng_pts.ix.iloc[-1]], [eng_pts.iy.iloc[-1]],
                        [eng_pts.iz.iloc[-1]],
                        c=INTERCEPTOR_COLOR, s=55, marker="^",
                        label="interceptor end")
            # LOS at terminal: interceptor end -> target end (truth)
            ax3.plot([eng_pts.ix.iloc[-1], df.px.iloc[-1]],
                     [eng_pts.iy.iloc[-1], df.py.iloc[-1]],
                     [eng_pts.iz.iloc[-1], df.pz.iloc[-1]],
                     color=LOS_COLOR, lw=0.8, ls=":",
                     alpha=0.7, label="LOS @ terminal")
        if summary and summary["outcome"] in ("KILL", "MISS"):
            cpa = summary["cpa_m"]
            cpa_txt = f"CPA={cpa:.2f} m" if np.isfinite(cpa) else ""
            title_extra = f"  ::  FC: {summary['outcome']}   {cpa_txt}"

    ax3.set_xlabel("x [m]", color=TEXT_COLOR)
    ax3.set_ylabel("y [m]", color=TEXT_COLOR)
    ax3.set_zlabel("z [m]", color=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.set_title(
        "AEGIS-LINK :: track -> lock -> engage" + title_extra,
        color=TEXT_COLOR, pad=12)
    ax3.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, loc="upper left", fontsize=8)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor(GRID_COLOR)

    # ---- Error panel ------------------------------------------------------
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.set_facecolor(BG_COLOR)
    ax_err.plot(df.t_s, df.err_norm * 1000, color=ESTIMATE_COLOR, lw=0.9)
    ax_err.axvspan(0, 10, color="orange", alpha=0.10, label="warm-up")
    if summary is not None:
        _shade_phase(ax_err, df, summary)
    ax_err.set_yscale("log")
    ax_err.set_ylabel("|err| [mm]", color=TEXT_COLOR)
    ax_err.tick_params(colors=TEXT_COLOR)
    ax_err.grid(alpha=0.3, color=GRID_COLOR)
    ax_err.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8, loc="upper right")
    ax_err.set_title("Position error (truth - EKF)",
                     color=TEXT_COLOR, fontsize=10)

    # ---- d^2 panel --------------------------------------------------------
    ax_d2 = fig.add_subplot(gs[1, 1])
    ax_d2.set_facecolor(BG_COLOR)
    ax_d2.plot(df.t_s, np.maximum(df.d2, 1e-2), color=TRUTH_COLOR, lw=0.9)
    ax_d2.axhline(16.81, color=KILL_COLOR, ls="--", lw=0.8, alpha=0.8)
    if summary is not None:
        _shade_phase(ax_d2, df, summary)
    ax_d2.set_yscale("log")
    ax_d2.set_xlabel("t [s]", color=TEXT_COLOR)
    ax_d2.set_ylabel(r"$d^2$", color=TEXT_COLOR)
    ax_d2.tick_params(colors=TEXT_COLOR)
    ax_d2.grid(alpha=0.3, color=GRID_COLOR)
    ax_d2.set_title(r"Mahalanobis $d^2$ + $\chi^2_{0.99}$ gate",
                    color=TEXT_COLOR, fontsize=10)

    fig.savefig(out_path, dpi=110, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"hero.png: {os.path.getsize(out_path) / 1024:.0f} KB"
          + (f"  [engagement overlay: {summary['outcome']}]"
             if summary else "  [EKF-only]"))


# --------------------------------------------------------------------------
#  Animated GIF
# --------------------------------------------------------------------------
def make_gif(df: pd.DataFrame, eng: pd.DataFrame | None,
             out_path: str) -> None:
    FPS = 20
    DUR = 9.0
    TRAIL = 220
    n_frames = int(FPS * DUR)
    idx = np.linspace(0, len(df) - 1, n_frames).astype(int)

    # Pre-align interceptor samples to run frames via ts_ns.
    eng_ix = eng_iy = eng_iz = None
    eng_state_per_frame = None
    eng_meta_per_frame = None     # (range, pred_miss, fc_state) per frame
    eng_first_idx = None          # first run-frame index where ENGAGED begins
    summary = None
    if eng is not None and not eng.empty:
        summary = _engagement_summary(eng)
        eng_sorted = eng.sort_values("ts_ns").reset_index(drop=True)
        eng_ts = eng_sorted["ts_ns"].to_numpy(dtype=np.int64)
        run_ts = df["ts_ns"].to_numpy(dtype=np.int64)
        # nearest engagement row at-or-before each run timestamp
        pos = np.searchsorted(eng_ts, run_ts, side="right") - 1
        pos = np.clip(pos, 0, len(eng_sorted) - 1)
        eng_ix = eng_sorted["ix"].to_numpy()[pos]
        eng_iy = eng_sorted["iy"].to_numpy()[pos]
        eng_iz = eng_sorted["iz"].to_numpy()[pos]
        eng_state_per_frame = eng_sorted["state"].to_numpy()[pos]
        # mask out frames *before* the first engagement row appears
        invalid = run_ts < eng_ts[0]
        if invalid.any():
            eng_state_per_frame = eng_state_per_frame.copy()
            eng_state_per_frame[invalid] = "IDLE"
        meta_range = (pd.to_numeric(eng_sorted.get("range"),
                                    errors="coerce").to_numpy()[pos]
                      if "range" in eng_sorted.columns else
                      np.full(len(pos), np.nan))
        meta_pmis  = (pd.to_numeric(eng_sorted.get("pred_miss"),
                                    errors="coerce").to_numpy()[pos]
                      if "pred_miss" in eng_sorted.columns else
                      np.full(len(pos), np.nan))
        eng_meta_per_frame = (meta_range, meta_pmis)
        engaged_mask = (eng_state_per_frame == "ENGAGED")
        if engaged_mask.any():
            eng_first_idx = int(np.argmax(engaged_mask))

    fig = plt.figure(figsize=(10, 5.5), facecolor=BG_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.22,
                          left=0.05, right=0.985, top=0.93, bottom=0.10)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.set_facecolor(BG_COLOR)
    axe = fig.add_subplot(gs[0, 1])
    axe.set_facecolor(BG_COLOR)

    # 3D limits: include interceptor extent so it stays in frame.
    all_x = [df.px.min(), df.px.max()]
    all_y = [df.py.min(), df.py.max()]
    all_z = [0.0,         df.pz.max() * 1.05]
    if eng_ix is not None:
        finite = np.isfinite(eng_ix) & np.isfinite(eng_iy) & np.isfinite(eng_iz)
        if finite.any():
            all_x = [min(all_x[0], np.nanmin(eng_ix[finite])),
                     max(all_x[1], np.nanmax(eng_ix[finite]))]
            all_y = [min(all_y[0], np.nanmin(eng_iy[finite])),
                     max(all_y[1], np.nanmax(eng_iy[finite]))]
            all_z = [min(all_z[0], np.nanmin(eng_iz[finite])),
                     max(all_z[1], np.nanmax(eng_iz[finite]) * 1.05)]
    ax3.set_xlim(*all_x); ax3.set_ylim(*all_y); ax3.set_zlim(*all_z)
    ax3.set_xlabel("x [m]", color=TEXT_COLOR)
    ax3.set_ylabel("y [m]", color=TEXT_COLOR)
    ax3.set_zlabel("z [m]", color=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    title = "Replay 3D - truth (green) / EKF (amber)"
    if eng_ix is not None:
        title += " / interceptor (cyan)"
    ax3.set_title(title, color=TEXT_COLOR, pad=10)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor(GRID_COLOR)

    axe.set_xlim(0, df.t_s.iloc[-1])
    axe.set_ylim(1, max(2, df.err_norm.iloc[200:].quantile(0.999) * 1000 * 1.2))
    axe.set_yscale("log")
    axe.set_xlabel("t [s]", color=TEXT_COLOR)
    axe.set_ylabel("|err| [mm] (log)", color=TEXT_COLOR)
    axe.tick_params(colors=TEXT_COLOR)
    axe.grid(alpha=0.3, color=GRID_COLOR)
    axe.set_title("Position error", color=TEXT_COLOR)
    if summary is not None:
        _shade_phase(axe, df, summary)

    l_truth, = ax3.plot([], [], [], color=TRUTH_COLOR, lw=1.5)
    l_est,   = ax3.plot([], [], [], color=ESTIMATE_COLOR, lw=1.0, alpha=0.95)
    h_truth  = ax3.scatter([df.px.iloc[0]], [df.py.iloc[0]], [df.pz.iloc[0]],
                           c=TRUTH_COLOR, s=45)
    h_est    = ax3.scatter([df.ex.iloc[0]], [df.ey.iloc[0]], [df.ez.iloc[0]],
                           c=ESTIMATE_COLOR, s=22)
    l_intc   = h_intc = l_los = None
    if eng_ix is not None:
        l_intc, = ax3.plot([], [], [], color=INTERCEPTOR_COLOR, lw=1.2,
                           alpha=0.95)
        h_intc  = ax3.scatter([eng_ix[0]], [eng_iy[0]], [eng_iz[0]],
                              c=INTERCEPTOR_COLOR, s=35, marker="^")
        l_los,  = ax3.plot([], [], [], color=LOS_COLOR, lw=0.7, ls=":",
                           alpha=0.6)
    l_err,   = axe.plot([], [], color=ESTIMATE_COLOR, lw=1.0)
    hud = fig.text(0.02, 0.96, "", color=TRUTH_COLOR, fontsize=10,
                   family="monospace", fontweight="bold")
    banner = fig.text(0.02, 0.02, "", color=TEXT_COLOR, fontsize=10,
                      family="monospace", fontweight="bold")

    def frame(k):
        i = idx[k]
        j = max(0, i - TRAIL)
        sub = df.iloc[j:i + 1]
        l_truth.set_data_3d(sub.px.values, sub.py.values, sub.pz.values)
        l_est.set_data_3d(sub.ex.values, sub.ey.values, sub.ez.values)
        h_truth._offsets3d = ([sub.px.iloc[-1]], [sub.py.iloc[-1]],
                              [sub.pz.iloc[-1]])
        h_est._offsets3d   = ([sub.ex.iloc[-1]], [sub.ey.iloc[-1]],
                              [sub.ez.iloc[-1]])
        l_err.set_data(df.t_s.iloc[:i + 1].values,
                       np.maximum(df.err_norm.iloc[:i + 1].values * 1000, 0.5))
        fc_state = "SEARCHING"
        if eng_ix is not None:
            # Only show the interceptor once it has actually launched.
            if eng_first_idx is not None and i >= eng_first_idx:
                ji = max(eng_first_idx, i - TRAIL)
                l_intc.set_data_3d(eng_ix[ji:i + 1], eng_iy[ji:i + 1],
                                   eng_iz[ji:i + 1])
                h_intc._offsets3d = ([eng_ix[i]], [eng_iy[i]], [eng_iz[i]])
                # LOS line from interceptor head to current EKF estimate.
                l_los.set_data_3d([eng_ix[i], df.ex.iloc[i]],
                                  [eng_iy[i], df.ey.iloc[i]],
                                  [eng_iz[i], df.ez.iloc[i]])
            else:
                l_intc.set_data_3d([], [], [])
                h_intc._offsets3d = ([], [], [])
                l_los.set_data_3d([], [], [])
            fc_state = str(eng_state_per_frame[i])

        hud_text = (f"t={df.t_s.iloc[i]:5.1f}s  "
                    f"|err|={df.err_norm.iloc[i] * 1000:7.1f} mm  "
                    f"d2={df.d2.iloc[i]:7.2f}")
        if eng_meta_per_frame is not None and fc_state == "ENGAGED":
            rng, pmis = eng_meta_per_frame[0][i], eng_meta_per_frame[1][i]
            if np.isfinite(rng) and np.isfinite(pmis):
                hud_text += f"   rng {rng:6.1f} m   pred_miss {pmis:5.2f} m"
        hud.set_text(hud_text)

        if fc_state in ("KILL", "MISS"):
            banner.set_text(f"FC: ** {fc_state} **")
            banner.set_color(KILL_COLOR if fc_state == "KILL" else "#ff9a3c")
        elif fc_state == "ENGAGED":
            banner.set_text("FC: ENGAGED");  banner.set_color(INTERCEPTOR_COLOR)
        elif fc_state == "ARMED":
            banner.set_text("FC: LOCKED");   banner.set_color(LOCK_COLOR)
        else:
            banner.set_text("FC: SEARCHING"); banner.set_color(TEXT_COLOR)

        artists = [l_truth, l_est, l_err]
        if l_intc is not None:
            artists += [l_intc, l_los]
        return artists

    print(f"Rendering {n_frames} frames -> {out_path} ...")
    anim = FuncAnimation(fig, frame, frames=n_frames,
                         interval=1000 / FPS, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)
    print(f"demo.gif:  {os.path.getsize(out_path) / 1024:.0f} KB"
          + (f"  [engagement overlay: {summary['outcome']}]"
             if summary else "  [EKF-only]"))


# --------------------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="run.csv",
                    help="Path to run.csv (ai_orchestrator output). "
                         "Default: ./run.csv")
    ap.add_argument("--engagement", default="engagement.csv",
                    help="Path to engagement.csv (engagement_engine output). "
                         "Default: ./engagement.csv")
    ap.add_argument("--out-dir", default="docs/img",
                    help="Directory where hero.png and demo.gif are written. "
                         "Default: docs/img")
    ap.add_argument("--no-engagement", action="store_true",
                    help="Force EKF-only assets even if engagement.csv exists "
                         "(legacy / sanity-check mode).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.run)
    df["t_s"] = (df.ts_ns - df.ts_ns.iloc[0]) * 1e-9
    for c in ["x", "y", "z"]:
        df[f"err_{c}"] = df[f"p{c}"] - df[f"e{c}"]
    df["err_norm"] = np.linalg.norm(
        df[["err_x", "err_y", "err_z"]].values, axis=1)

    eng = None if args.no_engagement else _load_engagement(args.engagement)
    if eng is None and not args.no_engagement:
        print(f"[assets] no usable {args.engagement}; "
              "producing EKF-only assets.", file=sys.stderr)

    make_hero(df, eng, os.path.join(args.out_dir, "hero.png"))
    make_gif(df,  eng, os.path.join(args.out_dir, "demo.gif"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
