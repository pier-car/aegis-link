"""
AEGIS-LINK :: Interactive Streamlit Demo
=========================================
A fully self-contained Python simulation of the AEGIS-LINK pipeline:

  SDE simulator → 9-D CA EKF tracker → AI Orchestrator (Mahalanobis + lock FSM)
  → Proportional-Navigation interceptor → MWIR IRST sensor

No ZeroMQ, Julia or C++ required — everything runs in-browser via NumPy.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS-LINK · Interactive Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global dark tactical skin ── */
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
div[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* ── Outcome banners ── */
.banner-kill {
    background: linear-gradient(135deg,#0f2d15,#1a4a1f);
    border: 2px solid #3fb950; border-radius: 10px;
    padding: 18px; text-align: center;
    font-size: 2rem; font-weight: 900; letter-spacing: 4px;
    color: #3fb950; text-shadow: 0 0 18px #3fb950;
}
.banner-miss {
    background: linear-gradient(135deg,#2d0f0f,#4a1a1a);
    border: 2px solid #f85149; border-radius: 10px;
    padding: 18px; text-align: center;
    font-size: 2rem; font-weight: 900; letter-spacing: 4px;
    color: #f85149; text-shadow: 0 0 18px #f85149;
}
.banner-nolock {
    background: linear-gradient(135deg,#1e1a0a,#3a300f);
    border: 2px solid #d29922; border-radius: 10px;
    padding: 18px; text-align: center;
    font-size: 2rem; font-weight: 900; letter-spacing: 4px;
    color: #d29922;
}

/* ── Metric cards ── */
.kpi-row { display:flex; gap:12px; margin-bottom:14px; }
.kpi { flex:1; background:#161b22; border:1px solid #30363d; border-radius:8px;
       padding:14px; text-align:center; }
.kpi-label { font-size:11px; color:#8b949e; text-transform:uppercase;
             letter-spacing:1px; font-weight:600; }
.kpi-value { font-size:22px; font-weight:700; color:#e6edf3; margin-top:4px; }
.kpi-green  { color:#3fb950 !important; }
.kpi-amber  { color:#e3b341 !important; }
.kpi-red    { color:#f85149 !important; }
.kpi-blue   { color:#58a6ff !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Physical constants ────────────────────────────────────────────────────────
GRAVITY  = 9.80665          # m/s²
G_VEC    = np.array([0.0, 0.0, -GRAVITY])
DT       = 0.01             # 100 Hz integration step [s]

# Orchestrator constants (mirror ai_orchestrator/main.py)
CHI2_99_DOF6   = 16.812
CHI2_999_DOF6  = 22.458
STREAK_ALERT   = 3
LOCK_HOLD_FRAMES = 50       # additional TRACKING frames before → LOCKED
R_DIAG_SENSOR  = np.array([0.05**2]*3 + [0.20**2]*3)  # sensor noise floor

LOCK_SEARCH   = 0
LOCK_TRACKING = 1
LOCK_LOCKED   = 2
_LOCK_NAME    = {LOCK_SEARCH: "SEARCH", LOCK_TRACKING: "TRACKING", LOCK_LOCKED: "LOCKED"}

# Colour palette
C_TRUTH  = "#3fb950"
C_EKF    = "#e3b341"
C_INTCPT = "#58a6ff"
C_IR     = "#f0883e"
C_ALERT  = "#f85149"
C_BG     = "#0d1117"
C_PANEL  = "#161b22"

# ─── EKF helpers ──────────────────────────────────────────────────────────────
# Observation: first 6 states (pos + vel) of the 9-D CA state vector.
H_OBS = np.zeros((6, 9))
H_OBS[:6, :6] = np.eye(6)


def _make_F(dt: float) -> np.ndarray:
    """State-transition matrix for 9-D Constant-Acceleration model."""
    F = np.eye(9)
    F[0:3, 3:6] = np.eye(3) * dt
    F[0:3, 6:9] = np.eye(3) * (0.5 * dt**2)
    F[3:6, 6:9] = np.eye(3) * dt
    return F


def _make_Q(dt: float, q_jerk: float) -> np.ndarray:
    """Process-noise matrix (Singer / jerk-input model)."""
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    Qb = q_jerk * np.array([
        [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
        [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
        [dt3 / 6.0,  dt2 / 2.0, dt],
    ])
    Q = np.zeros((9, 9))
    for i in range(3):
        idx = [i, i + 3, i + 6]
        Q[np.ix_(idx, idx)] = Qb
    return Q


def _ekf_predict(x: np.ndarray, P: np.ndarray,
                 F: np.ndarray, Q: np.ndarray):
    return F @ x, F @ P @ F.T + Q


def _ekf_update(x_p: np.ndarray, P_p: np.ndarray,
                z: np.ndarray, H: np.ndarray, R: np.ndarray):
    """Joseph-form EKF update for numerical stability."""
    y = z - H @ x_p
    S = H @ P_p @ H.T + R
    try:
        # K = P H^T S^{-1};  solve: S K^T = H P  =>  K = solve(S, H @ P_p).T
        K = np.linalg.solve(S, H @ P_p).T
    except np.linalg.LinAlgError:
        return x_p, P_p
    x = x_p + K @ y
    I_KH = np.eye(9) - K @ H
    P = I_KH @ P_p @ I_KH.T + K @ R @ K.T
    return x, P


# ─── Proportional-Navigation guidance ────────────────────────────────────────
def _pn_command(r_int: np.ndarray, v_int: np.ndarray,
                r_tgt: np.ndarray, v_tgt: np.ndarray,
                N_prime: float):
    """True PN acceleration command.

    Uses |V_c| for LOS-rate weighting so the guidance remains stable even
    during the initial boost phase when closing speed is negative (tail chase).

    Returns (a_cmd [m/s²], tgo [s], range [m], closing_speed [m/s]).
    """
    r_rel = r_tgt - r_int
    v_rel = v_tgt - v_int
    rng   = float(np.linalg.norm(r_rel))
    if rng < 1e-6:
        return np.zeros(3), 0.0, rng, 0.0
    u_los   = r_rel / rng
    omega   = np.cross(r_rel, v_rel) / (rng * rng)
    closing = -float(np.dot(u_los, v_rel))
    # Use |closing| so guidance steers toward the LOS-rate regardless of sign.
    # Standard PN inverts the command when Vc < 0, which is unstable in
    # tail-chase scenarios where the interceptor must first accelerate to
    # match the target speed before it can close.
    a_cmd   = N_prime * abs(closing) * np.cross(omega, u_los)
    tgo     = rng / max(closing, 1e-3) if closing > 0 else 1e9
    return a_cmd, tgo, rng, closing


# ─── Interceptor RK4 step ─────────────────────────────────────────────────────
def _rk4_step(pos: np.ndarray, vel: np.ndarray, mass: float, propellant: float,
              thrust: float, exhaust_vel: float,
              drag_coef: float, cross_sec: float,
              max_lat_acc: float, a_cmd_raw: np.ndarray, dt: float):
    """One RK4 integration step for a 3-DoF interceptor point-mass."""
    thrust_active = propellant > 0.0

    m_cmd = float(np.linalg.norm(a_cmd_raw))
    a_cmd = a_cmd_raw * (max_lat_acc / m_cmd) if m_cmd > max_lat_acc else a_cmd_raw.copy()

    def accel(v_):
        spd  = float(np.linalg.norm(v_))
        vhat = v_ / spd if spd > 1e-6 else np.zeros(3)
        T    = thrust if thrust_active else 0.0
        D    = drag_coef * cross_sec * spd * spd
        return (T - D) / max(mass, 1e-3) * vhat + a_cmd + G_VEC

    k1v = accel(vel);                    k1p = vel
    k2v = accel(vel + 0.5*dt*k1v);      k2p = vel + 0.5*dt*k1v
    k3v = accel(vel + 0.5*dt*k2v);      k3p = vel + 0.5*dt*k2v
    k4v = accel(vel + dt*k3v);           k4p = vel + dt*k3v

    new_pos = pos + (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    new_vel = vel + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    if thrust_active and exhaust_vel > 1e-3:
        burn       = min(thrust / exhaust_vel * dt, propellant)
        propellant -= burn
        mass       -= burn

    return new_pos, new_vel, mass, propellant


# ─── Main simulation ──────────────────────────────────────────────────────────
def run_simulation(p: dict) -> dict:          # noqa: C901 – long but sequential
    """
    Run the full AEGIS-LINK pipeline (batch, no ZMQ required).

    Parameters
    ----------
    p : dict  –  flat parameter dict assembled by the sidebar widgets.

    Returns
    -------
    dict of arrays + scalar metrics for the visualisation layer.
    """
    rng    = np.random.default_rng(int(p["seed"]))
    dt     = DT
    t_end  = float(p["duration"])
    N_max  = int(t_end / dt) + 2

    # ── Storage arrays (pre-allocated, trimmed at the end) ────────────────
    t_arr        = np.zeros(N_max)
    truth_arr    = np.zeros((N_max, 6))   # [px,py,pz,vx,vy,vz]
    est_arr      = np.zeros((N_max, 6))
    P_diag_arr   = np.zeros((N_max, 6))
    d2_arr       = np.zeros(N_max)
    lock_arr     = np.zeros(N_max, dtype=np.int8)
    sigma_p_arr  = np.zeros(N_max)
    maneuver_arr = np.zeros(N_max, dtype=bool)
    snr_arr      = np.zeros(N_max)
    tau_arr      = np.ones(N_max)
    pd_arr       = np.zeros(N_max)

    ir_det_t:   list[float] = []
    ir_det_pos: list[np.ndarray] = []
    ir_det_snr: list[float] = []
    ir_det_fa:  list[bool] = []

    int_t_list:      list[float] = []
    int_pos_list:    list[np.ndarray] = []
    int_fuel_list:   list[float] = []
    int_range_list:  list[float] = []
    int_close_list:  list[float] = []
    int_pmiss_list:  list[float] = []

    # ── Initial conditions ────────────────────────────────────────────────
    v0x, v0y, v0z = float(p["v0_x"]), float(p["v0_y"]), float(p["v0_z"])

    # SDE truth state: [rx,ry,rz, vx,vy,vz, Wx,Wy,Wz]
    sde = np.array([0.0, 0.0, 10.0, v0x, v0y, v0z, 0.0, 0.0, 0.0])

    # EKF: 9-D CA state, large initial covariance
    x_ekf = np.array([0.0, 0.0, 10.0, v0x, v0y, v0z, 0.0, 0.0, -GRAVITY])
    P_ekf = np.diag([500.0]*3 + [50.0]*3 + [5.0]*3)

    F = _make_F(dt)
    Q = _make_Q(dt, float(p["q_jerk"]))

    s_pos   = float(p["sigma_pos"])
    s_vel   = float(p["sigma_vel"])
    R_meas  = np.diag([s_pos**2]*3 + [s_vel**2]*3)

    # Orchestrator state
    chi2_gate    = float(p["chi2_gate"])
    lock_sigma_m = float(p["lock_sigma_m"])
    lock_streak  = int(p["lock_streak"])

    streak_d2   = 0
    lock_state  = LOCK_SEARCH
    in_gate_run = 0
    out_gate    = 0
    lock_acquired_at: Optional[float] = None

    # OU wind
    theta   = float(p["ou_theta"])
    sigma_w = float(p["ou_sigma"])

    # Interceptor parameters
    int_nav     = float(p["int_nav_ratio"])
    int_thrust  = float(p["int_thrust"])
    int_exhaust = 1900.0
    int_drag    = 0.30
    int_cross   = 0.012
    int_max_lat = float(p["int_max_g"]) * 9.81
    int_lethal  = float(p["int_lethal_r"])
    int_mass0   = float(p["int_mass"])
    int_prop0   = int_mass0 * float(p["int_prop_frac"])

    # Engagement state machine
    eng_state  = "IDLE"
    int_pos    = np.zeros(3)
    int_vel    = np.zeros(3)
    int_mass_c = int_mass0
    int_prop_c = int_prop0
    engaged_at: Optional[float] = None
    cpa        = float("inf")
    outcome    = "NO_LOCK"

    # IR sensor parameters
    ir_I0       = float(p["ir_I0"])
    ir_alpha    = float(p["ir_alpha"])
    ir_nei      = float(p["ir_nei"])
    ir_snr_thr  = float(p["ir_snr_thresh"])
    ir_lam_fa   = float(p["ir_lam_fa"])

    n_steps = N_max  # will be overwritten at termination

    # ── Main loop ─────────────────────────────────────────────────────────
    for i in range(N_max):
        t = i * dt

        # 1 ── SDE: Euler-Maruyama step ──────────────────────────────────
        r, v, W = sde[0:3], sde[3:6], sde[6:9]
        dW_noise = rng.standard_normal(3) * (sigma_w * np.sqrt(dt))

        sde[0:3] = r + v * dt
        sde[3:6] = v + (W + G_VEC) * dt
        sde[6:9] = W + (-theta * W * dt) + dW_noise

        # 2 ── Ground-impact termination ──────────────────────────────────
        if sde[2] <= 0.0 and i > 5:
            sde[2] = 0.0
            t_arr[i]      = t
            truth_arr[i]  = sde[0:6]
            est_arr[i]    = x_ekf[0:6]           # last EKF estimate
            P_diag_arr[i] = np.maximum(np.diag(P_ekf)[0:6], 0.0)
            lock_arr[i]   = lock_state
            sigma_p_arr[i] = float(np.sqrt(
                max(float(np.sum(np.maximum(np.diag(P_ekf)[0:3], 0.0))), 0.0)))
            n_steps = i + 1
            break

        # 3 ── Noisy measurement (truth + Gaussian sensor noise) ──────────
        z_meas       = sde[0:6].copy()
        z_meas[0:3] += rng.standard_normal(3) * s_pos
        z_meas[3:6] += rng.standard_normal(3) * s_vel

        # 4 ── EKF predict + update ───────────────────────────────────────
        x_ekf, P_ekf = _ekf_predict(x_ekf, P_ekf, F, Q)
        x_ekf, P_ekf = _ekf_update(x_ekf, P_ekf, z_meas, H_OBS, R_meas)

        est6 = x_ekf[0:6].copy()
        P6   = np.maximum(np.diag(P_ekf)[0:6], 0.0)

        # 5 ── AI Orchestrator: Mahalanobis + lock FSM ───────────────────
        delta  = sde[0:6] - est6
        S_diag = np.maximum(P6 + R_DIAG_SENSOR, 1e-12)
        d2     = float(np.sum(delta**2 / S_diag))

        is_alert  = d2 > chi2_gate
        streak_d2 = streak_d2 + 1 if is_alert else 0
        confirmed = streak_d2 >= STREAK_ALERT

        sigma_p = float(np.sqrt(max(float(np.sum(P6[0:3])), 0.0)))
        lock_ok = (not is_alert) and (sigma_p < lock_sigma_m)

        if lock_ok:
            in_gate_run += 1
            out_gate     = 0
            if lock_state == LOCK_SEARCH and in_gate_run >= lock_streak:
                lock_state = LOCK_TRACKING
            elif lock_state == LOCK_TRACKING and \
                    in_gate_run >= (lock_streak + LOCK_HOLD_FRAMES):
                lock_state = LOCK_LOCKED
                if lock_acquired_at is None:
                    lock_acquired_at = t
        else:
            out_gate += 1
            if out_gate >= 5 and lock_state != LOCK_SEARCH:
                lock_state  = LOCK_SEARCH
                in_gate_run = 0
            if sigma_p > 2.0 * lock_sigma_m:
                in_gate_run = 0

        # 6 ── Engagement engine ──────────────────────────────────────────
        if eng_state == "IDLE" and lock_state == LOCK_LOCKED:
            eng_state   = "ENGAGED"
            engaged_at  = t
            tgt_pos     = est6[0:3].copy()
            # Launch from a fixed ground site at the world origin (2 m altitude),
            # representing a surface-to-air launcher — not directly below the
            # target so the geometry is non-degenerate for PN guidance.
            int_pos     = np.array([0.0, 0.0, 2.0])
            dir_0       = tgt_pos - int_pos
            n_d         = float(np.linalg.norm(dir_0))
            int_vel     = dir_0 / n_d if n_d > 1e-6 else np.array([0., 0., 1.])
            int_mass_c  = int_mass0
            int_prop_c  = int_prop0

        if eng_state == "ENGAGED":
            r_tgt = est6[0:3]
            v_tgt = est6[3:6]
            a_cmd, tgo, rng_i, closing = _pn_command(
                int_pos, int_vel, r_tgt, v_tgt, int_nav)

            int_pos, int_vel, int_mass_c, int_prop_c = _rk4_step(
                int_pos, int_vel, int_mass_c, int_prop_c,
                int_thrust, int_exhaust,
                int_drag, int_cross, int_max_lat, a_cmd, dt)

            if rng_i < cpa:
                cpa = rng_i

            # Straight-line CPA prediction
            rr = r_tgt - int_pos
            rv = v_tgt - int_vel
            vv = float(np.dot(rv, rv))
            if vv > 1e-9 and tgo < 1e8:
                t_cpa_p  = max(0.0, -float(np.dot(rr, rv)) / vv)
                pred_miss = float(np.linalg.norm(rr + t_cpa_p * rv))
            else:
                pred_miss = float(np.linalg.norm(rr))

            fuel_frac = max(0.0, int_prop_c) / max(int_prop0, 1e-9)
            int_t_list.append(t)
            int_pos_list.append(int_pos.copy())
            int_fuel_list.append(fuel_frac)
            int_range_list.append(rng_i)
            int_close_list.append(closing)
            int_pmiss_list.append(min(pred_miss, 9_999.0))

            flight_t = t - engaged_at                    # type: ignore[operator]
            if rng_i <= int_lethal:
                eng_state = "KILL"
                outcome   = "KILL"
            elif flight_t > 30.0:
                eng_state = "MISS"
                outcome   = "MISS"

        # 7 ── MWIR IRST sensor ───────────────────────────────────────────
        rng_sensor = max(float(np.linalg.norm(sde[0:3])), 1.0)
        speed_t    = float(np.linalg.norm(sde[3:6]))
        intensity  = ir_I0 * (1.0 + 0.80 * (speed_t / 300.0)**2)
        tau_v      = float(np.exp(-ir_alpha * rng_sensor))
        snr_v      = intensity * tau_v / (rng_sensor**2) / ir_nei
        pd_v       = 1.0 / (1.0 + np.exp(-1.5 * (snr_v - ir_snr_thr)))

        if rng.random() < pd_v:
            sigma_r = max(0.05 * rng_sensor, 10.0)
            pos_ir  = sde[0:3] + rng.standard_normal(3) * sigma_r
            ir_det_t.append(t);   ir_det_pos.append(pos_ir.copy())
            ir_det_snr.append(snr_v); ir_det_fa.append(False)

        n_fa = rng.poisson(ir_lam_fa)
        for _ in range(n_fa):
            fa_pos = sde[0:3] + rng.standard_normal(3) * 500.0
            ir_det_t.append(t);   ir_det_pos.append(fa_pos.copy())
            ir_det_snr.append(0.0); ir_det_fa.append(True)

        # 8 ── Store ──────────────────────────────────────────────────────
        t_arr[i]        = t
        truth_arr[i]    = sde[0:6]
        est_arr[i]      = est6
        P_diag_arr[i]   = P6
        d2_arr[i]       = d2
        lock_arr[i]     = lock_state
        sigma_p_arr[i]  = sigma_p
        maneuver_arr[i] = confirmed
        snr_arr[i]      = snr_v
        tau_arr[i]      = tau_v
        pd_arr[i]       = pd_v

        n_steps = i + 1   # track in case loop finishes without break

    # ── Finalise outcome ──────────────────────────────────────────────────
    if outcome == "NO_LOCK" and eng_state == "ENGAGED":
        outcome = "MISS"

    # ── Trim arrays ───────────────────────────────────────────────────────
    sl = slice(0, n_steps)
    t_arr       = t_arr[sl]
    truth_arr   = truth_arr[sl]
    est_arr     = est_arr[sl]
    P_diag_arr  = P_diag_arr[sl]
    d2_arr      = d2_arr[sl]
    lock_arr    = lock_arr[sl]
    sigma_p_arr = sigma_p_arr[sl]
    maneuver_arr = maneuver_arr[sl]
    snr_arr     = snr_arr[sl]
    tau_arr     = tau_arr[sl]
    pd_arr      = pd_arr[sl]

    pos_err = np.linalg.norm(truth_arr[:, 0:3] - est_arr[:, 0:3], axis=1)

    # Convert interceptor lists to arrays
    int_pos_arr   = np.array(int_pos_list)   if int_pos_list   else None
    int_t_arr     = np.array(int_t_list)     if int_t_list     else None
    int_fuel_arr  = np.array(int_fuel_list)  if int_fuel_list  else None
    int_range_arr = np.array(int_range_list) if int_range_list else None
    int_close_arr = np.array(int_close_list) if int_close_list else None
    int_pmiss_arr = np.array(int_pmiss_list) if int_pmiss_list else None

    return {
        "t":              t_arr,
        "truth":          truth_arr,
        "estimate":       est_arr,
        "P_diag":         P_diag_arr,
        "d2":             d2_arr,
        "lock_state":     lock_arr,
        "sigma_p":        sigma_p_arr,
        "maneuver":       maneuver_arr,
        "pos_err":        pos_err,
        "snr":            snr_arr,
        "tau":            tau_arr,
        "pd":             pd_arr,
        # IR detections
        "ir_t":           np.array(ir_det_t)   if ir_det_t   else np.array([]),
        "ir_pos":         np.array(ir_det_pos) if ir_det_pos else np.zeros((0, 3)),
        "ir_snr":         np.array(ir_det_snr) if ir_det_snr else np.array([]),
        "ir_fa":          np.array(ir_det_fa)  if ir_det_fa  else np.array([], dtype=bool),
        # Interceptor trajectory
        "int_pos":        int_pos_arr,
        "int_t":          int_t_arr,
        "int_fuel":       int_fuel_arr,
        "int_range":      int_range_arr,
        "int_closing":    int_close_arr,
        "int_pred_miss":  int_pmiss_arr,
        # Summary scalars
        "outcome":        outcome,
        "cpa_m":          float(cpa)          if not np.isinf(cpa)        else None,
        "lock_acquired":  lock_acquired_at,
        "engaged_at":     engaged_at,
        "rmse_pos_m":     float(np.sqrt(np.mean(pos_err**2))),
        "median_d2":      float(np.median(d2_arr)),
    }


# ─── Plotly helpers ───────────────────────────────────────────────────────────
_DARK_LAYOUT = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_PANEL,
    font=dict(color="#c9d1d9", size=12),
    margin=dict(l=60, r=30, t=40, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
)

_DARK_SCENE = dict(
    bgcolor=C_BG,
    xaxis=dict(backgroundcolor=C_PANEL, gridcolor="#30363d", showbackground=True),
    yaxis=dict(backgroundcolor=C_PANEL, gridcolor="#30363d", showbackground=True),
    zaxis=dict(backgroundcolor=C_PANEL, gridcolor="#30363d", showbackground=True),
)


def _ds(arr: np.ndarray, step: int = 5) -> np.ndarray:
    """Downsample a 1-D or 2-D array by taking every `step`-th row."""
    return arr[::step]


def fig_3d(res: dict) -> go.Figure:
    """3-D tactical overview: truth trajectory, EKF estimate, interceptor, IR."""
    fig = go.Figure()
    st5 = 5   # downsample step for smooth Plotly render

    tr = res["truth"]
    es = res["estimate"]
    t  = res["t"]

    # Truth trajectory
    fig.add_trace(go.Scatter3d(
        x=_ds(tr[:, 0], st5), y=_ds(tr[:, 1], st5), z=_ds(tr[:, 2], st5),
        mode="lines", name="Truth (SDE)",
        line=dict(color=C_TRUTH, width=4),
    ))
    # Launch & impact markers
    fig.add_trace(go.Scatter3d(
        x=[tr[0, 0]], y=[tr[0, 1]], z=[tr[0, 2]],
        mode="markers", name="Launch",
        marker=dict(color=C_TRUTH, size=8, symbol="circle"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[tr[-1, 0]], y=[tr[-1, 1]], z=[tr[-1, 2]],
        mode="markers", name="Impact",
        marker=dict(color=C_TRUTH, size=8, symbol="x"),
        showlegend=False,
    ))

    # EKF estimate
    fig.add_trace(go.Scatter3d(
        x=_ds(es[:, 0], st5), y=_ds(es[:, 1], st5), z=_ds(es[:, 2], st5),
        mode="lines", name="EKF Estimate",
        line=dict(color=C_EKF, width=3, dash="dot"),
    ))

    # Interceptor trajectory
    if res["int_pos"] is not None:
        ip = res["int_pos"]
        fig.add_trace(go.Scatter3d(
            x=_ds(ip[:, 0], st5), y=_ds(ip[:, 1], st5), z=_ds(ip[:, 2], st5),
            mode="lines", name="Interceptor (PN)",
            line=dict(color=C_INTCPT, width=3),
        ))
        # Outcome marker on last interceptor position
        out_color = C_TRUTH if res["outcome"] == "KILL" else C_ALERT
        out_sym   = "diamond" if res["outcome"] == "KILL" else "cross"
        out_lbl   = res["outcome"]
        fig.add_trace(go.Scatter3d(
            x=[ip[-1, 0]], y=[ip[-1, 1]], z=[ip[-1, 2]],
            mode="markers+text",
            name=out_lbl,
            text=[f" {out_lbl}"],
            textfont=dict(color=out_color, size=13),
            marker=dict(color=out_color, size=10, symbol=out_sym),
        ))

    # IR true detections
    if len(res["ir_t"]) > 0:
        mask_true = ~np.array(res["ir_fa"], dtype=bool)
        mask_fa   =  np.array(res["ir_fa"], dtype=bool)
        ir_pos    =  res["ir_pos"]

        if mask_true.any():
            tp = ir_pos[mask_true]
            fig.add_trace(go.Scatter3d(
                x=tp[:, 0], y=tp[:, 1], z=tp[:, 2],
                mode="markers", name="IR Detection",
                marker=dict(color=C_IR, size=3, opacity=0.7),
            ))
        if mask_fa.any():
            fp = ir_pos[mask_fa]
            fig.add_trace(go.Scatter3d(
                x=fp[:, 0], y=fp[:, 1], z=fp[:, 2],
                mode="markers", name="IR False Alarm",
                marker=dict(color=C_ALERT, size=3, symbol="x", opacity=0.5),
            ))

    fig.update_layout(
        scene=dict(
            **_DARK_SCENE,
            xaxis_title="X [m] (East)",
            yaxis_title="Y [m] (North)",
            zaxis_title="Z [m] (Alt)",
        ),
        **_DARK_LAYOUT,
        title=dict(text="3-D Tactical Picture", font=dict(size=16)),
        height=600,
    )
    return fig


def fig_timeseries(res: dict) -> go.Figure:
    """4-panel time-series: position error, d², lock state, sigma_p."""
    t = res["t"]
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Position Error |Δr| [m]",
            "Mahalanobis d² (log scale)",
            "Lock State",
            "σ_p: EKF position 1-σ [m]",
        ),
        vertical_spacing=0.07,
    )
    fig.update_layout(**_DARK_LAYOUT, height=700,
                      title=dict(text="Time-Series Analysis", font=dict(size=16)))

    # 1 – Position error
    fig.add_trace(go.Scatter(
        x=t, y=res["pos_err"],
        mode="lines", name="‖Δr‖", line=dict(color=C_TRUTH, width=1.5),
        showlegend=False,
    ), row=1, col=1)

    # 2 – Mahalanobis d²
    d2 = np.clip(res["d2"], 1e-3, 1e5)
    fig.add_trace(go.Scatter(
        x=t, y=d2, mode="lines", name="d²",
        line=dict(color=C_EKF, width=1.5), showlegend=False,
    ), row=2, col=1)
    # Threshold lines
    fig.add_hline(y=CHI2_99_DOF6,  line=dict(color=C_IR,    dash="dash", width=1),
                  annotation_text="χ²₀.₉₉", annotation_position="top right", row=2, col=1)
    fig.add_hline(y=CHI2_999_DOF6, line=dict(color=C_ALERT, dash="dash", width=1),
                  annotation_text="χ²₀.₉₉₉", annotation_position="top right", row=2, col=1)
    # Manoeuvre alerts (filled area)
    alert_mask = res["maneuver"]
    if alert_mask.any():
        alert_t   = t[alert_mask]
        alert_d2  = np.clip(d2[alert_mask], 1e-3, 1e5)
        fig.add_trace(go.Scatter(
            x=alert_t, y=alert_d2, mode="markers",
            name="Manoeuvre alert", marker=dict(color=C_ALERT, size=3),
            showlegend=False,
        ), row=2, col=1)

    # 3 – Lock state (step)
    lock_vals = [_LOCK_NAME[int(s)] for s in res["lock_state"]]
    lock_num  = res["lock_state"].astype(float)
    fig.add_trace(go.Scatter(
        x=t, y=lock_num, mode="lines", name="Lock",
        line=dict(color=C_INTCPT, width=2, shape="hv"), showlegend=False,
    ), row=3, col=1)
    fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["SEARCH", "TRACKING", "LOCKED"],
        row=3, col=1,
    )

    # 4 – sigma_p
    fig.add_trace(go.Scatter(
        x=t, y=res["sigma_p"], mode="lines", name="σ_p",
        line=dict(color="#da8c62", width=1.5), showlegend=False,
    ), row=4, col=1)
    # Lock threshold line
    fig.add_hline(y=float(st.session_state.get("lock_sigma_m", 5.0)),
                  line=dict(color=C_EKF, dash="dash", width=1),
                  annotation_text="Lock σ_p", row=4, col=1)

    # Log-scale for d² panel
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=4, col=1,
                     gridcolor="#30363d", zerolinecolor="#30363d")
    for r in range(1, 5):
        fig.update_yaxes(gridcolor="#30363d", zerolinecolor="#30363d", row=r, col=1)
        fig.update_xaxes(gridcolor="#30363d", row=r, col=1)

    # Vertical markers for lock / engage
    if res["lock_acquired"] is not None:
        for r in range(1, 5):
            fig.add_vline(x=res["lock_acquired"],
                          line=dict(color=C_EKF, dash="dot", width=1),
                          row=r, col=1)
    if res["engaged_at"] is not None:
        for r in range(1, 5):
            fig.add_vline(x=res["engaged_at"],
                          line=dict(color=C_INTCPT, dash="dot", width=1),
                          row=r, col=1)
    return fig


def fig_engagement(res: dict) -> go.Figure:
    """Interceptor engagement panel: range, closing speed, fuel, pred-miss."""
    if res["int_t"] is None:
        return go.Figure().update_layout(
            **_DARK_LAYOUT,
            title="No engagement (lock not acquired)",
        )

    t_e = res["int_t"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "LOS Range [m]",
            "Closing Speed [m/s]",
            "Fuel Fraction",
            "Predicted Miss Distance [m]",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )
    fig.update_layout(**_DARK_LAYOUT, height=500,
                      title=dict(text="Engagement Engine Telemetry", font=dict(size=16)))

    kw = dict(mode="lines", showlegend=False)

    fig.add_trace(go.Scatter(x=t_e, y=res["int_range"],
                             line=dict(color=C_INTCPT, width=2), **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_e, y=res["int_closing"],
                             line=dict(color=C_EKF, width=2), **kw), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_e, y=res["int_fuel"],
                             line=dict(color=C_IR, width=2), **kw), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_e, y=res["int_pred_miss"],
                             line=dict(color="#da8c62", width=2), **kw), row=2, col=2)

    # Lethal-radius line on range panel
    fig.add_hline(y=float(st.session_state.get("int_lethal_r", 5.0)),
                  line=dict(color=C_TRUTH, dash="dash", width=1),
                  annotation_text="Lethal r", row=1, col=1)

    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_yaxes(gridcolor="#30363d", zerolinecolor="#30363d", row=r, col=c)
            fig.update_xaxes(gridcolor="#30363d", title_text="Time [s]", row=r, col=c)
    return fig


def fig_ir_sensor(res: dict) -> go.Figure:
    """IRST panel: SNR, atmospheric tau, P_D, and detection timeline."""
    t = res["t"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("SNR at detector", "Atmospheric τ (Beer–Lambert)",
                        "Detection probability P_D"),
        vertical_spacing=0.08,
    )
    fig.update_layout(**_DARK_LAYOUT, height=550,
                      title=dict(text="MWIR IRST Sensor", font=dict(size=16)))

    fig.add_trace(go.Scatter(
        x=t, y=res["snr"], mode="lines", name="SNR",
        line=dict(color=C_IR, width=1.5), showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=float(st.session_state.get("ir_snr_thresh", 4.0)),
                  line=dict(color=C_EKF, dash="dash", width=1),
                  annotation_text="SNR thresh", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=res["tau"], mode="lines", name="τ",
        line=dict(color=C_INTCPT, width=1.5), showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=res["pd"], mode="lines", name="P_D",
        line=dict(color=C_TRUTH, width=1.5), showlegend=False,
    ), row=3, col=1)

    # Detection scatter on P_D panel
    if len(res["ir_t"]) > 0:
        fa_flag = np.array(res["ir_fa"], dtype=bool)
        det_t_true = res["ir_t"][~fa_flag]
        det_t_fa   = res["ir_t"][fa_flag]
        if len(det_t_true) > 0:
            # Interpolate P_D at detection times
            pd_at = np.interp(det_t_true, t, res["pd"])
            fig.add_trace(go.Scatter(
                x=det_t_true, y=pd_at, mode="markers", name="True detect",
                marker=dict(color=C_IR, size=5, symbol="circle"),
                showlegend=False,
            ), row=3, col=1)
        if len(det_t_fa) > 0:
            pd_fa = np.zeros(len(det_t_fa))
            fig.add_trace(go.Scatter(
                x=det_t_fa, y=pd_fa, mode="markers", name="False alarm",
                marker=dict(color=C_ALERT, size=5, symbol="x"),
                showlegend=False,
            ), row=3, col=1)

    for r in range(1, 4):
        fig.update_yaxes(gridcolor="#30363d", zerolinecolor="#30363d", row=r, col=1)
        fig.update_xaxes(gridcolor="#30363d", row=r, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def _sidebar() -> dict:
    """Render sidebar widgets and return a flat params dict."""
    st.sidebar.markdown(
        "## 🛰️ AEGIS-LINK\n"
        "*Vary parameters and hit **Run** to simulate the full pipeline.*",
    )
    st.sidebar.divider()

    p: dict = {}

    # ── Target / Scenario ────────────────────────────────────────────────
    with st.sidebar.expander("🎯  Target · Scenario", expanded=True):
        p["v0_x"] = st.slider("v₀ₓ — East velocity [m/s]",   0, 300, 50, 5)
        p["v0_y"] = st.slider("v₀ᵧ — North velocity [m/s]",  0, 100,  0, 5)
        p["v0_z"] = st.slider("v₀z — Vertical velocity [m/s]", 50, 450, 250, 10)
        p["ou_theta"] = st.slider(
            "OU θ — wind mean-reversion [1/s]  (τ = 1/θ)", 0.1, 3.0, 0.5, 0.1)
        p["ou_sigma"] = st.slider(
            "OU σ_w — gust intensity [m/s²/√s]", 0.0, 10.0, 2.5, 0.1)
        p["duration"] = st.slider("Simulation duration [s]", 10, 120, 60, 5)
        p["seed"] = st.number_input("Random seed", 0, 2**31 - 1, 0xA3615, 1)

    # ── EKF Tracker ──────────────────────────────────────────────────────
    with st.sidebar.expander("📡  EKF Tracker", expanded=False):
        p["sigma_pos"] = st.slider(
            "σ_pos — measurement noise [m]", 0.001, 2.0, 0.05, 0.001,
            format="%.3f")
        p["sigma_vel"] = st.slider(
            "σ_vel — measurement noise [m/s]", 0.01, 2.0, 0.20, 0.01)
        p["q_jerk"] = st.slider(
            "q_jerk — process noise PSD [m²/s⁵]", 0.01, 20.0, 1.0, 0.1)

    # ── AI Orchestrator ───────────────────────────────────────────────────
    with st.sidebar.expander("🧠  AI Orchestrator", expanded=False):
        p["chi2_gate"] = st.slider(
            "χ² gate threshold (d² > this → alert)",
            5.0, 30.0, float(CHI2_99_DOF6), 0.5, format="%.1f")
        p["lock_sigma_m"] = st.slider(
            "Lock σ_p threshold [m]  (EKF position uncertainty)",
            0.5, 20.0, 5.0, 0.5)
        p["lock_streak"] = st.slider(
            "Lock streak [frames]  (consecutive in-gate to start tracking)",
            5, 60, 25, 1)
        # Store in session state for vline labels in subplots
        st.session_state["lock_sigma_m"] = p["lock_sigma_m"]

    # ── Interceptor / Engagement ──────────────────────────────────────────
    with st.sidebar.expander("🚀  Interceptor · Engagement", expanded=False):
        p["int_nav_ratio"] = st.slider("N′ — PN navigation ratio", 2.0, 8.0, 4.0, 0.5)
        p["int_thrust"]    = st.slider("Thrust [N]", 200, 5000, 1800, 50)
        p["int_mass"]      = st.slider("Initial mass [kg]", 4.0, 40.0, 12.0, 0.5)
        p["int_prop_frac"] = st.slider(
            "Propellant fraction", 0.1, 0.8, 0.50, 0.05, format="%.2f")
        p["int_max_g"]     = st.slider("Max lateral accel [G]", 10, 100, 40, 5)
        p["int_lethal_r"]  = st.slider("Lethal radius [m]", 1.0, 30.0, 5.0, 0.5)
        st.session_state["int_lethal_r"] = p["int_lethal_r"]

    # ── IR Sensor ──────────────────────────────────────────────────────────
    with st.sidebar.expander("🔭  MWIR IRST Sensor", expanded=False):
        p["ir_I0"]        = st.slider("Base intensity I₀ [W/sr]", 50.0, 1000.0, 200.0, 10.0)
        p["ir_alpha"]     = st.number_input(
            "Atmospheric α [m⁻¹]", value=2e-5, format="%.2e", step=1e-5)
        p["ir_nei"]       = st.number_input(
            "NEI — noise-equiv. irradiance [W/m²]",
            value=5e-11, format="%.2e", step=1e-11)
        p["ir_snr_thresh"] = st.slider("SNR detection threshold", 1.0, 12.0, 4.0, 0.5)
        p["ir_lam_fa"]    = st.number_input(
            "False-alarm rate λ [per frame]", value=5e-7, format="%.2e", step=1e-7)
        st.session_state["ir_snr_thresh"] = p["ir_snr_thresh"]

    st.sidebar.divider()
    run_btn = st.sidebar.button(
        "▶  Run Simulation",
        use_container_width=True,
        type="primary",
    )
    return p, run_btn


# ─── KPI helpers ──────────────────────────────────────────────────────────────
def _kpi_html(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {cls}">{value}</div></div>')


def _outcome_banner(outcome: str) -> str:
    if outcome == "KILL":
        return '<div class="banner-kill">💥 KILL</div>'
    if outcome == "MISS":
        return '<div class="banner-miss">✗ MISS</div>'
    return '<div class="banner-nolock">⚠️ NO LOCK</div>'


# ─── App ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Header
    st.markdown(
        "## 🛰️ AEGIS-LINK — Interactive Simulation Demo\n"
        "**SDE ballistic target → 9-D CA EKF → Mahalanobis lock FSM → "
        "PN interceptor → MWIR IRST** · all in pure Python/NumPy.",
    )
    st.caption(
        "Tune parameters in the sidebar, then click **Run Simulation**. "
        "Trajectories, EKF residuals, engagement telemetry and IR sensor data "
        "are displayed interactively with Plotly."
    )

    params, run_btn = _sidebar()

    # Session state
    if "results" not in st.session_state:
        st.session_state["results"] = None

    if run_btn:
        with st.spinner("Running 100-Hz pipeline simulation …"):
            t0  = __import__("time").perf_counter()
            res = run_simulation(params)
            dt_ = __import__("time").perf_counter() - t0
        st.session_state["results"] = res
        st.session_state["sim_time"] = dt_

    res = st.session_state.get("results")

    if res is None:
        st.info(
            "👈  Configure parameters in the sidebar and click **▶ Run Simulation** "
            "to start."
        )
        # Show architecture diagram from README as a teaser
        st.markdown("""
```
┌─────────────────┐   SDE (Ornstein-Uhlenbeck wind)   ┌──────────────────┐
│  SDE Simulator  │ ────────────────────────────────▶  │  9-D CA EKF      │
│  ballistic arc  │   noisy 6-D state @ 100 Hz         │  Joseph-form upd │
└─────────────────┘                                    └────────┬─────────┘
                                                                │
                                                  Mahalanobis d² + χ² gate
                                                                │
                                                       ┌────────▼─────────┐
                                                       │  AI Orchestrator  │
                                                       │  lock FSM:        │
                                                       │  SEARCH→TRACKING  │
                                                       │       →LOCKED     │
                                                       └────────┬─────────┘
                                                                │  LOCKED
                                                       ┌────────▼─────────┐
                                                       │  PN Interceptor  │
                                                       │  RK4 · thrust    │
                                                       │  drag · gravity  │
                                                       └────────┬─────────┘
                                                                │
                                                        KILL (CPA < r_lethal)
                                                        MISS  (timeout / diverge)
```
        """)
        return

    # ── Outcome banner ────────────────────────────────────────────────────
    st.markdown(_outcome_banner(res["outcome"]), unsafe_allow_html=True)
    st.markdown("")

    # ── KPI row ───────────────────────────────────────────────────────────
    cpa_str = (f'{res["cpa_m"]:.1f} m' if res["cpa_m"] is not None else "—")
    lock_str = (f'{res["lock_acquired"]:.1f} s'
                if res["lock_acquired"] is not None else "—")
    n_det = int(np.sum(~np.array(res["ir_fa"], dtype=bool))) if len(res["ir_t"]) else 0
    n_fa  = int(np.sum( np.array(res["ir_fa"], dtype=bool))) if len(res["ir_t"]) else 0
    sim_t = st.session_state.get("sim_time", 0)

    kpi_html = (
        '<div class="kpi-row">'
        + _kpi_html("Outcome",         res["outcome"],
                    "kpi-green" if res["outcome"] == "KILL"
                    else "kpi-red" if res["outcome"] == "MISS"
                    else "kpi-amber")
        + _kpi_html("CPA",             cpa_str,
                    "kpi-green" if res["cpa_m"] is not None and
                    res["cpa_m"] <= params["int_lethal_r"] else "kpi-red")
        + _kpi_html("EKF pos RMSE",   f'{res["rmse_pos_m"]*100:.1f} cm', "kpi-blue")
        + _kpi_html("Median d²",      f'{res["median_d2"]:.2f}',         "kpi-amber")
        + _kpi_html("Lock acquired",  lock_str,                           "kpi-blue")
        + _kpi_html("IR detections",  f'{n_det} true / {n_fa} FA',       "kpi-amber")
        + _kpi_html("Sim wall-time",  f'{sim_t*1000:.0f} ms',            "")
        + '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_3d, tab_ts, tab_eng, tab_ir, tab_data = st.tabs([
        "🗺️ 3-D Tactical",
        "📈 Time Series",
        "🚀 Engagement",
        "🔭 IRST Sensor",
        "📋 Raw Data",
    ])

    with tab_3d:
        st.plotly_chart(fig_3d(res), use_container_width=True)
        st.caption(
            f"**Truth** (green) · **EKF estimate** (amber dashed) · "
            f"**Interceptor** (blue) · "
            f"**IR true detections** (orange dots) · **false alarms** (red ×)"
        )

    with tab_ts:
        st.plotly_chart(fig_timeseries(res), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            la = res["lock_acquired"]
            if la is not None:
                st.markdown(
                    f"🔵 **Lock acquired at** t = {la:.2f} s  "
                    f"(amber dashed vertical line = lock instant, "
                    f"blue dashed = intercept launch)"
                )
            else:
                st.warning("Lock was never acquired — EKF uncertainty too high or "
                           "target hit ground first.")
        with col2:
            pct_alert = float(np.mean(res["maneuver"])) * 100
            st.markdown(
                f"📊 **Manoeuvre alert** on **{pct_alert:.1f}%** of frames  "
                f"(streak ≥ {STREAK_ALERT} consecutive d² > gate)"
            )

    with tab_eng:
        if res["int_t"] is None:
            st.warning("No engagement data — interceptor was never launched.")
            st.markdown(
                "To achieve lock, try: smaller **σ_pos / q_jerk** (tighter EKF), "
                "lower **lock σ_p threshold**, or shorter **lock streak**."
            )
        else:
            st.plotly_chart(fig_engagement(res), use_container_width=True)
            et = res["engaged_at"]
            total_t = (res["int_t"][-1] - et) if et else 0.0
            col1, col2, col3 = st.columns(3)
            col1.metric("Flight time",   f'{total_t:.2f} s')
            col2.metric("CPA",            cpa_str)
            col3.metric("Outcome",        res["outcome"])

    with tab_ir:
        st.plotly_chart(fig_ir_sensor(res), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("True detections",  n_det)
        col2.metric("False alarms",     n_fa)
        med_snr = float(np.median(res["snr"][res["snr"] > 0])) \
            if res["snr"].any() else 0.0
        col3.metric("Median SNR",  f"{med_snr:.1f}")
        st.caption(
            "SNR model: I(v) = I₀·(1 + 0.8·(|v|/300)²) — aerodynamic heating; "
            "τ(R) = exp(−α·R) — Beer–Lambert; P_D = σ(1.5·(SNR − thresh)) — "
            "Albersheim / Swerling-1 approximation."
        )

    with tab_data:
        st.markdown("### Export simulation data")
        # Build a tidy DataFrame (downsampled)
        ds = 10
        df = __import__("pandas").DataFrame({
            "t_s":        res["t"][::ds],
            "truth_px":   res["truth"][::ds, 0],
            "truth_py":   res["truth"][::ds, 1],
            "truth_pz":   res["truth"][::ds, 2],
            "truth_vx":   res["truth"][::ds, 3],
            "truth_vy":   res["truth"][::ds, 4],
            "truth_vz":   res["truth"][::ds, 5],
            "est_px":     res["estimate"][::ds, 0],
            "est_py":     res["estimate"][::ds, 1],
            "est_pz":     res["estimate"][::ds, 2],
            "est_vx":     res["estimate"][::ds, 3],
            "est_vy":     res["estimate"][::ds, 4],
            "est_vz":     res["estimate"][::ds, 5],
            "pos_err_m":  res["pos_err"][::ds],
            "d2":         res["d2"][::ds],
            "lock_state": res["lock_state"][::ds],
            "sigma_p":    res["sigma_p"][::ds],
            "maneuver":   res["maneuver"][::ds].astype(int),
            "snr":        res["snr"][::ds],
            "tau":        res["tau"][::ds],
            "pd":         res["pd"][::ds],
        })
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "aegis_link_simulation.csv",
            "text/csv",
            use_container_width=False,
        )
        st.dataframe(df.head(50), use_container_width=True)

        if res["int_t"] is not None:
            st.markdown("### Interceptor telemetry")
            df_int = __import__("pandas").DataFrame({
                "t_s":         res["int_t"],
                "range_m":     res["int_range"],
                "closing_mps": res["int_closing"],
                "fuel_frac":   res["int_fuel"],
                "pred_miss_m": res["int_pred_miss"],
            })
            st.download_button(
                "⬇ Download Engagement CSV",
                df_int.to_csv(index=False),
                "aegis_link_engagement.csv",
                "text/csv",
                use_container_width=False,
            )
            st.dataframe(df_int.head(50), use_container_width=True)

    st.divider()
    st.caption(
        "AEGIS-LINK · MIT License · "
        "[Source code](https://github.com/pier-car/aegis-link)"
    )


if __name__ == "__main__":
    main()
