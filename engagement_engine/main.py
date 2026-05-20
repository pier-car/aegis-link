"""
AEGIS-LINK :: engagement_engine/main.py
=======================================

Proportional-Navigation interceptor — the "knock-down" stage of the
track / hook / engage pipeline.

Subscribes to
  * tcp://127.0.0.1:5555  -> ground truth from the Julia simulator
                             (used ONLY to declare KILL/MISS after the
                              engagement ends — never as the guidance input)
  * tcp://127.0.0.1:5556  -> EKF estimate from the C++ tracker
                             (the guidance input: PN closes the loop on the
                              *filtered* track, not on the truth)

Publishes
  * tcp://127.0.0.1:5557  -> 128-byte `EngagementPacket` (binary-identical
                             layout to `TrackPacket`, see shared/messages.h)

Lock / launch logic
-------------------
The engine runs a small FSM mirroring the orchestrator's lock states:

    IDLE  ──(EKF flag AEGIS_FLAG_LOCKED rises OR sigma_p < LOCK_SIGMA_M)──▶ ARMED
    ARMED ──(first EKF frame in ARMED)─────────────────────────────────────▶ ENGAGED
    ENGAGED ──(CPA < lethal_radius)───────────────────────────────────────▶ KILL
    ENGAGED ──(timeout | fuel-out diverging | range divergence past CPA)──▶ MISS

The "lock arming" criterion is intentionally redundant with the
orchestrator's FSM: the engagement engine can run standalone (e.g. in a
unit-test loop) and will arm itself from the EKF covariance even if no
orchestrator is on the bus.

Guidance law — true Proportional Navigation
-------------------------------------------
Let r be the relative position of the target wrt the interceptor and
v_rel its time derivative. The line-of-sight (LOS) unit vector is
u = r / |r| and its rotation rate (LOS rate) is

    Omega_LOS = (r x v_rel) / |r|^2                              (1)

The closing velocity is V_c = -d|r|/dt = -(r . v_rel) / |r|.

True PN commands a lateral acceleration

    a_cmd = N' * V_c * (Omega_LOS x u_los_norm)                  (2)

with N' typically 3..5. We saturate |a_cmd_perp| at `max_lateral_g * g`.

Dynamics — 3-DoF point mass
---------------------------
The interceptor integrates

    dr/dt = v
    dv/dt = (T - D)/m * v_hat + a_cmd + g_vec
    dm/dt = -T / v_e        (while propellant remains, else T = 0)

with quadratic drag D = drag_coef * cross_section * |v|^2, and gravity
g_vec = (0, 0, -9.81). Integration uses a fixed-step RK4 at loop_hz.

Outcome packing into EngagementPacket
-------------------------------------
The 128-byte `EngagementPacket` reuses the `TrackPacket` layout:

    state[0:3]   = interceptor position [m]
    state[3:6]   = interceptor velocity [m/s]
    cov_diag[0]  = time-to-go [s]                  (geometric estimate)
    cov_diag[1]  = predicted miss-distance [m]     (current CPA prediction)
    cov_diag[2]  = LOS range to estimated target [m]
    cov_diag[3]  = closing speed [m/s]
    cov_diag[4]  = remaining fuel fraction [0,1]
    cov_diag[5]  = commanded |a_cmd| [m/s^2]
    flags        = LOCKED | ENGAGED | KILL | MISS  (bitfield)

This keeps every existing subscriber happy: it can `struct.unpack` the
same `_PKT_FMT` and read positions / "covariance-like" telemetry.
"""
from __future__ import annotations

import argparse
import os
import signal
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import zmq

try:
    import yaml
except ImportError:  # pragma: no cover - configurable fallback for headless tests
    yaml = None


# ---------------------------------------------------------------------------
#  Wire format (mirrors shared/messages.h, 128 bytes total)
# ---------------------------------------------------------------------------
PACKET_NBYTES = 128
SCHEMA_V      = 1
_PKT_FMT      = "<IIQ6d6dHH12s"
assert struct.calcsize(_PKT_FMT) == PACKET_NBYTES

PRODUCER_SIM         = 1
PRODUCER_TRACKER     = 2
PRODUCER_INTERCEPTOR = 4

# Mirrors AegisFlags in shared/messages.h.
FLAG_NONE     = 0x0000
FLAG_MANEUVER = 0x0001
FLAG_LOST_TRK = 0x0002
FLAG_LOCKED   = 0x0004
FLAG_ENGAGED  = 0x0008
FLAG_KILL     = 0x0010
FLAG_MISS     = 0x0020

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# Standalone arming threshold (used when no orchestrator is present).
DEFAULT_LOCK_SIGMA_M = 5.0
DEFAULT_LOCK_STREAK  = 25


@dataclass(frozen=True)
class TrackPacket:
    packet_id:      int
    producer_id:    int
    timestamp_ns:   int
    state:          np.ndarray
    cov_diag:       np.ndarray
    schema_version: int
    flags:          int

    @classmethod
    def unpack(cls, buf: bytes) -> "TrackPacket":
        if len(buf) != PACKET_NBYTES:
            raise ValueError(f"bad packet size {len(buf)}")
        f = struct.unpack(_PKT_FMT, buf)
        return cls(
            packet_id      = f[0],
            producer_id    = f[1],
            timestamp_ns   = f[2],
            state          = np.asarray(f[3:9],  dtype=np.float64),
            cov_diag       = np.asarray(f[9:15], dtype=np.float64),
            schema_version = f[15],
            flags          = f[16],
        )


def pack_engagement(
    packet_id:    int,
    timestamp_ns: int,
    pos:          np.ndarray,
    vel:          np.ndarray,
    tgo:          float,
    pred_miss:    float,
    los_range:    float,
    closing_spd:  float,
    fuel_frac:    float,
    a_cmd_mag:    float,
    flags:        int,
) -> bytes:
    """Build a 128-byte EngagementPacket."""
    state = np.concatenate([pos.astype(np.float64), vel.astype(np.float64)])
    cov = np.array(
        [tgo, pred_miss, los_range, closing_spd, fuel_frac, a_cmd_mag],
        dtype=np.float64,
    )
    return struct.pack(
        _PKT_FMT,
        int(packet_id) & 0xFFFFFFFF,
        PRODUCER_INTERCEPTOR,
        int(timestamp_ns) & 0xFFFFFFFFFFFFFFFF,
        *state.tolist(),
        *cov.tolist(),
        SCHEMA_V,
        int(flags) & 0xFFFF,
        b"\x00" * 12,
    )


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "interceptor": {
        "launch_offset_m":  [0.0, 0.0, 0.0],
        "launch_alt_m":     2.0,
        "mass_kg":         12.0,
        "propellant_kg":    6.0,
        "thrust_N":      1800.0,
        "exhaust_vel_mps": 1900.0,
        "drag_coef":        0.30,
        "cross_section_m2": 0.012,
        "max_lateral_g":   40.0,
    },
    "guidance": {
        "nav_ratio_N":       4.0,
        "command_latency_s": 0.020,
    },
    "engagement": {
        "lethal_radius_m":   5.0,
        "max_flight_time_s": 30.0,
        "abort_range_m":    50.0,
        "loop_hz":         100.0,
    },
    "network": {
        "truth_endpoint":    "tcp://127.0.0.1:5555",
        "estimate_endpoint": "tcp://127.0.0.1:5556",
        "publish_endpoint":  "tcp://*:5557",
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return _DEFAULT_CONFIG
    p = Path(path)
    if not p.is_file():
        print(f"[engage] config {path} not found, using defaults",
              file=sys.stderr)
        return _DEFAULT_CONFIG
    if yaml is None:
        print(f"[engage] PyYAML not installed; ignoring {path}, using defaults",
              file=sys.stderr)
        return _DEFAULT_CONFIG
    with p.open("r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    return _deep_merge(_DEFAULT_CONFIG, user)


# ---------------------------------------------------------------------------
#  Interceptor dynamics
# ---------------------------------------------------------------------------
@dataclass
class Interceptor:
    pos:           np.ndarray
    vel:           np.ndarray
    mass:          float
    propellant:    float
    thrust:        float
    exhaust_vel:   float
    drag_coef:     float
    cross_section: float
    max_lat_acc:   float           # m/s^2 (max_lateral_g * g)
    initial_mass:  float = field(init=False)
    initial_prop:  float = field(init=False)

    def __post_init__(self) -> None:
        self.initial_mass = float(self.mass)
        self.initial_prop = float(self.propellant)

    @property
    def fuel_fraction(self) -> float:
        return 0.0 if self.initial_prop <= 0.0 else \
            max(0.0, self.propellant) / self.initial_prop

    def _accel(self, vel: np.ndarray, a_cmd: np.ndarray,
               thrust_active: bool) -> np.ndarray:
        """Total acceleration: thrust along velocity, drag opposite, command, gravity."""
        speed = float(np.linalg.norm(vel))
        v_hat = vel / speed if speed > 1e-6 else np.zeros(3)
        T = self.thrust if thrust_active else 0.0
        # Quadratic drag along -v_hat.
        D = self.drag_coef * self.cross_section * speed * speed
        # Thrust + drag along v_hat (drag opposes motion).
        a_axial = (T - D) / max(self.mass, 1e-3) * v_hat
        return a_axial + a_cmd + GRAVITY

    def step(self, dt: float, a_cmd: np.ndarray) -> None:
        """RK4 integration over `dt` with constant `a_cmd` (zero-order hold)."""
        thrust_active = self.propellant > 0.0
        # Cap commanded lateral acceleration magnitude.
        m = float(np.linalg.norm(a_cmd))
        if m > self.max_lat_acc:
            a_cmd = a_cmd * (self.max_lat_acc / m)

        def deriv(p, v):
            return v, self._accel(v, a_cmd, thrust_active)

        p0, v0 = self.pos, self.vel
        k1p, k1v = deriv(p0,                 v0)
        k2p, k2v = deriv(p0 + 0.5*dt*k1p,    v0 + 0.5*dt*k1v)
        k3p, k3v = deriv(p0 + 0.5*dt*k2p,    v0 + 0.5*dt*k2v)
        k4p, k4v = deriv(p0 +     dt*k3p,    v0 +     dt*k3v)

        self.pos = p0 + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        self.vel = v0 + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)

        # Mass-burn.
        if thrust_active and self.exhaust_vel > 1e-3:
            m_dot = self.thrust / self.exhaust_vel
            burn  = m_dot * dt
            burn  = min(burn, self.propellant)
            self.propellant -= burn
            self.mass       -= burn


# ---------------------------------------------------------------------------
#  Proportional-Navigation guidance
# ---------------------------------------------------------------------------
def pn_command(r_int: np.ndarray, v_int: np.ndarray,
               r_tgt: np.ndarray, v_tgt: np.ndarray,
               nav_ratio: float) -> Tuple[np.ndarray, float, float, float]:
    """Compute the PN lateral acceleration command.

    Returns
    -------
    a_cmd       : (3,) acceleration command in world frame [m/s^2]
    tgo         : geometric time-to-go [s] (range / closing speed, clipped)
    los_range   : current range [m]
    closing_spd : -d(range)/dt [m/s]  (positive = closing)
    """
    r = r_tgt - r_int                       # relative position (LOS vector)
    v = v_tgt - v_int                       # relative velocity
    rng = float(np.linalg.norm(r))
    if rng < 1e-6:
        return np.zeros(3), 0.0, 0.0, 0.0
    u_los = r / rng
    # LOS rotation rate Omega_LOS = (r x v) / |r|^2
    omega = np.cross(r, v) / (rng * rng)
    # Closing speed (positive when range is shrinking).
    closing = -float(np.dot(u_los, v))
    # True PN: a_cmd = N' * V_c * (Omega_LOS x u_los)
    a_cmd = nav_ratio * closing * np.cross(omega, u_los)
    # Time-to-go: geometric (kinematic) estimate.
    tgo = rng / max(closing, 1e-3) if closing > 0 else float("inf")
    return a_cmd, tgo, rng, closing


def predict_miss(r_int: np.ndarray, v_int: np.ndarray,
                 r_tgt: np.ndarray, v_tgt: np.ndarray) -> float:
    """Closed-form straight-line miss-distance prediction.

    Assumes constant relative velocity over the engagement; gives a clean
    geometric prediction useful for the live HUD and CSV.
    """
    r = r_tgt - r_int
    v = v_tgt - v_int
    vv = float(np.dot(v, v))
    if vv < 1e-9:
        return float(np.linalg.norm(r))
    t_cpa = -float(np.dot(r, v)) / vv
    if t_cpa < 0.0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r + t_cpa * v))


# ---------------------------------------------------------------------------
#  Engagement state machine
# ---------------------------------------------------------------------------
ENG_IDLE    = 0
ENG_ARMED   = 1
ENG_ENGAGED = 2
ENG_KILL    = 3
ENG_MISS    = 4

_ENG_NAME = {
    ENG_IDLE: "IDLE", ENG_ARMED: "ARMED", ENG_ENGAGED: "ENGAGED",
    ENG_KILL: "KILL", ENG_MISS: "MISS",
}


@dataclass
class EngagementResult:
    """Summary of a single engagement; one row per Monte-Carlo sample."""
    outcome:        str               # "KILL" | "MISS" | "NO_LOCK"
    cpa_m:          float             # closest point of approach achieved [m]
    time_to_kill_s: float             # wall-time from ENGAGED to terminal [s]
    flight_time_s:  float             # interceptor airborne time [s]
    fuel_used_frac: float             # 1.0 - remaining propellant fraction
    pred_miss_m:    float             # last-published predicted miss [m]
    closing_spd_mps: float            # closing speed at CPA [m/s]


# ---------------------------------------------------------------------------
#  The engagement loop
# ---------------------------------------------------------------------------
class EngagementEngine:
    def __init__(self, cfg: dict, *,
                 print_csv:   bool = True,
                 csv_stream=None,
                 lock_sigma_m: float = DEFAULT_LOCK_SIGMA_M,
                 lock_streak:  int = DEFAULT_LOCK_STREAK) -> None:
        self.cfg = cfg
        self._print_csv = print_csv
        self._csv = csv_stream if csv_stream is not None else sys.stdout
        self._lock_sigma_m = lock_sigma_m
        self._lock_streak  = lock_streak

        ic = cfg["interceptor"]
        self._launch_offset = np.asarray(ic["launch_offset_m"], dtype=np.float64)
        self._launch_alt    = float(ic["launch_alt_m"])
        max_lat_acc = float(ic["max_lateral_g"]) * 9.81
        self._make_interceptor = lambda pos, vel: Interceptor(
            pos=pos.astype(np.float64).copy(),
            vel=vel.astype(np.float64).copy(),
            mass=float(ic["mass_kg"]),
            propellant=float(ic["propellant_kg"]),
            thrust=float(ic["thrust_N"]),
            exhaust_vel=float(ic["exhaust_vel_mps"]),
            drag_coef=float(ic["drag_coef"]),
            cross_section=float(ic["cross_section_m2"]),
            max_lat_acc=max_lat_acc,
        )

        gd = cfg["guidance"]
        self._nav_ratio = float(gd["nav_ratio_N"])

        eng = cfg["engagement"]
        self._lethal_r  = float(eng["lethal_radius_m"])
        self._max_t     = float(eng["max_flight_time_s"])
        self._abort_r   = float(eng["abort_range_m"])
        self._loop_dt   = 1.0 / float(eng["loop_hz"])

    # ----- public API -----------------------------------------------------
    def run(self, *, duration_s: Optional[float] = None) -> EngagementResult:
        """Run the engine until terminal state or `duration_s` seconds.

        Returns an `EngagementResult` summarising the outcome.
        """
        cfg_net = self.cfg["network"]
        ctx = zmq.Context.instance()
        sub_truth = ctx.socket(zmq.SUB)
        sub_truth.setsockopt(zmq.SUBSCRIBE, b"")
        sub_truth.setsockopt(zmq.RCVHWM, 64)
        sub_truth.connect(cfg_net["truth_endpoint"])

        sub_est = ctx.socket(zmq.SUB)
        sub_est.setsockopt(zmq.SUBSCRIBE, b"")
        sub_est.setsockopt(zmq.RCVHWM, 64)
        sub_est.connect(cfg_net["estimate_endpoint"])

        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 64)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(cfg_net["publish_endpoint"])

        poller = zmq.Poller()
        poller.register(sub_truth, zmq.POLLIN)
        poller.register(sub_est,   zmq.POLLIN)

        try:
            return self._loop(sub_truth, sub_est, pub, poller, duration_s)
        finally:
            for s in (sub_truth, sub_est, pub):
                try:
                    s.close(0)
                except Exception:
                    pass

    # ----- inner loop -----------------------------------------------------
    def _loop(self, sub_truth, sub_est, pub, poller,
              duration_s: Optional[float]) -> EngagementResult:
        truth_buf: Deque[TrackPacket] = deque(maxlen=512)
        last_est:  Optional[TrackPacket] = None
        in_gate:   int = 0

        state        = ENG_IDLE
        intercept:   Optional[Interceptor] = None
        t_engage_ns: int = 0
        t_term_ns:   int = 0
        cpa:         float = float("inf")
        cpa_closing: float = 0.0
        last_pred_miss: float = float("inf")
        last_a_cmd:  float = 0.0
        last_tgo:    float = float("inf")
        last_range:  float = float("inf")
        last_close:  float = 0.0
        pkt_id_out:  int = 0
        prev_range:  float = float("inf")

        running = True
        def _stop(signum, frame):
            nonlocal running
            running = False
        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)

        if self._print_csv:
            print("ts_ns,state,ix,iy,iz,vx,vy,vz,range,closing,"
                  "pred_miss,fuel_frac,a_cmd",
                  file=self._csv, flush=True)

        t_loop = time.perf_counter()
        deadline = (t_loop + duration_s) if duration_s else None

        while running:
            timeout_ms = max(1, int(self._loop_dt * 1000))
            events = dict(poller.poll(timeout=timeout_ms))

            if sub_truth in events:
                try:
                    pkt = TrackPacket.unpack(sub_truth.recv(zmq.NOBLOCK))
                    if pkt.schema_version == SCHEMA_V \
                            and pkt.producer_id == PRODUCER_SIM:
                        truth_buf.append(pkt)
                except (zmq.Again, ValueError):
                    pass

            new_est = False
            if sub_est in events:
                try:
                    est = TrackPacket.unpack(sub_est.recv(zmq.NOBLOCK))
                    if est.schema_version == SCHEMA_V \
                            and est.producer_id == PRODUCER_TRACKER:
                        last_est = est
                        new_est  = True
                except (zmq.Again, ValueError):
                    pass

            # ---- FSM updates that depend on a fresh estimate --------------
            if new_est and last_est is not None:
                sigma_p = float(np.sqrt(
                    max(np.sum(last_est.cov_diag[0:3]), 0.0)))
                lock_signal = (last_est.flags & FLAG_LOCKED) != 0
                if state == ENG_IDLE:
                    if lock_signal or sigma_p < self._lock_sigma_m:
                        in_gate += 1
                    else:
                        in_gate = 0
                    if lock_signal or in_gate >= self._lock_streak:
                        state = ENG_ARMED
                        print(f"[engage] state -> ARMED  "
                              f"(sigma_p={sigma_p:.2f} m, lock_flag={int(lock_signal)})",
                              file=sys.stderr, flush=True)

                if state == ENG_ARMED:
                    # Launch the interceptor below the target ground projection.
                    target_pos = last_est.state[0:3].copy()
                    launch_pos = np.array([
                        target_pos[0] + self._launch_offset[0],
                        target_pos[1] + self._launch_offset[1],
                        self._launch_alt + self._launch_offset[2],
                    ])
                    # Initial vertical impulse toward the target (1 m/s; PN
                    # commands take over immediately).
                    init_dir = target_pos - launch_pos
                    n = float(np.linalg.norm(init_dir))
                    init_vel = (init_dir / n) * 1.0 if n > 1e-6 \
                        else np.array([0.0, 0.0, 1.0])
                    intercept = self._make_interceptor(launch_pos, init_vel)
                    state = ENG_ENGAGED
                    t_engage_ns = int(last_est.timestamp_ns)
                    prev_range = float("inf")
                    print(f"[engage] state -> ENGAGED  "
                          f"launch={launch_pos.tolist()}",
                          file=sys.stderr, flush=True)

            # ---- Integrate interceptor at fixed loop_dt -------------------
            now = time.perf_counter()
            if state == ENG_ENGAGED and intercept is not None \
                    and last_est is not None and (now - t_loop) >= self._loop_dt:
                dt = now - t_loop
                t_loop = now
                # Cap dt to keep RK4 well-behaved on jittery scheduling.
                dt = min(dt, 5 * self._loop_dt)

                # Guidance on the EKF estimate (NOT the truth).
                r_tgt = last_est.state[0:3]
                v_tgt = last_est.state[3:6]
                a_cmd, tgo, rng, closing = pn_command(
                    intercept.pos, intercept.vel, r_tgt, v_tgt,
                    self._nav_ratio,
                )
                pred_miss = predict_miss(intercept.pos, intercept.vel,
                                         r_tgt, v_tgt)

                intercept.step(dt, a_cmd)

                last_a_cmd  = float(np.linalg.norm(a_cmd))
                last_tgo    = float(tgo)
                last_range  = float(rng)
                last_close  = float(closing)
                last_pred_miss = float(pred_miss)
                if rng < cpa:
                    cpa = rng
                    cpa_closing = closing

                # ---- Terminal conditions ---------------------------------
                flight_t = (int(last_est.timestamp_ns) - t_engage_ns) * 1e-9
                if cpa <= self._lethal_r:
                    state = ENG_KILL
                    t_term_ns = int(last_est.timestamp_ns)
                    print(f"[engage] !! KILL  cpa={cpa:.2f} m  "
                          f"t_flight={flight_t:.2f} s",
                          file=sys.stderr, flush=True)
                elif flight_t > self._max_t:
                    state = ENG_MISS
                    t_term_ns = int(last_est.timestamp_ns)
                    print(f"[engage] xx MISS (timeout)  cpa={cpa:.2f} m",
                          file=sys.stderr, flush=True)
                elif rng > prev_range and rng > (cpa + self._abort_r):
                    state = ENG_MISS
                    t_term_ns = int(last_est.timestamp_ns)
                    print(f"[engage] xx MISS (range diverged past CPA)  "
                          f"cpa={cpa:.2f} m  rng={rng:.2f} m",
                          file=sys.stderr, flush=True)
                prev_range = rng
            elif state != ENG_ENGAGED:
                # Keep t_loop fresh so the first dt after launch is small.
                t_loop = now

            # ---- Publish telemetry at every loop tick ---------------------
            self._publish(pub, intercept, state, pkt_id_out,
                          last_tgo, last_pred_miss, last_range, last_close,
                          last_a_cmd)
            pkt_id_out += 1

            # ---- Terminal: produce result & exit --------------------------
            if state in (ENG_KILL, ENG_MISS):
                # Try to refine CPA against truth (if we have a truth packet
                # near t_term_ns).
                truth_at = self._nearest_truth(truth_buf, t_term_ns)
                if truth_at is not None and intercept is not None:
                    truth_cpa = float(np.linalg.norm(
                        intercept.pos - truth_at.state[0:3]))
                    # Take the tighter of the two (we missed it if both say so).
                    cpa = min(cpa, truth_cpa)
                fuel_used = 1.0 - (intercept.fuel_fraction if intercept else 1.0)
                flight_t = (t_term_ns - t_engage_ns) * 1e-9
                return EngagementResult(
                    outcome=_ENG_NAME[state],
                    cpa_m=float(cpa),
                    time_to_kill_s=float(flight_t),
                    flight_time_s=float(flight_t),
                    fuel_used_frac=float(fuel_used),
                    pred_miss_m=float(last_pred_miss),
                    closing_spd_mps=float(cpa_closing),
                )

            if deadline is not None and now >= deadline:
                outcome = "MISS" if state == ENG_ENGAGED else "NO_LOCK"
                fuel_used = 1.0 - (intercept.fuel_fraction if intercept else 1.0)
                return EngagementResult(
                    outcome=outcome,
                    cpa_m=float(cpa),
                    time_to_kill_s=0.0,
                    flight_time_s=float((now - (t_loop - self._loop_dt))
                                        if state == ENG_ENGAGED else 0.0),
                    fuel_used_frac=float(fuel_used),
                    pred_miss_m=float(last_pred_miss),
                    closing_spd_mps=float(cpa_closing),
                )

        # Loop exited via signal.
        outcome = _ENG_NAME[state] if state in (ENG_KILL, ENG_MISS) else "NO_LOCK"
        fuel_used = 1.0 - (intercept.fuel_fraction if intercept else 1.0)
        return EngagementResult(
            outcome=outcome,
            cpa_m=float(cpa),
            time_to_kill_s=0.0,
            flight_time_s=0.0,
            fuel_used_frac=float(fuel_used),
            pred_miss_m=float(last_pred_miss),
            closing_spd_mps=float(cpa_closing),
        )

    # ----- helpers --------------------------------------------------------
    def _publish(self, pub, intercept: Optional[Interceptor], state: int,
                 pkt_id: int, tgo: float, pred_miss: float, rng: float,
                 closing: float, a_cmd: float) -> None:
        flags = FLAG_NONE
        if state >= ENG_ARMED:
            flags |= FLAG_LOCKED
        if state == ENG_ENGAGED:
            flags |= FLAG_ENGAGED
        if state == ENG_KILL:
            flags |= FLAG_KILL | FLAG_ENGAGED | FLAG_LOCKED
        if state == ENG_MISS:
            flags |= FLAG_MISS | FLAG_LOCKED

        if intercept is None:
            pos = np.zeros(3); vel = np.zeros(3); fuel = 1.0
        else:
            pos = intercept.pos; vel = intercept.vel; fuel = intercept.fuel_fraction

        # Sanitize non-finite telemetry before binary packing.
        def _finite(x: float, fallback: float = 0.0) -> float:
            return float(x) if np.isfinite(x) else fallback

        ts_ns = time.time_ns()
        buf = pack_engagement(
            packet_id=pkt_id,
            timestamp_ns=ts_ns,
            pos=pos, vel=vel,
            tgo=_finite(tgo, 0.0),
            pred_miss=_finite(pred_miss, 0.0),
            los_range=_finite(rng, 0.0),
            closing_spd=_finite(closing, 0.0),
            fuel_frac=_finite(fuel, 0.0),
            a_cmd_mag=_finite(a_cmd, 0.0),
            flags=flags,
        )
        try:
            pub.send(buf, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

        if self._print_csv:
            print(f"{ts_ns},{_ENG_NAME.get(state,'?')},"
                  f"{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},"
                  f"{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f},"
                  f"{_finite(rng):.3f},{_finite(closing):.3f},"
                  f"{_finite(pred_miss):.3f},{_finite(fuel):.3f},"
                  f"{_finite(a_cmd):.3f}",
                  file=self._csv, flush=True)

    @staticmethod
    def _nearest_truth(buf: Deque[TrackPacket],
                       ts_ns: int) -> Optional[TrackPacket]:
        if not buf:
            return None
        best = None
        best_dt = 10**18
        for t in buf:
            dt = abs(int(t.timestamp_ns) - int(ts_ns))
            if dt < best_dt:
                best_dt = dt
                best = t
        return best


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(
        Path(__file__).with_name("config.yaml")),
        help="YAML config (default: %(default)s)")
    p.add_argument("--duration", type=float, default=None,
        help="Stop after N seconds if no terminal state is reached")
    p.add_argument("--quiet", action="store_true",
        help="Suppress the CSV log on stdout")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    eng = EngagementEngine(cfg, print_csv=not args.quiet)
    result = eng.run(duration_s=args.duration)
    print(f"[engage] outcome={result.outcome} cpa={result.cpa_m:.2f} m  "
          f"t_flight={result.flight_time_s:.2f}s  "
          f"fuel_used={result.fuel_used_frac*100:.0f}%",
          file=sys.stderr, flush=True)
    return 0 if result.outcome == "KILL" else 1


if __name__ == "__main__":
    sys.exit(main())
