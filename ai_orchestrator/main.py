"""
AEGIS-LINK :: ai_orchestrator/main.py
=====================================

Real-time anomaly / manoeuvre detector.

Subscribes to:
  * tcp://127.0.0.1:5555  -> ground truth from the Julia simulator
  * tcp://127.0.0.1:5556  -> EKF estimate from the C++ tracker

For each (estimate, truth) pair (matched by `packet_id` is *not* possible
because they are independent streams; we instead match on the
producer-side TAI timestamp using a small bounded-latency buffer) we
compute the **squared Mahalanobis distance**

    d^2 = (z - x_hat)^T  S^{-1}  (z - x_hat)

where:
  z      -- observed state (truth)               (R^6)
  x_hat  -- predicted/estimated state (tracker)  (R^6)
  S      -- innovation covariance, here approximated by the diagonal
            covariance carried in the estimate packet plus a fixed
            sensor R (sub-pixel-class metrology heritage).

A target is flagged as MANEUVERING when d^2 exceeds the chi-square
99% gate for k=6 degrees of freedom (~16.81). A short consecutive
streak filter (default = 3) suppresses single-frame outliers.

The script also writes one line per fused sample to stdout in CSV
form for offline analysis (pandas, matplotlib, etc.).
"""
from __future__ import annotations

import ctypes
import signal
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import zmq

# ---------------------------------------------------------------------------
#  TrackPacket binary layout (mirrors shared/messages.h, 128 bytes total)
# ---------------------------------------------------------------------------
PACKET_NBYTES   = 128
SCHEMA_V        = 1

# struct format: little-endian, no padding (we pack manually to match the
# explicit layout from the C header).
#   < I I Q  6d  6d  H H 12s
_PKT_FMT = "<IIQ6d6dHH12s"
assert struct.calcsize(_PKT_FMT) == PACKET_NBYTES, struct.calcsize(_PKT_FMT)


@dataclass(frozen=True)
class TrackPacket:
    packet_id:      int
    producer_id:    int
    timestamp_ns:   int
    state:          np.ndarray   # shape (6,), float64
    cov_diag:       np.ndarray   # shape (6,), float64
    schema_version: int
    flags:          int

    @classmethod
    def unpack(cls, buf: bytes) -> "TrackPacket":
        if len(buf) != PACKET_NBYTES:
            raise ValueError(f"bad packet size {len(buf)}")
        fields = struct.unpack(_PKT_FMT, buf)
        return cls(
            packet_id      = fields[0],
            producer_id    = fields[1],
            timestamp_ns   = fields[2],
            state          = np.asarray(fields[3:9],  dtype=np.float64),
            cov_diag       = np.asarray(fields[9:15], dtype=np.float64),
            schema_version = fields[15],
            flags          = fields[16],
        )


# ---------------------------------------------------------------------------
#  Mahalanobis analyser
# ---------------------------------------------------------------------------
PRODUCER_SIM      = 1
PRODUCER_TRACKER  = 2

# Sensor-side noise floor (matches tracker assumptions). Position 5 cm,
# velocity 20 cm/s — Pirelli-style metrology lineage.
R_DIAG = np.array([0.05**2]*3 + [0.20**2]*3, dtype=np.float64)

CHI2_99_DOF6  = 16.812
CHI2_999_DOF6 = 22.458
STREAK_FOR_ALERT = 3

# Maximum age of a buffered truth packet before we give up matching it.
MAX_MATCH_LAG_NS = 50_000_000  # 50 ms


def mahalanobis_sq(delta: np.ndarray, cov_diag: np.ndarray) -> float:
    """Squared Mahalanobis distance for a *diagonal* covariance.

    delta    : residual vector (k,)
    cov_diag : variances on the diagonal (k,), strictly positive
    """
    # Numerical floor to avoid div-by-zero on a degenerate dimension.
    var = np.maximum(cov_diag, 1e-12)
    return float(np.sum((delta * delta) / var))


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
def run(sim_addr: str = "tcp://127.0.0.1:5555",
        trk_addr: str = "tcp://127.0.0.1:5556") -> None:

    ctx = zmq.Context.instance()

    sub_truth = ctx.socket(zmq.SUB)
    sub_truth.setsockopt(zmq.SUBSCRIBE, b"")
    sub_truth.setsockopt(zmq.RCVHWM, 64)
    sub_truth.connect(sim_addr)

    sub_est = ctx.socket(zmq.SUB)
    sub_est.setsockopt(zmq.SUBSCRIBE, b"")
    sub_est.setsockopt(zmq.RCVHWM, 64)
    sub_est.connect(trk_addr)

    poller = zmq.Poller()
    poller.register(sub_truth, zmq.POLLIN)
    poller.register(sub_est,   zmq.POLLIN)

    truth_buf: Deque[TrackPacket] = deque(maxlen=512)
    streak = 0
    n_total = n_alert = 0

    print("ts_ns,packet_id,d2,maneuver,px,py,pz,ex,ey,ez", flush=True)

    running = True
    def _stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        events = dict(poller.poll(timeout=200))

        if sub_truth in events:
            try:
                pkt = TrackPacket.unpack(sub_truth.recv(zmq.NOBLOCK))
                if pkt.schema_version == SCHEMA_V and pkt.producer_id == PRODUCER_SIM:
                    truth_buf.append(pkt)
            except (zmq.Again, ValueError):
                pass

        if sub_est in events:
            try:
                est = TrackPacket.unpack(sub_est.recv(zmq.NOBLOCK))
            except (zmq.Again, ValueError):
                continue
            if est.schema_version != SCHEMA_V or est.producer_id != PRODUCER_TRACKER:
                continue

            # Match: nearest-in-time truth packet, bounded by MAX_MATCH_LAG_NS.
            truth = _match(est, truth_buf)
            if truth is None:
                continue

            # Innovation covariance ~ P_diag (estimate) + R (sensor).
            S_diag = est.cov_diag + R_DIAG
            delta  = truth.state - est.state
            d2     = mahalanobis_sq(delta, S_diag)

            n_total += 1
            is_alert = d2 > CHI2_99_DOF6
            streak   = streak + 1 if is_alert else 0
            confirmed = streak >= STREAK_FOR_ALERT
            if confirmed:
                n_alert += 1

            print(f"{est.timestamp_ns},{est.packet_id},{d2:.4f},"
                  f"{int(confirmed)},"
                  f"{truth.state[0]:.3f},{truth.state[1]:.3f},{truth.state[2]:.3f},"
                  f"{est.state[0]:.3f},{est.state[1]:.3f},{est.state[2]:.3f}",
                  flush=True)

            if confirmed and d2 > CHI2_999_DOF6:
                # Strong manoeuvre — emit a human-readable warning to stderr.
                print(f"[orch] !! MANEUVER  d^2={d2:7.2f}  "
                      f"streak={streak}  pkt={est.packet_id}",
                      file=sys.stderr, flush=True)

    print(f"[orch] done. samples={n_total} alerts={n_alert}", file=sys.stderr)


def _match(est: TrackPacket,
           truth_buf: Deque[TrackPacket]) -> Optional[TrackPacket]:
    """Pop the closest-in-time truth packet within MAX_MATCH_LAG_NS."""
    best: Optional[TrackPacket] = None
    best_dt = MAX_MATCH_LAG_NS
    for t in truth_buf:
        dt = abs(int(t.timestamp_ns) - int(est.timestamp_ns))
        if dt < best_dt:
            best_dt = dt
            best    = t
    # Drop stale truth packets to keep the buffer bounded.
    while truth_buf and (int(est.timestamp_ns) - int(truth_buf[0].timestamp_ns)
                         > MAX_MATCH_LAG_NS):
        truth_buf.popleft()
    return best


if __name__ == "__main__":
    run()
