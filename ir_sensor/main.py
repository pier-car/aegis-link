"""
AEGIS-LINK :: ir_sensor/main.py
================================

Infrared Search and Track (IRST) sensor — standalone process.

Subscribes to  tcp://127.0.0.1:5555  (Julia truth stream, producer_id=1)
Publishes      tcp://*:5558          (IRPacket, 128-byte TrackPacket layout,
                                      producer_id=5 = AEGIS_PRODUCER_IR_SENSOR)

Sensor model
------------
Passive staring FPA in the MWIR band (3–5 μm), boresighted on the
surveillance hemisphere from a fixed ground site.

Sub-models
~~~~~~~~~~
1.  Radiometric target signature — aerodynamic skin-heating + base emitter

        I(v) = I₀ · (1 + κ · (|v| / v_ref)²)

    Aerodynamic heating scales roughly as |v|² (kinetic energy flux
    converted to surface temperature rise via Stefan–Boltzmann).  I₀ is
    the base MWIR radiant intensity of the target (~200 W/sr for a
    fast-moving small object).

2.  Atmospheric transmission — Beer–Lambert in the MWIR clear-air window

        τ(R) = exp(−α · R)

    α ≈ 2 × 10⁻⁵ m⁻¹ (≈ 0.02 dB/km) for a standard clear-air model;
    increases to ~5 × 10⁻⁵ m⁻¹ in moderate haze.

3.  SNR at the detector

        SNR = (I · τ / R²) / NEI

    where NEI is the noise-equivalent irradiance at the aperture [W/m²].
    I · τ / R² gives the signal irradiance; dividing by NEI converts to
    linear SNR.

4.  Detection probability — logistic (Albersheim / Swerling-1 approximation)

        P_D = σ(k · (SNR − SNR_thresh))

    SNR_thresh = 4 gives P_D ≈ 0.5 at threshold, rising steeply to 1.

5.  Angular noise — Cramér–Rao bound for focal-plane centroid estimation

        σ_θ = IFOV / (√2 · SNR),   clipped to [σ_min, σ_max]

    This captures the SNR-limited resolution limit of sub-pixel centroiding.

6.  False alarms — Poisson background clutter at rate λ_fa per frame.

Wire-format semantics (TrackPacket 128-byte layout)
----------------------------------------------------
  producer_id  = 5  (AEGIS_PRODUCER_IR_SENSOR)

  state[0:3]   = estimated 3-D position [m], ENU frame.
                 Derived from measured azimuth/elevation + a range estimate
                 (passive IRST has no active ranging; range uncertainty is
                 reflected in cov_diag).
  state[3:6]   = estimated velocity [m/s], finite-differenced over the two
                 most recent valid detections.

  cov_diag[0]  = σ²_x  [m²]   position variance, ENU x
  cov_diag[1]  = σ²_y  [m²]   position variance, ENU y
  cov_diag[2]  = σ²_z  [m²]   position variance, ENU z
                 The cross-range components are small (R·σ_θ)² while the
                 along-range component is large (σ_range_frac·R)².
  cov_diag[3]  = σ²_vel [m²/s²]  velocity variance (same on all 3 axes)
  cov_diag[4]  = SNR            diagnostic (not a covariance)
  cov_diag[5]  = τ              diagnostic atmospheric transmission

  flags bit AEGIS_FLAG_TEST  is set on injected false-alarm packets.

Note on missed detections
~~~~~~~~~~~~~~~~~~~~~~~~~
When the target is not detected on a given frame (P_D Bernoulli draw
fails), the IRST process publishes *nothing* for that frame.  Silence is
the absence of a detection; downstream consumers must handle gaps.
"""
from __future__ import annotations

import argparse
import math
import signal
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import zmq

# ---------------------------------------------------------------------------
#  Wire format (mirrors shared/messages.h  TrackPacket, 128 bytes total)
# ---------------------------------------------------------------------------
PACKET_NBYTES = 128
SCHEMA_V      = 1
_PKT_FMT      = "<IIQ6d6dHH12s"
assert struct.calcsize(_PKT_FMT) == PACKET_NBYTES, struct.calcsize(_PKT_FMT)

# Producer identifiers.
PRODUCER_SIM      = 1
PRODUCER_IR_SENSOR = 5   # AEGIS_PRODUCER_IR_SENSOR (shared/messages.h)

# Flag bits (mirror shared/messages.h::AegisFlags).
FLAG_NONE     = 0x0000
FLAG_TEST     = 0x8000   # set on false-alarm (synthetic) packets

# ---------------------------------------------------------------------------
#  IRST sensor physical parameters
# ---------------------------------------------------------------------------

# Aperture diameter (15 cm — representative of a medium-class IRST telescope).
APERTURE_DIAM_M   = 0.15          # [m]

# MWIR focal-plane IFOV — 0.1 mrad is typical for a 15 cm aperture at f/4
# with a 15 µm pitch detector (f=600 mm → IFOV = 15e-6/0.6 = 25 µrad; we
# round up to 0.1 mrad to account for pixel sampling, diffraction, and
# optical aberrations).
IFOV_MRAD         = 0.10          # [mrad]
IFOV_RAD          = IFOV_MRAD * 1e-3

# Target radiometric model.
TARGET_I0_W_SR    = 200.0         # base radiant intensity                 [W/sr]
AERO_HEAT_K       = 0.80          # aerodynamic heating scale coefficient
V_AERO_REF_M_S    = 300.0         # reference speed for heating model       [m/s]

# MWIR clear-air atmospheric extinction (Beer–Lambert).
ATMO_ALPHA_PER_M  = 2.0e-5        # ≈ 0.02 dB/km                           [1/m]

# System noise-equivalent irradiance at the aperture plane.
# NEI includes detector NETD, optics throughput, and integration time.
# 5×10⁻¹¹ W/m² is achievable for a cooled InSb/MCT FPA at 10 ms integration.
NEI_W_M2          = 5.0e-11       # [W/m²]

# Detection probability model.
SNR_DETECT_THRESH = 4.0           # SNR at which P_D = 50 %
SNR_LOGISTIC_K    = 1.50          # logistic curve steepness

# False alarm rate.
P_FA_PER_FRAME    = 5.0e-7        # Poisson clutter false-alarm rate per frame

# Angular noise bounds.
SIGMA_ANGLE_MIN_RAD = 0.02e-3     # 0.02 mrad — diffraction / centroid floor
SIGMA_ANGLE_MAX_RAD = 5.00e-3     # 5.00 mrad — faint-target / SNR floor

# Range estimation uncertainty fractions (passive IRST has no active ranging;
# these model accumulated kinematic observability over the track history).
SIGMA_RANGE_FRAC_INIT  = 0.50     # 50 % fractional uncertainty at first detection
SIGMA_RANGE_FRAC_TRACK = 0.30     # 30 % after the track is established
TRACK_ESTABLISHED_N    = 10       # consecutive detections for "established" status

# ---------------------------------------------------------------------------
#  Decoded packet dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackPacket:
    """Decoded 128-byte packet from any AEGIS-LINK producer."""
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


def pack_ir_packet(pkt_id: int, ts_ns: int,
                   state6: np.ndarray, cov6: np.ndarray,
                   flags: int) -> bytes:
    """Pack a 128-byte IRPacket (TrackPacket layout, producer_id=5)."""
    return struct.pack(
        _PKT_FMT,
        int(pkt_id) & 0xFFFFFFFF,
        PRODUCER_IR_SENSOR,
        int(ts_ns)  & 0xFFFFFFFFFFFFFFFF,
        *state6.tolist(),
        *cov6.tolist(),
        SCHEMA_V,
        int(flags)  & 0xFFFF,
        b"\x00" * 12,
    )


# ---------------------------------------------------------------------------
#  Radiometric sub-models
# ---------------------------------------------------------------------------

def target_radiant_intensity(speed_m_s: float) -> float:
    """MWIR radiant intensity of the target [W/sr].

    Aerodynamic skin heating contribution scales as |v|²:

        I(v) = I₀ · (1 + κ · (|v| / v_ref)²)
    """
    heating = AERO_HEAT_K * (speed_m_s / V_AERO_REF_M_S) ** 2
    return TARGET_I0_W_SR * (1.0 + heating)


def atmospheric_transmission(range_m: float) -> float:
    """MWIR one-way atmospheric transmission over slant range [m]."""
    return math.exp(-ATMO_ALPHA_PER_M * range_m)


def compute_snr(intensity_w_sr: float, tau: float, range_m: float) -> float:
    """Linear SNR at the focal plane.

    Signal irradiance at the aperture:
        E_sig = I · τ / R²   [W/m²]

    SNR = E_sig / NEI
    """
    if range_m < 1.0:
        return float("inf")
    e_sig = intensity_w_sr * tau / (range_m * range_m)
    return e_sig / NEI_W_M2


def detection_probability(snr: float) -> float:
    """P_D via logistic approximation (Albersheim / Swerling-1).

    P_D = 1 / (1 + exp(−k · (SNR − SNR_thresh)))

    Approaches 1 for SNR ≫ thresh, 0 for SNR ≪ thresh.
    """
    x = SNR_LOGISTIC_K * (snr - SNR_DETECT_THRESH)
    # Guard against overflow in exp for very large |x|.
    if x >  50.0: return 1.0
    if x < -50.0: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def angular_noise_sigma(snr: float) -> float:
    """1-σ angular measurement noise [rad], Cramér–Rao bound.

        σ_θ = IFOV / (√2 · SNR)

    Clipped to [σ_min, σ_max] to represent physical limits.
    """
    if snr <= 0.0:
        return SIGMA_ANGLE_MAX_RAD
    sigma = IFOV_RAD / (math.sqrt(2.0) * snr)
    return float(np.clip(sigma, SIGMA_ANGLE_MIN_RAD, SIGMA_ANGLE_MAX_RAD))


# ---------------------------------------------------------------------------
#  IRST sensor state
# ---------------------------------------------------------------------------

class IRSTSensor:
    """Encapsulates per-sensor state for one tracked target.

    Responsibilities:
      - Apply the radiometric / detection model to each truth frame.
      - Add IFOV-limited angular noise to the measured AZ/EL.
      - Reconstruct a 3-D position estimate from noisy angles + range prior.
      - Propagate an appropriate covariance that honestly reflects the
        bearing-only nature of a passive IRST (large along-range variance,
        small cross-range variance).
      - Stochastically inject false alarms from background clutter.
    """

    def __init__(self, sensor_pos: np.ndarray, rng: np.random.Generator) -> None:
        self._sensor_pos  = sensor_pos.copy()       # ENU [m]
        self._rng         = rng
        self._det_count   = 0                        # consecutive detection count
        self._prev_pos    : Optional[np.ndarray] = None
        self._prev_ts_ns  : Optional[int]        = None

    # ------------------------------------------------------------------
    def process_truth(
        self, truth: TrackPacket
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        """Try to detect the truth target and return IRPacket fields.

        Returns (state6, cov6, flags, snr, tau) if detected, else None.
        """
        rel     = truth.state[0:3] - self._sensor_pos     # relative position [m]
        range_m = float(np.linalg.norm(rel))
        if range_m < 1.0:
            return None

        speed   = float(np.linalg.norm(truth.state[3:6]))

        # Radiometric computation.
        intensity = target_radiant_intensity(speed)
        tau       = atmospheric_transmission(range_m)
        snr       = compute_snr(intensity, tau, range_m)

        # Stochastic detection draw.
        p_d = detection_probability(snr)
        if self._rng.random() >= p_d:
            self._det_count = 0    # reset consecutive-detection counter on miss
            return None

        self._det_count += 1

        # --- Angular measurement (AZ, EL) with IFOV-limited noise ---
        sigma_a   = angular_noise_sigma(snr)
        az_true   = math.atan2(rel[1], rel[0])
        el_true   = math.asin(float(np.clip(rel[2] / range_m, -1.0, 1.0)))

        az_meas   = az_true + self._rng.normal(0.0, sigma_a)
        el_meas   = el_true + self._rng.normal(0.0, sigma_a)

        # --- Range estimate (passive sensor — no active ranging) ---
        # Fractional uncertainty decreases once the track is established,
        # modelling accumulated kinematic observability from track history.
        frac = (SIGMA_RANGE_FRAC_TRACK
                if self._det_count >= TRACK_ESTABLISHED_N
                else SIGMA_RANGE_FRAC_INIT)
        sigma_r   = frac * range_m
        range_est = max(range_m + self._rng.normal(0.0, sigma_r), 1.0)

        # --- Reconstruct 3-D position from noisy angles + range estimate ---
        cos_el = math.cos(el_meas)
        los_measured = np.array([
            cos_el * math.cos(az_meas),
            cos_el * math.sin(az_meas),
            math.sin(el_meas),
        ])
        pos_est = self._sensor_pos + range_est * los_measured

        # --- Velocity estimate (finite difference over last two detections) ---
        vel_est = np.zeros(3)
        if self._prev_pos is not None and self._prev_ts_ns is not None:
            dt = (int(truth.timestamp_ns) - self._prev_ts_ns) * 1e-9
            if 0.0 < dt <= 1.0:
                vel_est = (pos_est - self._prev_pos) / dt

        self._prev_pos   = pos_est.copy()
        self._prev_ts_ns = int(truth.timestamp_ns)

        # --- Position covariance (diagonal approximation) ---
        # True LOS unit vector used to project angular and range uncertainties
        # into the ENU frame.
        los_true = rel / range_m

        # Cross-range 1-sigma [m] (angular noise projected to range).
        sigma_cross = range_est * sigma_a   # σ_θ · R

        # Full 3×3 position covariance:
        #   P_pos = σ_r²  · (los ⊗ los)   (along-range, large)
        #         + σ_c²  · (I − los ⊗ los) (cross-range, small)
        # Diagonal extracted for the TrackPacket cov_diag field.
        cov_pos_3x3 = (sigma_r ** 2) * np.outer(los_true, los_true) + \
                      (sigma_cross ** 2) * (np.eye(3) - np.outer(los_true, los_true))
        pos_var = np.diag(cov_pos_3x3)    # shape (3,)

        # Velocity variance — combine range-rate and cross-range-rate contributions.
        # This is an upper bound from the position noise propagated over one dt step.
        if self._prev_ts_ns is not None:
            dt_ref = max((int(truth.timestamp_ns) - self._prev_ts_ns) * 1e-9, 0.01)
        else:
            dt_ref = 0.01
        vel_var_val = max((sigma_r ** 2 + sigma_cross ** 2) / (dt_ref ** 2), 1.0)
        vel_var = np.full(3, vel_var_val)

        state6 = np.concatenate([pos_est, vel_est])
        # cov_diag[3] = velocity variance (same all axes)
        # cov_diag[4] = SNR   (diagnostic)
        # cov_diag[5] = tau   (diagnostic)
        cov6 = np.array([
            pos_var[0], pos_var[1], pos_var[2],
            vel_var_val,
            snr,
            tau,
        ])
        return state6, cov6, FLAG_NONE, snr, tau

    # ------------------------------------------------------------------
    def generate_false_alarm(
        self, ts_ns: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Stochastic false alarm from background clutter (Poisson).

        Returns (state6, cov6, FLAG_TEST) or None if no false alarm this frame.
        """
        if self._rng.random() >= P_FA_PER_FRAME:
            return None

        # Uniformly random angle measurement in the upper hemisphere.
        az_fa = self._rng.uniform(-math.pi, math.pi)
        el_fa = self._rng.uniform(0.0, math.pi / 2.0)
        r_fa  = self._rng.uniform(1.0e3, 30.0e3)    # [1 km, 30 km]

        cos_el = math.cos(el_fa)
        pos_fa = self._sensor_pos + r_fa * np.array([
            cos_el * math.cos(az_fa),
            cos_el * math.sin(az_fa),
            math.sin(el_fa),
        ])

        # Assign a very large covariance (max-noise angular estimate).
        sigma_fa = SIGMA_ANGLE_MAX_RAD * r_fa
        cov_fa   = np.array([
            sigma_fa ** 2, sigma_fa ** 2, sigma_fa ** 2,
            (sigma_fa / 0.01) ** 2,   # large velocity uncertainty
            0.0,                       # SNR unknown for false alarm
            0.0,                       # tau unknown for false alarm
        ])

        state6 = np.concatenate([pos_fa, np.zeros(3)])
        return state6, cov_fa, FLAG_TEST


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

def run(
    sub_addr: str   = "tcp://127.0.0.1:5555",
    pub_addr: str   = "tcp://*:5558",
    sensor_x: float = 0.0,
    sensor_y: float = 0.0,
    sensor_z: float = 0.0,
    rng_seed: int   = 0xBADF00D,
) -> None:
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 64)
    sub.connect(sub_addr)

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 16)
    pub.bind(pub_addr)

    print(f"[irst] subscribing to {sub_addr}", file=sys.stderr, flush=True)
    print(f"[irst] publishing on  {pub_addr}", file=sys.stderr, flush=True)

    sensor_pos = np.array([sensor_x, sensor_y, sensor_z], dtype=np.float64)
    rng        = np.random.default_rng(rng_seed)
    sensor     = IRSTSensor(sensor_pos, rng)

    pkt_id = 0
    n_rx   = 0
    n_det  = 0
    n_miss = 0
    n_fa   = 0

    running = True

    def _stop(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        try:
            raw = sub.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ZMQError as exc:
            if exc.errno in (zmq.ETERM, zmq.EINTR):
                break
            print(f"[irst] recv error: {exc}", file=sys.stderr, flush=True)
            continue

        n_rx += 1
        try:
            truth = TrackPacket.unpack(bytes(raw))
        except ValueError as exc:
            print(f"[irst] bad packet: {exc}", file=sys.stderr, flush=True)
            continue

        if truth.schema_version != SCHEMA_V or truth.producer_id != PRODUCER_SIM:
            continue

        ts_ns = truth.timestamp_ns

        # --- True-target detection attempt -----------------------------------
        result = sensor.process_truth(truth)
        if result is not None:
            state6, cov6, flags, snr, tau = result
            pkt_id += 1
            n_det  += 1
            buf = pack_ir_packet(pkt_id, ts_ns, state6, cov6, flags)
            try:
                pub.send(buf, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            if n_det % 200 == 0:
                r = float(np.linalg.norm(truth.state[0:3] - sensor_pos))
                print(
                    f"[irst] det={n_det:5d}  range={r/1e3:6.2f} km"
                    f"  SNR={snr:6.1f}  tau={tau:.3f}"
                    f"  miss={n_miss}  pos=({truth.state[0]:8.1f}"
                    f", {truth.state[1]:8.1f}, {truth.state[2]:8.1f})",
                    file=sys.stderr, flush=True,
                )
        else:
            n_miss += 1

        # --- False alarm injection -------------------------------------------
        fa = sensor.generate_false_alarm(ts_ns)
        if fa is not None:
            state6_fa, cov6_fa, flags_fa = fa
            pkt_id += 1
            n_fa   += 1
            buf_fa = pack_ir_packet(pkt_id, ts_ns, state6_fa, cov6_fa, flags_fa)
            try:
                pub.send(buf_fa, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            print(
                f"[irst] FALSE ALARM injected  n_fa={n_fa}",
                file=sys.stderr, flush=True,
            )

    print(
        f"[irst] done.  rx={n_rx}  detected={n_det}"
        f"  missed={n_miss}  false_alarms={n_fa}",
        file=sys.stderr, flush=True,
    )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AEGIS-LINK IRST sensor model — publishes IRPacket on :5558"
    )
    ap.add_argument(
        "--sub", default="tcp://127.0.0.1:5555",
        help="ZeroMQ address to subscribe for truth stream (default: %(default)s)",
    )
    ap.add_argument(
        "--pub", default="tcp://*:5558",
        help="ZeroMQ address to publish IRPacket stream (default: %(default)s)",
    )
    ap.add_argument(
        "--sensor-x", type=float, default=0.0, metavar="M",
        help="Sensor ENU x-position [m] (default: %(default)s)",
    )
    ap.add_argument(
        "--sensor-y", type=float, default=0.0, metavar="M",
        help="Sensor ENU y-position [m] (default: %(default)s)",
    )
    ap.add_argument(
        "--sensor-z", type=float, default=0.0, metavar="M",
        help="Sensor ENU z-position [m] (default: %(default)s)",
    )
    ap.add_argument(
        "--seed", type=lambda x: int(x, 0), default=0xBADF00D,
        metavar="HEX",
        help="RNG seed (hex with 0x prefix supported, default: 0xBADF00D)",
    )
    args = ap.parse_args()
    run(
        sub_addr = args.sub,
        pub_addr = args.pub,
        sensor_x = args.sensor_x,
        sensor_y = args.sensor_y,
        sensor_z = args.sensor_z,
        rng_seed = args.seed,
    )
