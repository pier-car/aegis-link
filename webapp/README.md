# AEGIS-LINK · Interactive Streamlit Demo

A fully self-contained web application that lets you explore the
**AEGIS-LINK** pipeline — stochastic ballistic simulation, 9-D CA EKF
tracking, Mahalanobis anomaly detection, Proportional-Navigation intercept
and MWIR IRST sensor — directly in your browser, **no Julia or C++ required**.

## Quick start

```bash
# from the repository root
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What you can vary

| Section | Parameters |
|---|---|
| 🎯 **Target / Scenario** | Launch velocity (Vₓ, Vᵧ, Vz), OU wind strength & correlation time, duration, random seed |
| 📡 **EKF Tracker** | Measurement noise (σ_pos, σ_vel), process-noise jerk PSD (q_jerk) |
| 🧠 **AI Orchestrator** | χ² gate threshold, lock σ_p threshold, lock streak length |
| 🚀 **Interceptor / Engagement** | PN navigation ratio N′, thrust, mass, propellant fraction, max lateral G, lethal radius |
| 🔭 **MWIR IRST Sensor** | Base radiant intensity I₀, atmospheric extinction α, detector NEI, SNR threshold, false-alarm rate λ |

## Output tabs

| Tab | Contents |
|---|---|
| 🗺️ **3-D Tactical** | Interactive Plotly 3-D scene: truth (green), EKF estimate (amber), interceptor (blue), IR detections (orange) |
| 📈 **Time Series** | Position error, Mahalanobis d² (log scale + χ² thresholds), lock-state step function, EKF σ_p |
| 🚀 **Engagement** | Interceptor range, closing speed, fuel fraction, predicted miss distance vs time |
| 🔭 **IRST Sensor** | SNR, atmospheric τ, detection probability P_D, true-detect / false-alarm events |
| 📋 **Raw Data** | Downloadable CSV files (trajectory + engagement telemetry) |

## Physics in 60 seconds

```
Stochastic SDE (Euler-Maruyama)
  dr = v dt
  dv = (a_cmd + W + g) dt
  dW = -θ W dt + σ_w dB(t)     ← Ornstein-Uhlenbeck coloured wind

9-D Constant-Acceleration EKF
  state x = [pos(3), vel(3), acc(3)]
  F(dt): closed-form CA transition
  Q(dt): Singer / jerk-input process noise
  Joseph-form update for numerical stability

AI Orchestrator
  d² = (z - x̂)ᵀ (P_diag + R_sensor)⁻¹ (z - x̂)
  lock FSM: SEARCH → TRACKING → LOCKED

True Proportional Navigation
  a_cmd = N′ · V_c · (Ω_LOS × û_los)
  interceptor: RK4 with thrust / drag / gravity

MWIR IRST
  I(v) = I₀ · (1 + 0.8 · (|v|/300)²)
  τ(R) = exp(−α R)
  SNR  = I τ / (R² NEI)
  P_D  = σ(1.5 · (SNR − thresh))
```

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `numpy` | All simulation numerics |
| `plotly` | Interactive 3-D and 2-D charts |
| `pandas` | CSV export |

No ZeroMQ, no Julia, no C++ compiler required.
