# Project AEGIS-LINK

> Distributed high-fidelity simulator for the interception of airborne threats.
> Julia (physics) ⇄ C++ (tracking) ⇄ Python (analysis), bridged by a
> 128-byte zero-copy data link over ZeroMQ.

```
┌──────────────────────────┐    PUB tcp://*:5555    ┌──────────────────────────┐
│ simulation_engine (Julia)│ ─────────────────────▶ │ tracking_system  (C++)   │
│  SDE + Ornstein–Uhlenbeck│   raw 128-byte frames  │  EKF, CA model, gating   │
└──────────────────────────┘                        └────────────┬─────────────┘
                                                                 │ PUB tcp://*:5556
                                                                 ▼
                                                ┌──────────────────────────────┐
                                                │ ai_orchestrator (Python 3.12)│
                                                │  Mahalanobis anomaly detector│
                                                └──────────────────────────────┘
```

## Modules

| Path                             | Role                                                     | Lang     |
|----------------------------------|----------------------------------------------------------|----------|
| [shared/messages.h](shared/messages.h)                       | C-ABI 128-B `TrackPacket` (cache-aligned, no padding) | C/C++  |
| [simulation_engine/main.jl](simulation_engine/main.jl)       | SDE physics with coloured (OU) wind noise             | Julia ≥ 1.11 (tested on 1.12) |
| [tracking_system/main.cpp](tracking_system/main.cpp)         | EKF with closed-form CA discretisation                | C++20  |
| [tracking_system/CMakeLists.txt](tracking_system/CMakeLists.txt) | CMake build, `-O3 -march=native`                  | CMake  |
| [ai_orchestrator/main.py](ai_orchestrator/main.py)           | Real-time Mahalanobis distance / manoeuvre flag       | Python 3.12 |
| [install_all.sh](install_all.sh)                              | One-shot environment provisioning (WSL2/Ubuntu)       | bash   |

## Quick start

```bash
./install_all.sh                                          # one-time setup
# 3 terminals:
julia --project=simulation_engine simulation_engine/main.jl
./tracking_system/build/aegis_tracker
source .venv/bin/activate && python ai_orchestrator/main.py | tee run.csv
```

---

## Clock synchronisation across the three asynchronous processes

The three processes are decoupled (different runtimes, schedulers and
languages). The EKF needs a correct `dt` between consecutive measurements,
and the orchestrator needs to *match* truth packets against estimate
packets. Naively reading `time()` in each process produces drift,
non-monotonicity (NTP step adjustments) and leap-second discontinuities —
all of which destabilise a Kalman filter.

We enforce three rules.

### 1. The producer stamps the packet, never the consumer

Every packet carries `timestamp_ns` written **at production time** by the
process that *generated* the data:

* The Julia simulator stamps when the SDE state was integrated.
* The C++ tracker stamps when the EKF update finished.

The consumer **never** uses its own wall-clock to derive `dt`. The C++
tracker computes
`dt = (in.timestamp_ns - last_in.timestamp_ns) * 1e-9`,
so the filter is immune to network jitter, ZMQ buffering and scheduler
preemption — the very things that ruin a wall-clock-based EKF.

### 2. One single, monotonic, leap-second-free time base: `CLOCK_TAI`

We use **`CLOCK_TAI`** (nanoseconds since the Unix epoch on the TAI
scale) everywhere:

* Linux exposes it via `clock_gettime(CLOCK_TAI, &ts)` in C/C++.
* Julia: `time()` returns POSIX (`CLOCK_REALTIME`); we add the current
  TAI–UTC offset (37 s as of 2026) to obtain the same scale.
* Python `pyzmq` consumers do not stamp anything — they read producer
  stamps.

`CLOCK_TAI` is monotonic *with respect to leap seconds* (POSIX time can
repeat a second; TAI never does), so packet ordering and `dt` remain
strictly positive even across a leap-second insertion.

To make `CLOCK_TAI` correct on Linux, the kernel must be told the
current UTC–TAI offset. `chrony` (installed by `install_all.sh`) sets
this automatically when it locks onto a stratum-1/2 source. On WSL2
without functional `chrony`, the producer code falls back to
`CLOCK_REALTIME + 37 s`, which is correct in steady state but vulnerable
to an unannounced future leap second — acceptable for development, **not**
for flight.

### 3. Disciplined wall clock + bounded matching window

For deployment, run `chrony` against a local PTP / GNSS source so the
host clock is held to ≤ 1 ms. Within the orchestrator we then:

* keep a small bounded buffer (≤ 50 ms) of recent truth packets;
* match each estimate packet to the closest-in-time truth packet by
  `|t_truth - t_est|`;
* drop any pair whose lag exceeds the window — the EKF was running on
  stale data and the comparison would be meaningless.

This is exactly the same pattern used in sensor-fusion stacks (radar +
EO/IR + LIDAR): one common time base, producer-side stamps, bounded
out-of-order tolerance.

### Practical recipe

| Layer                    | What we do                                              |
|--------------------------|---------------------------------------------------------|
| OS                       | `chrony` disciplines `CLOCK_REALTIME`; kernel TAI offset set |
| Producer (Julia / C++)   | Stamp **once**, with `CLOCK_TAI`, before sending        |
| Consumer (C++ tracker)   | Compute `dt` from packet stamps, never `clock_gettime`  |
| Consumer (Python orch.)  | Match by stamp within ±50 ms; drop stale pairs          |
| Schema                   | `timestamp_ns` is `uint64`; rolls over in year 2554     |

For sub-microsecond requirements (e.g. multi-host deployments) replace
`chrony` with **PTP (`linuxptp`)** on a hardware-timestamped NIC and
keep everything else identical — the data link is already designed for
that precision (the `timestamp_ns` field has nanosecond resolution).
