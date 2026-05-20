#!/usr/bin/env bash
# =============================================================================
#  AEGIS-LINK :: install_all.sh
# -----------------------------------------------------------------------------
#  Provisions a clean WSL2 / Ubuntu 22.04+ host with everything needed to
#  build and run the four modules:
#     - APT toolchain      (g++-12, cmake, ninja, pkg-config, git)
#     - libzmq + cppzmq + Eigen3 headers
#     - Python 3.12 venv with pyzmq + numpy
#     - Julia (latest stable via juliaup) packages: ZMQ, StaticArrays,
#       DifferentialEquations
#
#  Idempotent: re-running is safe.
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]   \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fatal]  \033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This installer targets Linux / WSL2 only."

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages via apt ..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential g++-12 gcc-12 \
    cmake ninja-build pkg-config git curl ca-certificates \
    libzmq3-dev libsodium-dev \
    libeigen3-dev \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    chrony

if command -v g++-12 >/dev/null; then
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120 || true
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 || true
fi

# ---------------------------------------------------------------------------
# 2. cppzmq (header-only) — install if not present
# ---------------------------------------------------------------------------
if [[ ! -f /usr/local/include/zmq.hpp && ! -f /usr/include/zmq.hpp ]]; then
    log "Fetching cppzmq header (zmq.hpp) ..."
    tmp="$(mktemp -d)"
    git clone --depth 1 --branch v4.10.0 https://github.com/zeromq/cppzmq.git "$tmp/cppzmq"
    sudo install -m 0644 "$tmp/cppzmq/zmq.hpp"        /usr/local/include/zmq.hpp
    sudo install -m 0644 "$tmp/cppzmq/zmq_addon.hpp"  /usr/local/include/zmq_addon.hpp || true
    rm -rf "$tmp"
else
    log "cppzmq already present, skipping."
fi

# ---------------------------------------------------------------------------
# 3. Julia — use the latest stable already installed; otherwise install via juliaup
# ---------------------------------------------------------------------------
# Make sure juliaup-managed binaries are visible even in non-interactive shells.
if [[ -d "$HOME/.juliaup/bin" ]]; then
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

if ! command -v julia >/dev/null 2>&1; then
    log "No Julia found on PATH — installing juliaup (latest stable channel) ..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel release
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

command -v julia >/dev/null 2>&1 \
    || die "Julia installation failed; install manually from https://julialang.org/downloads/"

# If juliaup is present, ensure 'release' channel is current and set as default.
if command -v juliaup >/dev/null 2>&1; then
    log "Updating juliaup 'release' channel to latest stable ..."
    juliaup update release  || true
    juliaup default release || true
fi

log "Using $(julia --version) at $(command -v julia)"

# ---------------------------------------------------------------------------
# 4. Julia packages — instantiate a project under simulation_engine/
# ---------------------------------------------------------------------------
log "Installing Julia packages (this can take several minutes the first time) ..."
julia --project="$ROOT/simulation_engine" --color=yes -e '
    using Pkg
    Pkg.add(["ZMQ", "StaticArrays", "DifferentialEquations", "StochasticDiffEq"])
    Pkg.precompile()
'

# ---------------------------------------------------------------------------
# 5. Python 3.12 virtualenv
# ---------------------------------------------------------------------------
log "Creating Python 3.12 venv at $ROOT/.venv ..."
python3.12 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install --quiet --upgrade pip wheel
pip install --quiet "pyzmq>=25" "numpy>=1.26" "pyyaml>=6.0"
deactivate

# ---------------------------------------------------------------------------
# 6. Build the C++ tracker
# ---------------------------------------------------------------------------
log "Configuring & building tracking_system (Release / -O3 -march=native) ..."
cmake -S "$ROOT/tracking_system" -B "$ROOT/tracking_system/build" \
      -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/tracking_system/build" --parallel

# ---------------------------------------------------------------------------
# 7. Chrony hint (CLOCK_TAI requires kernel offset set)
# ---------------------------------------------------------------------------
if ! systemctl is-active --quiet chrony 2>/dev/null; then
    warn "chrony is installed but not active under WSL2 systemd; CLOCK_TAI"
    warn "may default to CLOCK_REALTIME. The producer code falls back gracefully."
fi

cat <<EOF

------------------------------------------------------------------------
  AEGIS-LINK environment ready.

  Julia : $(julia --version)
  Python: $("$ROOT/.venv/bin/python" --version)
  C++   : $(g++ --version | head -n1)

  Run order (3 separate terminals):

    T1)   julia --project=simulation_engine simulation_engine/main.jl
    T2)   ./tracking_system/build/aegis_tracker
    T3)   source .venv/bin/activate
          python ai_orchestrator/main.py | tee run.csv

  See README.md ("Clock Synchronisation") for production-grade timing.
------------------------------------------------------------------------
EOF
