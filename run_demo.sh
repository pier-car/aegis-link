#!/usr/bin/env bash
# =============================================================================
#  AEGIS-LINK :: run_demo.sh
# -----------------------------------------------------------------------------
#  Orchestratore "one-click" per la prima esecuzione.
#  Avvia in sequenza:
#     1) Julia simulator   (background, log -> logs/sim.log)
#     2) C++ EKF tracker   (background, log -> logs/trk.log)
#     3) Python analyzer   (foreground, CSV -> run.csv)
#
#  Si ferma da solo dopo DURATION secondi (default 30) e fa stop pulito di
#  tutti i processi figli.
#
#  Uso:
#     ./run_demo.sh                # 30 s di simulazione
#     ./run_demo.sh 120            # 120 s di simulazione
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION="${1:-30}"
mkdir -p "$ROOT/logs"

c_cyan='\033[1;36m'; c_yel='\033[1;33m'; c_grn='\033[1;32m'; c_red='\033[1;31m'; c_off='\033[0m'
log()  { printf "${c_cyan}[demo]${c_off}  %s\n" "$*"; }
warn() { printf "${c_yel}[demo]${c_off}  %s\n" "$*" >&2; }

# --- Pre-flight ----------------------------------------------------------
[[ -x "$ROOT/tracking_system/build/aegis_tracker" ]] \
    || { warn "Tracker non buildato. Esegui prima ./install_all.sh"; exit 1; }
[[ -d "$ROOT/.venv" ]] \
    || { warn "venv assente. Esegui prima ./install_all.sh"; exit 1; }

# --- Cleanup hook --------------------------------------------------------
PIDS=()
cleanup() {
    log "Stop pulito dei processi figli..."
    for pid in "${PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 1) Simulatore Julia (PUB :5555) ------------------------------------
log "Avvio simulatore Julia (PUB tcp://*:5555)..."
( julia --project="$ROOT/simulation_engine" "$ROOT/simulation_engine/main.jl" \
    > "$ROOT/logs/sim.log" 2>&1 ) &
PIDS+=("$!")

# Attesa attiva: aspetta che la porta 5555 sia in LISTEN.
# (più affidabile del grep su log: Julia bufferizza stdout su file)
log "Attendo bind del simulatore (precompilazione Julia, ~10-20s)..."
WAIT=0
while ! ss -tnl 2>/dev/null | grep -q ':5555 '; do
    sleep 1; ((WAIT++)) || true
    if (( WAIT > 90 )); then
        warn "Simulatore non lega :5555 dopo 90s. Vedi logs/sim.log"
        exit 1
    fi
done
log "Simulatore pronto (in ${WAIT}s). Ulteriore secondo per stabilizzare..."
sleep 1

# --- 2) Tracker C++ (SUB :5555 / PUB :5556) -----------------------------
log "Avvio tracker C++ (SUB :5555 / PUB :5556)..."
( "$ROOT/tracking_system/build/aegis_tracker" \
    > "$ROOT/logs/trk.log" 2>&1 ) &
PIDS+=("$!")
sleep 2   # tempo per ricevere qualche frame prima dell'orchestrator

# --- 3) Orchestrator Python (SUB :5555 + :5556 -> run.csv) --------------
log "Avvio orchestrator Python (CSV -> run.csv) per ${DURATION}s..."
rm -f "$ROOT/run.csv"
( "$ROOT/.venv/bin/python" "$ROOT/ai_orchestrator/main.py" \
    > "$ROOT/run.csv" 2> "$ROOT/logs/orch.err" ) &
ORCH_PID=$!
PIDS+=("$ORCH_PID")

# --- Watchdog ------------------------------------------------------------
log "Simulazione in corso... (Ctrl-C per interrompere prima)"
SECS=0
while (( SECS < DURATION )); do
    sleep 1; ((SECS++)) || true
    # Heartbeat ogni 5 s
    if (( SECS % 5 == 0 )); then
        N=$(( $(wc -l < "$ROOT/run.csv" 2>/dev/null || echo 0) ))
        printf "${c_grn}[demo]${c_off}  t=%3ds  righe CSV=%d\n" "$SECS" "$N"
    fi
    # Sanity: se qualcuno è morto, esci
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || { warn "Processo PID $pid morto in anticipo"; break 2; }
    done
done

log "Tempo scaduto, chiusura..."
# (cleanup() viene chiamato dal trap EXIT)

# --- Riepilogo -----------------------------------------------------------
sleep 1
N_LINES=$(wc -l < "$ROOT/run.csv" 2>/dev/null || echo 0)
SIZE=$(du -h "$ROOT/run.csv" 2>/dev/null | cut -f1)

cat <<EOF

------------------------------------------------------------------------
  ✅ Run completata.

  CSV       : run.csv   ($N_LINES righe, $SIZE)
  Log sim   : logs/sim.log
  Log trk   : logs/trk.log
  Log orch  : logs/orch.err

  Prossimo passo (analisi grafica):

      source .venv/bin/activate
      pip install --quiet matplotlib pandas jupyterlab seaborn
      jupyter lab analysis.ipynb     # oppure aprilo da VS Code

------------------------------------------------------------------------
EOF
