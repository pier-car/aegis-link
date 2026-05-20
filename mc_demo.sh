#!/usr/bin/env bash
# =============================================================================
#  AEGIS-LINK :: mc_demo.sh
# -----------------------------------------------------------------------------
#  Monte-Carlo driver: runs N independent engagements end-to-end and
#  aggregates the outcomes into `mc_results.csv`.
#
#  Each run:
#     1) starts the Julia simulator on a fresh process (so the SDE re-seeds),
#     2) starts the C++ EKF tracker,
#     3) starts the orchestrator (writes the per-run CSV),
#     4) starts the engagement engine for at most ENGAGE_T_S seconds,
#     5) tears everything down, parses the engagement result from
#        `logs/engage.err` and appends a row to mc_results.csv.
#
#  Usage:
#     ./mc_demo.sh                 # 10 engagements (default)
#     ./mc_demo.sh 50              # 50 engagements
#     ./mc_demo.sh 20 60           # 20 engagements, 60 s wall budget each
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_RUNS="${1:-10}"
ENGAGE_T_S="${2:-45}"

mkdir -p "$ROOT/logs/mc"
OUT="$ROOT/mc_results.csv"

c_cyan='\033[1;36m'; c_yel='\033[1;33m'; c_grn='\033[1;32m'; c_red='\033[1;31m'; c_off='\033[0m'
log()  { printf "${c_cyan}[mc]${c_off}    %s\n" "$*"; }
warn() { printf "${c_yel}[mc]${c_off}    %s\n" "$*" >&2; }

[[ -x "$ROOT/tracking_system/build/aegis_tracker" ]] \
    || { warn "Tracker non buildato. Esegui prima ./install_all.sh"; exit 1; }
[[ -d "$ROOT/.venv" ]] \
    || { warn "venv assente. Esegui prima ./install_all.sh"; exit 1; }

echo "run_idx,outcome,cpa_m,flight_time_s,fuel_used_pct,pred_miss_m" > "$OUT"

# ---- helper: wait for a port to be in LISTEN (timeout in seconds) ----------
wait_port() {
    local port="$1" timeout="$2" w=0
    while ! ss -tnl 2>/dev/null | grep -q ":${port} "; do
        sleep 1; ((w++)) || true
        if (( w > timeout )); then return 1; fi
    done
    return 0
}

N_KILL=0
for ((i=1; i<=N_RUNS; i++)); do
    log "=========== run $i / $N_RUNS ==========="

    PIDS=()
    cleanup_run() {
        for pid in "${PIDS[@]}"; do
            kill -INT  "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${PIDS[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
    }
    trap cleanup_run RETURN

    # 1) Simulator
    ( julia --project="$ROOT/simulation_engine" \
            "$ROOT/simulation_engine/main.jl" \
            > "$ROOT/logs/mc/sim_${i}.log" 2>&1 ) &
    PIDS+=("$!")
    if ! wait_port 5555 90; then
        warn "  Simulatore non lega :5555 entro 90s — salto run $i."
        echo "$i,NO_SIM,0,0,0,0" >> "$OUT"
        cleanup_run; trap - RETURN; continue
    fi
    sleep 1

    # 2) Tracker
    ( "$ROOT/tracking_system/build/aegis_tracker" \
            > "$ROOT/logs/mc/trk_${i}.log" 2>&1 ) &
    PIDS+=("$!")
    sleep 2

    # 3) Orchestrator (CSV not retained per-run; only the log)
    ( "$ROOT/.venv/bin/python" "$ROOT/ai_orchestrator/main.py" \
            > "$ROOT/logs/mc/orch_${i}.csv" \
           2> "$ROOT/logs/mc/orch_${i}.err" ) &
    PIDS+=("$!")
    sleep 1

    # 4) Engagement engine, with a wall budget; runs in the foreground of
    #    this subshell so we can read its exit status (0 = KILL, 1 = MISS).
    ENGAGE_LOG="$ROOT/logs/mc/engage_${i}.err"
    ENGAGE_CSV="$ROOT/logs/mc/engage_${i}.csv"
    set +e
    "$ROOT/.venv/bin/python" "$ROOT/engagement_engine/main.py" \
            --config "$ROOT/engagement_engine/config.yaml" \
            --duration "$ENGAGE_T_S" \
            > "$ENGAGE_CSV" 2> "$ENGAGE_LOG"
    RC=$?
    set -e

    # 5) Parse outcome line from the engagement log.
    #    Expected format: "[engage] outcome=KILL cpa=2.45 m  t_flight=4.21s  fuel_used=82%"
    LINE="$(grep -E '^\[engage\] outcome=' "$ENGAGE_LOG" | tail -1 || true)"
    if [[ -z "$LINE" ]]; then
        OUTCOME="NO_RESULT"; CPA=0; TF=0; FUEL=0; PMISS=0
    else
        OUTCOME="$(printf '%s' "$LINE" | sed -E 's/.*outcome=([A-Z_]+).*/\1/')"
        CPA="$(printf    '%s' "$LINE" | sed -E 's/.*cpa=([0-9eE.+-]+).*/\1/')"
        TF="$(printf     '%s' "$LINE" | sed -E 's/.*t_flight=([0-9eE.+-]+).*/\1/')"
        FUEL="$(printf   '%s' "$LINE" | sed -E 's/.*fuel_used=([0-9]+)%.*/\1/')"
        PMISS=0  # not in the summary; could be parsed from engagement.csv if needed
    fi
    echo "$i,$OUTCOME,$CPA,$TF,$FUEL,$PMISS" >> "$OUT"
    if [[ "$OUTCOME" == "KILL" ]]; then
        ((N_KILL++)) || true
        printf "  ${c_grn}KILL${c_off}  cpa=%s m  t=%s s  rc=%d\n" "$CPA" "$TF" "$RC"
    else
        printf "  ${c_red}%s${c_off}  cpa=%s m  rc=%d\n" "$OUTCOME" "$CPA" "$RC"
    fi

    cleanup_run
    trap - RETURN
    sleep 1
done

# ---- Aggregate ------------------------------------------------------------
PK=$(awk -F, -v n="$N_RUNS" 'NR>1 && $2=="KILL"{k++} END{printf "%.3f", k/n}' "$OUT")

cat <<EOF

------------------------------------------------------------------------
  Monte-Carlo terminato.

    Engagements eseguiti : $N_RUNS
    KILL                 : $N_KILL
    Pk (kill probability): $PK

  Risultati:   $OUT
  Per-run log: $ROOT/logs/mc/

  Analisi suggerita (notebook):
      jupyter lab analysis.ipynb     # sezione "Engagement performance"

------------------------------------------------------------------------
EOF
