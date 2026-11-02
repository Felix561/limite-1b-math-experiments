#!/usr/bin/env bash
# Value-model study (docs/preregistration.md) on one GPU. MODE selects the job; every job writes to $RUN/out/.
#
#   MODE=parity  RUN=work/runs/parity  bash gpu/value_study.sh   # attention drop-in vs eager reference (gate for judge)
#   MODE=dev     RUN=work/runs/dev     bash gpu/value_study.sh   # dev pool: 3,000 problems, rollouts on the first 2,000
#   MODE=judges  RUN=work/runs/judges  bash gpu/value_study.sh   # judges on the 960 AIME26 rollouts + the dev pool
#   MODE=branch  RUN=work/runs/branch  bash gpu/value_study.sh   # commitment curve: 384 continuations from prefixes
#   MODE=hmmt START=0 END=17 RUN=work/runs/hmmt-1 bash gpu/value_study.sh   # HMMT Feb 2026 x 24 samples + judges
#
# Inputs (CPU, scripts/prepare_inputs.py) are read from $INPUTS (default work/inputs). Optional DEADLINE (unix time):
# each stage stops cleanly a few minutes before it and labels its output PARTIAL.
# The study ran on one NVIDIA RTX PRO 6000 Blackwell (96 GB) per job; HMMT was split into problems [0, 17) and [17, 33).
set -uo pipefail
source "$(dirname "$0")/setup.sh" || exit 1

MODE=${MODE:?set MODE}
RUN=${RUN:?set RUN (output directory)}
INPUTS=${INPUTS:-$WORK/inputs}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$RUN/out"
cp "$WORK/env/runtime.txt" "$RUN/out/runtime.txt" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
  > "$RUN/out/gpu-monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

deadline() {  # DEADLINE minus $1 minutes, or 0 (no limit)
  if [ -n "${DEADLINE:-}" ]; then python3 -c "import sys; print(float(sys.argv[1]) - 60 * float(sys.argv[2]))" "$DEADLINE" "$1"
  else echo 0; fi
}
run() {  # run one value_study.py subcommand with its own log
  local name=$1; shift
  echo "=== $name $(date -u +%FT%TZ)"
  "$PY" "$REPO/scripts/value_study.py" "$@" > "$RUN/out/$name.log" 2>&1 ||
    { echo "$name failed:"; tail -n 40 "$RUN/out/$name.log"; exit 1; }
}

case "$MODE" in
  parity)
    run parity parity --rollouts "$INPUTS/aime_smoke.jsonl.gz" --out "$RUN/out/parity"
    ;;
  dev)
    mkdir -p "$RUN/data"
    "$PY" "$REPO/scripts/build_dev_pool.py" --out "$RUN/data/pool.jsonl" > "$RUN/out/pool.log" 2>&1 ||
      { echo "build_dev_pool failed"; tail -n 40 "$RUN/out/pool.log"; exit 1; }
    "$PY" "$REPO/scripts/generate_dev_rollouts.py" --pool "$RUN/data/pool.jsonl" --max-problems 2000 \
      --out "$RUN/data/rollouts.jsonl" --metrics "$RUN/out/devgen.jsonl" --deadline "$(deadline 0)" \
      > "$RUN/out/devgen.log" 2>&1 || { echo "generate_dev_rollouts failed"; tail -n 40 "$RUN/out/devgen.log"; exit 1; }
    ;;
  judges)  # AIME26 rollouts at full length, then the dev pool (first 8,200 response tokens: checkpoints <= 8,192)
    run judge-aime judge --rollouts "$INPUTS/aime26.jsonl.gz" --out "$RUN/out/judge-aime" --deadline "$(deadline 12)"
    run judge-dev judge --rollouts "$INPUTS/dev.jsonl.gz" --out "$RUN/out/judge-dev" --max-response 8200 \
      --deadline "$(deadline 6)"
    ;;
  branch)
    run branch branch --plan "${PLAN:-$INPUTS/plan.jsonl.gz}" --out "$RUN/out/branch" \
      --seed-base "${SEED_BASE:-50000000}" --deadline "$(deadline 8)"
    ;;
  hmmt)
    run gen gen --taskset hmmt26 --start "${START:?}" --end "${END:?}" --samples "${SAMPLES:-24}" --out "$RUN/out/gen" \
      --deadline "$(deadline 30)"
    run judge-gen judge --rollouts "$RUN/out/gen/samples.jsonl" --out "$RUN/out/judge-gen" --deadline "$(deadline 8)"
    ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
echo "=== done $(date -u +%FT%TZ)"
