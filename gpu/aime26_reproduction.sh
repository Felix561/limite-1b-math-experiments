#!/usr/bin/env bash
# AIME 2026 reproduction with the upstream harness, as limite-violetto documents it:
#   uv run --locked limite-eval paradigma-inc/limite-1b-violetto --tasksets aime26
# Only `--tasksets aime26` (and `--limit 1` for the pilot) differ from the README's full-suite command; seeds,
# sampling, budget, concurrency, grader and checkpoint pin are the harness defaults. The pilot (problem 1 x 32) is a
# plumbing check; set SKIP_PILOT=1 to go straight to the full run.
#
#   bash gpu/aime26_reproduction.sh            # then: python scripts/analyze_reproduction.py <run dir> --env work/env ...
#
# Evidence in $WORK/env: nvidia-smi, runtime versions, gpu-monitor.csv (30 s), vllm-metrics.jsonl (engine Prometheus
# counters, 30 s). The harness writes its run dirs (every completion, all grader verdicts, summary) to
# $EV/outputs/eval/<run-id>.
set -uo pipefail
source "$(dirname "$0")/setup.sh" || exit 1

MODEL=paradigma-inc/limite-1b-violetto
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
cd "$EV" || exit 1

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 30 \
  > "$WORK/env/gpu-monitor.csv" 2>&1 &
MON=$!
( "$PY" - <<'EOF' ) > "$WORK/env/vllm-metrics.err" 2>&1 &
# Scrape the live engine's Prometheus endpoint (port read from the launch line the harness writes into its log).
import glob, json, os, re, time, urllib.request
work = os.environ["WORK"]
ev = os.environ["EV"]
out = open(os.path.join(work, "env", "vllm-metrics.jsonl"), "a")
while True:
    time.sleep(30)
    logs = sorted(glob.glob(f"{ev}/outputs/eval/*/artifacts/vllm-*.log"), key=os.path.getmtime)
    if not logs:
        continue
    try:
        port = re.search(r"--port (\d+)", open(logs[-1]).readline()).group(1)
        text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    except Exception:
        continue
    metrics = {}
    for line in text.splitlines():
        if line.startswith("vllm:"):
            name, _, value = line.rpartition(" ")
            try:
                metrics[name] = float(value)
            except ValueError:
                pass
    out.write(json.dumps({"t": time.time(), "run": logs[-1].split("/eval/")[1].split("/")[0], "m": metrics}) + "\n")
    out.flush()
EOF
SCRAPER=$!
trap 'kill $MON $SCRAPER 2>/dev/null' EXIT

if [ -z "${SKIP_PILOT:-}" ]; then
  echo "=== pilot $(date -u +%FT%TZ)"
  uv run --locked limite-eval "$MODEL" --tasksets aime26 --limit 1 --run-id "pilot-aime26-first1-$STAMP" \
    > "$WORK/env/pilot.log" 2>&1 || { echo "pilot failed (see $WORK/env/pilot.log)"; exit 1; }
fi
echo "=== full $(date -u +%FT%TZ)"
uv run --locked limite-eval "$MODEL" --tasksets aime26 --run-id "full-aime26-avg32-$STAMP" \
  > "$WORK/env/full.log" 2>&1 || { echo "full run failed (see $WORK/env/full.log)"; exit 1; }
echo "=== done $(date -u +%FT%TZ): $EV/outputs/eval/full-aime26-avg32-$STAMP"
