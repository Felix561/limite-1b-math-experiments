#!/usr/bin/env bash
# Upstream evaluation environment, exactly as limite-violetto documents it, plus this repository:
#   git clone limite-violetto @ 7ddad34; cd limite-evals; uv sync --locked; install this repo into that venv.
# Needs Linux x86-64, an NVIDIA driver with CUDA >= 13.0 (locked torch cu130 / vLLM cu129 stack), git, python3.
# Source it (`source gpu/setup.sh`): it exports WORK, EV (limite-evals dir) and PY (its venv python).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-$REPO/work}
UPSTREAM_COMMIT=7ddad34
EV=$WORK/limite-violetto/limite-evals
PY=$EV/.venv/bin/python
mkdir -p "$WORK/env"

# Hosted notebooks (e.g. Colab) point Python at their own site-packages and inline matplotlib backend, which shadows
# the locked venv; upstream assumes a clean host, so clear them.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONSTARTUP PYTHONWARNINGS PYTHONUTF8 PYDEVD_USE_FRAME_EVAL MPLBACKEND \
  TF_FORCE_GPU_ALLOW_GROWTH KMP_EXTRA_ARGS KMP_LISTEN_PORT KMP_TARGET_PORT
for v in $(env | grep -oE '^(UV|PIP)_[A-Z_]+' || true); do unset "$v"; done

nvidia-smi > "$WORK/env/nvidia-smi.txt" 2>&1 || { echo "nvidia-smi failed"; return 1 2>/dev/null || exit 1; }
cuda=$(grep -oP 'CUDA Version: \K[0-9.]+' "$WORK/env/nvidia-smi.txt")
python3 -c "import sys; v=tuple(map(int,'$cuda'.split('.'))); sys.exit(0 if v>=(13,0) else 1)" ||
  { echo "driver supports CUDA ${cuda:-unknown}; need >= 13.0"; return 1 2>/dev/null || exit 1; }

if [ ! -x "$PY" ]; then
  python3 -m pip install -q uv==0.13.5
  hash -r
  [ -d "$WORK/limite-violetto" ] || git clone -q https://github.com/paradigma-inc/limite-violetto.git "$WORK/limite-violetto"
  git -C "$WORK/limite-violetto" checkout -q "$UPSTREAM_COMMIT"
  (cd "$EV" && uv sync --locked) > "$WORK/env/uv-sync.log" 2>&1 ||
    { echo "uv sync --locked failed (see $WORK/env/uv-sync.log)"; return 1 2>/dev/null || exit 1; }
fi
uv pip install --python "$PY" --no-deps -e "$REPO" > "$WORK/env/install.log" 2>&1 ||
  { echo "installing this repo failed (see $WORK/env/install.log)"; return 1 2>/dev/null || exit 1; }
"$PY" - > "$WORK/env/runtime.txt" 2>&1 <<'EOF'
import platform, torch, vllm, transformers
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("vllm", vllm.__version__, "transformers", transformers.__version__)
EOF
git -C "$WORK/limite-violetto" rev-parse HEAD > "$WORK/env/upstream-head.txt"
# vLLM shells out to venv tools (FlashInfer's JIT needs `ninja` from the venv's bin/), as upstream's serve.py does
export PATH="$EV/.venv/bin:$PATH"
export WORK EV PY
echo "environment ready: $(head -n 3 "$WORK/env/runtime.txt" | tr '\n' ' ')"
