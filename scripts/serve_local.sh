#!/bin/bash
# Serve a model with vLLM on ONE local GPU, no SLURM.
#
# scripts/gpu_holder.sbatch does this on Unity, where the hard part is winning
# a GPU allocation. On a single box the hard part is the opposite -- picking a
# GPU nobody else is using -- so this refuses to start on a busy one rather
# than OOM half an hour later.
#
# Writes the same .gpu_holder stamp gpu_holder.sbatch does, so
# `source scripts/holder_env.sh` works identically afterwards.
#
#   scripts/serve_local.sh                    # auto-pick a free GPU, Qwen3-8B
#   scripts/serve_local.sh Qwen3-8B 5 8000    # explicit model / GPU / port
#   scripts/serve_local.sh --stop
#
# Carries the fixes FINDINGS D1-D3 cost real time to find:
#   D1  vLLM needs the python-3.12 venv_vllm, not venv
#   D2  Triton JIT needs a real compiler -> CC/CXX
# NOT carried: --enforce-eager. It was a D2 workaround for the Triton compiler
# failure, and fixing the real cause (D10, the distro python shipping no
# Python.h) made it obsolete -- CUDA graphs work. Keeping it cost decode
# throughput on every run until that was noticed. EAGER=1 restores it.
#   D3  absolute-path invocation hides venv_vllm/bin from PATH, so flashinfer's
#       JIT cannot find ninja and the engine dies at startup
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# One stamp per server. The default keeps the single-server behaviour every
# caller already relies on, but paired evaluation legitimately wants two
# engines -- one per GPU, same adapters, comparing checkpoints in parallel --
# and with a shared stamp the second invocation reads the first one's healthy
# endpoint and exits 0 without ever starting.
#   HOLDER_STAMP=$PWD/.gpu_holder_gpu5 scripts/serve_local.sh Qwen3-8B 5 8011
STAMP="${HOLDER_STAMP:-$P/.gpu_holder}"

if [ "${1:-}" = "--stop" ]; then
  if [ -f "$STAMP" ]; then
    . "$STAMP"
    kill "${HOLDER_PID:-0}" 2>/dev/null && echo "stopped pid ${HOLDER_PID}"
    rm -f "$STAMP"
  else
    echo "no stamp; nothing to stop"
  fi
  exit 0
fi

MODEL_KEY=${1:-Qwen3-8B}
GPU=${2:-}
PORT=${3:-8000}
FREE_MB=${FREE_MB:-100000}     # a GPU with less free than this is "in use"
# MAX_LEN defaults to 16384, not 40960. Reserving KV cache for context the
# episodes never use (they run ~6k tokens) costs concurrency: at 16384 vLLM
# holds far more sequences at once, which is what rollout collection is
# bottlenecked on. Raise it for genuinely long contexts.
# EAGER=1 restores --enforce-eager if a future vLLM/driver combination needs it.

if [ -f "$STAMP" ]; then
  . "$STAMP"
  if curl -sf --max-time 5 "http://127.0.0.1:${HOLDER_PORT}/health" >/dev/null 2>&1; then
    echo "already serving ${HOLDER_MODEL} on GPU ${HOLDER_GPU} -> ${HOLDER_BASE_URL}"
    exit 0
  fi
  echo "stale stamp (endpoint dead); taking over"
  rm -f "$STAMP"
fi

# Pick the emptiest GPU unless told otherwise. Other people share this box;
# claiming a busy GPU wastes their run as well as ours.
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
AVAIL=$((TOTAL - USED))
echo "GPU $GPU: ${AVAIL} MiB free of ${TOTAL} MiB"
if [ "$AVAIL" -lt "$FREE_MB" ]; then
  echo "FATAL: GPU $GPU has only ${AVAIL} MiB free (need ${FREE_MB})." >&2
  echo "       Pick another with: scripts/serve_local.sh $MODEL_KEY <gpu> $PORT" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
  exit 1
fi

CACHE="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
MODEL_DIR=$(find "$CACHE/hub/models--Qwen--${MODEL_KEY}/snapshots" \
                 -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && MODEL_DIR=$(find "$CACHE/hub" -maxdepth 3 -type d \
    -path "*${MODEL_KEY}*/snapshots/*" 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then
  echo "FATAL: no snapshot for $MODEL_KEY under $CACHE/hub" >&2
  exit 1
fi
# A snapshot dir can exist while the weights are still downloading (FINDINGS
# D5: waiting for a path is not waiting for the thing to be complete).
if ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "FATAL: $MODEL_DIR has no .safetensors -- download incomplete?" >&2
  exit 1
fi

mkdir -p "$P/logs"
LOG="$P/logs/vllm_${MODEL_KEY}_gpu${GPU}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$CACHE"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="/tmp/triton_$$" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$$"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PATH="$P/venv_vllm/bin:$PATH"            # D3

# Optional LoRA adapter:  LORA=/path/to/adapter scripts/serve_local.sh
# Serves the adapter alongside the base model; request it by the name below.
# --max-lora-rank must be >= the adapter's r (32 here) or vLLM refuses to load
# it, and the default is 16, so this is not optional for r=32.
LORA_ARGS=()
LORA_NAME=""
LORA_SPECS=()

# LORAS serves SEVERAL adapters from one engine:
#   LORAS="sft=/path/a rl100=/path/b" scripts/serve_local.sh
# Comparing checkpoints is the normal case -- a transfer curve is four
# adapters over the same base on the same problems -- and one server per
# adapter means a 2-3 minute vLLM start each time plus a fresh KV cache, for
# weights that differ by 175 MB. Adapters are swapped per request, so the same
# server answers for all of them and the comparison is exactly paired.
for spec in ${LORAS:-}; do
  name="${spec%%=*}"; path="${spec#*=}"
  [ -f "$path/adapter_model.safetensors" ] || {
    echo "FATAL: no adapter_model.safetensors in $path" >&2; exit 1; }
  LORA_SPECS+=("${name}=${path}")
done
if [ -n "${LORA:-}" ]; then
  [ -f "$LORA/adapter_model.safetensors" ] || {
    echo "FATAL: no adapter_model.safetensors in $LORA" >&2; exit 1; }
  LORA_NAME="${LORA_NAME_OVERRIDE:-$(basename "$(dirname "$LORA")")}"
  LORA_SPECS+=("${LORA_NAME}=${LORA}")
fi
if [ ${#LORA_SPECS[@]} -gt 0 ]; then
  # --max-lora-rank must be >= every adapter's r or vLLM refuses to load it,
  # and the default is 16, so this is not optional for r=32. Take the max
  # across all adapters rather than the first one's.
  LORA_RANK=$("$P/venv/bin/python" - "${LORA_SPECS[@]}" <<'PY'
import json, sys
from pathlib import Path
rs = []
for spec in sys.argv[1:]:
    cfg = Path(spec.split("=", 1)[1]) / "adapter_config.json"
    try:
        rs.append(int(json.loads(cfg.read_text())["r"]))
    except Exception:
        pass
print(max(rs) if rs else 32)
PY
)
  # vLLM keeps at most --max-loras adapters resident; below that it evicts and
  # reloads per request, which is correct but slow when a run alternates
  # between them.
  LORA_ARGS=(--enable-lora --max-lora-rank "$LORA_RANK"
             --max-loras "${#LORA_SPECS[@]}"
             --lora-modules "${LORA_SPECS[@]}")
  [ -z "$LORA_NAME" ] && LORA_NAME="${LORA_SPECS[0]%%=*}"
  echo "lora  : ${#LORA_SPECS[@]} adapter(s) at r<=$LORA_RANK"
  printf '        %s\n' "${LORA_SPECS[@]}"
fi

echo "model : $MODEL_DIR"
echo "log   : $LOG"
"$P/venv_vllm/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_KEY" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_LEN:-16384}" \
    --max-num-seqs "${MAX_SEQS:-64}" \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port "$PORT" \
    "${LORA_ARGS[@]}" ${EAGER:+--enforce-eager} > "$LOG" 2>&1 &
VLLM_PID=$!

echo "waiting for vLLM (pid $VLLM_PID) ..."
for i in $(seq 1 180); do
  if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ready after $((i*10))s"
    cat > "$STAMP" <<EOF
HOLDER_PID=$VLLM_PID
HOLDER_NODE=$(hostname)
HOLDER_GPU=$GPU
HOLDER_PORT=$PORT
HOLDER_MODEL=$MODEL_KEY
HOLDER_LORA=${LORA_NAME}
HOLDER_BASE_URL=http://127.0.0.1:$PORT/v1
HOLDER_STARTED=$(date -Is)
EOF
    echo "stamp : $STAMP"
    echo
    echo "next:  source scripts/holder_env.sh"
    echo "       cd harness && ../venv/bin/python runner.py \\"
    echo "           --agent llm:$MODEL_KEY:nothink --problems ../benchmarks/bixbench"
    exit 0
  fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup; see $LOG" >&2
                                     tail -25 "$LOG" >&2; exit 1; }
  sleep 10
done
echo "FATAL: not ready after 30 min; see $LOG" >&2
kill $VLLM_PID 2>/dev/null
exit 1
