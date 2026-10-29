#!/bin/bash
#SBATCH --job-name=gbp-eval
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/gbpeval_%j.out
#SBATCH --error=logs/gbpeval_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
#
# Serve a local model and evaluate it on the 10 public GeneBench-Pro problems,
# in ONE allocation. vLLM and the harness share a node and talk over localhost:
# no cross-job service discovery, nothing orphaned when the job ends, and no
# GPU sitting idle waiting on a separate queue.
#
#   sbatch eval_genebench.sbatch [MODEL_KEY] [TP]
#     MODEL_KEY  Qwen3-32B (default) | Qwen3-8B | Qwen3.5-9B
#     TP         tensor-parallel size, default = #GPUs requested

set -uo pipefail
P="$P"
# Shared model cache. Site-specific on Unity; override elsewhere with
#   export SHARED_LLM_CACHE=/path/to/hf/cache
L="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
MODEL_KEY=${1:-Qwen3-32B}
AGENT_SUFFIX=${3:-}   # e.g. ":nothink" to disable the reasoning trace
# Point at any problem set (default: the 10 public GeneBench-Pro problems).
# GBP_WORKERS controls concurrency -- vLLM batches well, so a single GPU
# supports far more than the 5 used for the 10-problem runs.
PROBLEMS=${GBP_PROBLEMS:-}
WORKERS=${GBP_WORKERS:-5}
PORT=${PORT:-8${SLURM_JOB_ID: -3}}

MODEL_DIR=$(find $L/models--Qwen--${MODEL_KEY}/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no snapshot for $MODEL_KEY"; exit 1; fi
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
TP=${2:-$NGPU}

echo "model : $MODEL_KEY"
echo "path  : $MODEL_DIR"
echo "gpus  : $NGPU (tp=$TP)   port: $PORT   node: $(hostname)"

# Keep HF fully offline: the weights are local and compute nodes should not
# need to reach huggingface.co to serve them.
export HF_HOME=$L/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

# Triton JIT-compiles a small CUDA shim at startup. The venv was built from the
# miniconda jupyterhub python, so it inherits that env's compiler, which cannot
# link libcuda.so.1 -- vLLM then dies during engine init. Force the system gcc
# and load a CUDA module matching the cu13 wheels vLLM installed.
module load cuda/13.1 2>/dev/null || module load cuda/12.8 2>/dev/null || true
# We invoke vllm by absolute path rather than activating the venv, so venv/bin
# is not on PATH -- flashinfer's JIT then cannot find `ninja` and the engine
# dies at startup. Put it on PATH explicitly.
export PATH="$P/venv_vllm/bin:$PATH"
# Belt and braces: skip the flashinfer sampler so no CUDA JIT is needed at all.
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/inductor_$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

$P/venv_vllm/bin/vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_KEY" \
    --tensor-parallel-size "$TP" \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --enforce-eager &
VLLM_PID=$!
trap 'echo "shutting down vLLM"; kill $VLLM_PID 2>/dev/null' EXIT

echo "waiting for vLLM health..."
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}0s"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "FATAL: vLLM died during startup"; exit 1; fi
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FATAL: vLLM never became healthy"; exit 1; }

export GBP_BASE_URL="http://127.0.0.1:$PORT/v1"
export GBP_API_KEY=EMPTY

cd $P/harness
echo "=== harness self-test (oracle must be 10/10) ==="
$P/venv/bin/python runner.py --agent oracle | tail -3

echo
echo "=== $MODEL_KEY on 10 public problems ==="
PROB_ARG=""
[ -n "$PROBLEMS" ] && PROB_ARG="--problems $PROBLEMS"
$P/venv/bin/python runner.py --agent "llm:${MODEL_KEY}${AGENT_SUFFIX:-}" \
    --workers "$WORKERS" --timeout 300 $PROB_ARG

echo "done: $(date)"
