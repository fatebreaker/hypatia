#!/bin/bash
#SBATCH --job-name=deep-holder
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/holder_%j.out
#SBATCH --error=logs/holder_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
#
# A long-lived GPU allocation that also serves vLLM, so experiments can start
# immediately instead of waiting in the queue each time.
#
# Deliberately on `gpu`, NOT `gpu-preempt`: a preemptible 14-day job would be
# killed and take the endpoint with it.
#
# The server runs continuously rather than the job merely sleeping. Holding an
# idle GPU for two weeks is antisocial; holding one that answers requests is
# the thing we actually need. Release it with `scancel` when done.
#
# Use it two ways:
#   1. point any client at the endpoint recorded in .gpu_holder
#   2. srun --jobid=<J> --overlap ... to run inside the same allocation
#
# Endpoint discovery: writes $PROJECT_ROOT/.gpu_holder, sourced by holder_env.sh

set -uo pipefail
source "$P"/scripts/env.sh
# Shared model cache. Site-specific on Unity; override elsewhere with
#   export SHARED_LLM_CACHE=/path/to/hf/cache
L="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
MODEL_KEY=${MODEL_KEY:-Qwen3-8B}

# We race several holders across GPU types because a 14-day request waits on
# priority, and which type frees up first is unpredictable. First one to start
# claims the stamp; the others exit immediately so we never hold two GPUs.
STAMP_PRE=$PROJECT_ROOT/.gpu_holder
if [ -f "$STAMP_PRE" ]; then
  . "$STAMP_PRE"
  if squeue -h -j "${HOLDER_JOBID:-0}" >/dev/null 2>&1 \
     && [ -n "$(squeue -h -j ${HOLDER_JOBID:-0} 2>/dev/null)" ]; then
    echo "holder $HOLDER_JOBID already live on $HOLDER_NODE -- exiting"
    exit 0
  fi
  echo "stale stamp from dead job ${HOLDER_JOBID:-?}; taking over"
  rm -f "$STAMP_PRE"
fi
PORT=${PORT:-7$((SLURM_JOB_ID % 1000))}

MODEL_DIR=$(find $L/models--Qwen--${MODEL_KEY}/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "FATAL: no snapshot for $MODEL_KEY"; exit 1; }

echo "holder job   : $SLURM_JOB_ID"
echo "node         : $(hostname)"
echo "model        : $MODEL_KEY"
echo "port         : $PORT"
echo "started      : $(date)"

export HF_HOME=$L/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
module load cuda/13.1 2>/dev/null || module load cuda/12.8 2>/dev/null || true
export PATH="$PROJECT_ROOT/venv_vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Bind to 0.0.0.0 so clients on the login node can reach it, not just localhost.
$PROJECT_ROOT/venv_vllm/bin/vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_KEY" \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enforce-eager &
VLLM_PID=$!

STAMP=$PROJECT_ROOT/.gpu_holder
cleanup() {
    echo "shutting down $(date)"
    kill $VLLM_PID 2>/dev/null
    rm -f "$STAMP"
}
trap cleanup EXIT

echo "waiting for vLLM ..."
for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready after ${i}0s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FATAL: never healthy"; exit 1; }

cat > "$STAMP" <<EOF
HOLDER_JOBID=$SLURM_JOB_ID
HOLDER_NODE=$(hostname)
HOLDER_PORT=$PORT
HOLDER_MODEL=$MODEL_KEY
HOLDER_BASE_URL=http://$(hostname):$PORT/v1
HOLDER_STARTED=$(date -Is)
EOF
echo "--- endpoint recorded in $STAMP ---"
cat "$STAMP"

# Keep the allocation alive, and notice if the server dies under us.
while kill -0 $VLLM_PID 2>/dev/null; do sleep 60; done
echo "vLLM exited unexpectedly at $(date)"
exit 1
