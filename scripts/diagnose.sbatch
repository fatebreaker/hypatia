#!/bin/bash
#SBATCH --job-name=gbp-diag
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
#
# Serve one model and probe the response shape. Answers whether zero code
# execution is a model limitation or a harness bug.
#
#   sbatch diagnose.sbatch Qwen3-8B --reasoning-model
#   sbatch diagnose.sbatch Qwen2.5-7B-Instruct

set -uo pipefail
P="$P"
# Shared model cache. Site-specific on Unity; override elsewhere with
#   export SHARED_LLM_CACHE=/path/to/hf/cache
L="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
MODEL_KEY=${1:-Qwen3-8B}
EXTRA=${2:-}
PORT=9${SLURM_JOB_ID: -3}

MODEL_DIR=$(find $L/models--Qwen--${MODEL_KEY}/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "FATAL: no snapshot for $MODEL_KEY"; exit 1; }
echo "model: $MODEL_KEY   port: $PORT   node: $(hostname)"

export HF_HOME=$L/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
module load cuda/13.1 2>/dev/null || module load cuda/12.8 2>/dev/null || true
export PATH="$P/venv_vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

$P/venv_vllm/bin/vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_KEY" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --enforce-eager &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "vLLM ready (${i}0s)"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died"; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FATAL: never healthy"; exit 1; }

# D3 needs a tiny data file so the "run code" instruction is actionable.
mkdir -p /tmp/diag_$SLURM_JOB_ID/data_files
printf "x\n1\n2\n3\n4\n" > /tmp/diag_$SLURM_JOB_ID/data_files/toy.tsv

cd $P/harness
$P/venv/bin/python diagnose.py --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL_KEY" $EXTRA
echo "done: $(date)"
