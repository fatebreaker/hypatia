#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/sfttrain_%j.out
#SBATCH --error=logs/sfttrain_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
#
# Needs its own GPU: the holder's vLLM has 90% of gpu009's memory reserved, so
# training cannot share it.
set -uo pipefail
source "$P"/scripts/env.sh
cd "$PROJECT_ROOT"
# Shared model cache. Site-specific on Unity; override elsewhere with
#   export SHARED_LLM_CACHE=/path/to/hf/cache
L="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
export HF_HOME=$L/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
module load cuda/13.1 2>/dev/null || module load cuda/12.8 2>/dev/null || true
export PATH="$PROJECT_ROOT/venv_vllm/bin:$PATH"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$PROJECT_ROOT/venv_vllm/bin/python rl/sft_train.py "$@"
