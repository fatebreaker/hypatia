#!/bin/bash

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
# Wait for the vLLM install to finish, then submit the GeneBench-Pro eval.
P="$P"
for i in $(seq 1 720); do          # up to 60 min
  if [ -x $P/venv_vllm/bin/vllm ] && $P/venv_vllm/bin/python -c "import vllm" 2>/dev/null; then
    echo "$(date) vLLM ready; submitting eval"
    sbatch $P/scripts/eval_genebench.sbatch Qwen3-32B
    exit 0
  fi
  sleep 5
done
echo "$(date) TIMEOUT waiting for vLLM install"; exit 1
