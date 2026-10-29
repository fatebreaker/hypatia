#!/bin/bash
#SBATCH --job-name=sft-gen
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/sftgen_%j.out
#SBATCH --error=logs/sftgen_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
set -uo pipefail
source "$P"/scripts/env.sh
cd "$PROJECT_ROOT"
W=${SLURM_CPUS_PER_TASK:-16}
# train: the SFT corpus. val: held out so format/pipeline transfer can be
# measured on instances never trained on.
$PY rl/sft_traces.py --split train --limit 800 --workers "$W"
$PY rl/sft_traces.py --split val   --limit 100 --workers "$W"
echo; ls -la "$BASE/sft/"
