#!/bin/bash
#SBATCH --job-name=gen-1k
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/gen1k_%j.out
#SBATCH --error=logs/gen1k_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
#
# Generate 1,000 cis-MVMR instances in disjoint train/val/test splits.
#
# Why 1,000: with n=10 a pass rate carries a ~50pp confidence interval and
# cannot separate a 20% model from a 60% one. ~335 problems are needed for 80%
# power at a 10pp gap; 1,000 gives ~+-3pp and room to report per-split.
#
# Splits use disjoint seed ranges so no instance can appear in two of them.
# Train is for RL, val for model selection, test reported once.

set -uo pipefail
source "$P"/scripts/env.sh
cd "$PROJECT_ROOT"

OUT=$BASE/generated/cis_mvmr_1k
W=${SLURM_CPUS_PER_TASK:-16}
echo "workers=$W   out=$OUT   started $(date)"

$PY generators/emit.py --n 800 --seed0 1000000 --workers "$W" \
    --prefix cis_mvmr_train --out "$OUT/train"
$PY generators/emit.py --n 100 --seed0 2000000 --workers "$W" \
    --prefix cis_mvmr_val   --out "$OUT/val"
$PY generators/emit.py --n 100 --seed0 3000000 --workers "$W" \
    --prefix cis_mvmr_test  --out "$OUT/test"

echo
echo "=== sizes ==="
du -sh "$OUT"/*/ 2>/dev/null
echo "total: $(du -sh "$OUT" | cut -f1)"
echo "done $(date)"
