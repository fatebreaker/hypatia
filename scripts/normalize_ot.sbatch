#!/bin/bash
#SBATCH --job-name=ot-norm
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/otnorm_%j.out
#SBATCH --error=logs/otnorm_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
set -uo pipefail
source "$P"/scripts/env.sh
cd "$PROJECT_ROOT"
export DUCKDB_MEM=48GB DUCKDB_THREADS=16
mkdir -p /tmp/duckdb_spill
$PY analysis/normalize_ot.py
