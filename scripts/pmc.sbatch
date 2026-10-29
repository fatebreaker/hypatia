#!/bin/bash
#SBATCH --job-name=ai4s-pmc
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --output=logs/pmc_%j.out
#SBATCH --error=logs/pmc_%j.err

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
export JOBS=48
bash "$P"/scripts/fetch_pmc.sh
