#!/bin/bash
#SBATCH --job-name=ai4s-core
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/core_%j.out
#SBATCH --error=logs/core_%j.err


P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
bash "$P"/scripts/fetch_core.sh
