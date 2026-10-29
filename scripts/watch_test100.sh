#!/bin/bash

P="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # submit from the repo root; works off-SLURM too
P="$P"; J=62405027
while squeue -h -j $J >/dev/null 2>&1 && [ -n "$(squeue -h -j $J 2>/dev/null)" ]; do sleep 60; done
echo "job $J finished $(date)" 
sed -n '/agent=llm/,$p' $P/logs/gbpeval_$J.out 2>/dev/null | grep -E "pass rate|mean score|cells run|ADJUSTED|WARNING|HARNESS"
