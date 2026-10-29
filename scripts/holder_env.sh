#!/bin/bash

# BASH_SOURCE, not $0: this file is meant to be SOURCED, and in a sourced
# script $0 is the calling shell ("bash"), so a $0-derived root silently
# resolves to the wrong directory and GBP_BASE_URL comes back empty.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Point a client at the long-lived GPU holder.
#   source scripts/holder_env.sh
# Sets GBP_BASE_URL / HOLDER_* from .gpu_holder, and verifies the endpoint is
# actually answering before you spend a run on it.
PROJECT_ROOT="$P"
STAMP=$PROJECT_ROOT/.gpu_holder
if [ ! -f "$STAMP" ]; then
  echo "no holder running: submit scripts/gpu_holder.sbatch" >&2
  return 1 2>/dev/null || exit 1
fi
set -a; . "$STAMP"; set +a
export GBP_BASE_URL="$HOLDER_BASE_URL"
export GBP_API_KEY=EMPTY
if curl -sf --max-time 10 "${HOLDER_BASE_URL%/v1}/health" >/dev/null 2>&1; then
  echo "holder $HOLDER_JOBID on $HOLDER_NODE serving $HOLDER_MODEL -> $GBP_BASE_URL"
else
  echo "WARN: holder recorded but $HOLDER_BASE_URL is not answering" >&2
fi
