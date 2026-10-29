#!/bin/bash
# Allocate the scratch workspace that holds all bulk data, and link it in.
#
# Bulk data lives OUTSIDE the repo because it is ~149 GB and reproducible. On
# Unity that means a `workspace`, which EXPIRES -- this sets a reminder and
# creates ./data as a convenience symlink. paths.py and env.sh resolve the root
# dynamically ($DEEP_DATA_ROOT -> ws_find -> ./data), so nothing hardcodes the
# mount point: an expired workspace re-allocated later can land on a different
# filesystem.
#
# On a cluster without ws_* tooling, skip this and just:
#   export DEEP_DATA_ROOT=/somewhere/with/200GB
#
# Usage: scripts/setup_workspace.sh [days] [email]
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=${WORKSPACE_NAME:-deep-ai4s}
DAYS=${1:-30}
MAIL=${2:-}

if ! command -v ws_allocate >/dev/null 2>&1; then
  echo "no ws_allocate on this system."
  echo "Set DEEP_DATA_ROOT to a directory with ~200 GB free and re-run fetch_all.sh:"
  echo "  export DEEP_DATA_ROOT=/path/with/space"
  exit 0
fi

ARGS=(-F workspace -n "$NAME" -d "$DAYS" -r 7 -c "AI4S deep research data")
[ -n "$MAIL" ] && ARGS+=(-m "$MAIL")

if ws_find "$NAME" >/dev/null 2>&1; then
  echo "workspace '$NAME' already exists; extending by $DAYS days"
  ws_allocate -x "${ARGS[@]}" >/dev/null
else
  ws_allocate "${ARGS[@]}" >/dev/null
fi

BASE="$(ws_find "$NAME" | grep '^/' | head -1)"
[ -z "$BASE" ] && { echo "FATAL: could not resolve workspace path"; exit 1; }

mkdir -p "$BASE"/{logs,opentargets,pubmed,pmc,aact,uniprot,chembl,clinvar,gwas_catalog,mgi,impc,gnomad}
ln -sfn "$BASE" "$P/data"

echo "workspace : $BASE"
echo "symlink   : $P/data -> $BASE"
ws_list 2>/dev/null | grep -A3 "id: $NAME" | grep -E "remaining|expiration" || true
echo
echo "NOTE: if this expires the data is DELETED, not archived."
echo "      ws_extend $NAME 30      # extensions are effectively unlimited"
echo "      python paths.py         # prints days remaining, warns under 7"
echo
echo "next: scripts/fetch_all.sh"
