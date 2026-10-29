#!/bin/bash
# Shared environment for all fetch/ETL scripts. Source this; never hardcode paths.
#
#   source "$(git rev-parse --show-toplevel)/scripts/env.sh"
#
# Mirrors paths.py: code lives with the repo, bulk data in a scratch workspace
# that EXPIRES and whose mount point is not stable across re-allocation.

# Derived from this script's own location ($PROJECT_ROOT/scripts/env.sh), so the
# repo runs unchanged wherever it is cloned. Mirrors paths.py, which derives
# PROJECT_ROOT from __file__ for the same reason.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_NAME=deep-ai4s

# Resolve data root: $DEEP_DATA_ROOT -> ws_find -> ./data symlink
if [ -n "${DEEP_DATA_ROOT:-}" ] && [ -d "${DEEP_DATA_ROOT:-}" ]; then
  BASE="$DEEP_DATA_ROOT"
elif command -v ws_find >/dev/null 2>&1 && ws_find "$WORKSPACE_NAME" >/dev/null 2>&1; then
  BASE="$(ws_find $WORKSPACE_NAME | grep '^/' | head -1)"
elif [ -d "$PROJECT_ROOT/data" ]; then
  BASE="$(readlink -f $PROJECT_ROOT/data)"
else
  echo "FATAL: data root not found; workspace '$WORKSPACE_NAME' may have expired." >&2
  echo "  ws_list ; ws_extend $WORKSPACE_NAME 30" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT WORKSPACE_NAME BASE
export DEEP_DATA_ROOT="$BASE"
export LOG="$BASE/logs"
export VENV="$PROJECT_ROOT/venv"
export PY="$VENV/bin/python"
mkdir -p "$LOG"

# Machine-specific overrides (caches, $TMPDIR, shared model cache). Not in git:
# every value in it is a path that exists on exactly one cluster.
# shellcheck source=/dev/null
[ -f "$PROJECT_ROOT/scripts/env.local.sh" ] && . "$PROJECT_ROOT/scripts/env.local.sh"

# Warn when the workspace is close to expiring.
if command -v ws_list >/dev/null 2>&1; then
  _left=$(ws_list 2>/dev/null | awk -v n="id: $WORKSPACE_NAME" '
    $0 ~ n {f=1} f && /remaining time/ {print; exit}')
  case "$_left" in
    *"remaining time"*)
      _d=$(echo "$_left" | grep -oE '[0-9]+ days' | grep -oE '[0-9]+')
      if [ -n "$_d" ] && [ "$_d" -lt 7 ]; then
        echo "*** WARNING: scratch workspace '$WORKSPACE_NAME' expires in $_d days." >&2
        echo "***          ws_extend $WORKSPACE_NAME 30" >&2
      fi
      ;;
  esac
fi
