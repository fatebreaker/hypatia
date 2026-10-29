#!/bin/bash
# THE V3.1 EVALUATION HARNESS (docs/RL_PLAN.md s.18, v3.1 ELICIT-THEN-REWARD): screen_v3.sh's harness -- KERNEL + notebook
# echo, NO library, experiments/harness_v2/screen_any.sh (config custom:kernel:V31) -- with EVERY row scored under the v3.1
# system prompt V31_SYSTEM (default science_check = the science prompt + one sentence, rl/systems_extra.py): the untrained
# base AND each checkpoint. Results under results/pods/harness_v31/ (V31/ = the base, V31__<run>_s<step>/ = a checkpoint),
# never under results/pods/harness_v3/ (screen_v3.sh's root, the science-prompt rows the reader compares against).
#   [CKPT=<run dir> STEP=<n>] [DESIGN_FAMS="..."] [ORDER="design test ext"] MODEL=Qwen3-8B \
#       experiments/harness_v3/screen_v31.sh PORT
# A wrapper, not an edit: screen_any.sh, screen_v3.sh and harness/agents.py are running files. The prompt reaches the
# harness through experiments/harness_v3/v31_site/sitecustomize.py (this directory first on PYTHONPATH): in every python
# process of the screen it registers rl/systems_extra.py's keys into harness/agents.py's SYSTEMS when `agents` is imported,
# sets GBP_SYSTEM=$V31_SYSTEM (screen_any.sh hard-sets science for a base row), and removes itself from the environment the
# sandbox cells inherit. Every other knob (MODEL, MTAG, WORKERS, BENCHES, SKIP_REAL, DESIGN_FAMS, ORDER) passes through as
# in screen_v3.sh. V31_RESULTS overrides the root (tests only).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
PORT="${1:?PORT}"
SYSK="${V31_SYSTEM:-science_check}"
HOOK="$P/experiments/harness_v3/v31_site"
[ -f "$HOOK/sitecustomize.py" ] || { echo "[hv31] missing $HOOK/sitecustomize.py" >&2; exit 2; }
venv/bin/python rl/systems_extra.py --check "$SYSK" || { echo "[hv31] V31_SYSTEM=$SYSK is not a key of rl/systems_extra.py" >&2; exit 2; }
if [ -n "${CKPT:-}" ] && [ -s "$CKPT/system.txt" ] && [ "$(cat "$CKPT/system.txt")" != "$SYSK" ]; then
  echo "[hv31] NOTE: $CKPT was trained under system '$(cat "$CKPT/system.txt")'; the V31 harness scores it under '$SYSK'"
fi
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL
export HV2_RESULTS="${V31_RESULTS:-$P/results/pods/harness_v31}"
export V31_SYSTEM="$SYSK"
export PYTHONPATH="$HOOK${PYTHONPATH:+:$PYTHONPATH}"
exec experiments/harness_v2/screen_any.sh custom:kernel:V31 "$PORT"
