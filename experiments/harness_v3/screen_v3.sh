#!/bin/bash
# THE V3 EVALUATION HARNESS (docs/RL_PLAN.md s.15): experiments/harness_v2/screen_any.sh with the v3 switches -- KERNEL,
# science prompt, notebook echo, NO library (config custom:kernel:V3) -- and its results under results/pods/harness_v3[_MTAG]/
# (screen_any.sh's HV2_RESULTS root, so no v3 directory ever lands among the v2/P14 screens that analyze.py and read_p14.py
# read). A wrapper, not an edit: screen_any.sh is executed by running P14 evaluation steps.
#   [MODEL=Qwen3-14B MTAG=q14] [CKPT=<run dir> STEP=<n>] [DESIGN_FAMS="..."] [ORDER="design test ext"] \
#       experiments/harness_v3/screen_v3.sh PORT
# -> results/pods/harness_v3_q14/V3/ (the base) or .../V3__<run>_s<step>/ (a checkpoint, P14 runs included: they are
#    scored in THIS harness for the matched comparison; screen_any.sh prints the training-harness difference).
# Every other knob (WORKERS, BENCHES, SKIP_REAL) passes through unchanged. V3_RESULTS overrides the root (tests only).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
PORT="${1:?PORT}"
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL
export HV2_RESULTS="${V3_RESULTS:-$P/results/pods/harness_v3}"
exec experiments/harness_v2/screen_any.sh custom:kernel:V3 "$PORT"
