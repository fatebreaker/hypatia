#!/bin/bash
# RL IN HARNESS V3.1 -- ELICIT-THEN-REWARD (docs/RL_PLAN.md s.18, registered 2026-09-26 16:35 UTC): experiments/harness_v3/
# run_v3.sh's rules LINE FOR LINE (czero arm: fresh LoRA on the untrained base, RL render, LR 1e-5, 8 x 6, ckpt every 25;
# KERNEL + notebook echo, NO library; the s.15 process reward PROC_LAMBDA / PROC_SHAM) with ONE change: the system prompt
# is GBP_SYSTEM=science_check (rl/systems_extra.py: the science prompt + one sentence asking for a robustness
# re-estimate), registered into harness/agents.py's SYSTEMS by rl/grpo_v31.py -- the trainer entry here -- which is
# rl/grpo_v3.py otherwise. A NEW file: run_v3.sh (running 8B/14B v3 arms) is not touched; tests/test_v31.py checks that
# this file is run_v3.sh with exactly the substitutions listed there.
#   experiments/harness_v3/run_v31.sh SEED PORT [STEPS]
#   env: CZ_FAM, PROC_LAMBDA (default 0.3), PROC_SHAM (off | perm | cols; default off), EXTRA_ARGS, MAXLEN, ON, LR,
#   WAITFOR, DRY_RUN=1 (print OUT/ARGS/SFT_MODEL/GBP_SYSTEM and exit), MODEL_TAG + SFT_MODEL -- all as run_v3.sh.
# Run dir: $DEEP_DATA_ROOT/ckpt/rl-czero-<FAM>-<FROM>-lr<LR>-s<SEED>, FROM = [q14]base-echo-sysscicheck2[-rc][-32k]-hv3<arm>
#   ("sysscicheck2" where run_v3.sh writes "sysscience", so a v3.1 run dir can never be a v3 one; the "2" is the process
#   DETECTOR v2, rl/process_signals_v31.py's reest -- docs/RL_PLAN.md s.18, the entry below 19:15 UTC -- so a run scored
#   by it can never share a directory with, or be read as, a v3.1 attempt under the old detector ("sysscicheck"), and
#   rl/grpo_v31.py refuses to resume a dir whose harness_v31.json names another detector):
#   ELICIT-PR    PROC_LAMBDA=0.3  -> rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0.3-lr1e-5-s1
#   ELICIT-PLAIN PROC_LAMBDA=0    -> rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0-lr1e-5-s1
# Its checkpoints are scored by experiments/harness_v3/screen_v31.sh (the same prompt), never by screen_v3.sh.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
for v in GBP_LIBRARY GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO; do
  val="${!v:-}"; val="${val,,}"
  case "$val" in ""|0|false|off|no) ;; *) echo "run_v31.sh: $v=${!v} -- harness v3 is KERNEL + science + echo only (no library, no other component)" >&2; exit 2 ;; esac
done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science_check GBP_ECHO_LAST=1
LAM="${PROC_LAMBDA:-0.3}"; SHAM="${PROC_SHAM:-off}"
LAMS="$(venv/bin/python -c 'import sys
x = float(sys.argv[1])
assert 0.0 <= x < 1.0, x
print(format(x, "g"))' "$LAM" 2>/dev/null)" || { echo "run_v31.sh: PROC_LAMBDA must be a number in [0, 1), got '$LAM'" >&2; exit 2; }
case "$SHAM" in
  off)  ARMTAG="pr$LAMS" ;;
  perm) ARMTAG="prsham$LAMS" ;;
  cols) ARMTAG="prcols$LAMS" ;;
  *) echo "run_v31.sh: PROC_SHAM must be off, perm or cols, got '$SHAM'" >&2; exit 2 ;;
esac
[ "$SHAM" != off ] && [ "$LAMS" = 0 ] && { echo "run_v31.sh: a sham arm needs PROC_LAMBDA > 0" >&2; exit 2; }
D="$(venv/bin/python -c 'import paths; print(paths.data_root())')"
ARM=czero; FAM="${CZ_FAM:?CZ_FAM (a training pool under generated/)}"; INIT_ARGS=(); RENDER=rl; FROM=base
LR="${LR:-1e-5}"
if [ -n "${MODEL_TAG:-}" ]; then
  { [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; } \
    || { echo "run_v31.sh: MODEL_TAG=$MODEL_TAG needs SFT_MODEL=<the non-8B base> (got '${SFT_MODEL:-}')" >&2; exit 2; }
  FROM="$MODEL_TAG$FROM"
elif [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; then
  echo "run_v31.sh: SFT_MODEL=$SFT_MODEL needs MODEL_TAG (its run dir would otherwise be an 8B run's)" >&2; exit 2
fi
FROM="$FROM-echo-sysscicheck2"
[ -n "${EXTRA_ARGS:-}" ] && read -r -a _xa <<< "$EXTRA_ARGS" && INIT_ARGS+=("${_xa[@]}")
for a in "${INIT_ARGS[@]}"; do
  case "$a" in --library*|--transfer-check*|--init-adapter*|--siblings*|--process-*)
    echo "run_v31.sh: $a does not belong in EXTRA_ARGS (v3 has no library/transfer check/init adapter; the process arm is PROC_LAMBDA/PROC_SHAM)" >&2; exit 2 ;;
  esac
done
[[ " ${EXTRA_ARGS:-} " == *" --require-code "* ]] && FROM="$FROM-rc"
[ "${ON:-0}" = 1 ] && { INIT_ARGS+=(--overflow-neutral); FROM="$FROM-on"; }
[ -n "${MAXLEN:-}" ] && { INIT_ARGS+=(--max-len "$MAXLEN"); FROM="$FROM-$((MAXLEN / 1024))k"; }
FROM="$FROM-hv3$ARMTAG"
OUT="$D/ckpt/rl-$ARM-$FAM-$FROM-lr$LR-s$SEED"
PARGS=(--process-lambda "$LAMS" --process-sham "$SHAM")
WAITFOR="${WAITFOR:-}"
if [ -n "$WAITFOR" ]; then
  while [ ! -e "$P/logs/pods/$WAITFOR.done" ] && [ ! -e "$P/logs/pods/$WAITFOR.failed" ]; do sleep 60; done
fi
echo "[hv31_rl] $(date -Is) arm=$ARM family=$FAM lr=$LR seed=$SEED steps=$STEPS out=$OUT process=lambda $LAMS sham $SHAM"
[ "${DRY_RUN:-0}" = 1 ] && { echo "OUT=$OUT"; echo "ARGS=--family $FAM --port $PORT --out $OUT --steps $STEPS --lr $LR --seed $SEED --ckpt-every 25 --render $RENDER ${INIT_ARGS[*]} ${PARGS[*]}"; echo "SFT_MODEL=${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"; echo "GBP_SYSTEM=$GBP_SYSTEM"; exit 0; }
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
# harness_v2.json: what screen_any.sh compares its own harness with (equal for the v3.1 eval harness: kernel, science_check, echo)
venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; json.dump(v2.HarnessConfig.from_env().as_dict(), open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json"
export SFT_MODEL="${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python rl/grpo_v31.py --family "$FAM" --port "$PORT" --out "$OUT" \
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --ckpt-every 25 --render "$RENDER" "${INIT_ARGS[@]}" "${PARGS[@]}"
