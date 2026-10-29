#!/bin/bash
# ASYNC-1 RL IN HARNESS V3 (docs/RL_PLAN.md 2026-09-29 23:35 UTC, registered): run_v3.sh's launch -- the same harness
# switches, refusals, run naming and trainer arguments -- with the trainer rl/grpo_async.py and GROUP STREAMING ON
# (rl/grpo_async.py docstring: FIFO-by-submission groups in flight, staleness cap, truncated-IS correction). A NEW file:
# run_v3.sh (used by running and queued items) is not touched.
#   experiments/harness_v3/run_v3async.sh SEED PORT [STEPS]
#   env: every run_v3.sh variable (CZ_FAM, PROC_LAMBDA, PROC_SHAM, EXTRA_ARGS, MAXLEN, ON, LR, WAITFOR, DRY_RUN=1, MODEL_TAG
#   + SFT_MODEL) plus ASYNC_INFLIGHT (groups in flight, default 16 = 2 x the 8 prompts per step), ASYNC_MAX_STALE (default
#   2), ASYNC_IS_CAP (default 2), ASYNC_CORRECTION (tis | none; default tis).
# Run dir: run_v3.sh's, with -async1 after -hv3<arm>: $D/ckpt/rl-czero-<FAM>-<FROM>-async1-lr<LR>-s<SEED>, e.g.
#   rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-async1-lr3e-6-s12 (HV3_151) next to HV3_119's synchronous
#   rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr3e-6-s12 -- "-async1" never occurs in a run_v3.sh name, so an
#   async run can never resume a synchronous one (rl/grpo_async.py refuses that as well, and the reverse).
#   A non-default setting is spelled out after -async1 (-f<N> in flight, -m<S> cap, -c<C> IS cap, -nocorr), so two settings
#   never share a directory (harness_async.json in the run dir refuses a resume with other settings in any case).
# Refused: --dyn-sample (not combined with ASYNC-1), --kl, --port2 / --port2-sleep (one colocated server), any --async-*
# in EXTRA_ARGS (use the ASYNC_* variables, so the name always says what ran).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
for v in GBP_LIBRARY GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO; do
  val="${!v:-}"; val="${val,,}"
  case "$val" in ""|0|false|off|no) ;; *) echo "run_v3async.sh: $v=${!v} -- harness v3 is KERNEL + science + echo only (no library, no other component)" >&2; exit 2 ;; esac
done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1
LAM="${PROC_LAMBDA:-0.3}"; SHAM="${PROC_SHAM:-off}"
LAMS="$(venv/bin/python -c 'import sys
x = float(sys.argv[1])
assert 0.0 <= x < 1.0, x
print(format(x, "g"))' "$LAM" 2>/dev/null)" || { echo "run_v3async.sh: PROC_LAMBDA must be a number in [0, 1), got '$LAM'" >&2; exit 2; }
case "$SHAM" in
  off)  ARMTAG="pr$LAMS" ;;
  perm) ARMTAG="prsham$LAMS" ;;
  cols) ARMTAG="prcols$LAMS" ;;
  *) echo "run_v3async.sh: PROC_SHAM must be off, perm or cols, got '$SHAM'" >&2; exit 2 ;;
esac
[ "$SHAM" != off ] && [ "$LAMS" = 0 ] && { echo "run_v3async.sh: a sham arm needs PROC_LAMBDA > 0" >&2; exit 2; }
# ASYNC-1 settings: validated numbers, canonical spellings (the run name is built from them)
AI="${ASYNC_INFLIGHT:-16}"; AS="${ASYNC_MAX_STALE:-2}"; AC="${ASYNC_IS_CAP:-2}"; AK="${ASYNC_CORRECTION:-tis}"
read -r AI AS AC <<< "$(venv/bin/python -c 'import sys
n, s, c = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
assert n >= 1 and s >= 0 and c >= 1.0, (n, s, c)
print(n, s, format(c, "g"))' "$AI" "$AS" "$AC" 2>/dev/null)" \
  || { echo "run_v3async.sh: ASYNC_INFLIGHT (int >= 1), ASYNC_MAX_STALE (int >= 0), ASYNC_IS_CAP (number >= 1) -- got '${ASYNC_INFLIGHT:-}' '${ASYNC_MAX_STALE:-}' '${ASYNC_IS_CAP:-}'" >&2; exit 2; }
[ -n "$AI" ] && [ -n "$AS" ] && [ -n "$AC" ] || { echo "run_v3async.sh: bad ASYNC_* settings" >&2; exit 2; }
case "$AK" in tis|none) ;; *) echo "run_v3async.sh: ASYNC_CORRECTION must be tis or none, got '$AK'" >&2; exit 2 ;; esac
ATAG="-async1"; [ "$AI" != 16 ] && ATAG="$ATAG-f$AI"; [ "$AS" != 2 ] && ATAG="$ATAG-m$AS"; [ "$AC" != 2 ] && ATAG="$ATAG-c$AC"
[ "$AK" != tis ] && ATAG="$ATAG-nocorr"
D="$(venv/bin/python -c 'import paths; print(paths.data_root())')"
ARM=czero; FAM="${CZ_FAM:?CZ_FAM (a training pool under generated/)}"; INIT_ARGS=(); RENDER=rl; FROM=base
LR="${LR:-1e-5}"
if [ -n "${MODEL_TAG:-}" ]; then
  { [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; } \
    || { echo "run_v3async.sh: MODEL_TAG=$MODEL_TAG needs SFT_MODEL=<the non-8B base> (got '${SFT_MODEL:-}')" >&2; exit 2; }
  FROM="$MODEL_TAG$FROM"
elif [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; then
  echo "run_v3async.sh: SFT_MODEL=$SFT_MODEL needs MODEL_TAG (its run dir would otherwise be an 8B run's)" >&2; exit 2
fi
FROM="$FROM-echo-sysscience"
[ -n "${EXTRA_ARGS:-}" ] && read -r -a _xa <<< "$EXTRA_ARGS" && INIT_ARGS+=("${_xa[@]}")
for a in "${INIT_ARGS[@]}"; do
  case "$a" in --library*|--transfer-check*|--init-adapter*|--siblings*|--process-*)
    echo "run_v3async.sh: $a does not belong in EXTRA_ARGS (v3 has no library/transfer check/init adapter; the process arm is PROC_LAMBDA/PROC_SHAM)" >&2; exit 2 ;;
  --dyn-sample*|--kl|--kl=*|--port2*|--async-*)
    echo "run_v3async.sh: $a is not accepted here (ASYNC-1 is built without dynamic sampling, KL or a second server; set the ASYNC_* variables, not --async-* flags)" >&2; exit 2 ;;
  esac
done
[[ " ${EXTRA_ARGS:-} " == *" --require-code "* ]] && FROM="$FROM-rc"
[ "${ON:-0}" = 1 ] && { INIT_ARGS+=(--overflow-neutral); FROM="$FROM-on"; }
[ -n "${MAXLEN:-}" ] && { INIT_ARGS+=(--max-len "$MAXLEN"); FROM="$FROM-$((MAXLEN / 1024))k"; }
FROM="$FROM-hv3$ARMTAG$ATAG"
TRAINER=rl/grpo_async.py
OUT="$D/ckpt/rl-$ARM-$FAM-$FROM-lr$LR-s$SEED"
PARGS=(--process-lambda "$LAMS" --process-sham "$SHAM")
AARGS=(--async-inflight "$AI" --async-max-stale "$AS" --async-is-cap "$AC" --async-correction "$AK")
WAITFOR="${WAITFOR:-}"
if [ -n "$WAITFOR" ]; then
  while [ ! -e "$P/logs/pods/$WAITFOR.done" ] && [ ! -e "$P/logs/pods/$WAITFOR.failed" ]; do sleep 60; done
fi
echo "[hv3_rl_async] $(date -Is) arm=$ARM family=$FAM lr=$LR seed=$SEED steps=$STEPS out=$OUT process=lambda $LAMS sham $SHAM async inflight=$AI max_stale=$AS is_cap=$AC correction=$AK"
[ "${DRY_RUN:-0}" = 1 ] && { echo "OUT=$OUT"; echo "ARGS=--family $FAM --port $PORT --out $OUT --steps $STEPS --lr $LR --seed $SEED --ckpt-every 25 --render $RENDER ${INIT_ARGS[*]} ${PARGS[*]} ${AARGS[*]}"; echo "SFT_MODEL=${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"; echo "TRAINER=$TRAINER"; exit 0; }
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
# harness_v2.json: what screen_any.sh compares its own harness with (equal for the v3 eval harness: kernel, science, echo)
venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; json.dump(v2.HarnessConfig.from_env().as_dict(), open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json"
export SFT_MODEL="${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python "$TRAINER" --family "$FAM" --port "$PORT" --out "$OUT" \
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --ckpt-every 25 --render "$RENDER" "${INIT_ARGS[@]}" "${PARGS[@]}" "${AARGS[@]}"
