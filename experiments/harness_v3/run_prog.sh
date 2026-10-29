#!/bin/bash
# ARM E -- RL IN HARNESS V3 WITH THE PORTABLE-PROGRAM REWARD (docs/RL_PLAN.md s.18): experiments/harness_v3/run_v3.sh's rules
# (run.sh's `czero` arm: fresh LoRA on the untrained base, RL render, LR 1e-5, 8 x 6, ckpt every 25; harness v3 = KERNEL +
# science prompt + notebook echo, NO library; the s.15 process reward PROC_LAMBDA / PROC_SHAM) through rl/grpo_prog.py,
# which adds `portable`: the rollout's executed cells re-run in a fresh sandbox on ONE sibling dataset of the same design
# and graded by the sibling's grader (weight PORT_MU; PORT_SHAM=perm permutes it within each GRPO group). A NEW file:
# run_v3.sh (running 8B/14B v3 arms) is not touched; everything below that is not about PORT_* is run_v3.sh's, line for line.
#   experiments/harness_v3/run_prog.sh SEED PORT [STEPS]
#   env: CZ_FAM, PROC_LAMBDA (default 0.3), PROC_SHAM (off | perm | cols; default off), PORT_MU (default 0.3; must be in
#   (0, 1): a mu = 0 arm is run_v3.sh's), PORT_SHAM (off | perm; default off), PORT_EXTRACT (json | printed; default json =
#   the registered spec, extract_answer on the re-run's stdout; printed = the number printed where the original run printed
#   its answer -- rl/grpo_prog.py), PORT_SIBLING (family | layout; default family = the registered spec, siblings.pick;
#   layout = a sibling with the same data-file names, headers and roles), EXTRA_ARGS, MAXLEN, ON, LR, WAITFOR,
#   DRY_RUN=1 (print OUT / ARGS / SFT_MODEL and exit before anything is written or waited on), MODEL_TAG + SFT_MODEL.
# Run dir: $DEEP_DATA_ROOT/ckpt/rl-czero-<FAM>-<FROM>-lr<LR>-s<SEED>, FROM = [q14]base-echo-sysscience[-rc][-32k]-hv3<arm><port>
#   with <arm> as run_v3.sh (pr<lambda> | prsham<lambda> | prcols<lambda>) and <port> = port[pos][lay][sham]<mu> ("pos" =
#   PORT_EXTRACT printed, "lay" = PORT_SIBLING layout, "sham" = PORT_SHAM perm): ...-hv3pr0.3port0.3-... (PORT, the spec) and
#   ...-hv3pr0.3portsham0.3-... (PORT-SHAM); ...-hv3pr0.3portposlay0.3-... / ...portposlaysham0.3-... (the measured-live
#   variant, docs/RL_PLAN.md s.18). "port" never occurs in a run_v3.sh name.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
for v in GBP_LIBRARY GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO; do
  val="${!v:-}"; val="${val,,}"
  case "$val" in ""|0|false|off|no) ;; *) echo "run_prog.sh: $v=${!v} -- harness v3 is KERNEL + science + echo only (no library, no other component)" >&2; exit 2 ;; esac
done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1
LAM="${PROC_LAMBDA:-0.3}"; SHAM="${PROC_SHAM:-off}"; MU="${PORT_MU:-0.3}"; PSHAM="${PORT_SHAM:-off}"; PEX="${PORT_EXTRACT:-json}"; PSIB="${PORT_SIBLING:-family}"
LAMS="$(venv/bin/python -c 'import sys
x = float(sys.argv[1])
assert 0.0 <= x < 1.0, x
print(format(x, "g"))' "$LAM" 2>/dev/null)" || { echo "run_prog.sh: PROC_LAMBDA must be a number in [0, 1), got '$LAM'" >&2; exit 2; }
MUS="$(venv/bin/python -c 'import sys
x = float(sys.argv[1])
assert 0.0 < x < 1.0, x
print(format(x, "g"))' "$MU" 2>/dev/null)" || { echo "run_prog.sh: PORT_MU must be a number in (0, 1), got '$MU' (mu = 0 is run_v3.sh)" >&2; exit 2; }
case "$SHAM" in
  off)  ARMTAG="pr$LAMS" ;;
  perm) ARMTAG="prsham$LAMS" ;;
  cols) ARMTAG="prcols$LAMS" ;;
  *) echo "run_prog.sh: PROC_SHAM must be off, perm or cols, got '$SHAM'" >&2; exit 2 ;;
esac
[ "$SHAM" != off ] && [ "$LAMS" = 0 ] && { echo "run_prog.sh: a sham arm needs PROC_LAMBDA > 0" >&2; exit 2; }
case "$PEX" in
  json)    PTAG=port ;;
  printed) PTAG=portpos ;;
  *) echo "run_prog.sh: PORT_EXTRACT must be json or printed, got '$PEX'" >&2; exit 2 ;;
esac
case "$PSIB" in
  family) ;;
  layout) PTAG="${PTAG}lay" ;;
  *) echo "run_prog.sh: PORT_SIBLING must be family or layout, got '$PSIB'" >&2; exit 2 ;;
esac
case "$PSHAM" in
  off)  ARMTAG="${ARMTAG}${PTAG}$MUS" ;;
  perm) ARMTAG="${ARMTAG}${PTAG}sham$MUS" ;;
  *) echo "run_prog.sh: PORT_SHAM must be off or perm, got '$PSHAM'" >&2; exit 2 ;;
esac
D="$(venv/bin/python -c 'import paths; print(paths.data_root())')"
ARM=czero; FAM="${CZ_FAM:?CZ_FAM (a training pool under generated/)}"; INIT_ARGS=(); RENDER=rl; FROM=base
LR="${LR:-1e-5}"
if [ -n "${MODEL_TAG:-}" ]; then
  { [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; } \
    || { echo "run_prog.sh: MODEL_TAG=$MODEL_TAG needs SFT_MODEL=<the non-8B base> (got '${SFT_MODEL:-}')" >&2; exit 2; }
  FROM="$MODEL_TAG$FROM"
elif [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; then
  echo "run_prog.sh: SFT_MODEL=$SFT_MODEL needs MODEL_TAG (its run dir would otherwise be an 8B run's)" >&2; exit 2
fi
FROM="$FROM-echo-sysscience"
[ -n "${EXTRA_ARGS:-}" ] && read -r -a _xa <<< "$EXTRA_ARGS" && INIT_ARGS+=("${_xa[@]}")
for a in "${INIT_ARGS[@]}"; do
  case "$a" in --library*|--transfer-check*|--init-adapter*|--siblings*|--process-*|--port-*)
    echo "run_prog.sh: $a does not belong in EXTRA_ARGS (v3 has no library/transfer check/init adapter; the process and portable arms are PROC_*/PORT_*)" >&2; exit 2 ;;
  esac
done
[[ " ${EXTRA_ARGS:-} " == *" --require-code "* ]] && FROM="$FROM-rc"
[ "${ON:-0}" = 1 ] && { INIT_ARGS+=(--overflow-neutral); FROM="$FROM-on"; }
[ -n "${MAXLEN:-}" ] && { INIT_ARGS+=(--max-len "$MAXLEN"); FROM="$FROM-$((MAXLEN / 1024))k"; }
FROM="$FROM-hv3$ARMTAG"
OUT="$D/ckpt/rl-$ARM-$FAM-$FROM-lr$LR-s$SEED"
PARGS=(--process-lambda "$LAMS" --process-sham "$SHAM" --port-mu "$MUS" --port-sham "$PSHAM" --port-extract "$PEX" --port-sibling "$PSIB")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "OUT=$OUT"
  echo "ARGS=--family $FAM --port $PORT --out $OUT --steps $STEPS --lr $LR --seed $SEED --ckpt-every 25 --render $RENDER ${INIT_ARGS[*]} ${PARGS[*]}"
  echo "SFT_MODEL=${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"
  exit 0
fi
WAITFOR="${WAITFOR:-}"
if [ -n "$WAITFOR" ]; then
  while [ ! -e "$P/logs/pods/$WAITFOR.done" ] && [ ! -e "$P/logs/pods/$WAITFOR.failed" ]; do sleep 60; done
fi
echo "[hv3_prog] $(date -Is) arm=$ARM family=$FAM lr=$LR seed=$SEED steps=$STEPS out=$OUT process=lambda $LAMS sham $SHAM portable=mu $MUS sham $PSHAM extract $PEX sibling $PSIB"
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
# harness_v2.json: what screen_any.sh compares its own harness with (equal for the v3 eval harness: kernel, science, echo)
venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; json.dump(v2.HarnessConfig.from_env().as_dict(), open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json"
export SFT_MODEL="${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python rl/grpo_prog.py --family "$FAM" --port "$PORT" --out "$OUT" \
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --ckpt-every 25 --render "$RENDER" "${INIT_ARGS[@]}" "${PARGS[@]}"
