#!/bin/bash
# run_v3rg.sh (2026-10-01, H-GROUNDED-RL): a COPY of run_v3.sh whose trainer is rl/grpo_rg.py (grpo_v3 unchanged + the
# GROUNDED reward: an answer not produced by the agent's own executed code earns 0) and whose run names carry -rg.
# run_v3.sh itself is not edited (runs are using it). Everything below is run_v3.sh's text except those two changes.
# RL IN HARNESS V3 (docs/RL_PLAN.md s.15): experiments/rl_foothold/run.sh's `czero` arm (fresh LoRA on the untrained base,
# RL render, LR 1e-5, 8 x 6, ckpt every 25) through rl/grpo_v3.py -- rl/grpo_onpolicy.py's trainer unchanged, its rollouts
# in the v3 harness (KERNEL + science prompt + notebook echo, NO library, no other v2 component) and its reward
# r_answer + bounded PROCESS term (harness/process_signals.py). A new file: run_v2.sh (used by the running P14 jobs) is
# not touched; the naming, MODEL_TAG rule and trainer arguments are run_v2.sh's.
#   experiments/harness_v3/run_v3.sh SEED PORT [STEPS]
#   env: CZ_FAM (a training pool under generated/, its train/), PROC_LAMBDA (default 0.3), PROC_SHAM (off | perm | cols;
#   default off), EXTRA_ARGS, MAXLEN, ON, LR, WAITFOR, DRY_RUN=1 (print OUT/ARGS and exit), MODEL_TAG + SFT_MODEL (a
#   non-8B base, as run_v2.sh: MODEL_TAG=q14 SFT_MODEL=/tmp/deeprs_models/Qwen3-14B -> ...-q14base-...).
#   The harness switches are SET here (GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1); any other GBP_* component in the
#   environment (library, profile, apidocs, siblings, checks) is refused, not silently dropped.
# Run dir: $DEEP_DATA_ROOT/ckpt/rl-czero-<FAM>-<FROM>-lr<LR>-s<SEED>, FROM = [q14]base-echo-sysscience[-rc][-32k]-hv3<arm>
#   with <arm> = pr<lambda> (process reward), prsham<lambda> (PROCESS-SHAM, permuted within group), prcols<lambda>
#   (column decoy; measurement only), pr0 (lambda 0 = answer-only in the v3 harness). "-hv3" never occurs in a v2/P14 name.
#   Recipe tags follow -hv3<arm>: -kl<BETA> (EXTRA_ARGS "--kl BETA"), then -ds<N> (EXTRA_ARGS "--dyn-sample N": DAPO-style
#   dynamic sampling through rl/grpo_ds.py, which wraps rl/grpo_v3.py; RL_PLAN 2026-09-29 recipe pilot arm B).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
for v in GBP_LIBRARY GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO; do
  val="${!v:-}"; val="${val,,}"
  case "$val" in ""|0|false|off|no) ;; *) echo "run_v3.sh: $v=${!v} -- harness v3 is KERNEL + science + echo only (no library, no other component)" >&2; exit 2 ;; esac
done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1
LAM="${PROC_LAMBDA:-0.3}"; SHAM="${PROC_SHAM:-off}"
LAMS="$(venv/bin/python -c 'import sys
x = float(sys.argv[1])
assert 0.0 <= x < 1.0, x
print(format(x, "g"))' "$LAM" 2>/dev/null)" || { echo "run_v3.sh: PROC_LAMBDA must be a number in [0, 1), got '$LAM'" >&2; exit 2; }
case "$SHAM" in
  off)  ARMTAG="pr$LAMS" ;;
  perm) ARMTAG="prsham$LAMS" ;;
  cols) ARMTAG="prcols$LAMS" ;;
  *) echo "run_v3.sh: PROC_SHAM must be off, perm or cols, got '$SHAM'" >&2; exit 2 ;;
esac
[ "$SHAM" != off ] && [ "$LAMS" = 0 ] && { echo "run_v3.sh: a sham arm needs PROC_LAMBDA > 0" >&2; exit 2; }
D="$(venv/bin/python -c 'import paths; print(paths.data_root())')"
ARM=czero; FAM="${CZ_FAM:?CZ_FAM (a training pool under generated/)}"; INIT_ARGS=(); RENDER=rl; FROM=base
LR="${LR:-1e-5}"
if [ -n "${MODEL_TAG:-}" ]; then
  { [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; } \
    || { echo "run_v3.sh: MODEL_TAG=$MODEL_TAG needs SFT_MODEL=<the non-8B base> (got '${SFT_MODEL:-}')" >&2; exit 2; }
  FROM="$MODEL_TAG$FROM"
elif [ -n "${SFT_MODEL:-}" ] && [ "$(basename "${SFT_MODEL%/}")" != Qwen3-8B ]; then
  echo "run_v3.sh: SFT_MODEL=$SFT_MODEL needs MODEL_TAG (its run dir would otherwise be an 8B run's)" >&2; exit 2
fi
FROM="$FROM-echo-sysscience"
[ -n "${EXTRA_ARGS:-}" ] && read -r -a _xa <<< "$EXTRA_ARGS" && INIT_ARGS+=("${_xa[@]}")
for a in "${INIT_ARGS[@]}"; do
  case "$a" in --library*|--transfer-check*|--init-adapter*|--siblings*|--process-*)
    echo "run_v3.sh: $a does not belong in EXTRA_ARGS (v3 has no library/transfer check/init adapter; the process arm is PROC_LAMBDA/PROC_SHAM)" >&2; exit 2 ;;
  esac
done
# INIT_ADAPTER (2026-09-30, H-MC-DISTILL stage 2; opt-in): start RL from an SFT adapter, sampled and scored in the SFT
# ("train") render it was trained in; INIT_TAG names it in the run dir (<tag>-base-...). Unset = unchanged.
if [ -n "${INIT_ADAPTER:-}" ]; then
  [ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "run_v3.sh: no adapter at $INIT_ADAPTER" >&2; exit 2; }
  # INIT_RENDER (2026-10-01): an RL-trained adapter (sampled in "rl") continues in "rl"; default "train" (SFT adapters).
  case "${INIT_RENDER:-train}" in rl|train) ;; *) echo "run_v3.sh: INIT_RENDER must be rl or train" >&2; exit 2 ;; esac
  INIT_ARGS+=(--init-adapter "$INIT_ADAPTER"); RENDER="${INIT_RENDER:-train}"; FROM="${INIT_TAG:-sft}-$FROM"; export HV3_INIT_ADAPTER=1   # grpo_v3.py refuses --init-adapter otherwise
fi
# THINK=1 (2026-10-01; opt-in): sample and score in the "train" render from the BASE model -- the generation prompt ends at
# "<|im_start|>assistant\n" with no empty think block, so Qwen3 reasons (<think>...</think>) before each reply, and the
# raw reply (reasoning included) stays in the history it is later scored in. Run dir: <...>-think-base-... Unset = unchanged.
if [ "${THINK:-0}" = 1 ]; then
  [ -z "${INIT_ADAPTER:-}" ] || { echo "run_v3.sh: THINK=1 is for runs from the base (INIT_ADAPTER already uses the train render)" >&2; exit 2; }
  RENDER=train; FROM="think-$FROM"
fi
[[ " ${EXTRA_ARGS:-} " == *" --require-code "* ]] && FROM="$FROM-rc"
FROM="$FROM-rg"   # run_v3rg.sh: the GROUNDED reward (rl/grpo_rg.py) is part of every run name
[ "${ON:-0}" = 1 ] && { INIT_ARGS+=(--overflow-neutral); FROM="$FROM-on"; }
[ -n "${MAXLEN:-}" ] && { INIT_ARGS+=(--max-len "$MAXLEN"); FROM="$FROM-$((MAXLEN / 1024))k"; }
FROM="$FROM-hv3$ARMTAG"
# 2026-09-29 (recipe pilot, RL_PLAN): a KL penalty to the base is part of the run name, so a --kl run never shares a
# dir with a plain run of the same seed. Only the "--kl BETA" spelling is accepted.
[[ " ${EXTRA_ARGS:-} " == *" --kl="* ]] && { echo "run_v3.sh: write --kl BETA (space), not --kl=BETA" >&2; exit 2; }
_klre=' --kl ([0-9.eE+-]+) '
if [[ " ${EXTRA_ARGS:-} " =~ $_klre ]]; then FROM="$FROM-kl${BASH_REMATCH[1]}"; fi
# 2026-09-29 (recipe pilot arm B, RL_PLAN): DAPO-style dynamic sampling ("--dyn-sample N": draw N prompts per step, train
# on the first --prompts-per-step of them whose total reward varies) runs through rl/grpo_ds.py and is part of the run
# name (-ds<N>, after -kl). Only the "--dyn-sample N" spelling, N a positive integer, given once. Without it the trainer
# is rl/grpo_v3.py and the name is unchanged.
[[ " ${EXTRA_ARGS:-} " == *" --dyn-sample"* ]] && { echo "run_v3rg.sh: --dyn-sample is not combined with the grounded reward" >&2; exit 2; }
TRAINER=rl/grpo_rg.py; _dsn=""
[[ " ${EXTRA_ARGS:-} " == *" --dyn-sample="* ]] && { echo "run_v3.sh: write --dyn-sample N (space), not --dyn-sample=N" >&2; exit 2; }
if [[ " ${EXTRA_ARGS:-} " == *" --dyn-sample"* ]]; then
  _dsre=' --dyn-sample ([1-9][0-9]*) '
  if [[ " ${EXTRA_ARGS:-} " =~ $_dsre ]]; then _dsn="${BASH_REMATCH[1]}"; fi
  if [ -z "$_dsn" ] || [[ " ${EXTRA_ARGS:-} " == *" --dyn-sample"*" --dyn-sample"* ]]; then
    echo "run_v3.sh: --dyn-sample takes one positive integer N, given once (EXTRA_ARGS='${EXTRA_ARGS:-}')" >&2; exit 2
  fi
  FROM="$FROM-ds$_dsn"; TRAINER=rl/grpo_ds.py
fi
OUT="$D/ckpt/rl-$ARM-$FAM-$FROM-lr$LR-s$SEED"
PARGS=(--process-lambda "$LAMS" --process-sham "$SHAM")
WAITFOR="${WAITFOR:-}"
if [ -n "$WAITFOR" ]; then
  while [ ! -e "$P/logs/pods/$WAITFOR.done" ] && [ ! -e "$P/logs/pods/$WAITFOR.failed" ]; do sleep 60; done
fi
echo "[hv3_rl] $(date -Is) arm=$ARM family=$FAM lr=$LR seed=$SEED steps=$STEPS out=$OUT process=lambda $LAMS sham $SHAM${_dsn:+ dyn_sample $_dsn}"
[ "${DRY_RUN:-0}" = 1 ] && { echo "OUT=$OUT"; echo "ARGS=--family $FAM --port $PORT --out $OUT --steps $STEPS --lr $LR --seed $SEED --ckpt-every 25 --render $RENDER ${INIT_ARGS[*]} ${PARGS[*]}"; echo "SFT_MODEL=${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"; [ -n "$_dsn" ] && echo "TRAINER=$TRAINER"; exit 0; }
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
# harness_v2.json: what screen_any.sh compares its own harness with (equal for the v3 eval harness: kernel, science, echo)
venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; json.dump(v2.HarnessConfig.from_env().as_dict(), open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json"
export SFT_MODEL="${SFT_MODEL:-/tmp/deeprs_models/Qwen3-8B}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python "$TRAINER" --family "$FAM" --port "$PORT" --out "$OUT" \
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --ckpt-every 25 --render "$RENDER" "${INIT_ARGS[@]}" "${PARGS[@]}"
