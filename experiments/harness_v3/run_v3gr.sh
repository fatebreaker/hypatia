#!/bin/bash
# H-GRDS RUNNER (docs/RL_PLAN.md 2026-09-29 05:35 UTC, registered before the build): experiments/harness_v3/run_v3.sh's
# configuration, refusals and naming -- run_v3.sh itself is NOT edited (running and queued RL items execute it) -- with the
# trainer rl/grpo_grds.py: answer credit ONLY if the answer is grounded (process_signals 'grounded', grpo_v3's own call),
# else 0; no process bonus; DAPO-style dynamic sampling (rl/grpo_ds.py) on the gated reward.
#   experiments/harness_v3/run_v3gr.sh SEED PORT [STEPS]
#   env: exactly run_v3.sh's (CZ_FAM, EXTRA_ARGS, MAXLEN, ON, LR, DRY_RUN, MODEL_TAG + SFT_MODEL); EXTRA_ARGS must hold
#   "--dyn-sample N"; PROC_LAMBDA / PROC_SHAM must be unset or 0 / off (there is no process bonus); WAITFOR is refused.
# HOW: run_v3.sh is asked for its DRY_RUN with PROC_LAMBDA=0 PROC_SHAM=off -- so every check, the run name and the trainer
# arguments are its own -- and must answer TRAINER=rl/grpo_ds.py with a run dir ...-hv3pr0-...-ds<N>-...; the run dir is that
# name with "-hv3pr0-" -> "-hv3gr-" (e.g. rl-czero-causal_rdd-base-echo-sysscience-rc-hv3gr-ds12-lr1e-5-s1: never an
# existing run's name, and no existing run's name changes), the trainer rl/grpo_grds.py, the arguments run_v3.sh's with
# only --out replaced. Then run_v3.sh's launch tail, line for line (its exports are not inherited from a DRY_RUN
# subprocess; experiments/harness_v3/tests/test_grds.py checks the lines are run_v3.sh's).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
case "${PROC_LAMBDA:-0}" in 0|0.0) ;; *) echo "run_v3gr.sh: H-GRDS has no process bonus -- PROC_LAMBDA must be unset or 0 (got '$PROC_LAMBDA')" >&2; exit 2 ;; esac
case "${PROC_SHAM:-off}" in off) ;; *) echo "run_v3gr.sh: H-GRDS has no process arm -- PROC_SHAM must be unset or off (got '$PROC_SHAM')" >&2; exit 2 ;; esac
[ -z "${WAITFOR:-}" ] || { echo "run_v3gr.sh: WAITFOR is not supported here (launch the item when the GPU is yours)" >&2; exit 2; }
[[ " ${EXTRA_ARGS:-} " == *" --dyn-sample "* ]] \
  || { echo "run_v3gr.sh: H-GRDS is registered with dynamic sampling -- EXTRA_ARGS needs '--dyn-sample N'" >&2; exit 2; }
DRY="$(env -u WAITFOR DRY_RUN=1 PROC_LAMBDA=0 PROC_SHAM=off experiments/harness_v3/run_v3.sh "$SEED" "$PORT" "$STEPS")" \
  || { echo "run_v3gr.sh: run_v3.sh refused this configuration (above)" >&2; exit 2; }
OUT0="$(sed -n 's/^OUT=//p' <<< "$DRY")"; ARGS0="$(sed -n 's/^ARGS=//p' <<< "$DRY")"
SFT="$(sed -n 's/^SFT_MODEL=//p' <<< "$DRY")"; TR0="$(sed -n 's/^TRAINER=//p' <<< "$DRY")"
[ "$TR0" = rl/grpo_ds.py ] && [ -n "$OUT0" ] && [ -n "$ARGS0" ] && [ -n "$SFT" ] \
  || { echo "run_v3gr.sh: unexpected run_v3.sh DRY_RUN (TRAINER '$TR0', OUT '$OUT0')" >&2; exit 2; }
B0="$(basename "$OUT0")"
[ "$(grep -o -- '-hv3pr0-' <<< "$B0" | wc -l)" = 1 ] \
  || { echo "run_v3gr.sh: run_v3.sh's run name $B0 does not hold '-hv3pr0-' exactly once" >&2; exit 2; }
OUT="$(dirname "$OUT0")/${B0/-hv3pr0-/-hv3gr-}"
read -r -a ARGS <<< "$ARGS0"
n_out=0; RENDER=""
for i in "${!ARGS[@]}"; do
  case "${ARGS[$i]}" in
    --out) [ "${ARGS[$((i + 1))]}" = "$OUT0" ] || { echo "run_v3gr.sh: --out in ARGS is not run_v3.sh's OUT" >&2; exit 2; }
           ARGS[$((i + 1))]="$OUT"; n_out=$((n_out + 1)) ;;
    --render) RENDER="${ARGS[$((i + 1))]}" ;;
  esac
done
[ "$n_out" = 1 ] && [ -n "$RENDER" ] || { echo "run_v3gr.sh: run_v3.sh's ARGS hold --out $n_out times / no --render" >&2; exit 2; }
TRAINER=rl/grpo_grds.py
echo "[hv3_rl] $(date -Is) H-GRDS (grounded-only answer credit, run_v3.sh's configuration of $B0) seed=$SEED steps=$STEPS out=$OUT"
[ "${DRY_RUN:-0}" = 1 ] && { echo "OUT=$OUT"; echo "ARGS=${ARGS[*]}"; echo "SFT_MODEL=$SFT"; echo "TRAINER=$TRAINER"; exit 0; }
# run_v3.sh's launch tail
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; json.dump(v2.HarnessConfig.from_env().as_dict(), open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json"
export SFT_MODEL="$SFT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python "$TRAINER" "${ARGS[@]}"
