#!/bin/bash
# H-V5RL RUNNER -- RL INSIDE HARNESS v5 (docs/RL_PLAN.md 2026-09-29 14:15 UTC, PHASE 2, registered before the build):
# experiments/harness_v3/run_v3.sh's configuration, refusals and naming -- run_v3.sh itself is NOT edited (running and queued
# RL items execute it) -- with the trainer rl/grpo_v5.py: training rollouts in harness v5 (experiments/harness_v3/
# program_first.py: submit solve(data_dir, task) [amended 15:55], run on the target + 3 layout-matched train siblings, <= 2 revisions; answer =
# solve(target) of the last valid submission), reward = r_answer + 0.3 x portability, no process bonus, DAPO-style dynamic
# sampling (rl/grpo_ds.py) on that reward.
#   experiments/harness_v3/run_v3v5.sh SEED PORT [STEPS]
#   env: exactly run_v3.sh's (CZ_FAM, EXTRA_ARGS, MAXLEN, ON, LR, DRY_RUN, MODEL_TAG + SFT_MODEL); EXTRA_ARGS must hold
#   "--dyn-sample N"; PROC_LAMBDA / PROC_SHAM must be unset or 0 / off (there is no process bonus); WAITFOR is refused.
# HOW (run_v3gr.sh's): run_v3.sh is asked for its DRY_RUN with PROC_LAMBDA=0 PROC_SHAM=off -- so every check, the run name and
# the trainer arguments are its own -- and must answer TRAINER=rl/grpo_ds.py with a run dir ...-hv3pr0-...-ds<N>-...; the run
# dir is that name with "-hv3pr0-" -> "-hv3v5port0.3-" (e.g. rl-czero-causal_rdd-base-echo-sysscience-rc-hv3v5port0.3-ds12-
# lr1e-5-s1: never an existing run's name, and no existing run's name changes), the trainer rl/grpo_v5.py, the arguments
# run_v3.sh's with only --out replaced. Then run_v3.sh's launch tail, line for line, with ONE difference: harness_v2.json is
# written by an interpreter carrying the v5 hook (experiments/harness_v3/v5_site, GBP_V5=1), so it records program_first_v5 --
# the harness the policy is trained in (screen_any.sh compares it with the harness a checkpoint is scored in: equal under
# screen_v3v5.sh, a printed difference under screen_v3.sh -- the registered v3 read of a v5-trained policy).
# HARNESS v5.1 (docs/RL_PLAN.md 2026-09-29 21:10 UTC, PHASE 2'), FLAG-GATED -- OFF unless exactly V51=1:
#   V51=1  the trainer is rl/grpo_v51.py (training rollouts in v5.1 = v5 + (a) LOOK FIRST; reward r_answer + 0.3 x LABELLED
#          portability, the siblings graded by the family's own grader behind the sealed-family guard), the run dir is
#          run_v3.sh's name with "-hv3pr0-" -> "-hv3v51lp0.3-" (e.g. rl-czero-causal_rdd-base-echo-sysscience-rc-hv3v51lp0.3-
#          ds12-lr3e-6-s1 with LR=3e-6), and harness_v2.json is written with GBP_V51=1 too, so it records program_first_v51.
#          Without V51=1 everything below is v5 exactly.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; PORT="${2:?PORT}"; STEPS="${3:-100}"
case "${PROC_LAMBDA:-0}" in 0|0.0) ;; *) echo "run_v3v5.sh: the v5 reward has no process bonus -- PROC_LAMBDA must be unset or 0 (got '$PROC_LAMBDA')" >&2; exit 2 ;; esac
case "${PROC_SHAM:-off}" in off) ;; *) echo "run_v3v5.sh: the v5 reward has no process arm -- PROC_SHAM must be unset or off (got '$PROC_SHAM')" >&2; exit 2 ;; esac
[ -z "${WAITFOR:-}" ] || { echo "run_v3v5.sh: WAITFOR is not supported here (launch the item when the GPU is yours)" >&2; exit 2; }
case "${V51:-}" in ""|1) ;; *) echo "run_v3v5.sh: V51 must be 1 or unset (got '$V51')" >&2; exit 2 ;; esac
V51ON="${V51:-}"; TAG=-hv3v5port0.3-; TRAINER=rl/grpo_v5.py; HOOKENV=(GBP_V5=1); HREC=program_first_v5; WHAT="H-V5RL (RL inside harness v5, reward r_answer + 0.3 x portability"
if [ "$V51ON" = 1 ]; then   # v5.1 (Phase 2')
  TAG=-hv3v51lp0.3-; TRAINER=rl/grpo_v51.py; HOOKENV=(GBP_V5=1 GBP_V51=1); HREC=program_first_v51
  WHAT="H-V51RL (RL inside harness v5.1 = v5 + look first, reward r_answer + 0.3 x labelled portability"
  [ -f "$P/rl/grpo_v51.py" ] || { echo "run_v3v5.sh: V51=1 but rl/grpo_v51.py is missing" >&2; exit 2; }
fi
[[ " ${EXTRA_ARGS:-} " == *" --dyn-sample "* ]] \
  || { echo "run_v3v5.sh: Phase 2 is registered with dynamic sampling -- EXTRA_ARGS needs '--dyn-sample N'" >&2; exit 2; }
[ -f "$P/experiments/harness_v3/program_first.py" ] && [ -f "$P/experiments/harness_v3/v5_site/sitecustomize.py" ] && [ -f "$P/rl/grpo_v5.py" ] \
  || { echo "run_v3v5.sh: missing experiments/harness_v3/program_first.py, v5_site/sitecustomize.py or rl/grpo_v5.py" >&2; exit 2; }
DRY="$(env -u WAITFOR DRY_RUN=1 PROC_LAMBDA=0 PROC_SHAM=off experiments/harness_v3/run_v3.sh "$SEED" "$PORT" "$STEPS")" \
  || { echo "run_v3v5.sh: run_v3.sh refused this configuration (above)" >&2; exit 2; }
OUT0="$(sed -n 's/^OUT=//p' <<< "$DRY")"; ARGS0="$(sed -n 's/^ARGS=//p' <<< "$DRY")"
SFT="$(sed -n 's/^SFT_MODEL=//p' <<< "$DRY")"; TR0="$(sed -n 's/^TRAINER=//p' <<< "$DRY")"
[ "$TR0" = rl/grpo_ds.py ] && [ -n "$OUT0" ] && [ -n "$ARGS0" ] && [ -n "$SFT" ] \
  || { echo "run_v3v5.sh: unexpected run_v3.sh DRY_RUN (TRAINER '$TR0', OUT '$OUT0')" >&2; exit 2; }
B0="$(basename "$OUT0")"
[ "$(grep -o -- '-hv3pr0-' <<< "$B0" | wc -l)" = 1 ] \
  || { echo "run_v3v5.sh: run_v3.sh's run name $B0 does not hold '-hv3pr0-' exactly once" >&2; exit 2; }
OUT="$(dirname "$OUT0")/${B0/-hv3pr0-/$TAG}"
read -r -a ARGS <<< "$ARGS0"
n_out=0; RENDER=""
for i in "${!ARGS[@]}"; do
  case "${ARGS[$i]}" in
    --out) [ "${ARGS[$((i + 1))]}" = "$OUT0" ] || { echo "run_v3v5.sh: --out in ARGS is not run_v3.sh's OUT" >&2; exit 2; }
           ARGS[$((i + 1))]="$OUT"; n_out=$((n_out + 1)) ;;
    --render) RENDER="${ARGS[$((i + 1))]}" ;;
  esac
done
[ "$n_out" = 1 ] && [ -n "$RENDER" ] || { echo "run_v3v5.sh: run_v3.sh's ARGS hold --out $n_out times / no --render" >&2; exit 2; }
echo "[hv3_rl] $(date -Is) $WHAT, run_v3.sh's configuration of $B0) seed=$SEED steps=$STEPS out=$OUT"
[ "${DRY_RUN:-0}" = 1 ] && { echo "OUT=$OUT"; echo "ARGS=${ARGS[*]}"; echo "SFT_MODEL=$SFT"; echo "TRAINER=$TRAINER"; exit 0; }
# run_v3.sh's launch tail
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO HV2_FROZEN HV2_TRAIN_LIBRARY
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1
mkdir -p "$OUT"
echo "$RENDER" > "$OUT/render.txt"
echo "$GBP_SYSTEM" > "$OUT/system.txt"
# (the one difference: the v5 hook is on this interpreter's PYTHONPATH, so harness_v2.json records program_first_v5)
env -u GBP_V51 PYTHONPATH="$P/experiments/harness_v3/v5_site${PYTHONPATH:+:$PYTHONPATH}" "${HOOKENV[@]}" venv/bin/python -c 'import json,sys; sys.path.insert(0,"harness"); import v2; d = v2.HarnessConfig.from_env().as_dict(); assert sys.argv[2] in d, "the v5 hook did not install"; json.dump(d, open(sys.argv[1],"w"), indent=1)' "$OUT/harness_v2.json" "$HREC" \
  || { echo "run_v3v5.sh: could not write a v5 harness_v2.json" >&2; exit 2; }
export SFT_MODEL="$SFT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv_vllm/bin/python "$TRAINER" "${ARGS[@]}"
