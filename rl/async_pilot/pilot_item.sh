L_="$(dirname "$0")/../_lib.sh"
[ -f "$L_" ] || L_=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad/pod_queue/_lib.sh
source "$L_"
# ASYNC RL PILOT (docs/RL_PLAN.md s.10, docs/ASYNC_PILOT.md): condition R1 ported to AReaL v2.0.0.
# Same pool (rlmix6), reward (per-instance S, c=0.5, --require-code, GBP_ECHO_LAST=1, wall budget 600 s), Qwen3-8B,
# fresh LoRA r=128/alpha=256, LR 1e-5, 8 groups x 6 rollouts per update, 150 updates, R1 seed-1 instance stream.
# Asynchronous: vLLM (with runtime LoRA) on GPU 0, FSDP trainer on GPU 1; groups start at most STALENESS (=2) policy
# versions ahead of the update that consumes them; decoupled PPO objective (AReaL).
# NEEDS THE WHOLE DOUBLE POD: both GPUs must be free (the item waits for both, and fails rather than start on a busy
# card). Checkpoints: $OUT/step{25,50,...,150} in PEFT layout -> experiments/rl_plan/eval.sh $OUT PORT 50 as for R1.
# Logs: $OUT/history.jsonl (per update: time, staleness, live episodes, LoRA-B norm), $OUT/rollouts.jsonl (per
# episode), logs/pods/<step>.log (AReaL + vLLM). Read with: venv/bin/python rl/async_pilot/summarize.py $OUT
# REGISTERED (RL_PLAN s.10): parity at step 50 (TRAIN and HELD-OUT means within 5 points of the R1 seeds' range);
# speed >= 3x a capped R1 seed's updates/hour; staleness: invalid if most trained samples are > 2 versions stale.
SEED=${PILOT_SEED:-1}; STALE=${PILOT_STALENESS:-2}; STEPS=${PILOT_STEPS:-150}
OUT=$D/ckpt/rl-async-rlmix6-base-echo-rc-wb600-lr1e-5-st$STALE-s$SEED
N=asyncR1st${STALE}s${SEED}_$J
[ "$(scontrol show job $J | grep -o 'gres/gpu=[0-9]*' | head -1 | cut -d= -f2)" -ge 2 ] 2>/dev/null \
  || { echo "$(date -Is) [async_pilot] job $J is not a double-GPU pod; this item needs two GPUs"; exit 1; }
boot                                                  # Qwen3-8B (pinned revision) -> /tmp/deeprs_models/Qwen3-8B
for g in 0 1; do                                      # vLLM takes 0.85 x 80 GB on one card, the trainer ~55-70 GB
  G=$g GPU_NEED_MIB=70000 GPU_WAIT_TRIES=${PILOT_GPU_WAIT_TRIES:-144} gpu_wait \
    || { echo "$(date -Is) [async_pilot] GPU $g never freed; not starting"; exit 1; }
done
G=0
# AReaL's fileroot (per-version LoRA copies for vLLM, logs, kernel caches) on the pod-private /tmp when it has room
# (the node's /tmp is shared with other jobs -- memory: node-tmp-disk-full), else on scratch.
TMPFREE=$(timeout 60 scripts/pod_quick.sh $J df -Pk /tmp 2>/dev/null < /dev/null | awk 'NR==2{print int($4/1048576)}')
AR=/tmp/deeprs_areal_s$SEED; [ "${TMPFREE:-0}" -ge 30 ] || AR=$D/tmp/areal_${J}_s$SEED
echo "$(date -Is) [async_pilot] start $N (out $OUT, areal root $AR, /tmp free ${TMPFREE:-?}G)"
scripts/pod_exec.sh $J $C $N -- env GPUS=1,0 STALENESS=$STALE AREAL_ROOT=$AR OUT=$OUT \
  rl/async_pilot/launch.sh $SEED $STEPS >/dev/null
# Watchdog instead of a bare wait_marker: an engine that cannot generate writes $OUT/FATAL and exits, but a hung
# controller would hold the pod forever. No new update or episode for STALL_MIN minutes -> cancel our step.
STALL_MIN=${PILOT_STALL_MIN:-150}
until [ -e logs/pods/$N.done ] || [ -e logs/pods/$N.failed ]; do
  sleep 300
  if [ -e $OUT/FATAL ]; then echo "$(date -Is) [async_pilot] FATAL: $(cat $OUT/FATAL)"; scripts/pod_cancel.sh $J $N; break; fi
  last=$(stat -c %Y $OUT/history.jsonl $OUT/rollouts.jsonl logs/pods/$N.log 2>/dev/null | sort -n | tail -1)
  if [ -n "$last" ] && [ $(( $(date +%s) - last )) -gt $(( STALL_MIN * 60 )) ]; then
    echo "$(date -Is) [async_pilot] no progress for ${STALL_MIN} min -- cancelling $N"; scripts/pod_cancel.sh $J $N
    echo 1 > logs/pods/$N.failed; break
  fi
done
echo "$(date -Is) [async_pilot] $N finished: $(ls logs/pods/$N.done logs/pods/$N.failed 2>/dev/null | xargs -n1 basename)"
venv/bin/python rl/async_pilot/summarize.py $OUT 2>&1 | tail -25
[ -e logs/pods/$N.done ] && [ -e $OUT/DONE ]
