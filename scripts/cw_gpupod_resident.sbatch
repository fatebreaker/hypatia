#!/bin/bash
#SBATCH --job-name=cwr_pod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/work/pi_hongyu_umass_edu/hieutran_umass_edu/deeprs_pods/logs/resident/pod_%j.out
#
# A held GPU pod whose BATCH STEP runs the pod-resident worker (scripts/pod_resident.sh) -- 2026-09-30.
# Same allocation role as crossworld's cw_gpupod14.sbatch (a 14-day slot, --requeue, TERM 120 s before the end), but
# the work runs INSIDE this batch step: the resident claims queue items and runs them, their servers, trainers, GPU
# checks and monitoring samples as local processes. The pod needs NO job step at all, and no srun client anywhere --
# nothing outside the pod can take its runs down with it.
#
#   sbatch --job-name=cwr_a100_80g_2 --partition=gpu --gres=gpu:2 --cpus-per-task=28 --mem=250G --constraint=a100-80g \
#          --chdir=/project/pi_hongyu_umass_edu/hieutran/deep/deeprs scripts/cw_gpupod_resident.sbatch [KINDS [PORT_OFF]]
#   KINDS: "auto" (default: 2 GPUs -> rl per GPU, H200 -> "h200 rl", one 80 GB -> "single l40s", L40S -> l40s),
#          or e.g. rl | "single l40s" | double (one worker for the whole pod) | "-" (no queue work). PORT_OFF: auto.
#
# JOB NAME: use cwr_<tier> (NOT cw_<tier>). pod_watch.sh, keepalive_guard.sh and fleet_monitor.sh match
# ^cw_(a100|h100|h200|evl40s) and would otherwise arm this pod the old way (keepalive steps, a podagent step, old
# workers claiming from pod_queue). The resident does all of that itself; monitor it with scripts/pod_status.sh.
# The --output directory must exist before submitting (mkdir -p logs/pods/resident).
#
# The loop below is the allocation's life: it restarts the resident if it ever exits (60 s later; the script is
# re-read from /project each time, so a fix is picked up), and on SIGTERM (preemption, time limit via --signal, scancel)
# passes it on and gives the resident up to 110 s to put its running items back in their lanes before releasing.
# `touch logs/pods/resident/<J>/control/exit` stops the resident without releasing the pod (it is not restarted while
# the file exists; remove it to restart within a minute).
set -uo pipefail
R=${POD_REPO:-/work/pi_hongyu_umass_edu/hieutran_umass_edu/deeprs}   # 10-06: the /work live tree (was /project); POD_REPO overrides
J=${SLURM_JOB_ID:?}
KINDS=${1:-auto}; OFF=${2:-auto}
echo "[podres] host=$(hostname) job=$J gpu=${CUDA_VISIBLE_DEVICES:-?} part=${SLURM_JOB_PARTITION:-?}" \
     "cpus=${SLURM_CPUS_ON_NODE:-?} kinds=[$KINDS] start $(date -u '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[podres] gpu: /'
term=0; rp=""
trap 'term=1; echo "[podres] signalled $(date -u "+%F %T"); passing it to the resident"; [ -n "$rp" ] && kill -TERM "$rp" 2>/dev/null' TERM INT
while [ "$term" = 0 ]; do
  RS=${POD_RESIDENT_DIR:-$R/logs/pods/resident}/$J
  if [ -e "$RS/control/exit" ]; then sleep "${POD_LAUNCHER_RESTART:-60}" & wait $!; continue; fi
  # 2026-10-09 (user: 'use pod resident agent instead of srun'): started as an srun STEP (an anchor) in a pod whose
  # resident already runs in the pod's own BATCH step (a cwr_ pod), this launcher is redundant -- end the step at once
  # instead of retrying every minute ('already running', rc 3). The batch launcher (no SLURM_STEP_ID) never exits here.
  # A cwr_ job is one: its batch step runs this script (decided by the job NAME, so an anchor step cannot win the race
  # at pod start and leave the resident inside srun); otherwise exit only if the live resident already runs in step_batch.
  if [ -n "${SLURM_STEP_ID:-}" ]; then
    op=$(cat "$RS/supervisor.pid" 2>/dev/null); jn=$(timeout 30 squeue -h -j "$J" -o %j 2>/dev/null)
    if [[ "$jn" == cwr_* ]] || { [ -n "$op" ] && grep -qs "/step_batch/" "/proc/$op/cgroup"; }; then
      echo "[podres] $(date -u '+%F %T') job $J ($jn): the resident belongs in this pod's batch step; srun step $SLURM_STEP_ID is redundant -- exiting"
      exit 0
    fi
  fi
  bash "$R/scripts/pod_resident.sh" "$J" "$KINDS" auto "$OFF" &
  rp=$!
  wait "$rp"; rc=$?
  while kill -0 "$rp" 2>/dev/null; do wait "$rp"; rc=$?; done      # a trapped signal cuts wait short
  rp=""
  [ "$term" = 1 ] && break
  echo "[podres] resident exited (rc $rc) $(date -u '+%F %T'); restarting in ${POD_LAUNCHER_RESTART:-60} s"
  sleep "${POD_LAUNCHER_RESTART:-60}" & wait $!
done
# SIGTERM: give the resident time to return items (it moves them first, then stops their processes)
for _ in $(seq 110); do [ -n "$rp" ] && kill -0 "$rp" 2>/dev/null || break; sleep 1; done
echo "[podres] releasing $(date -u '+%F %T')"
exit 0
