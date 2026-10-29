#!/bin/bash
#SBATCH --job-name=cwr_anchor
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=/project/pi_hongyu_umass_edu/hieutran/deep/deeprs/logs/pods/resident/anchor_%j.out
#
# ANCHOR for ALREADY-RUNNING pods (2026-09-30). A pod started with crossworld's cw_gpupod14.sbatch runs only a sleep
# loop in its batch step, so its resident worker (scripts/pod_resident.sh) must be ONE job step. A step needs an srun
# client somewhere; this tiny CPU job is that somewhere -- never a login node, and not the orchestrator (which restarts,
# hands over to a standby, and ran the clients that died on 09-29). It does nothing else.
#
#   sbatch scripts/pod_resident_anchor.sbatch CONF        CONF lines as orch_pods.conf:  J|kinds|n_gpus|port_off
#
# Every 60 s, for each pod in CONF that is RUNNING, has no live resident (scripts/pod_status.sh fresh) and no srun of
# ours alive for it: start
#   srun --jobid=J --overlap -n1 -c<NumCPUs> --input=none --output=<resident dir>/step_J_%s.out --job-name=podresident_J \
#        bash scripts/cw_gpupod_resident.sbatch KINDS PORT_OFF
# The step's main process is the LAUNCHER's restart loop, not the supervisor: when a step's main process exits, Slurm
# kills everything left in that step, so a supervisor that crashed or was OOM-killed would take every run on the pod
# with it. Under the loop it is simply restarted a minute later and adopts its workers and their items. (NGPU is
# counted on the pod: orch_pods.conf's n is 1 for a whole-pod `double` line, which is not the pod's GPU count.)
# At most 3 starts per pod per 6 h (a pod that refuses steps must not be hammered with more). A pod whose resident
# was told to stop (logs/pods/resident/J/control/exit) is left alone. Remove a line from CONF to stop tending a pod.
# If THIS job ends, its srun clients forward SIGTERM to the residents, which put their items back in their lanes.
MYJOB=${SLURM_JOB_ID:-none}
for v in $(env | grep -oE "^SLURM[A-Z_]*"); do unset "$v"; done      # so srun --jobid=<pod> works from inside this job
R=${POD_REPO:-/project/pi_hongyu_umass_edu/hieutran/deep/deeprs}   # POD_REPO: tests only
CONF=${1:?CONF (J|kinds|n|off lines)}
RD=${POD_RESIDENT_DIR:-$R/logs/pods/resident}
cd "$R" || exit 1
declare -A PID STARTS REFUSED STARTAT
term=0; trap 'term=1' TERM INT
echo "[anchor] $(date -Is) job $MYJOB on $(hostname), conf $CONF"
while [ "$term" = 0 ]; do
  RUN=" $(timeout 60 squeue -u "$USER" -h -t R -o %i 2>/dev/null | tr '\n' ' ') "
  while IFS='|' read -r J kinds n off; do
    [ -z "$J" ] && continue; case "$J" in \#*) continue ;; esac
    case "$RUN" in *" $J "*) ;; *) continue ;; esac
    [ -e "$RD/$J/control/exit" ] && continue
    [ -n "${PID[$J]:-}" ] && kill -0 "${PID[$J]}" 2>/dev/null && continue
    bash scripts/pod_status.sh fresh "$J" && continue      # via bash: no dependence on the file mode
    # 2026-09-30 15:55: a gpu-preempt pod inside its preemption GraceTime is still RUNNING but its /tmp is gone and it
    # ends within minutes -- a resident started there burns every item it claims (65016268 / 65016465). Skip it; and a
    # REQUEUED pod (new StartTime) starts a fresh 3-per-6-h budget, since its earlier starts were a previous life.
    jinfo=$(timeout 60 scontrol show job "$J" 2>/dev/null)
    grep -q "PreemptTime=[0-9]" <<< "$jinfo" && continue
    st=$(grep -o 'StartTime=[^ ]*' <<< "$jinfo" | head -1)
    if [ -n "$st" ] && [ "${STARTAT[$J]:-}" != "$st" ]; then STARTAT[$J]=$st; STARTS[$J]=""; REFUSED[$J]=""; fi
    now=$(date +%s); recent=""; for t in ${STARTS[$J]:-}; do [ $((now - t)) -lt 21600 ] && recent="$recent $t"; done
    STARTS[$J]=$recent
    if [ "$(echo $recent | wc -w)" -ge 3 ]; then
      [ -z "${REFUSED[$J]:-}" ] && echo "[anchor] $(date -Is) $J: 3 starts in 6 h -- not starting again (check the pod)"
      REFUSED[$J]=1; continue
    fi
    REFUSED[$J]=""
    ncpu=$(grep -o 'NumCPUs=[0-9]*' <<< "$jinfo" | cut -d= -f2)
    [[ "$ncpu" =~ ^[0-9]+$ ]] || continue
    mkdir -p "$RD/$J"
    srun --jobid="$J" --overlap -n1 -c"$ncpu" --input=none --output="$RD/$J/step_${J}_%s.out" --job-name="podresident_$J" \
         bash "$R/scripts/cw_gpupod_resident.sbatch" "$kinds" "${off:-auto}" &
    PID[$J]=$!; STARTS[$J]="$recent $now"
    echo "[anchor] $(date -Is) $J: resident step started (srun pid ${PID[$J]}, kinds [$kinds], n $n, off $off)"
  done < "$CONF"
  sleep "${ANCHOR_EVERY:-60}" & wait $!
done
echo "[anchor] $(date -Is) SIGTERM: passing it to the resident steps"
for J in "${!PID[@]}"; do kill -TERM "${PID[$J]}" 2>/dev/null; done
wait
