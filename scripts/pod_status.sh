#!/bin/bash
# pod_status.sh -- read the resident workers' status files (scripts/pod_resident.sh) instead of sampling pods with srun.
# Monitoring a resident pod needs NO job step: the resident writes logs/pods/resident/J/status every ~30 s and touches
# heartbeat every 10 s; a status or heartbeat older than POD_STATUS_STALE s (120) means the resident is dead.
#
#   pod_status.sh fresh J         exit 0 iff J's resident is alive (status and heartbeat fresh, not ended)
#   pod_status.sh sample J        exactly the 4 lines fleet_monitor.sh's srun sampler printed, plus a 5th (what runs):
#                                   "<max util>, <mem MiB>;..." / "<disk>%" / <episodes> / "<cur> <max> <pinned> <oomkills>" / itxt
#                                 exit 1 (and no output) when the status is missing or stale -- a drop-in for
#                                   srun --jobid=J ... > .fm_s_J   in fleet_monitor.sh
#   pod_status.sh line J          one fleet-status line for J (STALE flagged)
#   pod_status.sh table           a line for every pod with a resident status written in the last day
#   pod_status.sh check           alert lines (fleet_monitor's style), for the caller to dedupe and append:
#                                   RESIDENT-STALE J  status/heartbeat older than the limit while squeue says RUNNING
#                                   RESIDENT-ERROR J  the resident cannot use the queue root from its node (no workers)
#                                   GPU-IDLE J gpuN   max utilisation over the last ~10 min <= 2%
#                                   WORKER-DEAD J W   a worker process is gone
#                                   GPU-HELD J W      a worker has waited > 30 min for its GPU with work in its lanes
#                                   IDLE J W          a worker idle > 10 min
#                                   MEM-HIGH / MEM-OOM / DISK as fleet_monitor
#   pod_status.sh sweep [--apply] claims ($POD_PQ/claims/J_<item>) of pods that are not RUNNING and whose resident is
#                                 stale: print them; with --apply put each item back in its lane (the resident does this
#                                 itself on SIGTERM and at start; this covers a pod killed without either).
# Env: POD_RESIDENT_DIR (default <repo>/logs/pods/resident), POD_PQ (default <scratchpad>/pq), POD_STATUS_STALE (120).
# Slurm calls: only `squeue` (check, sweep) -- an RPC, never a step.
set -uo pipefail
P="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RD="${POD_RESIDENT_DIR:-$P/logs/pods/resident}"
PQ="${POD_PQ:-/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad/pq}"
STALE="${POD_STATUS_STALE:-120}"
kv() { sed -n "s/^$1=//p" "$2" 2>/dev/null | head -1; }
age() { local f=$1; [ -e "$f" ] || { echo 999999; return; }; echo $(( $(date +%s) - $(stat -c %Y "$f") )); }
fresh() {
  local d="$RD/$1"
  [ -f "$d/status" ] && [ "$(age "$d/status")" -le "$STALE" ] && [ "$(age "$d/heartbeat")" -le "$STALE" ] || return 1
  [ "$(kv supervisor_alive "$d/status")" = 1 ] || return 1
  # written by the CURRENT supervisor (right after a restart the last status is still the previous incarnation's)
  [ "$(kv resident_pid "$d/status")" = "$(cat "$d/supervisor.pid" 2>/dev/null)" ] || return 1
  ! grep -qxE 'state=(terminating|ended|exited)' "$d/state" 2>/dev/null
}
running_jobs() { timeout 60 squeue -u "$USER" -h -t R -o %i 2>/dev/null; }
line() {
  local J=$1 d="$RD/$1" f; f="$d/status"
  [ -f "$f" ] || { echo "$J  no resident status"; return; }
  local fl=""; fresh "$J" || fl="  STALE(status $(age "$f")s, heartbeat $(age "$d/heartbeat")s, state $(kv state "$d/state"))"
  read -r mc mm mp mo <<< "$(kv mem "$f")"
  printf "%s  gpus[%s]  10m[%s]  eps %-3s disk %s%%  mem %s/%sG (pinned %sG)  [%s]  %s%s\n" "$J" "$(kv gpus "$f")" \
    "$(kv gpus_max10m "$f")" "$(kv episodes "$f")" "$(kv disk_tmp "$f")" "$mc" "$mm" "$mp" "$(kv kinds "$f")" \
    "$(kv itxt "$f" | cut -c1-90)" "$fl"
}
case "${1:-}" in
  fresh) fresh "${2:?J}" ;;
  sample)
    J=${2:?J}; fresh "$J" || exit 1; f="$RD/$J/status"
    kv gpus "$f"; echo "$(kv disk_tmp "$f")%"; kv episodes "$f"; kv mem "$f"; kv itxt "$f" ;;
  line) line "${2:?J}" ;;
  table)
    for d in "$RD"/*/; do J=$(basename "$d"); [ -f "$d/status" ] && [ "$(age "$d/status")" -lt 86400 ] && line "$J"; done ;;
  check)
    now=$(date +%s); R=" $(running_jobs | tr '\n' ' ') "
    for d in "$RD"/*/; do
      J=$(basename "$d"); f="$d/status"; [ -f "$f" ] || continue
      case "$R" in *" $J "*) ;; *) continue ;; esac                      # only pods that are RUNNING
      er=$(kv error "$f"); [ -n "$er" ] && echo "RESIDENT-ERROR $J $er -- it runs no workers"
      if ! fresh "$J"; then
        echo "RESIDENT-STALE $J status $(age "$f")s / heartbeat $(age "$d/heartbeat")s old, state $(kv state "$d/state") -- the resident is dead or hung (the pod runs; nothing claims or monitors on it)"
        continue
      fi
      gi=0; for u in $(kv gpus_max10m "$f" | tr ';' ' '); do
        [ "$u" -le 2 ] 2>/dev/null && [ "$(wc -l < "$d/.gpu_hist" 2>/dev/null || echo 0)" -ge 20 ] && echo "GPU-IDLE $J gpu$gi <=2% for >=10 min: $(kv itxt "$f" | cut -c1-80)"
        gi=$((gi + 1)); done
      while IFS= read -r w; do
        n=${w%%=*}; n=${n#worker_}; IFS='|' read -r st it ln sn wy <<< "${w#*=}"
        s=$(date -d "$sn" +%s 2>/dev/null || echo "$now")
        case "$st" in
          dead*) echo "WORKER-DEAD $J $n its process is gone (was $st, item ${it})" ;;
          waiting-gpu) q=0; for k in $(kv kinds "$f"); do q=$((q + $(ls "$PQ/$k" 2>/dev/null | grep -vc '^\.'))); done
                       [ $((now - s)) -gt 1800 ] && [ "$q" -gt 0 ] && echo "GPU-HELD $J $n waiting $(( (now - s) / 60 )) min for its GPU with $q item(s) queued: $wy" ;;
          idle) [ $((now - s)) -gt 600 ] && echo "IDLE $J $n worker idle since $sn" ;;
        esac
      done < <(grep '^worker_' "$f")
      read -r mc mm mp mo <<< "$(kv mem "$f")"
      [ "${mm:-0}" -gt 0 ] 2>/dev/null && [ $(( mp * 100 / mm )) -ge 70 ] && echo "MEM-HIGH $J pinned ${mp}G of ${mm}G job limit (cache ${mc}G)"
      po=$(cat "$d/.oom_seen" 2>/dev/null || echo "${mo:-0}"); [ "${mo:-0}" -gt "${po:-0}" ] 2>/dev/null && echo "MEM-OOM $J oom_kill $po -> $mo"
      echo "${mo:-0}" > "$d/.oom_seen" 2>/dev/null
      dk=$(kv disk_tmp "$f"); [ "$dk" -ge 92 ] 2>/dev/null && echo "DISK $J node /tmp at ${dk}%"
    done ;;
  sweep)
    apply=0; [ "${2:-}" = --apply ] && apply=1
    R=$(running_jobs); [ -n "$R" ] || { echo "sweep: squeue gave no answer (or nothing runs) -- not judging any pod"; exit 1; }
    R=" $(echo "$R" | tr '\n' ' ') "
    for c in "$PQ"/claims/*; do
      [ -e "$c" ] || continue; it=$(basename "$c"); J=${it%%_*}; ln=$(kv lane "$c"); base=${it#"${J}"_}
      case "$R" in *" $J "*) continue ;; esac                         # the pod runs: its resident owns the claim
      [ "$(age "$RD/$J/status")" -le "$STALE" ] && continue             # its resident wrote very recently: wait
      if [ ! -f "$PQ/started/$it" ]; then [ $apply = 1 ] && rm -f "$c"; echo "stale claim $it (no started/ file)"; continue; fi
      if [ $apply = 1 ]; then
        mv "$PQ/started/$it" "$PQ/$ln/$base" && rm -f "$c" && echo "$(date -Is) returned $base to $ln (pod $J not running, resident stale)"
      else echo "would return $base to $ln (pod $J not running, resident stale)"; fi
    done ;;
  *) sed -n '2,26p' "$0"; exit 2 ;;
esac
