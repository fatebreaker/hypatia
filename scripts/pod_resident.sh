#!/bin/bash
# POD-RESIDENT WORKER (2026-09-30). Runs INSIDE pod J and does there, as local processes, what the orchestrator used to
# do from outside with one Slurm step per action.
#
# WHY. Every fleet-monitor sample, liveness check, GPU wait, server poll, launch and cancel was an `srun --jobid=J`,
# i.e. a new job step: an H200 pod reached ~98 steps in 2.5 h, stopped accepting steps and hung its runs (twice). And
# every step's srun CLIENT lived on the worker host (the orchestrator's CPU node, earlier a login node), so when that
# host or its job died, every run on every pod died with it. Here the items, their servers and trainers, the GPU
# checks and the monitoring samples are children of one process inside the pod: ZERO steps when it runs in the pod's
# batch step (scripts/cw_gpupod_resident.sbatch), ONE when started on an already-running pod, and no client anywhere.
#
#   scripts/pod_resident.sh J [KINDS [NGPU [PORT_OFF [CPUS]]]]      the supervisor; must run inside job J
#     KINDS     queue lanes, e.g. "rl", "single l40s", "h200 rl", "double"; "auto" (default) decides as pod_watch.sh
#               did (2 GPUs -> rl, H200 -> "h200 rl", one 80 GB GPU -> "single l40s", a smaller card -> l40s); "-" = no
#               queue work (status, keepalive and the remote inbox only).
#     NGPU      the pod's GPUs ("auto" = count them). NGPU >= 2 and KINDS != double -> one worker per GPU with GPU=g,
#               CPUS=14 and PORT_OFF=off+25g, exactly as orchestrator.sbatch's ensure_workers; otherwise ONE worker for
#               the whole pod (GPU unset, CPUS unset), exactly as there.
#     PORT_OFF  as in orch_pods.conf; "auto" = the first of 0 50 100 ... whose ports nothing on the node listens on.
#
# WHAT IT DOES (state under logs/pods/resident/J/ -- $POD_RESIDENT_DIR/J):
#   * one worker per GPU (or one for the pod) claims items from the NEW queue root $POD_PQ (default <scratchpad>/pq,
#     lanes rl double single l40s h200) by atomic mv into $POD_PQ/started/J_<item> -- ONLY when its GPU(s) are actually
#     free (memory.used <= POD_FREE_USED_MIB, default 2000; the tiny keepalive is ~0.3 GB) -- and runs it locally:
#     `bash started/J_<item> J` with POD_LOCAL=J, so pod_exec/pod_quick/pod_cancel run as local processes, no srun.
#     A claim record $POD_PQ/claims/J_<item> names the lane, host, worker and pids, so the item can always go back.
#   * work.log: the old pod_worker.sh lines ("<iso> worker J: start|finished ... (exit N)|queue empty") + item output.
#   * status (every ~30 s, atomic): per-GPU utilisation (max and mean over ~10 reads, and the max over the last 5 and
#     10 minutes) and memory, the job cgroup's memory (current / limit / 'pinned' = anon+shmem+dirty+writeback / OOM
#     kills), /tmp use, live episodes, what every worker is doing, the local commands and keepalives. heartbeat is
#     touched every 10 s. scripts/pod_status.sh reads these -- a stale file = the resident is dead -> alert.
#   * a tiny keepalive per GPU (scripts/bin/gpu_keepalive_tiny; none on an H200 -- the keepalive rule) whenever none of
#     any origin runs on that GPU. POD_KEEPALIVE=0 turns it off, =1 forces it.
#   * a remote inbox, so the session / orchestrator can act on the pod with no step: scripts/pod_exec.sh, pod_quick.sh,
#     pod_cancel.sh with POD_VIA_RESIDENT=1 write inbox/NAME.req (+ the caller's environment), quick/ID.sh, cancel/NAME.
#   * control/: `drain` (no new claims; running items finish), `drain_g1` (one worker), `exit` (workers leave when idle,
#     then the supervisor; a launcher does not restart it while the file exists). Remove the file to undo.
#   * SELF-RELOAD: workers re-exec this file between items when it changed (and `bash -n` passes); the supervisor and
#     the status loop at their next pass. Same PIDs, same arguments -- a fix reaches every pod without killing an item.
#   * SIGTERM (Slurm: preemption, the pod's end, the step cancelled; or the launcher passing on --signal=B:TERM@120):
#     each worker FIRST puts its running item back at the front of its lane (so it survives Slurm's SIGKILL 30 s
#     later), then TERMs the item's process group and every process tagged with it, KILLs what is left after
#     POD_TERM_GRACE s (45). A requeued pod keeps its job id: at start the supervisor returns every claim of J whose
#     worker and item are both gone (pod_status.sh sweep does the same, from outside, for a pod that is not RUNNING).
#   * a supervisor or worker that dies is restarted (at most once a minute each); a restarted worker ADOPTS the item
#     its predecessor was running instead of starting another.
#
# Internal modes: pod_resident.sh --worker J W GPUS KINDS PORT_OFF CPUS | --status J
set -uo pipefail
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
P="${POD_ROOT:-$(cd "$(dirname "$SELF")/.." && pwd)}"   # 10-22: POD_ROOT = a shadow repo root (aarch64 GH200 pods: venv and
                                                          # venv_vllm there point to the ARM builds; every other entry links here)
cd "$P" || exit 1
S_DEFAULT=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
PQ="${POD_PQ:-$S_DEFAULT/pq}"
L="$P/logs/pods"
NVSMI="${POD_NVSMI:-nvidia-smi}"
KT="${POD_KT:-$P/scripts/bin/gpu_keepalive_tiny}"
LANES="rl double single l40s h200"
FREE_MIB="${POD_FREE_USED_MIB:-2000}"
IDLE_POLL="${POD_IDLE_POLL:-60}"; BUSY_POLL="${POD_BUSY_POLL:-30}"; GRACE="${POD_TERM_GRACE:-45}"
# shellcheck source=pod_local.sh
. "$P/scripts/pod_local.sh"
# a plain `cmd &` in a script starts cmd with SIGINT/SIGQUIT ignored; workers and items get every signal at its default
DSIG=(); env --default-signal true 2>/dev/null && DSIG=(env --default-signal)

ts() { date -Is; }
nap() { sleep "$1" & wait $! 2>/dev/null; }                 # a sleep a trapped signal can interrupt
mtime() { stat -c %Y "$SELF" 2>/dev/null || echo 0; }
kv() { sed -n "s/^$1=//p" "$2" 2>/dev/null | head -1; }      # kv KEY FILE
put() { local f=$1; shift; printf '%s\n' "$@" > "$f.tmp.$$" && mv -f "$f.tmp.$$" "$f"; }   # atomic key=value file
is_ours() {  # is_ours PID J MODE -- PID is a live pod_resident.sh process of job J in MODE (sup|--worker|--status)
  local p=$1 c; [ -n "$p" ] && pl_alive "$p" && pl_injob "$p" "$2" || return 1
  c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null); [[ "$c" == *pod_resident.sh* ]] || return 1
  case "$3" in sup) [[ "$c" != *" --worker "* && "$c" != *" --status "* ]] ;; *) [[ "$c" == *" $3 "* ]] ;; esac
}
item_alive() {  # item_alive PID ITEM -- the item's own bash is still running
  pl_alive "$1" && grep -qzF -- "started/$2" "/proc/$1/cmdline" 2>/dev/null
}
kt_alive() {  # kt_alive J G -- a keepalive (ours, any origin: this resident or a keepalive_guard step) runs on GPU G
  local p
  for p in $(pgrep -u "$(id -u)" -f "$(basename "$KT")" 2>/dev/null); do
    pl_injob "$p" "$1" && timeout 5 grep -qzFx "CUDA_VISIBLE_DEVICES=$2" "/proc/$p/environ" 2>/dev/null && return 0
  done
  return 1
}

# ======================================================================================================== worker
worker_main() {
  J=$1; W=$2; GPUS=$3; KINDS=$4; OFFW=$5; CPUSW=$6
  RS=$(pl_rs "$J"); WLOG="$RS/work.log"
  echo $$ > "$RS/workers/$W.pid"
  export POD_LOCAL="$J" POD_LOCAL_HOST; POD_LOCAL_HOST="$(hostname)"
  export POD_JOBINFO="$RS/jobinfo" PORT_OFF="$OFFW"
  [ "$W" != all ] && export GPU="$GPUS"
  [ -n "$CPUSW" ] && export CPUS="$CPUSW"
  local JS=$J; [ "${GPU:-0}" != 0 ] && JS=${J}g$GPU
  local ME; ME=$(mtime)
  WTERM=0; trap 'WTERM=1' TERM INT HUP
  local cur="" curitem="" since why item lane f k ipid rc left names logged_idle=0 logged_busy=""
  wstate() {  # wstate STATE [ITEM LANE WHY] -- written only when something changed ('since' = when the state began)
    local s=$1 it=${2:-} ln=${3:-} wy=${4:-}
    if [ "$s|$it" != "$cur|$curitem" ]; then since=$(ts); cur=$s; curitem=$it; fi
    put "$RS/workers/$W.state" "state=$s" "item=$it" "lane=$ln" "since=$since" "why=$wy" "gpus=$GPUS" "kinds=$KINDS" "pid=$$" "port_off=$OFFW"
  }
  kill_item() {  # kill_item ITEM IPID -- TERM the item's group + every process tagged with it; KILL what is left
    local it=$1 ip=$2 i ps
    [ -n "$ip" ] && kill -TERM -- "-$ip" 2>/dev/null
    ps=$(pl_tagged "POD_ITEM=$it" "$J"); [ -n "$ps" ] && kill -TERM $ps 2>/dev/null
    for ((i = 0; i < GRACE; i++)); do
      ps=$(pl_tagged "POD_ITEM=$it" "$J")
      [ -z "$ps" ] && ! { [ -n "$ip" ] && item_alive "$ip" "$it"; } && return 0
      sleep 1
    done
    [ -n "$ps" ] && kill -KILL $ps 2>/dev/null
    [ -n "$ip" ] && kill -KILL -- "-$ip" 2>/dev/null
    echo "$(ts) worker $J: KILLed what was left of $it after ${GRACE}s" >> "$WLOG"
  }
  on_term() {  # the pod is ending (or the resident is being stopped): the item goes back FIRST, then its processes
    local it=$1 ln=$2 ip=$3 base=${1#"${J}"_}
    if mv "$PQ/started/$it" "$PQ/$ln/$base" 2>/dev/null; then
      echo "$(ts) worker $J: SIGTERM (pod ending or resident stopped) while running $it; returned $base to $ln" >> "$WLOG"
    else
      echo "$(ts) worker $J: SIGTERM while running $it; could NOT return it to $ln (started/ file missing?)" >> "$WLOG"
    fi
    rm -f "$PQ/claims/$it" "$RS/workers/$W.item"
    kill_item "$it" "$ip"
    wstate ended; exit 0
  }
  run_wait() {  # run_wait ITEM LANE IPID ADOPTED -- wait for the item; SIGTERM returns it
    local it=$1 ln=$2 ip=$3 ad=$4 r=0
    if [ "$ad" = 0 ]; then
      while :; do wait "$ip"; r=$?; [ "$WTERM" = 1 ] && break; pl_alive "$ip" || break; done
    else  # adopted from a dead predecessor: not our child, so poll (its exit status is unknowable)
      while item_alive "$ip" "$it" && [ "$WTERM" = 0 ]; do nap 5; done; r="?"
    fi
    [ "$WTERM" = 1 ] && on_term "$it" "$ln" "$ip"
    echo "$(ts) worker $J: finished $it (exit $r)" >> "$WLOG"
    rm -f "$PQ/claims/$it" "$RS/workers/$W.item"
    left=$(pl_tagged "POD_ITEM=$it" "$J")
    if [ -n "$left" ]; then
      names=$(for p in $left; do tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n "s/^POD_CMD=$J://p"; done | sort -u | tr '\n' ' ')
      echo "$(ts) worker $J: $it left $(echo $left | wc -w) process(es) running (${names:-untagged}); they keep their GPU memory until cancelled" >> "$WLOG"
      [ "${POD_REAP_AFTER_ITEM:-0}" = 1 ] && kill_item "$it" ""
    fi
  }

  # a restarted worker adopts the item its predecessor was running (its claim names this worker and a live item pid)
  if read -r ipid item lane 2>/dev/null < "$RS/workers/$W.item" && [ -n "${item:-}" ] && item_alive "$ipid" "$item"; then
    echo "$(ts) worker $J: adopting $item (pid $ipid) from a previous worker $W" >> "$WLOG"
    wstate running "$item" "$lane"; run_wait "$item" "$lane" "$ipid" 1
  fi
  wstate idle

  while :; do
    [ "$WTERM" = 1 ] && { wstate ended; exit 0; }
    # SELF-RELOAD between items (same PID, same arguments)
    if [ "$(mtime)" != "$ME" ] && bash -n "$SELF" 2>/dev/null; then
      echo "$(ts) worker $J: pod_resident.sh changed; reloading worker $W" >> "$WLOG"
      exec bash "$SELF" --worker "$J" "$W" "$GPUS" "$KINDS" "$OFFW" "$CPUSW"
    fi
    if [ -e "$RS/control/exit" ]; then echo "$(ts) worker $J: exit requested (control/exit); worker $W leaves" >> "$WLOG"; wstate exited; exit 0; fi
    if [ -e "$RS/control/drain" ] || [ -e "$RS/control/drain_$W" ]; then wstate draining "" "" "control/drain"; nap "$IDLE_POLL"; continue; fi
    # claim ONLY when this worker's GPU(s) are free: an item never waits in gpu_wait holding a lane slot
    why=$(gpus_busy "$GPUS")
    if [ -n "$why" ]; then
      [ "$logged_busy" != "$why" ] && echo "$(ts) worker $J: not claiming on GPU(s) $GPUS: $why" >> "$WLOG"; logged_busy=$why
      wstate waiting-gpu "" "" "$why"; nap "$BUSY_POLL"; continue
    fi
    logged_busy=""
    # 2026-09-30 15:55: never claim on a pod that is being PREEMPTED (gpu-preempt GraceTime: the job stays RUNNING but its
    # private /tmp is gone and it ends within minutes -- 65016268 / 65016465 burned four items in 4 min this way), nor
    # when /tmp is not writable for any other reason (every item stages its model there).
    if timeout 30 scontrol show job "$J" 2>/dev/null | grep -q "PreemptTime=[0-9]"; then
      [ "$logged_busy" != preempt ] && echo "$(ts) worker $J: pod is being preempted; not claiming" >> "$WLOG"; logged_busy=preempt
      wstate draining "" "" "preempting"; nap "$BUSY_POLL"; continue
    fi
    if ! { [ -d /tmp ] && [ -w /tmp ]; }; then
      [ "$logged_busy" != notmp ] && echo "$(ts) worker $J: /tmp missing or not writable; not claiming" >> "$WLOG"; logged_busy=notmp
      wstate draining "" "" "no /tmp"; nap "$BUSY_POLL"; continue
    fi
    item=""; lane=""
    for k in $KINDS; do
      for f in $(ls "$PQ/$k" 2>/dev/null | sort); do
        case "$f" in .*|*.tmp) continue ;; esac
        if mv "$PQ/$k/$f" "$PQ/started/${J}_$f" 2>/dev/null; then item=${J}_$f; lane=$k; break 2; fi
      done
    done
    if [ -z "$item" ]; then
      [ "$logged_idle" = 0 ] && echo "$(ts) worker $J: queue empty; worker $W idle on GPU(s) $GPUS [$KINDS], ASSIGN WORK" >> "$WLOG"
      logged_idle=1; wstate idle; nap "$IDLE_POLL"; continue
    fi
    logged_idle=0
    put "$PQ/claims/$item" "lane=$lane" "host=$(hostname)" "worker=$W" "pid=$$" "since=$(ts)" "ipid="
    # the legacy server names of this worker, as pod_worker.sh stopped before every item (only if one runs locally)
    for n in serveQ_$JS serveM_$JS serveQ32_$JS; do
      [ -f "$L/local/$J/$n.pid" ] && "$P/scripts/pod_cancel.sh" "$J" "$n" >/dev/null 2>&1 < /dev/null
    done
    echo "$(ts) worker $J: start $item" >> "$WLOG"
    wstate running "$item" "$lane"
    POD_ITEM="$item" setsid "${DSIG[@]}" bash "$PQ/started/$item" "$J" >> "$WLOG" 2>&1 < /dev/null &
    ipid=$!
    echo "$ipid $item $lane" > "$RS/workers/$W.item"
    put "$PQ/claims/$item" "lane=$lane" "host=$(hostname)" "worker=$W" "pid=$$" "since=$(ts)" "ipid=$ipid"
    run_wait "$item" "$lane" "$ipid" 0
    wstate idle
  done
}

gpus_busy() {  # gpus_busy "0 1" -- empty when every listed GPU has <= FREE_MIB used; else the reason
  local out g u
  out=$(timeout 30 $NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo "nvidia-smi failed"; return; }
  for g in $1; do
    u=$(echo "$out" | awk -F', *' -v g="$g" '$1 == g {print $2+0}')
    if [ -z "$u" ]; then echo "GPU $g not visible"; return; fi
    [ "$u" -gt "$FREE_MIB" ] && { echo "GPU $g has ${u}MiB in use"; return; }
  done
}

# ======================================================================================================== status
status_main() {
  J=$1; RS=$(pl_rs "$J"); echo $$ > "$RS/workers/status.pid"
  trap 'exit 0' TERM INT HUP
  local ME; ME=$(mtime)
  local EVERY=${POD_STATUS_EVERY:-30} READS=${POD_STATUS_READS:-10} gap hist="$RS/.gpu_hist"
  gap=$(( EVERY / READS )); [ "$gap" -lt 1 ] && gap=1
  local cg; cg=$(cut -d: -f3 /proc/self/cgroup); cg="/sys/fs/cgroup${cg%%/step_*}"
  while :; do
    [ "$(mtime)" != "$ME" ] && bash -n "$SELF" 2>/dev/null && exec bash "$SELF" --status "$J"
    if [ -e "$RS/control/exit" ] && ! is_ours "$(cat "$RS/supervisor.pid" 2>/dev/null)" "$J" sup; then exit 0; fi
    # --- GPUs: READS reads over ~EVERY seconds; per GPU the max and mean utilisation and the last memory figures
    unset MX SM MU MT; declare -A MX SM MU MT; local n=0 i u m t r
    for ((r = 0; r < READS; r++)); do
      while IFS=', ' read -r i u m t; do
        [[ "$i" =~ ^[0-9]+$ ]] || continue
        u=${u%%[^0-9]*}; u=${u:-0}; [ "$u" -gt "${MX[$i]:-0}" ] && MX[$i]=$u; [ -z "${MX[$i]:-}" ] && MX[$i]=$u
        SM[$i]=$(( ${SM[$i]:-0} + u )); MU[$i]=$m; MT[$i]=$t
      done < <(timeout 20 $NVSMI --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
      n=$((n + 1)); [ $r -lt $((READS - 1)) ] && sleep "$gap"
    done
    local gl="" gm="" g
    for g in $(printf '%s\n' "${!MX[@]}" | sort -n); do
      gl="$gl${MX[$g]}, ${MU[$g]};"; gm="$gm$g:$(( SM[$g] / n )),${MU[$g]},${MT[$g]};"
    done
    # rolling maxima over the last 10 and 20 passes (~5 and ~10 min at the default 30 s)
    { tail -19 "$hist" 2>/dev/null; echo "$(date +%s) $gl"; } > "$hist.tmp" && mv -f "$hist.tmp" "$hist"
    local m5 m10
    m5=$(tail -10 "$hist" | cut -d' ' -f2- | awk -F';' '{for(i=1;i<NF;i++){split($i,a,","); if(a[1]+0>x[i]||!(i in x)) x[i]=a[1]+0; k=(NF-1>k?NF-1:k)}} END{for(i=1;i<=k;i++) printf "%d;", x[i]}')
    m10=$(cut -d' ' -f2- "$hist" | awk -F';' '{for(i=1;i<NF;i++){split($i,a,","); if(a[1]+0>x[i]||!(i in x)) x[i]=a[1]+0; k=(NF-1>k?NF-1:k)}} END{for(i=1;i<=k;i++) printf "%d;", x[i]}')
    # --- job cgroup memory (GiB): current, limit, pinned = anon+shmem+dirty+writeback (what cannot be dropped), OOM kills
    local mem="? ? ? ?"
    [ -r "$cg/memory.stat" ] && mem=$(awk -v cur="$(cat "$cg/memory.current")" -v max="$(cat "$cg/memory.max")" \
         -v ok="$(awk '/^oom_kill /{print $2}' "$cg/memory.events" 2>/dev/null)" \
         '/^(anon|shmem|file_dirty|file_writeback) /{p+=$2} END{if(max=="max") max=0; printf "%d %d %d %d", cur/2^30, max/2^30, p/2^30, ok+0}' "$cg/memory.stat")
    local disk eps=0 p
    disk=$(df -P /tmp 2>/dev/null | awk 'NR==2{print $5}' | tr -d %)
    for p in $(pgrep -u "$(id -u)" -f '[_]episode' 2>/dev/null); do pl_injob "$p" "$J" && eps=$((eps + 1)); done
    # --- workers, local commands, keepalives, supervisor, queue
    local wl=() itxt_run="" itxt_idle="" wf st it ln sn wy
    for wf in "$RS"/workers/*.state; do
      [ -e "$wf" ] || continue
      st=$(kv state "$wf"); it=$(kv item "$wf"); ln=$(kv lane "$wf"); sn=$(kv since "$wf"); wy=$(kv why "$wf")
      is_ours "$(kv pid "$wf")" "$J" --worker || { [ "$st" = exited ] || st="dead($st)"; }
      wl+=("worker_$(basename "$wf" .state)=$st|${it:--}|${ln:--}|$sn|$wy")
      case "$st" in running) itxt_run="$itxt_run${itxt_run:+ + }$it" ;; *) itxt_idle="$itxt_idle${itxt_idle:+, }$(basename "$wf" .state):$st" ;; esac
    done
    local cmds="" f nm kts="" ng=${#MX[@]}
    for f in "$L/local/$J"/*.pid; do [ -e "$f" ] || continue; nm=$(basename "$f" .pid); pl_running "$J" "$nm" "$L" && cmds="$cmds$nm "; done
    for ((g = 0; g < ng; g++)); do if kt_alive "$J" "$g"; then kts="${kts}g$g:up "; else kts="${kts}g$g:none "; fi; done
    local supok=0; is_ours "$(cat "$RS/supervisor.pid" 2>/dev/null)" "$J" sup && supok=1
    local qd="" k; for k in $LANES; do qd="$qd$k:$(ls "$PQ/$k" 2>/dev/null | grep -vc '^\.')"$' '; done
    local itxt
    if [ -n "$itxt_run" ]; then itxt="start $itxt_run${itxt_idle:+ ($itxt_idle)}"; elif [ -n "$itxt_idle" ]; then itxt="queue empty / $itxt_idle"; else itxt="no workers"; fi
    put "$RS/status" "v=1" "job=$J" "host=$(hostname)" "time=$(date -u +%Y-%m-%dT%H:%M:%SZ)" "epoch=$(date +%s)" \
        "resident_pid=$(cat "$RS/supervisor.pid" 2>/dev/null)" "supervisor_alive=$supok" "script_mtime=$ME" \
        "state=$(kv state "$RS/state")" "kinds=$(kv kinds "$RS/config")" "port_off=$(kv port_off "$RS/config")" "error=$(kv error "$RS/config")" \
        "gpus=$gl" "gpus_mean=$gm" "gpus_max5m=$m5" "gpus_max10m=$m10" "disk_tmp=${disk:-?}" "episodes=$eps" "mem=$mem" \
        "cmds=$cmds" "keepalive=$kts" "queue=$qd" "${wl[@]}" "itxt=$itxt"
    nap $(( gap > 2 ? gap : 2 ))
  done
}

# ======================================================================================================== supervisor
sup_main() {
  J=${1:?JOBID}; local KINDS=${2:-auto} NG=${3:-auto} OFF=${4:-auto} WCPUS=${5:-}
  # 2026-10-11: a pod submitted with its kinds unquoted (cw_gpupod_resident.sbatch h200 rl, job 65510704) passes the second
  # kind in the port-offset slot; fold it back into the kinds rather than fail on arithmetic with a word.
  if [ "$OFF" != auto ] && ! [[ "$OFF" =~ ^[0-9]+$ ]]; then KINDS="$KINDS $OFF"; OFF=auto; fi
  if [ "${SLURM_JOB_ID:-}" != "$J" ]; then echo "pod_resident.sh: not inside job $J (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}); refusing" >&2; exit 2; fi
  RS=$(pl_rs "$J")
  mkdir -p "$RS/inbox" "$RS/quick" "$RS/cancel" "$RS/control" "$RS/workers" "$L/local/$J" || exit 1
  rlog() { echo "$(ts) [resident $J] $*" >> "$RS/resident.log"; }
  # the queue root must be reachable FROM THIS NODE (items now run here, not on the orchestrator's host); if it is not,
  # run no workers and say so in config/status rather than sit idle looking healthy
  local d qerr=""
  for d in $LANES started claims; do mkdir -p "$PQ/$d" 2>/dev/null || qerr="cannot create $PQ/$d"; done
  [ -z "$qerr" ] && { touch "$PQ/claims/.probe_$J" 2>/dev/null && rm -f "$PQ/claims/.probe_$J"; } || qerr=${qerr:-"$PQ/claims not writable"}
  [ -z "$qerr" ] && [ ! -e "$PQ/_lib.sh" ] && { ln -s "$P/scripts/pod_resident_lib.sh" "$PQ/_lib.sh" 2>/dev/null || [ -e "$PQ/_lib.sh" ] || qerr="cannot link $PQ/_lib.sh"; }
  if [ -n "$qerr" ]; then rlog "ERROR: queue root unusable from $(hostname) ($qerr) -- running NO workers (status, keepalive, inbox only)"; KINDS="-"; fi
  local old; old=$(cat "$RS/supervisor.pid" 2>/dev/null)
  if [ -n "$old" ] && [ "$old" != "$$" ] && is_ours "$old" "$J" sup; then echo "pod_resident.sh: already running for $J (pid $old)" >&2; exit 3; fi
  echo $$ > "$RS/supervisor.pid"
  local reloaded=${POD_RESIDENT_RELOADED:-0}; unset POD_RESIDENT_RELOADED
  # --- the pod: job record (one slurmctld call; items read it instead of calling scontrol), GPUs, lanes, ports
  if [ "$reloaded" = 0 ] || [ ! -s "$RS/jobinfo" ]; then
    timeout 60 scontrol show job "$J" > "$RS/jobinfo.tmp" 2>/dev/null && grep -q "JobId=$J " "$RS/jobinfo.tmp" && mv -f "$RS/jobinfo.tmp" "$RS/jobinfo"
    rm -f "$RS/jobinfo.tmp"
  fi
  local gi ngv tot0 h200=0
  gi=$(timeout 60 $NVSMI --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null)
  ngv=$(echo "$gi" | grep -c '^[0-9]'); tot0=$(echo "$gi" | head -1 | awk -F', *' '{print $3+0}')
  echo "$gi" | grep -q H200 && h200=1
  [ "$NG" = auto ] || [ "$NG" = 0 ] && NG=$ngv
  if [ "$ngv" -eq 0 ]; then rlog "WARNING: nvidia-smi shows no GPU; workers wait for their GPU(s) (ngpu=$NG)"
  elif [ "$ngv" -lt "$NG" ]; then rlog "WARNING: $NG GPU(s) expected but $ngv visible -- serving the visible ones"; NG=$ngv; fi
  if [ "$KINDS" = auto ]; then
    if [ $h200 = 1 ]; then KINDS="h200 rl"; elif [ "$NG" -ge 2 ]; then KINDS=rl
    elif [ "${tot0:-0}" -gt 0 ] && [ "$tot0" -lt 79000 ]; then KINDS=l40s; else KINDS="single l40s"; fi
  fi
  local KA=${POD_KEEPALIVE:-auto}; [ "$KA" = auto ] && { [ $h200 = 1 ] && KA=0 || KA=1; }
  # workers: (name gpus port_off cpus)
  local WN=() WG=() WO=() WC=() g base ports
  if [ "$OFF" = auto ]; then
    local busy others o clash q
    busy=$(ss -ltnH 2>/dev/null | awk '{sub(/.*:/, "", $4); print $4}')
    others=$(for f in "${POD_RESIDENT_DIR:-$P/logs/pods/resident}"/*/config; do [ "$f" = "$RS/config" ] && continue
               [ "$(kv host "$f")" = "$(hostname)" ] && kv ports "$f"; done 2>/dev/null | tr ' ' '\n')
    OFF=0
    for o in 0 50 100 150 200 250 300 350 400 450; do
      clash=0
      for ((g = 0; g < (NG > 1 ? NG : 1); g++)); do q=$((17000 + J % 1000 + o + 25 * g))
        echo "$busy" | grep -qx "$q" && clash=1; echo "$others" | grep -qx "$q" && clash=1; done
      [ $clash = 0 ] && { OFF=$o; break; }
    done
  fi
  if [ "$KINDS" != "-" ]; then
    if [ "$NG" -ge 2 ] && [ "$KINDS" != double ]; then
      for ((g = 0; g < NG; g++)); do WN+=("g$g"); WG+=("$g"); WO+=($((OFF + 25 * g))); WC+=("${WCPUS:-14}"); done
    else
      WN+=(all); WG+=("$(seq -s ' ' 0 $(( (NG > 0 ? NG : 1) - 1 )))"); WO+=("$OFF"); WC+=("$WCPUS")
    fi
  fi
  ports=""; for ((g = 0; g < ${#WN[@]}; g++)); do ports="$ports $((17000 + J % 1000 + ${WO[$g]}))"; done
  put "$RS/config" "job=$J" "host=$(hostname)" "kinds=$KINDS" "ngpu=$NG" "port_off=$OFF" "cpus=$WCPUS" "keepalive=$KA" \
      "workers=${WN[*]:-}" "ports=${ports# }" "pq=$PQ" "gpus=$(echo "$gi" | tr '\n' ';')" "error=$qerr"
  put "$RS/state" "state=running" "since=$(ts)"
  rm -f "$RS/workers/"*.pid.tmp.* 2>/dev/null
  # state files of workers outside this layout (an 'all' worker from an earlier `double` run, say) whose process is gone
  local sf wn keep
  for sf in "$RS"/workers/*.state; do
    [ -e "$sf" ] || continue; wn=$(basename "$sf" .state); keep=0
    for g in "${WN[@]}"; do [ "$g" = "$wn" ] && keep=1; done
    [ $keep = 0 ] && ! is_ours "$(kv pid "$sf")" "$J" --worker && rm -f "$sf" "$RS/workers/$wn.pid"   # .item kept: adoption
  done
  if [ "$reloaded" = 1 ]; then rlog "reloaded (pid $$)"
  else rlog "started pid $$ on $(hostname): kinds=[$KINDS] ngpu=$NG port_off=$OFF keepalive=$KA workers=[${WN[*]:-none}] queue=$PQ"; fi

  # --- claims of this job whose worker AND item are both gone (a requeued pod keeps its job id; a crashed resident)
  local c it ln wp ip
  for c in "$PQ/claims/${J}_"*; do
    [ -e "$c" ] || continue; it=$(basename "$c"); ln=$(kv lane "$c"); wp=$(kv pid "$c"); ip=$(kv ipid "$c")
    if [ "$(kv host "$c")" = "$(hostname)" ] && { is_ours "$wp" "$J" --worker || item_alive "$ip" "$it"; }; then continue; fi
    if [ -f "$PQ/started/$it" ] && [ -n "$ln" ] && mv "$PQ/started/$it" "$PQ/$ln/${it#"${J}"_}"; then
      rlog "returned ${it#"${J}"_} to $ln (its worker and item are gone: requeued pod or resident restart)"
    fi
    rm -f "$c"
  done

  local -A LASTSTART=()
  ensure() {  # start (or adopt) every worker and the status loop
    local i w pidf now; now=$(date +%s)
    for ((i = 0; i < ${#WN[@]}; i++)); do
      w=${WN[$i]}; pidf="$RS/workers/$w.pid"
      is_ours "$(cat "$pidf" 2>/dev/null)" "$J" --worker && continue
      [ -e "$RS/control/exit" ] && continue
      [ $(( now - ${LASTSTART[$w]:-0} )) -lt "${POD_RESTART_EVERY:-60}" ] && continue
      LASTSTART[$w]=$now
      setsid "${DSIG[@]}" bash "$SELF" --worker "$J" "$w" "${WG[$i]}" "$KINDS" "${WO[$i]}" "${WC[$i]}" >> "$RS/resident.log" 2>&1 < /dev/null &
      echo $! > "$pidf"; rlog "worker $w started (pid $!, GPU(s) ${WG[$i]}, lanes [$KINDS], port_off ${WO[$i]})"
    done
    if ! is_ours "$(cat "$RS/workers/status.pid" 2>/dev/null)" "$J" --status && [ $(( now - ${LASTSTART[status]:-0} )) -ge "${POD_RESTART_EVERY:-60}" ]; then
      LASTSTART[status]=$now
      setsid "${DSIG[@]}" bash "$SELF" --status "$J" >> "$RS/resident.log" 2>&1 < /dev/null &
      echo $! > "$RS/workers/status.pid"
    fi
  }
  keepalive() {
    [ "$KA" = 1 ] || return 0
    local g
    for ((g = 0; g < NG; g++)); do
      kt_alive "$J" "$g" && continue
      [ -x "$KT" ] || { rlog "no keepalive binary at $KT"; return; }
      POD_LOCAL=$J POD_LOCAL_HOST=$(hostname) "$P/scripts/pod_exec.sh" "$J" 1 "kt_${J}_g$g" -- env CUDA_VISIBLE_DEVICES=$g "$KT" >/dev/null 2>&1 < /dev/null \
        && rlog "tiny keepalive started on GPU $g (kt_${J}_g$g)"
    done
  }
  merge_env() {  # the caller's environment (NUL-separated) minus Slurm / GPU / local-mode keys, plus this pod's
    local e arm=0
    # 10-22: on an aarch64 pod (GH200) the caller's HOST-specific settings (PATH to x86 conda binaries, compilers, library paths,
    # modules) break items ('sed: Exec format error') -- drop them and keep this resident's own clean PATH and HOME instead
    [ "$(uname -m)" = aarch64 ] && arm=1
    if [ -f "$1" ]; then
      while IFS= read -r -d '' e; do
        case "$e" in SLURM_*|SLURMD_*|CUDA_VISIBLE_DEVICES=*|POD_LOCAL=*|POD_LOCAL_HOST=*|POD_CMD=*|POD_ITEM=*) continue ;; esac
        if [ "$arm" = 1 ]; then
          case "$e" in PATH=*|LD_LIBRARY_PATH=*|LIBRARY_PATH=*|CPATH=*|C_INCLUDE_PATH=*|CPLUS_INCLUDE_PATH=*|PKG_CONFIG_PATH=*|CC=*|CXX=*|CPP=*|FC=*|CFLAGS=*|CXXFLAGS=*|CPPFLAGS=*|LDFLAGS=*|CONDA_*|_CE_*|PYTHONHOME=*|PYTHONPATH=*|MODULEPATH=*|LOADEDMODULES=*|_LMFILES_=*|__LMOD_*|LMOD_*|MANPATH=*|HOME=*|GBP_SANDBOX_PYTHON=*) continue ;; esac
        fi
        printf '%s\0' "$e"
      done < "$1"
    fi
    env -0 | while IFS= read -r -d '' e; do
      case "$e" in SLURM_*|SLURMD_*|CUDA_VISIBLE_DEVICES=*) printf '%s\0' "$e" ;; esac
      [ "$arm" = 1 ] && case "$e" in PATH=*|HOME=*) printf '%s\0' "$e" ;; esac
    done
    printf '%s\0' "POD_LOCAL=$J" "POD_LOCAL_HOST=$(hostname)"
    # 10-22 15:2x: the trainers default the cell interpreter to Path(__file__).resolve()/../../venv/bin/python, and resolve() follows
    # the shadow root's symlinks back into /work = the x86 venv ('Exec format error' at preflight) -- point cells at this root's venv
    [ "$arm" = 1 ] && printf '%s\0' "GBP_SANDBOX_PYTHON=$P/venv/bin/python"
  }
  inbox() {  # remote requests: exec (NAME.req + NAME.env), cancel/NAME, quick/ID.sh
    local r NAME CMDF C envm q id x
    for r in "$RS"/inbox/*.req; do
      [ -e "$r" ] || continue
      NAME=$(basename "$r" .req); CMDF=""; C=""; { read -r CMDF; read -r C; } < "$r"
      envm="$RS/inbox/.$NAME.envm.$$"; merge_env "$RS/inbox/$NAME.env" > "$envm"; rm -f "$RS/inbox/$NAME.env" "$r"
      if [ "$CMDF" != "$L/$NAME.cmd.sh" ] || [ ! -x "$CMDF" ]; then
        rlog "inbox: refused $NAME (command file '$CMDF' is not $L/$NAME.cmd.sh or not executable)"; rm -f "$envm"
        { echo "[pod_exec] $(ts) job=$J resident refused the request (bad command file)"; echo "[pod_exec] $(ts) exit 2"; } >> "$L/$NAME.log"
        echo 2 > "$L/$NAME.failed"; continue
      fi
      pl_launch "$J" "${C:-1}" "$NAME" "$CMDF" "$L" "$envm"; rlog "inbox: started $NAME"
    done
    for x in "$RS"/cancel/*; do
      [ -e "$x" ] || continue; NAME=$(basename "$x"); rm -f "$x"
      ( pl_kill "$J" "$NAME" "$L" >/dev/null 2>&1 ) & rlog "inbox: cancel $NAME"
    done
    for q in "$RS"/quick/*.sh; do
      [ -e "$q" ] || continue; id=${q%.sh}; mv -f "$q" "$id.run" 2>/dev/null || continue
      ( timeout 60 bash "$id.run" > "$id.out" 2>&1 < /dev/null; echo $? > "$id.rc.tmp"; mv -f "$id.rc.tmp" "$id.rc"; rm -f "$id.run" ) &
    done
  }
  shutdown() {
    put "$RS/state" "state=terminating" "since=$(ts)"
    rlog "SIGTERM: the pod is ending (or the resident is being stopped); workers return their items"
    local i p pids=()
    for ((i = 0; i < ${#WN[@]}; i++)); do p=$(cat "$RS/workers/${WN[$i]}.pid" 2>/dev/null)
      is_ours "$p" "$J" --worker && { kill -TERM "$p" 2>/dev/null; pids+=("$p"); }; done
    for ((i = 0; i < GRACE + 20; i++)); do
      local alive=0; for p in "${pids[@]}"; do pl_alive "$p" && alive=1; done; [ $alive = 0 ] && break; sleep 1; done
    p=$(cat "$RS/workers/status.pid" 2>/dev/null); is_ours "$p" "$J" --status && kill -TERM "$p" 2>/dev/null
    put "$RS/state" "state=ended" "since=$(ts)"; rlog "ended"
    exit 0
  }

  SUPTERM=0; trap 'SUPTERM=1' TERM INT HUP
  local ME last_hb=0 last_ens=0 last_ka=0 last_gc=0 t alive
  ME=$(mtime)
  while :; do
    [ "$SUPTERM" = 1 ] && shutdown
    t=$(date +%s)
    [ $((t - last_hb)) -ge 10 ] && { touch "$RS/heartbeat"; last_hb=$t; }
    inbox
    if [ -e "$RS/control/exit" ]; then
      alive=0; for g in "${WN[@]}"; do is_ours "$(cat "$RS/workers/$g.pid" 2>/dev/null)" "$J" --worker && alive=1; done
      if [ $alive = 0 ]; then put "$RS/state" "state=exited" "since=$(ts)"; rlog "control/exit: all workers left; supervisor exits"; exit 0; fi
    fi
    [ $((t - last_ens)) -ge 10 ] && { ensure; last_ens=$t; }
    [ $((t - last_ka)) -ge 60 ] && { keepalive; last_ka=$t; }
    if [ $((t - last_gc)) -ge 600 ]; then   # answers nobody collected, requests nobody took
      find "$RS/quick" "$RS/inbox" -maxdepth 1 -type f -mmin +10 -delete 2>/dev/null; last_gc=$t; fi
    if [ "$(mtime)" != "$ME" ] && bash -n "$SELF" 2>/dev/null; then
      rlog "pod_resident.sh changed; reloading the supervisor"
      exec env POD_RESIDENT_RELOADED=1 bash "$SELF" "$J" "$KINDS" "$NG" "$OFF" "$WCPUS"
    fi
    nap 2
  done
}

case "${1:-}" in
  --worker) shift; worker_main "$@" ;;
  --status) shift; status_main "$@" ;;
  -h|--help|"") sed -n '2,50p' "$SELF"; exit 0 ;;
  *) sup_main "$@" ;;
esac
