#!/bin/bash
# pod_local.sh -- SOURCED (never run) by scripts/pod_exec.sh, pod_quick.sh, pod_cancel.sh and pod_resident.sh.
#
# LOCAL MODE (2026-09-30). Run, check and cancel commands as LOCAL processes of the pod we are already inside, with the
# same contract as a pod_exec step and ZERO srun. Why: every srun is a Slurm job step, and a pod that had taken ~98
# steps in 2.5 h (monitor samples, liveness checks, GPU waits, server polls, launches, cancels) stopped accepting
# steps and hung its runs -- twice. And every step's srun CLIENT lived on the worker host, so that host dying killed
# every run on every pod. A resident worker inside the pod (scripts/pod_resident.sh) runs everything as its own
# children instead; these helpers are how.
#
# Used only when asked for (callers check first, so with POD_LOCAL / POD_VIA_RESIDENT unset nothing here runs):
#   POD_LOCAL=J          (exported by the resident worker to everything it runs) -- we are inside job J: act locally.
#                        Refused unless $SLURM_JOB_ID is J too (and the host is $POD_LOCAL_HOST when that is set).
#   POD_VIA_RESIDENT=1   (a caller OUTSIDE the pod) -- if J's resident worker is alive, hand the request to it through
#                        files under its state dir; otherwise the caller carries on exactly as before.
#
# The contract kept from pod_exec.sh's step mode:
#   logs/pods/NAME.cmd.sh   the command file (pod_exec.sh writes it before it gets here)
#   logs/pods/NAME.log      "[pod_exec] <iso> job=J cpus=C local :: <cmd>" ... output ... "[pod_exec] <iso> exit RC"
#   logs/pods/NAME.done     or NAME.failed (containing RC)
# Local bookkeeping: logs/pods/local/J/NAME.pid = "<pgid> <wrapper pid> <epoch>" while the command runs (removed at
# its exit). Every process of the command carries POD_CMD=J:NAME in its environment -- inherited by all descendants,
# including any that setsid away from the process group -- so a cancel finds all of them, the way scancel finds a
# step's cgroup. The wrapper outlives signals sent to the whole pod so that it can record the exit, as the srun client
# did; the command itself starts with every signal at its default (a plain `cmd &` would ignore SIGINT/SIGQUIT).

PL_P="${PL_P:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

pl_inside() {  # pl_inside J -- local mode was asked for job J and this shell really is inside it
  [ -n "${POD_LOCAL:-}" ] && [ "$POD_LOCAL" = "$1" ] && [ "${SLURM_JOB_ID:-}" = "$1" ] \
    && { [ -z "${POD_LOCAL_HOST:-}" ] || [ "$POD_LOCAL_HOST" = "$(hostname)" ]; }
}

pl_rs() { echo "${POD_RESIDENT_DIR:-$PL_P/logs/pods/resident}/$1"; }   # a pod's resident state dir

pl_alive() {  # a live, non-zombie process
  local s
  s=$(awk '/^State:/{print $2; exit}' "/proc/$1/status" 2>/dev/null) || return 1
  [ -n "$s" ] && [ "$s" != Z ] && [ "$s" != X ]
}

pl_injob() {  # pl_injob PID J -- PID belongs to Slurm job J (pods of ours on one node see each other's processes)
  [ "${POD_CGROUP_CHECK:-1}" = 0 ] && return 0
  grep -q "/job_$2/" "/proc/$1/cgroup" 2>/dev/null
}

pl_tagged() {  # pl_tagged KEY=VALUE J -- our live processes in job J whose environment holds KEY=VALUE (never this shell)
  local f p
  # timeout: reading a process's environ takes its mmap lock, which a process stuck in the kernel can hold
  for f in $(timeout 20 grep -lzFx -- "$1" /proc/[0-9]*/environ 2>/dev/null); do
    p=${f#/proc/}; p=${p%/environ}
    [ "$p" = "$$" ] && continue; [ "$p" = "${BASHPID:-$$}" ] && continue
    pl_alive "$p" && pl_injob "$p" "$2" && echo "$p"
  done
}

pl_has_tag() {  # pl_has_tag PID KEY=VALUE
  timeout 5 grep -qzFx -- "$2" "/proc/$1/environ" 2>/dev/null
}

pl_count_exits() {  # pl_count_exits L NAME -- exit lines the wrapper(s) wrote so far (pod_cancel waits for one more)
  local n; n=$(grep -c '^\[pod_exec\] .* exit ' "$1/$2.log" 2>/dev/null); echo "${n:-0}"
}

# The wrapper: its own session; ignores TERM/INT/HUP (so a signal to the whole pod lets it record the exit, as the srun
# client did) while the command gets every signal back at its default through `env --default-signal`.
# shellcheck disable=SC2016
PL_WRAP='
J=$1 C=$2 NAME=$3 CMDF=$4 L=$5 D=$6 ENVF=$7
echo "[pod_exec] $(date -Is) job=$J cpus=$C local :: $(tail -1 "$CMDF")"
DS=(); [ -z "${POD_NO_DEFAULT_SIGNAL:-}" ] && env --default-signal true 2>/dev/null && DS=(--default-signal)
if [ ${#DS[@]} -gt 0 ]; then trap "" TERM INT HUP; else trap : TERM INT HUP; fi
if [ -n "$ENVF" ]; then
  E=(); [ -f "$ENVF" ] && readarray -d "" E < "$ENVF"
  setsid env -i "${DS[@]}" "${E[@]}" POD_CMD="$J:$NAME" "$CMDF" &
else
  POD_CMD="$J:$NAME" setsid env "${DS[@]}" "$CMDF" &
fi
c=$!
printf "%s %s %s\n" "$c" "$$" "$(date +%s)" > "$D/$NAME.pid.tmp.$$" && mv -f "$D/$NAME.pid.tmp.$$" "$D/$NAME.pid"
[ -n "$ENVF" ] && rm -f "$ENVF"      # the caller environment dump is read; do not leave it (tokens) lying around
if [ ${#DS[@]} -gt 0 ]; then
  wait "$c"; rc=$?
else  # no --default-signal: a handler (not an ignore) interrupts wait; wait again until the command is really gone
  while :; do wait "$c"; rc=$?; kill -0 "$c" 2>/dev/null || break; done
  wait "$c" 2>/dev/null; r2=$?; [ "$r2" -ne 127 ] && rc=$r2   # the status bash saved if a signal cut the last wait short
fi
[ "$(cut -d" " -f2 "$D/$NAME.pid" 2>/dev/null)" = "$$" ] && rm -f "$D/$NAME.pid"
echo "[pod_exec] $(date -Is) exit $rc"
if [ $rc -eq 0 ]; then touch "$L/$NAME.done"; else echo $rc > "$L/$NAME.failed"; fi
'

pl_launch() {  # pl_launch J C NAME CMDF L [ENVFILE] -- start CMDF as a local process group; returns at once
  local J=$1 C=$2 NAME=$3 CMDF=$4 L=$5 ENVF=${6:-} D="$5/local/$1"
  mkdir -p "$D"; rm -f "$D/$NAME.pid"
  setsid bash -c "$PL_WRAP" pl_wrap "$J" "$C" "$NAME" "$CMDF" "$L" "$D" "$ENVF" >> "$L/$NAME.log" 2>&1 < /dev/null &
}

pl_pids() {  # pl_pids J NAME L -- every live process of command NAME (its process group leader and all tagged ones)
  local pg=""
  read -r pg _ 2>/dev/null < "$3/local/$1/$2.pid"
  { [ -n "$pg" ] && pl_alive "$pg" && pl_injob "$pg" "$1" && pl_has_tag "$pg" "POD_CMD=$1:$2" && echo "$pg"
    pl_tagged "POD_CMD=$1:$2" "$1"; } | sort -un
}

pl_running() {  # pl_running J NAME L -- the group leader alone answers it cheaply; else scan for tagged processes
  local pg=""; read -r pg _ 2>/dev/null < "$3/local/$1/$2.pid"
  [ -n "$pg" ] && pl_alive "$pg" && pl_injob "$pg" "$1" && pl_has_tag "$pg" "POD_CMD=$1:$2" && return 0
  [ -n "$(pl_tagged "POD_CMD=$1:$2" "$1")" ]
}

pl_kill() {  # pl_kill J NAME L -- TERM the command's group and every tagged process, KILL what is left; 1 = nothing ran
  local J=$1 NAME=$2 L=$3 pg="" ps i
  # a cancel right after a launch can beat the wrapper's pid file by milliseconds: give a just-started command 5 s
  if [ ! -f "$L/local/$J/$NAME.pid" ] && grep -q '^\[pod_exec\] .* local :: ' "$L/$NAME.log" 2>/dev/null \
     && ! tail -1 "$L/$NAME.log" | grep -q '^\[pod_exec\] .* exit '; then
    for i in 1 2 3 4 5; do [ -f "$L/local/$J/$NAME.pid" ] && break; sleep 1; done
  fi
  ps=$(pl_pids "$J" "$NAME" "$L")
  [ -n "$ps" ] || return 1
  read -r pg _ 2>/dev/null < "$L/local/$J/$NAME.pid"
  if [ -n "$pg" ] && pl_has_tag "$pg" "POD_CMD=$J:$NAME"; then kill -TERM -- "-$pg" 2>/dev/null; fi
  # shellcheck disable=SC2086
  kill -TERM $ps 2>/dev/null
  echo "cancelled $J.local.${pg:-?} ($NAME)"
  for ((i = 0; i < ${POD_KILL_GRACE:-30}; i++)); do ps=$(pl_pids "$J" "$NAME" "$L"); [ -z "$ps" ] && return 0; sleep 1; done
  # shellcheck disable=SC2086
  [ -n "$ps" ] && kill -KILL $ps 2>/dev/null
  [ -n "$pg" ] && pl_has_tag "$pg" "POD_CMD=$J:$NAME" && kill -KILL -- "-$pg" 2>/dev/null
  return 0
}

pl_resident_fresh() {  # pl_resident_fresh J -- J's resident worker is alive (heartbeat < POD_RESIDENT_FRESH s) and not ending
  local rs; rs=$(pl_rs "$1")
  [ -f "$rs/heartbeat" ] || return 1
  [ $(( $(date +%s) - $(stat -c %Y "$rs/heartbeat" 2>/dev/null || echo 0) )) -lt "${POD_RESIDENT_FRESH:-60}" ] || return 1
  ! grep -qxE 'state=(terminating|ended)' "$rs/state" 2>/dev/null
}

pl_request_exec() {  # pl_request_exec J C NAME CMDF L -- hand a command to J's resident worker, with the caller's environment
  local J=$1 C=$2 NAME=$3 CMDF=$4 L=$5 rs i; rs=$(pl_rs "$1")
  mkdir -p "$rs/inbox" || return 1
  env -0 > "$rs/inbox/.$NAME.env.$$" && mv -f "$rs/inbox/.$NAME.env.$$" "$rs/inbox/$NAME.env" || return 1
  printf '%s\n%s\n' "$CMDF" "$C" > "$rs/inbox/.$NAME.req.$$" && mv -f "$rs/inbox/.$NAME.req.$$" "$rs/inbox/$NAME.req" || return 1
  # wait until the resident takes it (it polls every 2 s); if it never does, fail the way a refused srun did
  for ((i = 0; i < ${POD_RESIDENT_PICKUP:-30}; i++)); do [ -e "$rs/inbox/$NAME.req" ] || return 0; sleep 1; done
  rm -f "$rs/inbox/$NAME.req" "$rs/inbox/$NAME.env"
  { echo "[pod_exec] $(date -Is) job=$J resident did not take the request within ${POD_RESIDENT_PICKUP:-30}s"
    echo "[pod_exec] $(date -Is) exit 1"; } >> "$L/$NAME.log"
  echo 1 > "$L/$NAME.failed"
  return 0
}

pl_request_quick() {  # pl_request_quick J cmd... -- a short command through J's resident worker; prints output, returns rc
  local J=$1 rs id out rc i; shift; rs=$(pl_rs "$J"); id="q$$_$RANDOM"
  mkdir -p "$rs/quick" || return 1
  { printf '%q ' "$@"; echo; } > "$rs/quick/.$id.tmp" && mv -f "$rs/quick/.$id.tmp" "$rs/quick/$id.sh" || return 1
  for ((i = 0; i < 90; i++)); do [ -f "$rs/quick/$id.rc" ] && break; sleep 1; done
  [ -f "$rs/quick/$id.rc" ] || { rm -f "$rs/quick/$id.sh"; echo "pod_quick: resident worker of $J did not answer"; return 124; }
  out=$(cat "$rs/quick/$id.out" 2>/dev/null); rc=$(cat "$rs/quick/$id.rc"); rm -f "$rs/quick/$id.out" "$rs/quick/$id.rc"
  printf '%s' "$out"; [ -n "$out" ] && echo
  return "$rc"
}

# ---------------------------------------------------------------------------------------------------------------------
# The three entry points. Each script calls its one from a clause appended to an EXISTING line (no line added, so even
# bash's own diagnostics keep their line numbers):
#   [ -n "${POD_LOCAL:-}${POD_VIA_RESIDENT:-}" ] && . "$P/scripts/pod_local.sh" && pl_mode_<x> ...
# With both variables unset that clause is false and this file is never read. A function below `exit`s the script when
# it handled the call; when it returns, the script carries on exactly as before (POD_LOCAL names another job, or
# POD_VIA_RESIDENT=1 but J has no live resident).

pl_mode_exec() {  # from pod_exec.sh, after its executable check, marker reset and command file: J C NAME CMDF L
  local J=$1 C=$2 NAME=$3 CMDF=$4 L=$5
  if [ "${POD_LOCAL:-}" = "$J" ]; then
    pl_inside "$J" || { echo "pod_exec.sh: POD_LOCAL=$J but this shell is not inside job $J on its host (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}, host $(hostname)); refusing" >&2; exit 2; }
    pl_launch "$J" "$C" "$NAME" "$CMDF" "$L"
    echo "launched $NAME on job $J (log: logs/pods/$NAME.log)"
    exit 0
  fi
  if [ "${POD_VIA_RESIDENT:-}" = 1 ] && [[ "$NAME" != podagent_* ]] && pl_resident_fresh "$J"; then
    pl_request_exec "$J" "$C" "$NAME" "$CMDF" "$L" || { echo "pod_exec.sh: could not hand $NAME to the resident worker of $J" >&2; exit 2; }
    echo "launched $NAME on job $J through its resident worker (log: logs/pods/$NAME.log)"
    exit 0
  fi
  return 0
}

pl_mode_quick() {  # from pod_quick.sh: J cmd... -- output and exit status as the step path prints them
  local J=$1 out rc; shift
  if [ "${POD_LOCAL:-}" = "$J" ]; then
    pl_inside "$J" || { echo "pod_quick: POD_LOCAL=$J but this shell is not inside job $J on its host (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}); refusing"; exit 2; }
    out="$(timeout 60 "$@" 2>&1 < /dev/null)"; rc=$?
    printf '%s' "$out"; [ -n "$out" ] && echo
    exit $rc
  fi
  if [ "${POD_VIA_RESIDENT:-}" = 1 ] && pl_resident_fresh "$J"; then pl_request_quick "$J" "$@"; exit $?; fi
  return 0
}

pl_mode_cancel() {  # from pod_cancel.sh: J NAME L -- same messages, same wait for the wrapper's exit line, same reset
  local J=$1 NAME=$2 L=$3 before rs
  if [ "${POD_LOCAL:-}" = "$J" ]; then
    pl_inside "$J" || { echo "pod_cancel: POD_LOCAL=$J but this shell is not inside job $J on its host (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}); refusing"; exit 2; }
    before=$(pl_count_exits "$L" "$NAME")
    pl_kill "$J" "$NAME" "$L" || { echo "no running step named $NAME in job $J"; exit 1; }
  elif [ "${POD_VIA_RESIDENT:-}" = 1 ] && [ -f "$L/local/$J/$NAME.pid" ] && pl_resident_fresh "$J"; then
    before=$(pl_count_exits "$L" "$NAME"); rs=$(pl_rs "$J"); mkdir -p "$rs/cancel"
    touch "$rs/cancel/$NAME"; echo "asked the resident worker of $J to cancel $NAME"
  else
    return 0    # a step (or nothing): pod_cancel.sh's own sacct / scancel path
  fi
  for _ in $(seq 60); do [ "$(pl_count_exits "$L" "$NAME")" -gt "$before" ] && break; sleep 1; done
  rm -f "$L/$NAME.done" "$L/$NAME.failed"
  echo "markers for $NAME cleared"
  exit 0
}
