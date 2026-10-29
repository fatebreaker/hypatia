#!/bin/bash
# A long-lived command runner INSIDE a pod, so the pod stays usable after Slurm stops letting us create new steps in it.
#
# Pods submitted with --time-min=1-00:00:00 refuse every new job step once they have run for TimeMin ("Unable to create
# step ... Job/step already completing or completed"; seen on eight pods, 2026-09-13/14), while steps already running
# carry on. Started as ONE step before that point (scripts/pod_exec.sh JOBID CPUS podagent_JOBID -- scripts/pod_agent.sh
# JOBID), this loop runs our commands as its own children instead of as new steps. scripts/pod_exec.sh, pod_cancel.sh and
# the queue's up() check use it whenever its heartbeat is fresh:
#
#   logs/pods/agent/J/inbox/NAME.req   a pod_exec request: first line = the command file logs/pods/NAME.cmd.sh. Run in the
#                                      background; log, NAME.done / NAME.failed markers and exit lines exactly as pod_exec
#   logs/pods/agent/J/cancel/NAME      kill that command's process group (pod_cancel)
#   logs/pods/agent/J/quick/ID.sh      run now, at most 60 s: output ID.out, exit status ID.rc (short checks, e.g. curl)
#   logs/pods/agent/J/heartbeat        touched every few seconds
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:?JOBID}"; L="$P/logs/pods"; AG="$L/agent/$J"
mkdir -p "$AG/inbox" "$AG/pids" "$AG/cancel" "$AG/quick"
echo "[pod_agent] $(date -Is) job=$J host=$(hostname) pid=$$"
while true; do
  touch "$AG/heartbeat"
  for r in "$AG"/inbox/*.req; do
    [ -e "$r" ] || continue
    NAME="$(basename "$r" .req)"; CMDF="$(head -1 "$r")"; rm -f "$r"
    setsid bash -c "
      echo \"[pod_exec] \$(date -Is) job=$J agent :: \$(tail -1 '$CMDF')\"
      bash '$CMDF'
      rc=\$?
      echo \"[pod_exec] \$(date -Is) exit \$rc\"
      if [ \$rc -eq 0 ]; then touch '$L/$NAME.done'; else echo \$rc > '$L/$NAME.failed'; fi
    " >> "$L/$NAME.log" 2>&1 < /dev/null &
    echo $! > "$AG/pids/$NAME"
    echo "[pod_agent] $(date -Is) started $NAME (pid $!)"
  done
  for c in "$AG"/cancel/*; do
    [ -e "$c" ] || continue
    NAME="$(basename "$c")"; rm -f "$c"
    pid="$(cat "$AG/pids/$NAME" 2>/dev/null)"
    [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null && echo "[pod_agent] $(date -Is) cancelled $NAME (group $pid)"
  done
  for q in "$AG"/quick/*.sh; do
    [ -e "$q" ] || continue
    id="${q%.sh}"; timeout 60 bash "$q" > "$id.out" 2>&1; echo $? > "$id.rc.tmp"; mv "$id.rc.tmp" "$id.rc"; rm -f "$q"
  done
  sleep 2
done
