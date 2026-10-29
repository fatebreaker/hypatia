#!/bin/bash
# Cancel one of OUR pod steps (launched by scripts/pod_exec.sh) without leaving a
# stale marker behind.
#
#   scripts/pod_cancel.sh JOBID NAME
#
# Why: when a step is cancelled, its pod_exec wrapper records the non-zero exit
# as NAME.failed a moment later -- and any job waiting on NAME (WAITFOR=NAME)
# treats .failed as "go". Removing the marker right after scancel races that
# write; this waits for the wrapper's own "exit" line first. Never use it on a
# step we did not launch (see ~/cosmo_world/UNITY_HANDBOOK.md).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:?JOBID}"; NAME="${2:?NAME}"; L="$P/logs/pods"; [ -n "${POD_LOCAL:-}${POD_VIA_RESIDENT:-}" ] && . "$P/scripts/pod_local.sh" && pl_mode_cancel "$J" "$NAME" "$L"  # local / resident modes (2026-09-30): scripts/pod_local.sh; nothing unless POD_LOCAL / POD_VIA_RESIDENT is set
steps=$(sacct -j "$J" --format=JobID%24,JobName%60,State%12 --noheader \
        | awk -v n="$NAME.cmd.sh" '$2 == n && $3 == "RUNNING" {print $1}')
count_exits() { local n; n=$(grep -c '^\[pod_exec\] .* exit ' "$L/$NAME.log" 2>/dev/null); echo "${n:-0}"; }
AG="$L/agent/$J"; apid="$(cat "$AG/pids/$NAME" 2>/dev/null)"
if [ -z "$steps" ] && [ -n "$apid" ] && [ -f "$AG/heartbeat" ]; then   # run by the pod's agent (scripts/pod_agent.sh)
  before=$(count_exits); touch "$AG/cancel/$NAME"; echo "asked the agent of $J to cancel $NAME"
else
  [ -n "$steps" ] || { echo "no running step named $NAME in job $J"; exit 1; }
  before=$(count_exits)
  for s in $steps; do scancel "$s" && echo "cancelled $s ($NAME)"; done
fi
for _ in $(seq 60); do
  now=$(count_exits)
  [ "$now" -gt "$before" ] && break
  sleep 1
done
rm -f "$L/$NAME.done" "$L/$NAME.failed"
echo "markers for $NAME cleared"
