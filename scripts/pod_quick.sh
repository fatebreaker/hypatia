#!/bin/bash
# Run a short command inside a pod and print its output: a new Slurm step when the pod still allows one, otherwise its
# agent (scripts/pod_agent.sh; pods submitted with --time-min refuse new steps after TimeMin). At most ~60 s.
#
#   scripts/pod_quick.sh JOBID command args...
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:?JOBID}"; shift; [ -n "${POD_LOCAL:-}${POD_VIA_RESIDENT:-}" ] && . "$P/scripts/pod_local.sh" && pl_mode_quick "$J" "$@"  # local / resident modes (2026-09-30): scripts/pod_local.sh; nothing unless POD_LOCAL / POD_VIA_RESIDENT is set
out="$(timeout 60 srun --jobid="$J" --overlap -n1 -c1 "$@" 2>&1 < /dev/null)"; rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "Unable to create step"; then
  AG="$P/logs/pods/agent/$J"
  # 180 s, not 60: an agent busy launching a step lags its heartbeat, and at 60 s a live agent was judged dead so
  # the caller received srun's error text as its "answer" (2026-09-20, two working pods reported idle).
  [ -f "$AG/heartbeat" ] && [ $(( $(date +%s) - $(stat -c %Y "$AG/heartbeat") )) -lt 180 ] || { echo "$out"; exit $rc; }
  id="q$$_$RANDOM"
  printf '%q ' "$@" > "$AG/quick/$id.sh.tmp"; echo >> "$AG/quick/$id.sh.tmp"; mv "$AG/quick/$id.sh.tmp" "$AG/quick/$id.sh"
  for _ in $(seq 90); do [ -f "$AG/quick/$id.rc" ] && break; sleep 1; done
  [ -f "$AG/quick/$id.rc" ] || { echo "pod_quick: agent of $J did not answer"; exit 124; }
  cat "$AG/quick/$id.out"; rc=$(cat "$AG/quick/$id.rc"); rm -f "$AG/quick/$id.out" "$AG/quick/$id.rc"
  exit "$rc"
fi
printf '%s' "$out"; [ -n "$out" ] && echo
exit $rc
