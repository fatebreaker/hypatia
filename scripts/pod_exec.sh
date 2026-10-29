#!/bin/bash
# Run one repo command inside a held SLURM pod, detached from this shell, with the
# repo environment (data root, caches, TMPDIR on the payloads' filesystem).
#
#   scripts/pod_exec.sh JOBID CPUS NAME -- command args...
#
# CPUS: pass the job's FULL count (8 on the H200 singles). A smaller -c pins the
# step to the job's lowest-numbered cores -- the same ones every other --overlap
# step lands on -- so a vLLM server, an eval and 48 RL sandboxes ended up sharing
# two cores at 100% while six sat idle. With the full count the kernel spreads them.
#
# Each SLURM job has a PRIVATE /tmp, even on a shared node: stage per job.
# Output goes to logs/pods/NAME.log; a NAME.done / NAME.failed marker records the
# exit status, so a watcher can wait on files instead of on processes. Survives
# this shell and a login-node reboot (the step lives on the pod).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:?JOBID}"; C="${2:?CPUS}"; NAME="${3:?NAME}"; shift 3
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "usage: pod_exec.sh JOBID CPUS NAME -- cmd..." >&2; exit 2; }
L="$P/logs/pods"; mkdir -p "$L"

# A repo script that lost its executable bit fails on the pod with exit 126, minutes later, having burned a queue slot
# (it has happened twice: 2026-09-11 and 2026-09-16, both times after an "atomic" f.tmp + mv created a 0644 file).
# Catch it here, before the step is launched.
for a_ in "$@"; do
  case "$a_" in
    experiments/*.sh|scripts/*.sh|rl/*.sh|harness/*.sh)
      [ -x "$P/$a_" ] || { echo "pod_exec.sh: $a_ is not executable (chmod +x it; an atomic edit drops the mode)" >&2; exit 2; } ;;
  esac
done

rm -f "$L/$NAME.done" "$L/$NAME.failed"

# Write the command to a file: quoting a command through nohup, setsid, srun and
# bash -c is where these launchers usually break.
CMDF="$L/$NAME.cmd.sh"
{
  echo '#!/bin/bash'
  echo "cd '$P' && source scripts/env.sh >/dev/null"
  echo 'export PATH="'"$P"'/venv_vllm/bin:$PATH" PYTHONUNBUFFERED=1'
  # FINDINGS D2: Triton JIT-compiles with $CC, and a login shell inherits
  # conda's cross-compiler, which cannot link here. Always the system compiler.
  echo 'export CC=/usr/bin/gcc CXX=/usr/bin/g++'
  printf '%q ' "$@"; echo
} > "$CMDF"
chmod +x "$CMDF"; [ -n "${POD_LOCAL:-}${POD_VIA_RESIDENT:-}" ] && . "$P/scripts/pod_local.sh" && pl_mode_exec "$J" "$C" "$NAME" "$CMDF" "$L"  # local / resident modes (2026-09-30): scripts/pod_local.sh; nothing unless POD_LOCAL / POD_VIA_RESIDENT is set

# A pod submitted with --time-min stops allowing new steps after TimeMin; if its agent (scripts/pod_agent.sh, one step
# started earlier) is alive, hand the command to it instead: same log, markers and exit lines.
AG="$L/agent/$J"
use_agent=0
if [[ "$NAME" != podagent_* ]] && [ -f "$AG/heartbeat" ] && [ $(( $(date +%s) - $(stat -c %Y "$AG/heartbeat") )) -lt 60 ]; then
  # capture srun's output (a pipe into grep would fail under pipefail whenever srun does)
  probe="$(timeout 60 srun --jobid="$J" --overlap -n1 -c1 true 2>&1 < /dev/null)" || { [[ "$probe" == *"Unable to create step"* ]] && use_agent=1; }
fi
if [ "$use_agent" = 1 ]; then
  echo "$CMDF" > "$AG/inbox/$NAME.req.tmp" && mv "$AG/inbox/$NAME.req.tmp" "$AG/inbox/$NAME.req"
  echo "launched $NAME on job $J through its agent (log: logs/pods/$NAME.log)"
  exit 0
fi
nohup setsid bash -c "
  echo \"[pod_exec] \$(date -Is) job=$J cpus=$C :: \$(tail -1 '$CMDF')\"
  srun --jobid=$J --overlap -n1 -c$C '$CMDF'
  rc=\$?
  echo \"[pod_exec] \$(date -Is) exit \$rc\"
  if [ \$rc -eq 0 ]; then touch '$L/$NAME.done'; else echo \$rc > '$L/$NAME.failed'; fi
" >> "$L/$NAME.log" 2>&1 < /dev/null &
echo "launched $NAME on job $J (log: logs/pods/$NAME.log)"
