#!/bin/bash
# pod_resident_lib.sh -- SOURCED by the queue items a resident worker runs (scripts/pod_resident.sh). Items begin with
#   source "$(dirname "$0")/../_lib.sh"
# and the resident's queue root holds _lib.sh as a symlink to THIS file. The helpers themselves (boot, stage, gpu_wait,
# up, wait_marker, reap_orphan_server, J/P/C/G/JS/NG/TG/SU) are pod_queue/_lib.sh's own -- sourced from there, one copy,
# so a fix reaches items of both queue roots. In a resident they act locally with no srun because the worker exports
# POD_LOCAL=J (scripts/pod_exec.sh, pod_quick.sh and pod_cancel.sh then run as local processes).
#
# The one addition: `scontrol show job <this pod>` (NumCPUs, gres/gpu, Partition -- fields that do not change while the
# pod runs) is answered from the snapshot the resident took at start ($POD_JOBINFO), so an item makes no slurmctld call.
# Any other scontrol call goes to the real command.
#
# 2026-10-27 (Unity's GPU report; user: 'try to maximize gpu usage'): Slurm accounts a step's GPU and CPU use only for
# processes in the step's process tree (sstat/sacct TRESUsageIn gres/gpuutil, gres/gpumem). An item starts its servers and
# trainers through scripts/pod_exec.sh, which returns at once, so they were re-parented to PID 1, outside the tree: every
# deeprs pod read gres/gpuutil=0 while busy (MLSys's inbox-started runs on a lent pod read 185%). So the item makes itself a
# child subreaper (prctl PR_SET_CHILD_SUBREAPER, kept across exec): what it starts and orphans is re-parented to the item,
# inside the batch step's tree. The item is re-executed once, in the same process (it has done nothing before this source
# line); if python3 or the prctl is unavailable it runs exactly as before.
if [ -n "${POD_LOCAL:-}" ] && [ -z "${POD_ITEM_SUBREAPER:-}" ] && [ -f "$0" ]; then
  case "$0" in
    */held/*.sh|*/started/*.sh)
      export POD_ITEM_SUBREAPER=1
      if python3 -c 'import ctypes' 2>/dev/null; then
        shopt -s execfail
        exec python3 -c 'import ctypes, os, sys
ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)   # PR_SET_CHILD_SUBREAPER
os.execv("/bin/bash", ["bash"] + sys.argv[1:])' "$0" "$@"
        shopt -u execfail
      fi ;;
  esac
fi
if [ -n "${POD_LOCAL:-}" ] && [ -s "${POD_JOBINFO:-}" ]; then
  scontrol() {
    if [ $# -eq 3 ] && [ "$1" = show ] && [ "$2" = job ] && [ "$3" = "$POD_LOCAL" ]; then cat "$POD_JOBINFO"
    else command scontrol "$@"; fi
  }
fi
# shellcheck source=/dev/null
source "${POD_SHARED_LIB:-/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad/pod_queue/_lib.sh}"
