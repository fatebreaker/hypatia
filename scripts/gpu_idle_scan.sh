#!/bin/bash
# Which of our pods is holding a GPU without computing? Unity warns after an hour of no GPU activity and kills at two.
#
#   scripts/gpu_idle_scan.sh [MIN_PCT]        default 5
#
# Why this exists, and why it is written the way it is -- three things it must get right, each learned the hard way on
# 2026-09-17:
#
#   1. A WORKER LOG IS NOT EVIDENCE. During a /project outage pod_exec.sh could not write its step log
#      ("Permission denied"), so the step never launched -- while the queue worker had already printed "start <item>"
#      and considered itself busy. Every log looked healthy; the GPU sat at 0% for an hour until Unity mailed about
#      64445094. Utilisation is the signal, not bookkeeping.
#   2. ONE SAMPLE IS WORTHLESS. Training alternates GPU steps with CPU phases, so a busy pod reads 0% between batches
#      (D's runs sample 100/0/0/100). Take the max over a window.
#   3. LOW UTILISATION IS NOT IDLENESS. An agentic benchmark (QRData, DABench) spends most of its wall clock running
#      pandas/numpy cells on CPU between model calls; 64356260 sat at 0% for half an hour mid-QRData while finishing
#      412 answers. Require BOTH low utilisation and no recent step-log write before calling a pod idle.
#
# Pods are sampled in PARALLEL: serially this took ~90 s each and was too slow to run on demand, which is the surest
# way for a check to never be run. All 30 finish in about three minutes.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$P"
MIN="${1:-5}"
SKIP="${SKIP_PODS:-64001124 64001126}"          # D's pods, running D's own work
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/g" "$TMP/w"   # per-GPU readings and in-pod work flags, kept OUT of the pod loop's directory
total=0
# Suspects carry over between runs so a one-sweep reading is never reported. Kept beside the queue, not in $TMP.
STATE="${SCRATCHPAD:-$P/../scratchpad}/gpu_idle_suspects"
SEEN_PREV="$STATE.prev"; SEEN_NOW="$STATE.now"
: > "$SEEN_NOW"

# GPU jobs only. squeue lists every running job of this user, including CPU-only work that has nothing to do with
# the pod fleet -- a `cyber_cpu_pe` job on a CPU node was reported UNREACHABLE (no srun, no agent, no GPU reading)
# on 2026-09-18, which is exactly right and entirely useless. %b carries the generic resources; a pod without
# "gpu" in it is not ours to watch.
for J in $(squeue -u "$USER" -h -t R -o "%i %b" 2>/dev/null | awk 'tolower($2) ~ /gpu/ {print $1}'); do
  case " $SKIP " in *" $J "*) continue;; esac
  total=$((total+1))
  (
    # Sample across a full MINUTE. An agentic run (the general-agent loop, QRData) calls the model every few seconds
    # between code cells, so a 12-second window lands entirely in CPU time and reads 0% on a perfectly busy pod --
    # 64356393 was flagged that way while finishing selfver_astro_transit. 60 s is long enough to catch the calls.
    # 45 s of sampling, NOT 60. Pods that refuse srun are reached through scripts/pod_quick.sh, which hands the
    # command to the pod's agent -- and the agent kills a quick command at 60 s. A 60 s loop is therefore always
    # killed, returns nothing, and silently degrades the check to a heartbeat test. That is exactly how 64356393 sat
    # idle for two hours with a fresh heartbeat until Unity warned (2026-09-18). Long enough to catch an agentic
    # workload's model calls, short enough to survive the cap.
    r=$(timeout 90 scripts/pod_quick.sh "$J" bash -c \
          'pgrep -f "dispersion\.py|harness/runner\.py|grpo_onpolicy|sft_train|verl" >/dev/null && echo WORKPROC || echo NOWORK
           for i in $(seq 8); do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; sleep 5; done' \
          < /dev/null 2>/dev/null | tr -d ' %' | tr -d 'MiB')
    # Peak utilisation AND peak memory, PER GPU INDEX. Per pod is not enough: a double pod's busy GPU masks an idle
    # one, because the pod-level maximum is whatever the busy card reads. Two items died that way on 2026-09-18 --
    # 4f on 64356395 and 4m on 63648609 -- each leaving a 70 GB vLLM server at 0% on GPU 0 while GPU 1 ran at 99%
    # and 100%. Neither pod ever looked idle, and neither worker took another item for hours.
    # The work-process check runs FIRST and the sample is 8x5s, not 9x5s. Both because the agent kills a quick
    # command at 60 s and 9 samples land exactly on it: the check emitted its line, the loop then ran out the clock,
    # and the line was cut off -- so a pod running three dispersion.py processes was reported as an orphan holding
    # 129 GB (2026-09-18, 64392460). A guard that races the timeout it was written to survive is no guard.
    # Is a work process actually running IN the pod? The only liveness signal that does not go stale. Utilisation
    # reads 0% through a CPU-bound phase; sacct shows nothing for an agent-dispatched step; and a step log gets ONE
    # LINE PER FAMILY, so a healthy run whose family takes 87 minutes looks 74 minutes silent. All three failed
    # together on 64356387 (2026-09-18) and the scan called a live run an orphan -- one command short of killing a
    # server that was mid-scrna_1k.
    case "$r" in *WORKPROC*) echo live > "$TMP/w/$J";; esac
    r=$(echo "$r" | grep -vE 'WORKPROC|NOWORK')
    u=$(echo "$r" | cut -d, -f2 | sort -rn | head -1)
    m=$(echo "$r" | cut -d, -f3 | sort -rn | head -1)
    # per-GPU lines: "<index> <max util> <max mem>"
    echo "$r" | awk -F, 'NF==3 {i=$1+0; if ($2+0 > u[i]) u[i]=$2+0; if ($3+0 > m[i]) m[i]=$3+0; seen[i]=1}
                         END {for (i in seen) print i, u[i], m[i]}' | sort -n > "$TMP/g/$J"
    # pod_quick can answer with an ERROR STRING ("agent of <J> did not answer") rather than a number; anything
    # non-numeric must fall through to the agent/step path, not be compared as a utilisation (2026-09-17).
    case "$u" in ''|*[!0-9]*) u=NA;; esac
    case "$m" in ''|*[!0-9]*) m=NA;; esac
    echo "$u $m" > "$TMP/$J"
  ) &
done
wait

# Is this pod doing work? Used by BOTH the low-GPU branch and the no-reading branch, so "healthy" means the same
# thing in each: a running work step, or a step log of this job's written within the hour.
_has_work() {
  local j="$1" st fr
  [ -f "$TMP/w/$j" ] && return 0   # a work process seen inside the pod settles it outright
  st=$(timeout 45 sacct -j "$j" --format=JobName%40,State%12 --noheader 2>/dev/null \
       | awk '$2 == "RUNNING"' | grep -vE "batch|extern|^ *ka_|^ *cw_|podagent_|serveQ_" | head -1 | awk '{print $1}')
  [ -n "$st" ] && return 0
  fr=$(find -L "$P/logs/pods" -maxdepth 1 -name "*.log" -mmin -60 2>/dev/null \
       | xargs -r grep -l "job=$j" 2>/dev/null | head -1)
  [ -n "$fr" ]
}

idle=0; busy=0; answered=0; agentonly=0
for f in "$TMP"/*; do
  [ -e "$f" ] || continue
  [ -d "$f" ] && continue          # "$TMP/g" holds the per-GPU readings, one file per job -- not a pod
  J="$(basename "$f")"; read -r u m < "$f"
  if [ "$u" = NA ]; then
    # No GPU reading: neither srun nor the pod's agent ran the sampling command. A FRESH HEARTBEAT DOES NOT CLEAR
    # THIS. An agent can keep writing its heartbeat while refusing to execute anything -- 64356396 did exactly that
    # on 2026-09-18, heartbeat 0 minutes old, `pod_quick true` failing, its worker declining to claim any item --
    # and treating the heartbeat as proof of health is how 64356393 sat idle for two hours until Unity warned.
    # So fall through to the same evidence used for every other pod: is a work step running, or a step log fresh?
    hb="$P/logs/pods/agent/$J/heartbeat"
    age=-1
    [ -f "$hb" ] && age=$(( ( $(date +%s) - $(stat -c %Y "$hb") ) / 60 ))
    if [ "$age" -lt 0 ]; then
      echo "UNREACHABLE  $J  no srun, no agent heartbeat, and no GPU reading"; idle=$((idle+1)); continue
    fi
    if [ "$age" -gt 15 ]; then
      echo "STALE AGENT  $J  heartbeat ${age}m old and no GPU reading"; idle=$((idle+1)); continue
    fi
    if _has_work "$J"; then
      agentonly=$((agentonly+1))
    else
      echo "MUTE AGENT   $J  heartbeat ${age}m old but it did not run the sampling command, and no work step or"
      echo "                 step log in an hour -- the agent is alive and not executing; its worker will not claim"
      idle=$((idle+1))
    fi
    continue
  fi
  answered=$((answered+1))
  # Per-GPU orphan check, independent of whether the POD looks busy. A card holding a model's worth of memory at ~0%
  # for 45 s is either a server nobody owns any more or one still loading weights -- and loading finishes in minutes
  # while an orphan does not, so a suspect is only reported once it has survived TWO consecutive sweeps. Without that
  # patience two pods mid-startup were flagged the first time this ran.
  # A pod with a work process running in it is NOT orphaned, whatever its cards read. Without this gate the
  # per-GPU check fired on memory and utilisation alone and reported pods that were mid-run: 64460161 was flagged
  # twice while `rl/dispersion.py --family causal_rdd` was running inside it, because its vLLM server sits at 0%
  # between an agentic run's model calls. The in-pod check existed by then and was wired into only one branch --
  # this one never saw it.
  if [ -s "$TMP/g/$J" ] && ! _has_work "$J"; then
    while read -r gi gu gm; do
      [ -n "$gm" ] || continue
      if [ "$gm" -gt 40000 ] && [ "$gu" -lt 3 ]; then
        if grep -qx "$J:$gi" "$SEEN_PREV" 2>/dev/null; then
          echo "ORPHAN?      $J gpu$gi  ${gu}%, holding ${gm} MiB across two sweeps -- a server with no owner"
          echo "                 find it with --query-compute-apps=gpu_uuid,pid and kill that PID; the pod may look"
          echo "                 busy because its OTHER gpu is working, which is how two of these ran for hours"
          idle=$((idle+1))
        else
          echo "watching     $J gpu$gi  ${gu}%, holding ${gm} MiB (one sweep; could be weight loading)"
        fi
        echo "$J:$gi" >> "$SEEN_NOW"
      fi
    done < "$TMP/g/$J"
  fi
  if [ "$u" -ge "$MIN" ]; then busy=$((busy+1)); continue; fi
  # Is a work step actually running? This is what separates the two cases. 64445094's step never launched (its log
  # write was denied), so it had NO work step and was genuinely idle; 64356260 was mid-KramaBench with a step running
  # and simply CPU-bound. A step log alone is too coarse to tell them apart -- it gets one line per stage completed,
  # and benchmark stages are more than 20 minutes apart, so a healthy pod looks silent.
  # The in-pod work-process flag outranks both of the signals below; check it first so one definition of "busy"
  # applies on every path.
  if _has_work "$J"; then
    echo "low-GPU ok   $J  ${u}%, a work process is running inside the pod"
    continue
  fi
  step=$(timeout 45 sacct -j "$J" --format=JobName%40,State%12 --noheader 2>/dev/null \
         | awk '$2 == "RUNNING"' | grep -vE "batch|extern|^ *ka_|^ *cw_|podagent_|serveQ_" | head -1 | awk '{print $1}')
  # A pod that runs its work through scripts/pod_agent.sh shows only "podagent_<J>" to sacct, so the absence of an
  # srun step proves nothing there. Fall back to whether any step log of this pod's was written recently -- 60 minutes,
  # because a step log gets one line per STAGE, and a benchmark stage (QRData is 411 questions) runs far longer than 20.
  # -L because logs/pods is a SYMLINK to the scratch workspace, and find does not follow one by default: without
  # it this returns nothing at all and every low-GPU pod looks idle (2026-09-17; the same trap as results/pods).
  fresh=$(find -L "$P/logs/pods" -maxdepth 1 -name "*.log" -mmin -60 2>/dev/null \
          | xargs -r grep -l "job=$J" 2>/dev/null | head -1)
  if [ -n "$step" ]; then
    echo "low-GPU ok   $J  ${u}%, but ${step%.cmd.sh} is running (CPU-bound phase, e.g. a benchmark's code cells)"
  elif [ -n "$fresh" ]; then
    echo "low-GPU ok   $J  ${u}%, agent-run; $(basename "$fresh" .log) wrote within the hour"
  elif [ "$m" != NA ] && [ "$m" -gt 40000 ]; then
    # Idle AND still holding the card: a server the item that started it no longer owns. This is the 124 GB vLLM
    # process left behind when item 4k was killed on 64392460, which then failed every later item on that pod with
    # "Engine core initialization failed" -- a pod that looks merely idle but is actually poisoned for the next item.
    # It is deliberately NOT a check on memory alone: a healthy run holds just as much between model calls, and two
    # pods mid-run were flagged that way the first time this ran. Only the idle side of the step/log test qualifies.
    echo "ORPHAN?      $J  ${u}%, no work step and no step log in an hour, still holding ${m} MiB"
    echo "                 kill that pod's VLLM::EngineCore only -- leave the keepalive alone -- or the next item OOMs"
    idle=$((idle+1))
  else
    echo "IDLE         $J  ${u}%, no work step and no step log in an hour$([ "$m" != NA ] && echo ", ${m} MiB held")"
    tail -2 "${SCRATCHPAD:-$P/../scratchpad}/work_$J.log" 2>/dev/null | sed 's/^/                 /'
    idle=$((idle+1))
  fi
done

# The all-clear must never be reachable without evidence. Until this line existed, a scan in which NO pod answered
# -- every pod_quick timing out, a login node wedged, squeue returning nothing -- fell through the loop with idle=0
# and printed "every pod is computing": a clean bill of health from zero readings, which is the failure mode a
# monitor can least afford. Coverage is now reported on every run and a thin scan says so in words the watcher greps.
mv -f "$SEEN_NOW" "$SEEN_PREV" 2>/dev/null

echo
echo "coverage: $total pods in squeue, $answered answered with a GPU reading ($busy computing), $agentonly agent-only"
if [ "$total" -gt 0 ] && [ "$((answered + agentonly))" -lt "$(( (total + 1) / 2 ))" ]; then
  echo "SCAN BROKEN  only $((answered + agentonly)) of $total pods answered -- this run proves nothing about the rest"
fi
if [ "$idle" = 0 ]; then echo "every pod that answered is computing"; else
  echo "$idle pod(s) idle or stale. Free the worker so the queue hands it the next item: kill the item script."
  echo "Do NOT write or delete markers by hand -- requeue the item file instead."
fi
