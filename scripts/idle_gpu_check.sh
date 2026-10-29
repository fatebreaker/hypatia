#!/bin/bash
# Which of OUR pods are burning a GPU allocation without doing GPU work?
#
# This is the failure Unity emails about (2026-09-18: two pods flagged, "killed after two hours of no GPU compute
# activity"). The tell is the pod's newest step: a `ka_*` keepalive step, or a step that already wrote its .done /
# .failed marker, means nothing is computing. The gpu_keepalive does NOT satisfy Unity's detector, so a pod sitting
# on one is idle in the only sense that matters.
#
# EXCLUDED, because they are idle-but-not-ours-to-fill:
#   - cw_l40s / cw_pod14 : D's pods. D runs flowgauge_rl on them; we take one only when D is not using it.
#   - 64158117           : released to SESSION F (2026-09-18). Must stay empty.
set -uo pipefail
cd /project/pi_hongyu_umass_edu/hieutran/deep/deeprs
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
# pods released to SESSION F (2026-09-18/19): one H200, two double-A100s. Idle there is correct, not a fault.
FREE_PODS="64158117 64356260 64356261 64460161"   # SESSION F (user, 2026-09-21: "leave h100 for F, and their h200 pod, the rest of the fleet are yours"): 3 H100s on uri-gpu009 + the H200 64158117. 63648614/615 dropped -- those jobs ended at walltime.
# ACCEPTED_IDLE: pods idle by explicit decision, not by fault -- alerting on them is noise. 2026-09-19:
#   63648609 -- the 8B work of the programme is exhausted; it loses every claim race to faster-polling
#   workers; ~11h walltime left; keepalives running per policy. If Unity reaps it we lose nothing usable.
# ACCEPTED_IDLE: pods deliberately left idle. DANGER: this list goes STALE the moment the premise changes.
# On 2026-09-19 I listed four pods here as "the 8B programme is complete", then found more 8B work, refed two of
# them, and left the entries in place -- so the checker silently excluded 64356387/64445170 while they idled for
# 3.5h and Unity flagged them. A pod belongs here ONLY if it is near walltime end AND has no possible work.
# Prefer an empty list: if a pod is idle and alive, the honest answer is to find it work or admit there is none.
ACCEPTED_IDLE=""
now=$(date +%s); idle=0; busy=0; out=""
while IFS='|' read -r j nm; do
  [ -z "$j" ] && continue
  case " $FREE_PODS " in *" $j "*) continue;; esac
  case " $ACCEPTED_IDLE " in *" $j "*) continue;; esac
  case "$nm" in cw_l40s|cw_pod14) continue;; esac
  # keepalives now run on every non-h200 GPU by policy (user, 2026-09-19), so a ka_* log is no longer news:
  # judge the pod by its newest NON-keepalive step, and only call it "keepalive only" when that is all it has.
  # podagent_* is the in-pod AGENT, not work -- counting its log as "the current step" made two pods look busy
  # for 2.6h while Unity (correctly) flagged them idle (2026-09-19, 64356387/64356392). Exclude it like ka_*.
  # INFRASTRUCTURE steps are never evidence of work and must not be picked as "the current step":
  #   ka_*       keepalive (2026-09-19)
  #   podagent_* the in-pod command runner (2026-09-19: made 3 idle pods look busy for 2.6h)
  #   serve*     a vLLM server; items launch one per work step and cancel it with CLEARED markers, so a dead
  #              server looks like a running step forever (2026-09-19: missed 64356393/64443616 for ~2.5h,
  #              Unity flagged both before this check did -- the third variant of this same bug).
  # A pod can have SEVERAL non-infra steps, and the newest is not necessarily the live one: a short step that
  # failed fast (e.g. an engine-init collision) sits on top of a long step still running underneath, and picking
  # only the newest reported three working pods as idle (2026-09-20). Prefer ANY step that has no marker yet.
  f=""
  for cand in $(grep -l "job=$j" logs/pods/*.log 2>/dev/null | grep -vE "/(ka_|podagent_|serve)" | xargs -r ls -t 2>/dev/null); do
    cn=$(basename "$cand" .log)
    if [ ! -f "logs/pods/$cn.done" ] && [ ! -f "logs/pods/$cn.failed" ]; then f="$cand"; break; fi
  done
  # none unmarked -> fall back to the newest, which will read as "finished" and count the pod idle
  [ -z "$f" ] && f=$(grep -l "job=$j" logs/pods/*.log 2>/dev/null | grep -vE "/(ka_|podagent_|serve)" | xargs -r ls -t 2>/dev/null | head -1)
  [ -z "$f" ] && f=$(grep -l "job=$j" logs/pods/*.log 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  if [ -z "$f" ]; then out="$out\n  IDLE $j $nm (no step log at all)"; idle=$((idle+1)); continue; fi
  n=$(basename "$f" .log); age=$(( (now - $(stat -c %Y "$f" 2>/dev/null || echo "$now")) / 60 ))
  case "$n" in
    ka_*) out="$out\n  IDLE $j $nm (keepalive only, ${age}m)"; idle=$((idle+1)); continue;;
  esac
  if [ -f "logs/pods/$n.done" ] || [ -f "logs/pods/$n.failed" ]; then
    out="$out\n  IDLE $j $nm (last step $n finished, ${age}m ago)"; idle=$((idle+1))
  elif [ "$age" -ge "${STALE_MIN:-45}" ]; then
    # No marker and the step log is quiet -- but that is NOT evidence of idleness: a running step writes its
    # progress to results/pods/<OUT_DIR>/*.log, not to its pod_exec log, so healthy 32B runs look "stale" for
    # hours (2026-09-19: this rule flagged three pods that were at 100% GPU with 71-130GB loaded). So VERIFY
    # the suspects -- and only the suspects, to keep the sweep cheap -- with an actual GPU reading.
    # 75s and one retry: the probe goes through the pod agent, which is slow to answer while it is busy
    # serving a 32B model -- a 45s single attempt reported UNKNOWN for a pod verified at 100% GPU (2026-09-19).
    probe=$(timeout 75 scripts/pod_quick.sh "$j" nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$probe" ] && probe=$(timeout 75 scripts/pod_quick.sh "$j" nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1)
    u=$(echo "$probe" | cut -d, -f1 | tr -d ' '); m=$(echo "$probe" | cut -d, -f2 | tr -d ' ')
    # A probe that answers with TEXT is a failed probe. On 2026-09-20 two pods that had started refusing srun
    # ("Job/step already completing or completed") returned that error string; the checker parsed it as a
    # utilization reading, the numeric test failed, and it reported two pods at 100%/71GB as IDLE.
    [[ "$u" =~ ^[0-9]+$ ]] || u=""
    if [ -z "$u" ]; then
      # pod_quick.sh (srun, then the agent's quick lane) answered nothing. Before giving up, ask through the SLOW agent
      # lane that scripts/pod_exec.sh uses -- it is the door that keeps working when srun says "Job/step already
      # completing" (2026-09-20: 64445170 was at 100%/71GB running a step, srun refused, the quick lane answered
      # nothing, and a marker-age heuristic then declared it idle. That heuristic was WRONG -- a running step has no
      # marker by design -- and has been removed.)
      rm -f logs/pods/idleprobe_$j.done logs/pods/idleprobe_$j.failed logs/pods/idleprobe_$j.log
      scripts/pod_exec.sh "$j" 1 idleprobe_$j -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >/dev/null 2>&1 < /dev/null
      for _ in $(seq 1 12); do [ -e logs/pods/idleprobe_$j.done ] && break; sleep 5; done
      probe=$(grep -vE '^\[pod_exec\]' logs/pods/idleprobe_$j.log 2>/dev/null | head -1)
      u=$(echo "$probe" | cut -d, -f1 | tr -d ' '); m=$(echo "$probe" | cut -d, -f2 | tr -d ' ')
      [[ "$u" =~ ^[0-9]+$ ]] || u=""
    fi
    if [ -z "$u" ]; then
      out="$out\n  UNKNOWN $j $nm (step $n quiet ${age}m; no GPU answer after 2 probes + agent -- NOT counted idle)"
    elif [ "$u" -ge 5 ] 2>/dev/null || [ "${m:-0}" -ge 5000 ] 2>/dev/null; then
      busy=$((busy+1))
    else
      out="$out\n  IDLE $j $nm (step $n quiet ${age}m AND gpu ${u}%/${m}MiB)"; idle=$((idle+1))
    fi
  else
    busy=$((busy+1))
  fi
done < <(squeue -u hieutran_umass_edu -h -o "%i|%T|%b|%j" 2>/dev/null | awk -F'|' '$2=="RUNNING" && tolower($3)~/gpu/{print $1"|"$4}')
q=$(ls "$S"/pod_queue/single "$S"/pod_queue/h200 "$S"/pod_queue/double 2>/dev/null | grep -c '\.sh$')
# ALERT mode (--alert): print only what needs acting on, so this can drive a monitor without spamming.
#   * a pod idle >= ALERT_MIN minutes  -> Unity kills at 120m of no GPU compute, so warn well before
#   * queue below ALERT_Q              -> ~30 GPUs chew through items in 1-3h; 10 is under an hour of work
if [ "${1:-}" = "--alert" ]; then
  AM=${ALERT_MIN:-75}; AQ=${ALERT_Q:-8}
  # portable: 3-arg match() is a gawk extension and fails silently elsewhere (it printed 17m rows at a 75m
  # threshold on first test). Extract the minutes with sed and compare in the shell instead.
  hot=""
  while IFS= read -r ln; do
    [ -z "$ln" ] && continue
    # UNKNOWN lines are explicitly "NOT counted idle" -- they must not raise an alert either. The minute-extractor
    # below matched the "quiet NNm" inside them and fired on a pod that was merely slow to probe (2026-09-20).
    case "$ln" in *"  UNKNOWN "*) continue;; esac
    mins=$(printf "%s" "$ln" | sed -n 's/.*, \([0-9]\{1,\}\)m).*/\1/p')
    [ -z "$mins" ] && mins=$(printf "%s" "$ln" | sed -n 's/.*[^0-9]\([0-9]\{1,\}\)m.*/\1/p')
    [ -n "$mins" ] && [ "$mins" -ge "$AM" ] 2>/dev/null && hot="$hot\n$ln"
  done <<EOT
$(printf "%b" "$out")
EOT
  [ -n "$hot" ] && printf "IDLE GPU >=%sm (Unity kills at 120m):%b\n" "$AM" "\n$hot"
  [ "$q" -lt "$AQ" ] && echo "QUEUE LOW: $q items for $((busy+idle)) pods -- refill to 25+"
  exit 0
fi
echo "busy=$busy idle=$idle queue=$q"
[ "$idle" -gt 0 ] && printf "%b\n" "$out"
exit 0
