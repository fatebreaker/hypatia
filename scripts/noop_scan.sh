#!/bin/bash
# Which finished steps did not actually do anything? A step that skips its work still exits 0, still writes a .done
# marker, and still makes the worker log "finished ... (exit 0)". Nothing distinguishes it from real work except how
# long it took.
#
#   scripts/noop_scan.sh [MAX_MINUTES] [HOURS_BACK]     default: flag steps under 5 min, looking back 12 h
#
# Three of these on 2026-09-17/18, each a measurement that silently did not happen:
#   3x   experiments/general_agent/run.sh skips an arm whose output file exists -- a surv_cox re-run at a different n
#        would have found the 24x4 files and exited (caught before it ran; OUT_DIR added to run.sh).
#   4a   probed "$CK/final" then "$CK" for an adapter that lives at the top level, gave up, exited 0 in 64 seconds.
#   52   passed OUT_DIR to run_model.sh, which hardcoded its output directory and ignored it -- found the first
#        sample's files, skipped both arms, done in two minutes (OUT_DIR added there too).
#
# The tell is always DURATION, never the logs. Anything that trains, serves a model or scores 96+ rollouts takes
# longer than a few minutes; boot/stage/keepalive steps are the legitimate exceptions and are skipped by name.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$P"
MAXMIN="${1:-5}"; HOURS="${2:-12}"
SKIP_RE='^(boot_|stage_|ka_|probe_|gs_|pq_|pw|px|pz|podagent_|serveQ_|agenttest)'

echo "steps that finished in under ${MAXMIN} min in the last ${HOURS} h (a skip looks exactly like success):"
found=0
# -L: logs/pods is a SYMLINK to the scratch workspace and find does not follow one by default
while IFS= read -r m; do
  name="$(basename "$m" .done)"
  [[ "$name" =~ $SKIP_RE ]] && continue
  log="$P/logs/pods/$name.log"
  [ -f "$log" ] || continue
  # pod_exec brackets every step: "[pod_exec] <iso> job=..." at the start, "[pod_exec] <iso> exit N" at the end
  s=$(grep -m1 -oE '^\[pod_exec\] [0-9T:+-]+' "$log" 2>/dev/null | awk '{print $2}')
  e=$(grep -oE '^\[pod_exec\] [0-9T:+-]+ exit' "$log" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$s" ] && [ -n "$e" ] || continue
  ss=$(date -d "$s" +%s 2>/dev/null) || continue
  ee=$(date -d "$e" +%s 2>/dev/null) || continue
  mins=$(( (ee - ss) / 60 ))
  if [ "$mins" -lt "$MAXMIN" ]; then
    echo "  ${mins}m  $name"
    # the most useful next line is whatever the step said just before exiting
    grep -vE '^\[pod_exec\]' "$log" 2>/dev/null | tail -1 | sed 's/^/          /' | cut -c1-140
    found=$((found+1))
  fi
done < <(find -L "$P/logs/pods" -maxdepth 1 -name '*.done' -mmin "-$((HOURS*60))" 2>/dev/null)

echo
if [ "$found" = 0 ]; then echo "nothing suspiciously fast"; else
  echo "$found step(s) to check. A fast finish is not proof of a no-op -- a resumed run or an already-complete"
  echo "arm is legitimately quick -- but confirm each produced the result files it was supposed to."
fi
