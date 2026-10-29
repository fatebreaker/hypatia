#!/bin/bash
# Requeue queue items whose in-pod step FAILED but which the worker recorded as a success.
#
# THE BUG THIS EXISTS FOR (2026-09-20). pod_queue/_lib.sh's wait_marker() returns 0 whether the step wrote
# .done or .failed:
#     wait_marker() { until [ -e logs/pods/$1.done ] || [ -e logs/pods/$1.failed ]; do sleep 120; done; ... }
# so an item whose vLLM server died ("Engine core initialization failed") ends with `wait_marker` returning 0,
# pod_worker.sh logs "finished <item> (exit 0)", the item stays filed under started/, and the measurement is
# LOST SILENTLY. Four surv_tvc legs disappeared this way in one night (v16tvc4, v20tvc2, v21tvc2, v23tvc4)
# while the queue drained to empty -- work vanishing and GPUs going idle are the SAME fault.
#
# Requeues at most RETRY_MAX times per item; a fifth copy of a step that fails every time is a broken item,
# not a transient, and should be read rather than re-run.
set -uo pipefail
cd /project/pi_hongyu_umass_edu/hieutran/deep/deeprs
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
Q=$S/pod_queue; RETRY_MAX=${RETRY_MAX:-2}; n=0

# EXISTENCE IS THE WRONG TEST; READABILITY IS THE RIGHT ONE (2026-09-21). A result file with >=10% of its rollouts
# never reaching the server is refused by ga_table.score and is exactly as useless as a missing file -- but it
# EXISTS, so the per-arm check below counted it as done. general_agent_iv8_d2/selfver20_causal_iv.json sat that
# way with 69/240 dead, quietly dropping that whole draw out of every content-vs-placebo reading. Sweep the tree
# ONCE here and treat an unreadable file as absent. One python start-up for the whole run, not one per item.
UNREADABLE=$(venv/bin/python - <<'PYSCAN' 2>/dev/null
import sys, glob, os, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from ga_table import score
for f in sorted(glob.glob("results/pods/general_agent*/*.json")):
    if os.path.basename(f).startswith("."):
        continue
    try:
        score(f)
    except Exception:
        print(f)
PYSCAN
)
[ -n "$UNREADABLE" ] && echo "[requeue] $(printf "%s" "$UNREADABLE" | wc -l) result file(s) exist but are unreadable; treating as missing:" && printf "%s\n" "$UNREADABLE" | sed 's/^/[requeue]   /'
readable() { [ -f "$1" ] && ! printf "%s\n" "$UNREADABLE" | grep -qxF "$1"; }
for f in $Q/started/*.sh; do
  [ -e "$f" ] || continue
  b=$(basename "$f")                      # <job>_<item>.sh
  item=${b#*_}                            # <item>.sh
  base=$(printf "%s" "$item" | sed -E 's/\.r[0-9]+\.sh$/.sh/; s/\.sh$//')
  # How many retries of THIS item already exist? Counting only the suffix on the file in hand was wrong: the
  # started/ copy always reads as r0, so every sweep minted another ".r1" for an item that had already been
  # retried and had failed again -- an unbounded loop (2026-09-20, v16tvc4). Count every copy that exists
  # anywhere instead, which is what the cap is actually about.
  r=$(ls -d $Q/*/${base}.r*.sh $Q/*/*_${base}.r*.sh 2>/dev/null | wc -l)
  [ "$r" -ge "$RETRY_MAX" ] && continue
  # the step name this item waits on, and whether that step failed
  step=$(grep -oE 'wait_marker [A-Za-z0-9_]+\$JS' "$f" 2>/dev/null | tail -1 | sed 's/wait_marker //; s/\$JS//')
  [ -z "$step" ] && continue
  job=${b%%_*}
  m=""
  for cand in logs/pods/${step}${job}.failed logs/pods/${step}${job}g0.failed logs/pods/${step}${job}g1.failed; do
    [ -e "$cand" ] && m="$cand"
  done
  [ -z "$m" ] && continue
  # A failure the monitor did not requeue within hours is not a failure it should requeue now: retiring an
  # ancient unmarked orphan by hand-writing its .failed marker resurrected a two-day-old item (6a_v7_breadth,
  # 2026-09-20). Genuine failures are swept within 5 minutes; anything older than 6h is history.
  # Age of the STEP LOG, not the marker: a hand-written marker is minutes old on a step that died days ago.
  lg="${m%.failed}.log"; [ -e "$lg" ] || lg="$m"
  if [ $(( $(date +%s) - $(stat -c %Y "$lg") )) -gt 21600 ]; then continue; fi
  # Did it nevertheless produce results? Then it is not lost and must not re-run.
  #
  # "ANY json present" was too coarse (2026-09-21). With run.sh/run_model.sh now exiting 4 the moment their vLLM
  # server dies, the COMMON failure is a sweep that finished two arms of four: three files exist, the item is
  # .failed, and the old rule skipped it forever -- losing exactly the arms the guard was added to protect. Check
  # every expected <arm>_<family>.json instead; the re-run's own SKIP guard then measures only what is missing,
  # and an item that turns out to be complete exits 3 (NOOP) harmlessly.
  od=$(grep -oE 'OUT_DIR=[A-Za-z0-9_]+' "$f" | head -1 | cut -d= -f2)
  if [ -n "$od" ]; then
    arms=$(grep -oE 'ARMS="[^"]+"' "$f" | head -1 | sed 's/^ARMS="//; s/"$//')
    fams=$(grep -oE 'FAMS="[^"]+"' "$f" | head -1 | sed 's/^FAMS="//; s/"$//')
    if [ -n "$arms" ] && [ -n "$fams" ]; then
      missing=0
      for a_ in $arms; do
        for fm_ in $fams; do
          readable "results/pods/$od/${a_%%:*}_$fm_.json" || missing=1
        done
      done
      [ "$missing" = 0 ] && continue
    elif [ -n "$(ls results/pods/$od/*.json 2>/dev/null)" ]; then
      continue                      # arms not enumerable from the script: fall back to the old all-or-nothing rule
    fi
  fi
  # Already sitting in a lane waiting to re-run? NOTE: `ls a b c` exits non-zero when ANY argument is missing,
  # even when another matched -- so the obvious three-glob test almost never fired and the sweeper minted a
  # second copy of items it had just requeued (2026-09-20). Count matches instead of trusting ls's exit code.
  if [ "$(ls -d $Q/single/${base}*.sh $Q/double/${base}*.sh $Q/h200/${base}*.sh 2>/dev/null | wc -l)" -gt 0 ]; then continue; fi
  # Or already CLAIMED and in flight on some pod: a retry that is sitting in _lib.sh's gpu_wait has not failed,
  # it is queued behind a live run. Requeuing on the strength of the PREVIOUS attempt's .failed marker sends the
  # same measurement to two pods, which then collide on the OUT_DIR skip guard (2026-09-20, 301_head14_a).
  live=0
  for other in $Q/started/*_${base}*.sh; do
    [ -e "$other" ] || continue
    ojob=$(basename "$other"); ojob=${ojob%%_*}
    ostep=$(grep -oE 'wait_marker [A-Za-z0-9_]+\$JS' "$other" 2>/dev/null | tail -1 | sed 's/wait_marker //; s/\$JS//')
    [ -z "$ostep" ] && continue
    # A started copy with NO marker of any kind is in flight. Requiring a step .log here was wrong: an item
    # blocked in _lib.sh's gpu_wait has not reached pod_exec.sh yet, so it has no log, and the sweeper read it
    # as absent and dispatched the same measurement a second time (2026-09-20).
    marked=0
    for g in "" g0 g1; do
      [ -e "logs/pods/${ostep}${ojob}${g}.done" ] && marked=1
      [ -e "logs/pods/${ostep}${ojob}${g}.failed" ] && marked=1
    done
    [ "$marked" = 0 ] && live=1
  done
  [ "$live" = 1 ] && continue
  lane=single; grep -q "qwen32\|Qwen3-32B" "$f" && lane=h200
  new="$Q/$lane/${base}.r$((r+1)).sh"
  # DRY=1 prints what would be requeued and writes nothing -- the way to check a change to the rules above
  # against the live started/ tree without minting copies.
  if [ -n "${DRY:-}" ]; then
    # DRY must model the side effect it is skipping, or it over-reports: the retry count `r` is read from the lane
    # on every iteration, so a real run that just wrote .r2 sees r=2 on the next copy of the same item and stops.
    # Without this the dry run claims two requeues where a real one makes one file.
    case " ${dry_done:-} " in *" $base "*) continue;; esac
    dry_done="${dry_done:-} $base"
    n=$((n+1)); echo "WOULD REQUEUE $b -> $(basename $new)  (step $step$job failed)"; continue
  fi
  cp "$f" "$new.tmp" && mv "$new.tmp" "$new" && { n=$((n+1)); echo "REQUEUED $b -> $(basename $new)  (step $step$job failed)"; }
done
echo "requeued=$n"
