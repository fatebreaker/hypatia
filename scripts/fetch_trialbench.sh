#!/bin/bash
# TrialBench (Zenodo 15455785, CC-BY-4.0) -- the external eval axis for arm A.
#
# fetch_core.sh already pulls the whole Zenodo files-archive as one opaque zip.
# This fetches the per-task archives and UNPACKS them, which is what
# analysis/trialbench.py actually reads. Both can coexist; this is resumable
# and skips anything already unpacked.
#
# 9 task archives, ~875 MB total. The two that bear on target quality are
# fetched by default:
#   trial-approval-forecasting          did the trial succeed
#   trial-failure-reason-identification why it stopped
# Pass --all for the other six (dropout, adverse events, duration, mortality,
# dose, eligibility design).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$LOG" "$BASE/trialbench"
exec > >(tee -a "$LOG/fetch_trialbench.log") 2>&1

REC=15455785
CORE="trial-approval-forecasting trial-failure-reason-identification"
EXTRA="patient-dropout-event-forecasting serious-adverse-event-forecasting \
trial-duration-forecasting mortality-event-prediction drug-dose-prediction \
eligibility-criteria-design"

TASKS="$CORE"
[ "${1:-}" = "--all" ] && TASKS="$CORE $EXTRA"

echo "=================================================="
echo "TrialBench start: $(date)"
echo "=================================================="

for t in $TASKS; do
  if [ -d "$BASE/trialbench/$t" ]; then
    echo "--- $t already unpacked, skipping"
    continue
  fi
  echo "--- $t"
  z="$BASE/trialbench/$t.zip"
  wget -c -nv --tries=5 --timeout=60 --waitretry=10 \
       -O "$z" "https://zenodo.org/records/$REC/files/$t.zip?download=1" \
    || { echo "  WARN: download failed for $t"; continue; }
  unzip -q -o "$z" -d "$BASE/trialbench" || { echo "  WARN: unzip failed for $t"; continue; }
  echo "    -> $(find "$BASE/trialbench/$t" -name '*.csv' | wc -l) csv files"
done

echo
echo "=== coverage ==="
fail=0
for t in $CORE; do
  n=$(find "$BASE/trialbench/$t" -name "*_x.csv" 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then
    echo "  OK   $t  ($n feature files)"
  else
    echo "  MISS $t  ($n feature files, expected >= 4)"; fail=1
  fi
done
echo "total: $(du -sh "$BASE/trialbench" 2>/dev/null | cut -f1)"
echo "=================================================="
echo "TrialBench complete: $(date)"
[ "$fail" -eq 0 ] || echo "FAIL -- rerun; downloads are resumable"
exit $fail
