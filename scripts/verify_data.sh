#!/bin/bash
# Check that the data actually landed, with expected counts.
#
# This exists because three separate download jobs in this project "succeeded"
# while fetching nothing -- EBI's robots.txt silently defeated `wget -r`, and
# Open Targets' Hive-partitioned evidence left five empty directories behind.
# Both exited 0. A size check would have caught either instantly.
#
# Exit 0 = everything present. Exit 1 = something is missing or suspiciously
# small; the report says which.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P"
# shellcheck source=/dev/null
source scripts/env.sh

fail=0
ok()   { printf "  \033[32mOK\033[0m   %-34s %s\n" "$1" "$2"; }
bad()  { printf "  \033[31mMISS\033[0m %-34s %s\n" "$1" "$2"; fail=1; }
warn() { printf "  \033[33mWARN\033[0m %-34s %s\n" "$1" "$2"; }

# check_count <label> <glob> <min-expected> <note>
check_count() {
  local label="$1" pat="$2" min="$3" note="${4:-}"
  local n; n=$(eval "ls -d $pat 2>/dev/null | wc -l")
  if [ "$n" -ge "$min" ]; then ok "$label" "$n (>= $min) $note"
  else bad "$label" "$n found, expected >= $min $note"; fi
}

# check_size <label> <path> <min-MB>
check_size() {
  local label="$1" path="$2" min="$3"
  if [ ! -e "$path" ]; then bad "$label" "absent"; return; fi
  local mb; mb=$(du -sm "$path" 2>/dev/null | cut -f1)
  if [ "${mb:-0}" -ge "$min" ]; then ok "$label" "${mb} MB (>= ${min})"
  else bad "$label" "${mb} MB, expected >= ${min} MB"; fi
}

echo "data root: $BASE"
echo
echo "Open Targets releases"
for r in 21.06 22.06 23.06 24.06 25.06 26.06; do
  # Evidence is the dataset most likely to be silently empty: it is
  # Hive-partitioned, so a naive glob creates dirs without files.
  d="$BASE/opentargets/$r"
  if [ ! -d "$d" ]; then bad "  $r" "release dir absent"; continue; fi
  np=$(find "$d" -name "*.parquet" 2>/dev/null | wc -l)
  if [ "$r" = "26.06" ]; then
    ne=$(ls -d "$d"/evidence_* 2>/dev/null | wc -l)
  else
    ne=$(ls -d "$d"/evidence/sourceId=* 2>/dev/null | wc -l)
  fi
  eb=$(find "$d" -path "*evidence*" -name "*.parquet" 2>/dev/null | wc -l)
  if [ "$np" -lt 50 ]; then bad "  $r" "$np parquet files -- too few"
  elif [ "$ne" -lt 15 ]; then bad "  $r" "$ne evidence sources (expect >=15); Hive partitions missing?"
  elif [ "$eb" -lt 15 ]; then bad "  $r" "evidence dirs exist but only $eb parquet inside -- EMPTY PARTITIONS"
  else ok "  $r" "$np parquet, $ne evidence sources"; fi
done

echo
echo "other sources"
check_size "chembl"          "$BASE/chembl"                  4000
check_size "uniprot"         "$BASE/uniprot"                 7000
check_size "clinvar"         "$BASE/clinvar"                  400
check_size "gwas_catalog"    "$BASE/gwas_catalog"             100
check_size "gnomad constraint" "$BASE/gnomad"                  20
check_size "impc"            "$BASE/impc"                     300
check_size "mgi"             "$BASE/mgi"                       10
check_count "aact snapshots" "$BASE/aact/*.zip"                 5 "yearly trial registry"
check_count "pubmed baseline" "$BASE/pubmed/baseline/*.xml.gz" 1300 "expect 1334"

echo
echo "optional / large"
if [ -d "$BASE/pmc/txt" ]; then
  n=$(find "$BASE/pmc/txt" -name "*.txt" 2>/dev/null | wc -l)
  if [ "$n" -ge 1800000 ]; then ok "pmc full text" "$n files"
  else warn "pmc full text" "$n files (expect ~1,862,506; still fetching?)"; fi
else
  warn "pmc full text" "absent -- fine if you ran --no-pmc"
fi

echo
echo "derived artefacts"
check_count "normalized evidence" "$BASE/normalized/evidence_*.parquet" 6 "one per release"
check_count "generated train"     "$BASE/generated/cis_mvmr_1k/train/cis_mvmr_*" 700
check_count "generated val"       "$BASE/generated/cis_mvmr_1k/val/cis_mvmr_*"    90
check_count "generated test"      "$BASE/generated/cis_mvmr_1k/test/cis_mvmr_*"   90
if [ -f "$BASE/sft/cis_mvmr_train.jsonl" ]; then
  n=$(wc -l < "$BASE/sft/cis_mvmr_train.jsonl")
  ok "sft traces" "$n (audit with: rl/sft_traces.py --audit <jsonl>)"
else
  warn "sft traces" "absent"
fi
check_count "qrdata problems"   "$P/benchmarks/qrdata/problems/*" 411 \
  "venv/bin/python scripts/fetch_qrdata.py"
check_count "benchmark problems" "$P/benchmarks/genebench-pro-public/problems/*" 10 \
  "run scripts/fetch_benchmarks.sh"

echo
if command -v ws_list >/dev/null 2>&1; then
  d=$(ws_list 2>/dev/null | awk -v n="id: ${WORKSPACE_NAME:-deep-ai4s}" \
        '$0 ~ n {f=1} f && /remaining time/ {print; exit}')
  [ -n "$d" ] && echo "workspace ${d#*: }"
fi
echo "total: $(du -sh "$BASE" 2>/dev/null | cut -f1)"
echo
if [ "$fail" -eq 0 ]; then echo "verify: PASS"; else
  echo "verify: FAIL -- re-run the relevant sbatch; all fetchers are resumable."; fi
exit $fail
