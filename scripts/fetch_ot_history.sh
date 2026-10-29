#!/bin/bash
# Phase B: historical Open Targets releases -- the backtest time machine.
#
# Layout eras (verified 2026-07-28):
#   <=20.11      output/es_snapshot/            Elasticsearch binary. NOT fetched.
#   21.02        transitional (ETL + ETL_parquet + es_snapshot)
#   21.04-24.09  output/etl/parquet/<camelCase>/
#   25.03-26.06  output/<snake_case>/
#
# So parquet-native coverage runs 21.04 -> 26.06: ~5.4 years of annual
# checkpoints. Anything earlier would require restoring an ES snapshot.
#
# knownDrugsAggregated is the label source: it records clinical phase per
# drug-target-disease AS OF that release, which is what makes a leak-free
# time split possible.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
mkdir -p $LOG
exec > >(tee -a $LOG/fetch_ot_history.log) 2>&1

echo "=================================================="
echo "Phase B start: $(date)"
echo "=================================================="

W="wget -c -nv --tries=5 --timeout=60 --waitretry=10"
ROOT=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform

fetch_dir () {
  local url="$1" dest="$2"
  mkdir -p "$dest"
  local files
  files=$(curl -sL --max-time 120 "$url" \
          | grep -oE 'href="[^"]*\.parquet"' | sed 's/href="//;s/"$//')
  if [ -z "$files" ]; then echo "    WARN: no parquet at $url"; return 1; fi
  local n=0
  for f in $files; do
    $W -P "$dest" "${url}${f}" || echo "    WARN: ${url}${f}"
    n=$((n+1))
  done
  echo "    -> $n files"
}

# camelCase era: annual checkpoints
ETL_DATASETS="evidence targets diseases knownDrugsAggregated indication \
mechanismOfAction molecule drugWarnings associationByDatasourceDirect \
associationByOverallDirect mousePhenotypes diseaseToPhenotype interaction"

for rel in 21.06 22.06 23.06 24.06; do
  echo "--- Open Targets $rel (etl/parquet era)"
  for d in $ETL_DATASETS; do
    echo "  $rel/$d"
    fetch_dir "$ROOT/$rel/output/etl/parquet/$d/" "$BASE/opentargets/$rel/$d"
  done
done

# snake_case era: 25.06 (26.06 already pulled in Phase A)
SNAKE_DATASETS="evidence_gwas_credible_sets evidence_gene_burden evidence_impc \
evidence_europepmc evidence_eva evidence_clingen evidence_clinical_precedence \
target disease drug_molecule drug_mechanism_of_action drug_warning \
clinical_indication clinical_target l2g_prediction \
association_overall_direct association_by_datasource_direct mouse_phenotype"

for rel in 25.06; do
  echo "--- Open Targets $rel (snake_case era)"
  for d in $SNAKE_DATASETS; do
    echo "  $rel/$d"
    fetch_dir "$ROOT/$rel/output/$d/" "$BASE/opentargets/$rel/$d"
  done
done

echo "=================================================="
echo "Phase B complete: $(date)"
du -sh $BASE/opentargets/*/ 2>/dev/null
echo "=================================================="
