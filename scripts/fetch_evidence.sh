#!/bin/bash
# Phase D: the `evidence` dataset for every historical release.
#
# WHY THIS IS A SEPARATE PHASE: Open Targets stores `evidence` Hive-partitioned
# as evidence/sourceId=<datasource>/*.parquet. Phase B's fetcher only globbed
# *.parquet at the top level, so it silently created empty dirs (512 b) for the
# single most important dataset. This descends into the sourceId= partitions.
#
# The partitioning is a gift: evidence arrives pre-split by datasource
# (chembl, europepmc, eva, gene_burden, gwas_credible_sets, impc, clingen,
# crispr, ...), which is exactly the provenance separation the adjudication
# task needs. Keep the partition structure on disk -- do not flatten.
#
# Layout eras:
#   21.04-24.09  output/etl/parquet/evidence/sourceId=*/
#   25.03-25.12  output/evidence/sourceId=*/
#   26.03+       already split into top-level evidence_<source>/ dirs (Phase A)

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
mkdir -p $LOG
exec > >(tee -a $LOG/fetch_evidence.log) 2>&1

echo "=================================================="
echo "Phase D start: $(date)"
echo "=================================================="

W="wget -c -nv --tries=5 --timeout=60 --waitretry=10"
ROOT=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform

# Download all *.parquet directly under a URL.
grab_parquet () {
  local url="$1" dest="$2"
  mkdir -p "$dest"
  local files n=0
  files=$(curl -sL --max-time 180 "$url" | grep -oE 'href="[^"]*\.parquet"' | sed 's/href="//;s/"$//')
  [ -z "$files" ] && return 1
  for f in $files; do
    $W -P "$dest" "${url}${f}" || echo "      WARN: ${url}${f}"
    n=$((n+1))
  done
  echo "      -> $n parquet"
}

# Walk evidence/sourceId=*/ partitions.
fetch_evidence () {
  local url="$1" dest="$2" label="$3"
  echo "--- $label"
  local parts
  parts=$(curl -sL --max-time 180 "$url" \
          | grep -oE 'href="sourceId=[^"]*/"' | sed 's/href="//;s/"$//')
  if [ -z "$parts" ]; then echo "    WARN: no sourceId= partitions at $url"; return 1; fi
  for p in $parts; do
    echo "    $p"
    grab_parquet "${url}${p}" "${dest}/${p%/}" || echo "      WARN: empty $p"
  done
}

# ---- etl/parquet era ----
for rel in 21.06 22.06 23.06 24.06; do
  fetch_evidence "$ROOT/$rel/output/etl/parquet/evidence/" \
                 "$BASE/opentargets/$rel/evidence" "OT $rel evidence"
done

# ---- 25.06 (unified evidence + differently-named drug/clinical datasets) ----
fetch_evidence "$ROOT/25.06/output/evidence/" \
               "$BASE/opentargets/25.06/evidence" "OT 25.06 evidence"

# 25.06 names these known_drug / drug_indication, not clinical_* as in 26.06.
for d in known_drug drug_indication literature expression; do
  echo "--- OT 25.06/$d"
  grab_parquet "$ROOT/25.06/output/$d/" "$BASE/opentargets/25.06/$d" \
    || echo "    WARN: 25.06/$d missing"
done

echo "=================================================="
echo "Phase D complete: $(date)"
for r in 21.06 22.06 23.06 24.06 25.06; do
  printf "%-7s evidence=%s\n" "$r" "$(du -sh $BASE/opentargets/$r/evidence 2>/dev/null|cut -f1)"
done
echo "TOTAL: $(du -sh $BASE 2>/dev/null|cut -f1)"
echo "=================================================="
