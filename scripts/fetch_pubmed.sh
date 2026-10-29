#!/bin/bash
# Phase C: PubMed 2026 baseline + update files (~30 GB).
#
# This is the literature layer. Every citation carries an immutable EDAT
# (Entrez Date), which is what lets us reconstruct "literature as of year T"
# without leaking future publications into a frozen evidence horizon.
#
# NOTE: full-text PMC is deliberately NOT fetched here. The PMC S3 bucket is
# keyed per-article-version (millions of objects, no bulk tarballs), so a full
# mirror is a multi-million-request crawl. Abstracts + Open Targets'
# evidence_europepmc cover most of what the agent needs; revisit full text only
# if abstract-level retrieval proves insufficient.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
mkdir -p $LOG $BASE/pubmed/baseline $BASE/pubmed/updatefiles
exec > >(tee -a $LOG/fetch_pubmed.log) 2>&1

echo "=================================================="
echo "Phase C start: $(date)"
echo "=================================================="

W="wget -c -nv --tries=5 --timeout=60 --waitretry=10"

for sub in baseline updatefiles; do
  url="https://ftp.ncbi.nlm.nih.gov/pubmed/$sub/"
  echo "--- PubMed $sub"
  files=$(curl -sL --max-time 180 "$url" | grep -oE 'pubmed26n[0-9]+\.xml\.gz' | sort -u)
  n=0
  for f in $files; do
    $W -P "$BASE/pubmed/$sub" "${url}${f}" || echo "  WARN: $f"
    n=$((n+1))
    [ $((n % 100)) -eq 0 ] && echo "  ... $n files done"
  done
  echo "  -> $n files in $sub"
  # checksums for integrity verification
  $W -P "$BASE/pubmed/$sub" "${url}" -O /dev/null 2>/dev/null
done

echo "=================================================="
echo "Phase C complete: $(date)"
du -sh $BASE/pubmed/*/ 2>/dev/null
echo "=================================================="
