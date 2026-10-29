#!/bin/bash
# Phase E: UniProt (release 2026_02, Swiss-Prot 10-Jun-2026).
#
# Swiss-Prot only -- TrEMBL is ~100x larger and unreviewed. Swiss-Prot is the
# curated set where every functional annotation carries an evidence code and a
# literature citation, which is precisely the material the adjudication task
# needs (an assertion you can trace to a source and grade).
#
# idmapping_selected (7 GB) is the join table: UniProt <-> Ensembl <-> GeneID
# <-> PDB <-> PubMed. Needed to link OT targets (Ensembl gene IDs) to protein
# annotations.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
mkdir -p $LOG $BASE/uniprot
exec > >(tee -a $LOG/fetch_uniprot.log) 2>&1

echo "=================================================="
echo "Phase E (UniProt) start: $(date)"
echo "=================================================="

W="wget -c -nv --tries=5 --timeout=60 --waitretry=10"
U=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase

$W -P $BASE/uniprot "$U/complete/reldate.txt"

# Swiss-Prot: .dat is the richest (full annotation + evidence codes),
# .xml is easier to parse programmatically, .fasta for sequence work.
for f in uniprot_sprot.dat.gz uniprot_sprot.xml.gz uniprot_sprot.fasta.gz; do
  echo "--- $f"
  $W -P $BASE/uniprot "$U/complete/$f" || echo "  WARN: $f"
done

echo "--- idmapping_selected.tab.gz (7 GB)"
$W -P $BASE/uniprot "$U/idmapping/idmapping_selected.tab.gz" || echo "  WARN: idmapping"

echo "=================================================="
echo "Phase E complete: $(date)"
du -sh $BASE/uniprot/ 2>/dev/null
ls -la $BASE/uniprot/
echo "=================================================="
