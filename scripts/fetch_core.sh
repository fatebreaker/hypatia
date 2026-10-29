#!/bin/bash
# Phase A: core evidence layer for the target-ID backtest.
# All sources verified 2026-07-28. Permissive licenses only (CC0 / CC-BY / public domain).
# Resumable: every wget uses -c, so re-running skips completed files.
#
# NOTE: ftp.ebi.ac.uk serves "robots.txt: Disallow: /", which silently kills
# `wget -r`. We enumerate parquet links from the directory index instead --
# more robust, and it gives real per-file progress.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
mkdir -p $LOG
exec > >(tee -a $LOG/fetch_core.log) 2>&1

echo "=================================================="
echo "Phase A start: $(date)"
echo "=================================================="

W="wget -c -nv --tries=5 --timeout=60 --waitretry=10"

# Download every file matching a suffix from an Apache-style directory index.
fetch_dir () {
  local url="$1" dest="$2" suffix="${3:-.parquet}"
  mkdir -p "$dest"
  local files
  files=$(curl -sL --max-time 120 "$url" \
          | grep -oE "href=\"[^\"]*${suffix}\"" \
          | sed 's/href="//;s/"$//')
  if [ -z "$files" ]; then
    echo "  WARN: no ${suffix} files found at $url"
    return 1
  fi
  local n=0
  for f in $files; do
    $W -P "$dest" "${url}${f}" || echo "  WARN: failed ${url}${f}"
    n=$((n+1))
  done
  echo "  -> $n files into $dest"
}

# ---------- 1. Open Targets 26.06 (current release) ----------
# Every evidence_* dir is kept separately: datasource provenance is the
# whole point for the adjudication task.
OT=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output
OTD=$BASE/opentargets/26.06

OT_DIRS="target disease drug_molecule drug_mechanism_of_action drug_warning \
clinical_indication clinical_target clinical_report \
l2g_prediction credible_set colocalisation variant study \
association_overall_direct association_overall_indirect \
association_by_datasource_direct association_by_datatype_direct \
mouse_phenotype target_prioritisation target_essentiality \
disease_phenotype disease_hpo interaction go so biosample pharmacogenomics \
evidence_gwas_credible_sets evidence_gene_burden evidence_impc \
evidence_europepmc evidence_eva evidence_eva_somatic evidence_clingen \
evidence_clinical_precedence evidence_crispr evidence_crispr_screen \
evidence_expression_atlas evidence_gene2phenotype evidence_genomics_england \
evidence_intogen evidence_orphanet evidence_reactome \
evidence_uniprot_literature evidence_uniprot_variants \
evidence_cancer_biomarkers evidence_cancer_gene_census"

for d in $OT_DIRS; do
  echo "--- OT 26.06/$d"
  fetch_dir "$OT/$d/" "$OTD/$d" ".parquet"
done

# ---------- 2. ChEMBL 37 (5.8 GB) ----------
echo "--- ChEMBL 37 sqlite (5.8 GB)"
mkdir -p $BASE/chembl
$W -P $BASE/chembl https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz
$W -P $BASE/chembl https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/checksums.txt

# ---------- 3. GWAS Catalog (current; historical state comes from OT releases) ----------
echo "--- GWAS Catalog"
mkdir -p $BASE/gwas_catalog
G=https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest
for f in gwas-catalog-associations_ontology-annotated-full.zip \
         gwas-catalog-associations-full.zip \
         gwas-catalog-studies.tsv gwas-catalog-ancestry.tsv; do
  $W -P $BASE/gwas_catalog "$G/$f"
done

# ---------- 4. ClinVar ----------
echo "--- ClinVar"
mkdir -p $BASE/clinvar
$W -P $BASE/clinvar https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
$W -P $BASE/clinvar https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz

# ---------- 5. gnomAD constraint (gene-level LoF; NOT the 0.76 TB VCFs) ----------
echo "--- gnomAD constraint"
mkdir -p $BASE/gnomad
$W -P $BASE/gnomad "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv"

# ---------- 6. Mouse KO phenotypes ----------
echo "--- MGI"
mkdir -p $BASE/mgi
for f in MGI_PhenoGenoMP.rpt MGI_GenePheno.rpt MGI_Geno_Disease.rpt HMD_HumanPhenotype.rpt; do
  $W -P $BASE/mgi "https://www.informatics.jax.org/downloads/reports/$f" || echo "  WARN: MGI $f"
done

echo "--- IMPC release-24.0"
mkdir -p $BASE/impc
fetch_dir "https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/release-24.0/results/" "$BASE/impc" ".csv.gz"

# ---------- 7. AACT: trial-status time machine, one snapshot per year ----------
# ClinicalTrials.gov's own API is current-state only, so these archives are
# the only way to get status as-of a past date without leaking the future.
echo "--- AACT yearly snapshots (~2.45 GB each)"
mkdir -p $BASE/aact
for y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  f="${y}0601_pipe-delimited-export.zip"
  url="https://aact.ctti-clinicaltrials.org/static/exported_files/monthly/$f"
  code=$(curl -s -o /dev/null -w "%{http_code}" -L -r 0-100 --max-time 45 "$url" 2>/dev/null)
  if [ "$code" = "200" ] || [ "$code" = "206" ]; then
    echo "  fetching AACT $y"
    $W -O "$BASE/aact/$f" "$url" || echo "  WARN: AACT $y failed"
  else
    echo "  AACT $y unavailable (HTTP $code) - skipping"
  fi
done

# ---------- 8. TrialBench (CC-BY-4.0 trial outcomes incl. failure reasons) ----------
echo "--- TrialBench"
mkdir -p $BASE/trialbench
$W -O $BASE/trialbench/trialbench-files-archive.zip \
   "https://zenodo.org/records/15455785/files-archive" || echo "  WARN: TrialBench"

echo "=================================================="
echo "Phase A complete: $(date)"
du -sh $BASE/*/ 2>/dev/null
echo "TOTAL: $(du -sh $BASE 2>/dev/null | cut -f1)"
echo "=================================================="
