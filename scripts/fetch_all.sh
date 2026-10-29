#!/bin/bash
# Submit every download and derived-artefact job, wired with SLURM dependencies.
#
#   scripts/setup_workspace.sh          # once, allocates scratch
#   scripts/fetch_all.sh                # this
#   scripts/verify_data.sh              # after everything finishes
#
# Dependency graph (parallel where possible):
#
#   core ──┐
#   othist ─┼──> normalize_ot ──> decision_set
#   evid ──┘
#   pubmed, uniprot, pmc, benchmarks    (independent)
#   generate_1k ──> sft_traces          (pure simulation, needs no download)
#
# Total ~149 GB. PMC dominates wall-clock (1.86M small objects, ~4 h at 48
# workers). Everything is resumable -- re-running skips what is already on
# disk, so a preempted or failed job costs nothing but time.
#
# Flags:
#   --no-pmc     skip the 1.86M full texts (~70 GB); PubMed abstracts plus
#                Open Targets evidence_europepmc cover most retrieval needs
#   --sim-only   only the simulation chain (generate_1k -> sft_traces), which
#                needs no downloaded data at all
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P"
# shellcheck source=/dev/null
source scripts/env.sh

WANT_PMC=1; SIM_ONLY=0
for a in "$@"; do
  case "$a" in
    --no-pmc)   WANT_PMC=0 ;;
    --sim-only) SIM_ONLY=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

sub() {  # sub <script> [dependency-jobid...] -> prints jobid
  local s="$1"; shift
  local dep=""
  if [ "$#" -gt 0 ]; then
    local list; list=$(printf ":%s" "$@")
    dep="--dependency=afterok${list}"
  fi
  sbatch $dep --parsable "$s"
}

echo "data root: $BASE"
echo

if [ "$SIM_ONLY" -eq 0 ]; then
  CORE=$(sub scripts/core.sbatch)        ; echo "core        $CORE  Open Targets 26.06, ChEMBL, ClinVar, GWAS, gnomAD, MGI, IMPC, AACT"
  HIST=$(sub scripts/othist.sbatch)      ; echo "othist      $HIST  OT 21.06 22.06 23.06 24.06 25.06"
  EVID=$(sub scripts/evid.sbatch)        ; echo "evid        $EVID  evidence/sourceId=* partitions (slow, ~2 h)"
  PUBM=$(sub scripts/pubmed.sbatch)      ; echo "pubmed      $PUBM  2026 baseline + updates (~63 GB)"
  UNIP=$(sub scripts/uniprot.sbatch)     ; echo "uniprot     $UNIP  Swiss-Prot 2026_02 + idmapping (8.3 GB)"
  if [ "$WANT_PMC" -eq 1 ]; then
    PMC=$(sub scripts/pmc.sbatch)        ; echo "pmc         $PMC  1.86M full texts (~70 GB, ~4 h)"
  else
    echo "pmc         SKIPPED (--no-pmc)"
  fi

  # normalize_ot needs every release's evidence present
  NORM=$(sub scripts/normalize_ot.sbatch "$CORE" "$HIST" "$EVID")
  echo "normalize   $NORM  -> normalized/ (132.5M rows; needs ~48 GB RAM)"
fi

# Simulation chain: independent of all downloads.
GEN=$(sub scripts/generate_1k.sbatch)    ; echo "generate    $GEN  1000 cis-MVMR instances (train/val/test)"
SFT=$(sub scripts/sft_traces.sbatch "$GEN")
echo "sft_traces  $SFT  execution-verified traces"
echo "                  NOTE: the current generator is degenerate -- see"
echo "                  FINDINGS C7 and ROADMAP step 1 before training on this."

echo
echo "benchmarks (not a batch job, ~18 MB from HuggingFace):"
echo "  bash scripts/fetch_benchmarks.sh"
echo
echo "watch:   squeue -u \$USER"
echo "verify:  scripts/verify_data.sh"
