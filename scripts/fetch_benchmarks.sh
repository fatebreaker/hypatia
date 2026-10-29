#!/bin/bash
# GeneBench-Pro public package: 10 of 129 problems, CC-BY-4.0.
# Kept out of git (third-party); this restores it into benchmarks/.
#
# The other 119 are NOT obtainable: 50 went to Artificial Analysis for
# third-party model benchmarking, 69 are an internal holdout. The 10 public
# ones are a format spec and a smoke test -- n=10 gives a ~50pp confidence
# interval, so never quote a pass rate from them as a capability score.
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$P/benchmarks/genebench-pro-public"
mkdir -p "$P/benchmarks"
"${PY:-$P/venv/bin/python}" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="ajh-oai/genebench-pro-public-package",
                      repo_type="dataset", local_dir="$DEST")
print("downloaded to", p)
PY
echo "problems: $(ls "$DEST/problems" 2>/dev/null | wc -l)  (expect 10)"
echo
echo "Regenerate the DGP/ablation specs (also gitignored):"
echo "  \$PY scripts/extract_dgp.py    # -> specs/, 10 DGPs + 130 ablation rows"
