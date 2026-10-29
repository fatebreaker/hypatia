#!/bin/bash
# Build the anonymised code supplement for double-blind review (ARR allows one).
#
#   bash scripts/make_supplement.sh [out.zip]
#
# Copies the code that produces every number in the paper -- harness, generators,
# RL/SFT training, benchmark adapters and graders, the results index -- and none
# of the narrative docs, which name the cluster, the lab and the authors. Author
# handles are replaced, and the build FAILS if any identifying string survives:
# a supplement that de-anonymises the paper is worse than no supplement.
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$P/supplement.zip}"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE" 2>/dev/null || true' EXIT
D="$STAGE/code"

cd "$P"
# --cached --others: tracked files plus new untracked ones (never ignored files),
# without touching the index. Minus identifying narrative and site-only scripts.
git ls-files -z --cached --others --exclude-standard -- \
    harness generators rl scripts benchmarks results_index \
    paths.py requirements.txt requirements-train.txt \
    docs/make_fig1.py docs/make_figdata_exercise.py docs/figdata \
  | grep -zvE '^scripts/(upload_hf|upload_hf_adapters|make_supplement)\.|\.sbatch$|gpu_holder|holder_env|launch_when_ready|env\.local' \
  | while IFS= read -r -d '' f; do mkdir -p "$D/$(dirname "$f")"; cp "$f" "$D/$f"; done

# Replace handles and site-specific names, rewriting each file IN PLACE. Not
# `sed -i`: on NFS it unlinks the file while still holding it open, the client
# "silly-renames" the unscrubbed original to .nfsXXXX in the same directory, and
# that copy -- author handles and all -- would end up in the zip.
python3 - "$D" <<'SCRUB'
import pathlib, re, sys
SUBS = [(r"hieutran81", "ANONYMOUS"), (r"fatebreaker", "ANONYMOUS"),
        (r"/home/htran[^ \"']*", "/path/to/data"),
        (r"/project/pi_hongyu_umass_edu[^ \"']*", "/path/to/project"),
        (r"/scratch[0-9]*/workspace[^ \"']*", "/path/to/scratch"),
        (r"UMass Unity", "the cluster"), (r"\bUnity\b", "the cluster")]
for f in pathlib.Path(sys.argv[1]).rglob("*"):
    if f.is_file() and f.suffix in {".py", ".sh", ".md", ".yaml", ".txt", ".csv", ".json"}:
        t = f.read_text(errors="replace")
        u = t
        for pat, rep in SUBS:
            u = re.sub(pat, rep, u)
        if u != t:
            with open(f, "r+") as fh:        # same inode: no rename, no .nfs copy
                fh.seek(0)
                fh.write(u)
                fh.truncate()
SCRUB

cat > "$D/README.md" <<'EOF'
# Code supplement (anonymised for review)

Reproduces every number in the paper from the stored rollout index, and every
experiment from scratch given the public benchmarks.

- `harness/`      sandboxed multi-turn agent loop, grading, behavioural counters
- `generators/`   one simulator per public GeneBench-Pro problem, with trap screens
- `rl/`           SFT (LoRA) and GRPO training, step-attributed reward, trace builders
- `benchmarks/`   BixBench / QRData adapters and the vendored graders
- `results_index/` per-(run, problem) scores behind every table
- `docs/make_fig1.py` regenerates Figure 1 from `docs/figdata/`; each row of
  `fig1b_rl_targets.csv` names the index runs it comes from, and
  `docs/make_figdata_exercise.py` rebuilds panel (c) from the index

Setup: `pip install -r requirements.txt` (analysis, harness) and
`-r requirements-train.txt` (training, serving; Python 3.12).
Benchmark payloads are not included (licences); fetch them with
`scripts/fetch_benchmarks.sh`, `scripts/fetch_bixbench.py`, `scripts/fetch_qrdata.py`.
Set `DEEP_DATA_ROOT` to a directory for data and checkpoints.
EOF

# Rule 1 of this repo, applied to anonymity: assert, don't assume. Scan every
# file that will be zipped (NFS placeholders are excluded from the zip below).
if grep -rIl --exclude='.nfs*' -iE 'hieutran|htran|fatebreaker|umass|hongyu|michael\.tran|unity\.rc' "$D"; then
  echo "FATAL: identifying strings survive in the files above" >&2; exit 1
fi
n=$(find "$D" -type f ! -name '.nfs*' | wc -l)
[ "$n" -gt 50 ] || { echo "FATAL: only $n files staged" >&2; exit 1; }

rm -f "$OUT"; (cd "$STAGE" && zip -qr "$OUT" code -x '*/.nfs*')
if unzip -l "$OUT" | grep -q '\.nfs'; then echo "FATAL: NFS placeholder in zip" >&2; exit 1; fi
echo "supplement: $OUT ($n files, $(du -h "$OUT" | cut -f1)); anonymity scan clean"
