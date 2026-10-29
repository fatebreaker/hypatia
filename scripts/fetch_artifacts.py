"""Download the project's own artifacts from HuggingFace.

This is the counterpart to scripts/upload_hf*.py. It fetches what THIS project
produced -- generated problems, SFT corpora, and trained adapters. For the
upstream raw data (PMC, GWAS Catalog, benchmark payloads) see docs/DATA.md and
scripts/fetch_*.sh.

    venv/bin/python scripts/fetch_artifacts.py --what adapters
    venv/bin/python scripts/fetch_artifacts.py --what problems   # gated, needs access
    venv/bin/python scripts/fetch_artifacts.py --what all --only tenrl-step150-lora

WHY YOU CANNOT JUST REGENERATE THE PROBLEMS
-------------------------------------------
Re-running a generator with the same seed does NOT reproduce the stored
instances: the generators were edited after the corpora were built (instances
2026-08-10, generators 2026-09-01). Verified by regenerating popgen seed 0 and
diffing ground truth. Every number in the paper was measured on the stored
instances, so download them rather than regenerate, or your numbers will drift
for reasons that look like real effects.

The adapter repo is public. The problems repo is access-gated, because those
instances encode data-generating processes transcribed from OpenAI's
GeneBench-Pro case-study reports; request access on the repo page.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

USER = os.environ.get("DEEPRS_HF_USER", "hieutran81")
ADAPTERS_REPO = f"{USER}/deeprs-adapters"
PROBLEMS_REPO = f"{USER}/deeprs-problems"
# Resolved the same way as everything else ($DEEP_DATA_ROOT -> ws_find -> ./data),
# so this lands beside the upstream data whether or not the env var is exported.
DATA = data_root()

# The four that reproduce the headline; --what adapters gets all 120.
KEY = ["qwen3-8b-tenfam-lora", "qwen3-8b-tenfam2-lora",
       "tenrl-step150-lora", "tenrl-step100-lora"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["adapters", "problems", "key", "all"], default="key",
                    help="key = the 4 adapters behind the headline (2.8 GB); "
                         "adapters = all 120 (124 GB); problems = instances + SFT corpora "
                         "(2.6 GB, gated); all = everything")
    ap.add_argument("--only", help="restrict to one adapter directory name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    tok = os.environ.get("HF_TOKEN")
    jobs = []
    if a.what in ("key", "adapters", "all"):
        pats = [f"{a.only}/**"] if a.only else (
            None if a.what in ("adapters", "all") else [f"{k}/**" for k in KEY])
        jobs.append((ADAPTERS_REPO, "model", DATA / "ckpt", pats))
    if a.what in ("problems", "all"):
        jobs.append((PROBLEMS_REPO, "dataset", DATA, None))

    for repo, kind, dest, pats in jobs:
        print(f"{'[dry-run] ' if a.dry_run else ''}{repo}  ->  {dest}"
              + (f"   (patterns: {pats})" if pats else ""))
        if a.dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        try:
            p = snapshot_download(repo_id=repo, repo_type=kind, local_dir=str(dest),
                                  allow_patterns=pats, token=tok)
            print(f"    -> {p}")
        except Exception as e:
            msg = str(e).split("\n")[0][:160]
            print(f"    FAILED: {type(e).__name__} {msg}", file=sys.stderr)
            if kind == "dataset":
                print("    the problems repo is access-gated; request access at "
                      f"https://huggingface.co/datasets/{repo}", file=sys.stderr)
            return 1
    print("\nNote: adapters land under $DEEP_DATA_ROOT/ckpt/<name>/ and are loaded "
          "directly (they are the `final/` contents, flattened).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
