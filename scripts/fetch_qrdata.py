"""Adapt QRData into harness-compatible problems.

QRData (Liu et al., ACL Findings 2024, https://arxiv.org/abs/2402.17644) is 411
statistical and causal reasoning questions over 195 real CSV data sheets from
textbooks, courses and papers. GPT-4 reaches 58 %.

WHY THIS BENCHMARK, SPECIFICALLY
--------------------------------
Every transfer test in this project so far was measured at the floor: base
Qwen3-8B scores 0.000 on eight of ten GeneBench-Pro problems, 0.074/0.059 on
the other two, and 0.064 on held-out BixBench -- where the BixBench authors
themselves report agents at "no better than random guessing" in the
multiple-choice setting. A null result on an instrument that cannot register a
positive is weak evidence. QRData sits mid-range (GPT-4 58 %, smaller open
models well below), so a real transfer effect -- in either direction -- has
room to show.

GRADING
-------
Ported from the official benchmark/eval.py so scores are comparable to
published numbers:
  * numerical      -- correct if within a 3 % RELATIVE band of the gold value,
                      taking the first number in the response
  * multiple_choice -- case-insensitive containment of the gold string
Note the 3 % band is relative and two-sided, and the official code compares
with strict inequalities; both are reproduced rather than "improved".

ONE PROBLEM PER QUESTION
------------------------
Unlike BixBench capsules, QRData questions are individually answerable and
share small CSVs (20 MB total for all 195), so copying per question is cheap
and keeps one question per graded key.

NO ANSWER LEAKAGE
-----------------
data.zip holds CSVs only -- no notebooks, no solution scripts. verify() asserts
that, because the BixBench adapter learned the hard way that shipped analysis
artefacts can hand over the answer key (7 of 7 answers recoverable from one
capsule's executed notebook).

Run:
  venv/bin/python scripts/fetch_qrdata.py               # all 411
  venv/bin/python scripts/fetch_qrdata.py --limit 5     # smoke test
  venv/bin/python scripts/fetch_qrdata.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

REPO = "https://github.com/xxxiaol/QRData.git"
# Mirrors fetch_bixbench.py: the repo location is derived, never hardcoded, so
# a clone anywhere works. The clone cache is bulk data, so it goes under the
# data root -- which is itself resolved, not a fixed scratch path.
ROOT = PROJECT_ROOT / "benchmarks" / "qrdata"
CACHE = data_root() / "cache" / "qrdata_repo"

BANNED = (".ipynb", ".ipynb_checkpoints", "solution", "answer", ".py", ".r", ".rmd")


def fetch_repo() -> Path:
    if not (CACHE / "benchmark" / "QRData.json").exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(CACHE, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(CACHE)], check=True)
    return CACHE / "benchmark"


def build(limit: int | None) -> int:
    src = fetch_repo()
    qs = json.loads((src / "QRData.json").read_text())
    if limit:
        qs = qs[:limit]

    zpath = src / "data.zip"
    extract = CACHE / "data_extracted"
    if not extract.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract)
    csv_root = next((p for p in extract.rglob("*.csv")), None)
    assert csv_root is not None, "no CSVs found in data.zip"
    csv_dir = csv_root.parent

    ROOT.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, q in enumerate(qs):
        pid = f"qr-{i:03d}"
        pdir = ROOT / "problems" / pid
        ddir = pdir / "data_files"
        ddir.mkdir(parents=True, exist_ok=True)
        missing = False
        for f in q["data_files"]:
            s = csv_dir / f
            if not s.exists():
                missing = True
                break
            if not (ddir / f).exists():
                shutil.copy2(s, ddir / f)
        if missing:
            shutil.rmtree(pdir, ignore_errors=True)
            continue

        qtype = q["meta_data"].get("question_type", "numerical")
        task = (
            "You are given the data files from a published statistical analysis.\n"
            "Answer the question below by computing the result from the data.\n\n"
            f"{q['data_description']}\n\n"
            f"Files in data_files/:\n  " + "\n  ".join(q["data_files"]) + "\n\n"
            f"Question: {q['question']}\n\n"
            'Return your answer as JSON: {"answer": {"' + pid + '-q1": <value>}}'
        )
        cfg = {
            "id": pid,
            "task": task,
            "data_files": q["data_files"],
            "ground_truth": {f"{pid}-q1": q["answer"]},
            "grader": {
                "type": "qrdata_official",
                "config": {"keys": {f"{pid}-q1": {"question_type": qtype,
                                                  "error_scale": 0.03}}},
            },
            "_provenance": {
                "benchmark": "QRData",
                "paper": "arXiv:2402.17644",
                "reference": q["meta_data"].get("reference"),
                "keywords": q["meta_data"].get("keywords", []),
            },
        }
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        made += 1
    return made


def verify() -> None:
    bad = []
    for f in ROOT.rglob("*"):
        if f.is_file() and any(b in f.name.lower() for b in BANNED):
            if "data_files" in str(f):
                bad.append(f)
    assert not bad, f"ANSWER-LEAKAGE RISK: analysis artefacts reached data_files/: {bad[:5]}"
    n = len(list((ROOT / "problems").glob("*/eval_config.json")))
    print(f"verify OK: {n} problems, no analysis artefacts in data_files/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        print(f"built {build(a.limit)} problems under {ROOT}")
    verify()
