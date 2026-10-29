"""EHRSQL-2024 RL training pool (healthcare RL, second clinical task; docs/RL_PLAN.md 2026-09-30 H-EHRSQL-RL).

generated/ehrsql2024/train/<ehrt-XXXX>/ from the shared task's labelled TRAIN split (commit 5ddaaa7, the version the
test set benchmarks/ehrsql2024 was built from), in the same problem format and task text, graded by the same vendored
official scorer against the same read-only database (rl/reward.py -> benchmarks/ehrsql2024/reference_grader.py).

  * DISJOINT EXAMPLES: the train split is shuffled (fixed seed) into an RL pool of --n questions and an examples file
    holding every OTHER train row. The pool's own gold SQL is never in the train_examples.json its instances see
    (at test time the agent sees all 5,124 rows; here it sees 5,124 minus the pool).
  * CONTAMINATION: train rows whose question text (case/space-normalised) equals a TEST question of either released
    version (shared-task 5ddaaa7, v1.1.2 f9e1aa0) are dropped from both the pool and the examples.
  * Gold results are executed by the vendored official code; rows whose gold SQL errors are dropped.
  * The data files are one copy under generated/ehrsql2024/data (hard-linked into each instance, same filesystem).

Run:  venv/bin/python scripts/build_ehrsql_train.py [--n 1200] [--seed 0] [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fetch_ehrsql2024 import TASK, VERSIONS, _gold, _link, fetch_repo, show  # noqa: E402
from paths import PROJECT_ROOT, data_root  # noqa: E402


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    fetch_repo()
    data_commit, _, pp, _ = VERSIONS["ehrsql2024"]
    bench = PROJECT_ROOT / "benchmarks" / "ehrsql2024"          # its grader + read-only DB execute the gold SQL
    tr_q = json.loads(show(data_commit, "train/data.json"))["data"]
    tr_l = json.loads(show(data_commit, "train/label.json"))
    test_q = set()
    for name in VERSIONS:
        c = VERSIONS[name][0]
        test_q |= {norm(x["question"]) for x in json.loads(show(c, "test/data.json"))["data"]}
    clean = [x for x in tr_q if norm(x["question"]) not in test_q]
    random.Random(a.seed).shuffle(clean)
    cand, rest = clean[: a.n + a.n // 5], clean[a.n + a.n // 5:]     # spare candidates cover gold-SQL errors
    with mp.get_context("fork").Pool(a.workers) as pool:
        gold = dict(pool.map(_gold, [(str(bench), x["id"], tr_l[x["id"]], pp) for x in cand]))
    ok = [x for x in cand if not str(gold[x["id"]]).startswith("error_")]
    rl, spare = ok[: a.n], [x for x in cand if x not in ok[: a.n]]
    examples = [{"question": x["question"], "query": tr_l[x["id"]]} for x in rest + spare]

    out = data_root() / "generated" / "ehrsql2024"
    ddir = out / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    db_src = bench / "data" / "mimic_iv.sqlite"
    for fname, write in (("mimic_iv.sqlite", lambda p: shutil.copy2(db_src, p)),
                         ("tables.json", lambda p: shutil.copy2(bench / "data" / "tables.json", p)),
                         ("train_examples.json", lambda p: p.write_text(json.dumps(examples, indent=0)))):
        tmp = ddir / (fname + ".part")
        tmp.unlink(missing_ok=True)
        write(tmp)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(tmp, ddir / fname)
    db_sha = hashlib.sha256((ddir / "mimic_iv.sqlite").read_bytes()).hexdigest()
    bench_cfg = json.loads((bench / "problems" / "ehr-0000" / "eval_config.json").read_text())["grader"]["config"]
    assert db_sha == bench_cfg["db_sha256"], "the copied DB is not the benchmark's"

    root = out / "train"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for i, x in enumerate(rl):
        pid = f"ehrt-{i:04d}"
        key = f"{pid}-q1"
        pdir = root / pid
        (pdir / "data_files").mkdir(parents=True)
        for f in ("mimic_iv.sqlite", "tables.json", "train_examples.json"):
            _link(ddir / f, pdir / "data_files" / f)
        cfg = {"id": pid, "task": TASK.format(question=x["question"], key=key),
               "data_files": ["mimic_iv.sqlite", "tables.json", "train_examples.json"],
               "ground_truth": {key: tr_l[x["id"]]},
               "grader": {"type": "ehrsql2024_official", "config": {
                   "db_sha256": bench_cfg["db_sha256"], "db_bytes": bench_cfg["db_bytes"],
                   "keys": {key: {"gold_result": gold[x["id"]], "postprocess": pp, "ehrsql_id": x["id"]}}}},
               "_provenance": {"benchmark": "ehrsql2024_train", "data_commit": data_commit, "postprocess": pp,
                               "ehrsql_id": x["id"], "answerable": tr_l[x["id"]] != "null"}}
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    n_una = sum(tr_l[x["id"]] == "null" for x in rl)
    print(f"[ehrsql_train] {len(tr_q)} train rows; {len(tr_q) - len(clean)} dropped (question in a test set); "
          f"{len(cand) - len(ok)} candidates dropped (gold SQL error); pool {len(rl)} ({len(rl) - n_una} answerable, "
          f"{n_una} unanswerable); examples file {len(examples)} rows (disjoint from the pool) -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
