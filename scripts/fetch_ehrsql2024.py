"""Adapt EHRSQL-2024 into harness-compatible problems (healthcare benchmark: reliable question answering over an EHR).

SOURCE    github.com/glee4810/ehrsql-2024 (Lee, Kweon, Bae, Choi; "Overview of the EHRSQL 2024 Shared Task on Reliable
          Text-to-SQL Modeling on Electronic Health Records", ClinicalNLP @ NAACL 2024, aclanthology 2024.clinicalnlp-1.62)
LICENSE   repo CC-BY-4.0; the database is the MIMIC-IV Clinical Database DEMO v2.2 (100 patients, OPEN access, ODbL
          v1.0) as preprocessed by the organisers -- NOT credentialed MIMIC-IV.
PINNED    one clone at REPO_REV; every file is read with `git show <commit>:<path>` from the commits below.
TASK      question about the EHR -> a SQLite query whose execution answers it, OR the string "null" to abstain when the
          question cannot be answered from the database (asks for data it does not hold, external knowledge, or
          something beyond SQL). ~20% of questions are unanswerable by design.

TWO VERSIONS, BOTH BUILT (the release was re-labelled after the shared task: only 807 of 1,167 test ids survive and
about half the labels changed, so the two are different test sets, not a patch)
  benchmarks/ehrsql2024       THE SHARED-TASK TEST SET: test questions/labels as released at 5ddaaa7 (version string
                              "mimic_iv_demo_1.0_test"), the database committed at 91a1e13, the postprocessing the
                              leaderboard was scored with. PRIMARY -- this is the one the published numbers are on.
  benchmarks/ehrsql2024_v112  the current cleaned release v1.1.2 (f9e1aa0: test, database, postprocessing).
  Both: n = 1,167 = 934 answerable + 233 unanswerable (Table 1 of the overview paper: 934 / 233 / 1,167).
  The TEST split is used; its labels are public in the repo (test/label.json + test/answer.json).

WHAT THE AGENT GETS (data_files/, hard links to one read-only copy in benchmarks/<name>/data/)
  mimic_iv.sqlite       the database the official scorer executes against
  tables.json           the release's Spider-format schema
  train_examples.json   the labelled TRAIN split (5,124 question -> gold SQL, "null" when unanswerable) of the same
                        version -- the supervision every shared-task participant had (every ranked system fine-tuned
                        on it or retrieved examples from it). The VALID labels were hidden during the shared task's
                        test phase (released at 5ddaaa7, after it), so they are not given.
  The task text states the dataset's clock convention (now = 2100-12-31 23:59:00, which the official postprocessing
  substitutes) and the scoring rule, both published with the task.

GRADING (reference_grader.py = scripts/ehrsql2024_reference_grader.py; official scoring.py logic, vendored)
  execution-result match of the submitted SQL against the gold SQL (sorted rows, 3-decimal rounding, first 100 rows),
  with 'null' as abstention; per item +1 correct / 0 abstain-on-answerable / -1 wrong or answered-unanswerable.
  grade["passed"] = item score +1, so the harness pass rate IS RS(0). RS(5), RS(10) (the primary metric) and RS(N):
      venv/bin/python scripts/ehrsql2024_score.py <runner outdir>
  Build-time check: the gold SQL executed by the vendored code reproduces the release's answer.json on ALL 1,167
  items of each version (asserted below; a mismatch aborts the build).

PUBLISHED NUMBERS (overview paper, Table 3 -- the official leaderboard, shared-task test set = benchmarks/ehrsql2024;
text-to-SQL SYSTEMS trained on the train split, not tool-using agents):
      team                 RS_0   RS_10    RS_N     model
      LG AI Research&KAIST 88.17  81.32   -711.83   ChatGPT (ensemble, not fine-tuned; retrieval from train)
      PromptMind           82.6   74.89   -817.4    GPT-4, ChatGPT, Claude Opus (ensemble, fine-tuned)
      ProbGate             81.92  74.21   -818.08   ChatGPT (fine-tuned)
      KU-DMIS              72.07  59.21   -1427.93  ChatGPT (fine-tuned)
      AIRINLP              68.89  44.04   -2831.11  T5-3B + logistic regression
      LTRC-IIITH           66.84  43.7    -2633.16  SQLCoder-7b-2 (fine-tuned)
      Saama Technologies   53.21  36.08   -1946.79  decision trees, CodeLlama-7b, ChatGPT
      TEAM_optimist        14.14  -713.37 -84.9K    SQLCoder-7b-2 (not fine-tuned)
      ABSTAIN-ALL          20.0   20.0    20.0      baseline (reproduced exactly by ehrsql2024_score.py --abstain-all)
  No published numbers on v1.1.2 were found on the source pages.

Run:
  venv/bin/python scripts/fetch_ehrsql2024.py              # both versions, all 1,167 test items each
  venv/bin/python scripts/fetch_ehrsql2024.py --limit 5    # smoke
  venv/bin/python scripts/fetch_ehrsql2024.py --verify-only
Evaluate (from harness/):
  ../venv/bin/python runner.py --agent llm:MODEL --problems ../benchmarks/ehrsql2024 --workers 8 --outdir OUT
  ../venv/bin/python ../scripts/ehrsql2024_score.py OUT
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

REPO = "https://github.com/glee4810/ehrsql-2024.git"
REPO_REV = "f9e1aa02160d39e3f8df52bf5c69c5cf2e472499"      # HEAD on 2026-09-30 ("Update README.md", 2026-06-25)
CACHE = data_root() / "cache" / "ehrsql2024" / "repo"
GRADER = Path(__file__).resolve().parent / "ehrsql2024_reference_grader.py"

VERSIONS = {
    # name: (commit for test+train files, commit for the sqlite + tables.json, postprocessing version, label)
    "ehrsql2024": ("5ddaaa7", "91a1e13", "2024-03", "shared-task test (mimic_iv_demo_1.0_test)"),
    "ehrsql2024_v112": ("f9e1aa0", "f9e1aa0", "v1.1.2", "release v1.1.2 (test_v1.1.2)"),
}
# tables.json was only added to data/ at 98a747d (2024-01) and revised at 2782cf9 (v1.0.4): the DB commit's copy is
# the one that matches that DB.


def git(*args, text=True):
    return subprocess.run(["git", "-C", str(CACHE), *args], check=True, capture_output=True, text=text).stdout


def fetch_repo() -> None:
    if not (CACHE / ".git").exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(CACHE, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", REPO, str(CACHE)], check=True)   # full history: old commits are read
    try:
        git("cat-file", "-e", REPO_REV)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "-C", str(CACHE), "fetch", "-q", "origin"], check=True)
    git("checkout", "-q", REPO_REV)


def show(commit: str, path: str, binary=False):
    return git("show", f"{commit}:data/mimic_iv/{path}", text=not binary)


def load_grader(root: Path):
    spec = importlib.util.spec_from_file_location(f"ehrsql_grader_{root.name}", root / "reference_grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gold(args):
    root, key, sql, pp = args
    g = load_grader(Path(root))
    if sql == "null":
        return key, "null"
    return key, g.execute_sql_wrapper(g.POST_PROCESS[pp](sql), "real", Path(root) / "data" / "mimic_iv.sqlite")


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


TASK = """You are given an electronic health record (EHR) database and a question about it.

  data_files/mimic_iv.sqlite      SQLite build of the MIMIC-IV Clinical Database Demo (100 patients), as preprocessed for
                                  the EHRSQL 2024 shared task
  data_files/tables.json          its schema (Spider format: tables, columns, types, keys)
  data_files/train_examples.json  the shared task's labelled training set: {{"question", "query"}} pairs, where "query"
                                  is the gold SQL, or "null" for questions that cannot be answered from this database

Conventions of this dataset:
  * Times in the database are shifted into the 2100s. Treat the CURRENT time as 2100-12-31 23:59:00 ("this year" is
    2100, "last year" 2099, ...). Before your query is executed, the official post-processing replaces every
    `current_time` in it with '2100-12-31 23:59:00'; when you test a query yourself, SQLite's own `current_time` is
    today's clock, so test with the literal instead.
  * Some questions cannot be answered from this database (they need data it does not contain, outside knowledge, or
    something SQL cannot do). For those, the correct response is to ABSTAIN.

Question: {question}

Produce ONE SQLite query whose execution result answers the question, or abstain. Explore the database with Python
(sqlite3 / pandas) and test your query before answering.

Scoring (the official EHRSQL-2024 reliability score): your query is executed and its result compared with the gold
query's result (rows sorted, numbers rounded to 3 decimals, first 100 rows; the query text itself does not matter).
A matching result scores +1; abstaining on an answerable question scores 0; a wrong result -- or any query at all for
an unanswerable question -- scores -10 (RS(10), the shared task's primary metric). A correct abstention scores +1.

Return your answer as JSON: {{"answer": {{"{key}": "<your SQLite query>"}}}}
or, to abstain: {{"answer": {{"{key}": "null"}}}}"""


def build(name: str, limit: int | None, workers: int) -> int:
    data_commit, db_commit, pp, label = VERSIONS[name]
    root = PROJECT_ROOT / "benchmarks" / name
    ddir = root / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GRADER, root / "reference_grader.py")

    db = ddir / "mimic_iv.sqlite"
    if not db.exists():
        tmp = ddir / "mimic_iv.sqlite.part"
        tmp.write_bytes(show(db_commit, "mimic_iv.sqlite", binary=True))
        tmp.rename(db)
    os.chmod(db, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)       # the grader reads this one copy: keep it pristine
    db_sha = hashlib.sha256(db.read_bytes()).hexdigest()
    tr_q = json.loads(show(data_commit, "train/data.json"))["data"]
    tr_l = json.loads(show(data_commit, "train/label.json"))
    train = [{"question": x["question"], "query": tr_l[x["id"]]} for x in tr_q]
    for fname, text in (("tables.json", show(db_commit, "tables.json")),
                        ("train_examples.json", json.dumps(train, indent=0))):
        tmp = ddir / (fname + ".part")          # a new inode: never write through the problems' hard links
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(tmp, ddir / fname)

    te = json.loads(show(data_commit, "test/data.json"))
    qs = te["data"]
    labels = json.loads(show(data_commit, "test/label.json"))
    answers = json.loads(show(data_commit, "test/answer.json"))
    assert len(qs) == len(labels) == len(answers) == 1167, (len(qs), len(labels), len(answers))
    if limit:
        qs = qs[:limit]

    # Gold results by the vendored official code, checked against the release's own answer.json.
    g = load_grader(root)
    with mp.get_context("fork").Pool(workers) as pool:
        gold = dict(pool.map(_gold, [(str(root), x["id"], labels[x["id"]], pp) for x in qs]))
    bad = [k for k in gold if gold[k] != g.process_answer(answers[k])]
    assert not bad, f"{name}: gold SQL does not reproduce answer.json for {len(bad)} items, e.g. {bad[:3]}"

    if (root / "problems").exists():
        shutil.rmtree(root / "problems")
    for i, x in enumerate(qs):
        pid = f"ehr-{i:04d}"
        key = f"{pid}-q1"
        pdir = root / "problems" / pid
        (pdir / "data_files").mkdir(parents=True)
        for f in ("mimic_iv.sqlite", "tables.json", "train_examples.json"):
            _link(ddir / f, pdir / "data_files" / f)
        cfg = {
            "id": pid,
            "task": TASK.format(question=x["question"], key=key),
            "data_files": ["mimic_iv.sqlite", "tables.json", "train_examples.json"],
            "ground_truth": {key: labels[x["id"]]},        # gold SQL, or "null": what the oracle submits
            "grader": {"type": "ehrsql2024_official", "config": {
                "db_sha256": db_sha, "db_bytes": db.stat().st_size,
                "keys": {key: {"gold_result": gold[x["id"]], "postprocess": pp, "ehrsql_id": x["id"]}}}},
            "_provenance": {
                "benchmark": f"EHRSQL-2024 {label}", "repo": f"{REPO}@{REPO_REV}",
                "data_commit": data_commit, "db_commit": db_commit, "postprocess": pp,
                "dataset_version": te.get("version"), "ehrsql_id": x["id"],
                "answerable": labels[x["id"]] != "null", "paper": "aclanthology 2024.clinicalnlp-1.62",
            },
        }
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    return len(qs)


def verify(name: str) -> None:
    root = PROJECT_ROOT / "benchmarks" / name
    g = load_grader(root)
    cfgs = [json.loads(p.read_text()) for p in sorted((root / "problems").glob("*/eval_config.json"))]
    g.check_db(cfgs[0])
    n, ok, rel_gold, rel_null, rel_abst = len(cfgs), 0, [], [], []
    for cfg in cfgs:
        key = next(iter(cfg["ground_truth"]))
        r = g.evaluate(cfg, {"answer": dict(cfg["ground_truth"])})
        ok += r["passed"]
        rel_gold.append(r["reliability"])
        bad = g.evaluate(cfg, {"answer": {key: "SELECT 'banana'"}})
        assert not bad["passed"], cfg["id"]
        rel_null.append(g.evaluate(cfg, {"answer": {key: ""}})["reliability"])        # the harness null agent
        rel_abst.append(g.evaluate(cfg, {"answer": {key: "null"}})["reliability"])
    assert ok == n, f"{name}: the ground truth passes only {ok}/{n}"
    assert not any(r == 1 for r in rel_null), f"{name}: the empty-string (null agent) answer scored"
    n_una = sum(c["ground_truth"][next(iter(c["ground_truth"]))] == "null" for c in cfgs)
    rs_abst = g.reliability_scores(rel_abst)
    print(f"[ehrsql] {name}: {n} problems ({n - n_una} answerable, {n_una} unanswerable); ground truth passes {ok}/{n} "
          f"(RS10 {g.reliability_scores(rel_gold)['RS10']:.2f}); 'SELECT banana' fails all; empty answer fails all; "
          f"ABSTAIN-ALL RS0/RS10/RSN = {rs_abst['RS0']:.2f}/{rs_abst['RS10']:.2f}/{rs_abst['RSN']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=sorted(VERSIONS), default=None)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    names = [a.only] if a.only else list(VERSIONS)
    if not a.verify_only:
        fetch_repo()
        for nm in names:
            print(f"[ehrsql] {nm}: built {build(nm, a.limit, a.workers)} problems")
    for nm in names:
        verify(nm)
