"""Adapt DataBench (SemEval-2025 Task 8 test split) into harness-compatible problems.

DataBench (Osés Grijalba et al., LREC-COLING 2024; huggingface.co/datasets/cardiffnlp/databench,
MIT), the dataset of SemEval-2025 Task 8 (question answering over tabular data), is 80 real tables
(Kaggle, government and company open data): 1,308 train/dev questions and a 522-question TEST split
over 15 held-out tables, whose answers were released after the competition. Answers are exact:
boolean, category, number, list[category] or list[number].

WHY THIS BENCHMARK
------------------
A second external format nothing in the corpora resembles: one real table per question (40 to
239,461 rows, 5 to 35 columns) in parquet, a one-line question in the benchmark's
own words, and boolean, categorical and list answers -- where every corpus question asks for a
named numeric estimate. It is closer to the floor of analysis skill than KramaBench, so it registers
transfer that a harder benchmark would leave at zero.

WHICH QUESTIONS
---------------
All 522 of the official test split (tables 066-080), no filtering: it is the split published
leaderboards report, and never part of any train/dev release. The full table (all.parquet) is
provided, i.e. DataBench proper, not DataBench-lite's 20-row sample.

GRADING
-------
Ported from the official evaluator, databench_eval.Evaluator.default_compare
(github.com/jorses/databench_eval @ 65f92cf), which compares str(answer) with the gold string:
  boolean          true/yes/y vs false/no/n, case-insensitive
  category         string equality after stripping [ ] ' " and spaces, or the same date
  number           equal after truncation to two decimals: trunc(100 x)
  list[category]   same length, same set, after the same stripping (or the same dates)
  list[number]     same length, same set, each truncated to two decimals
Deliberately NOT "improved" -- truncation rather than rounding is the published rule. The prompt
therefore says numbers are compared to two decimals, so an unrounded answer is not penalised for
formatting. The answer TYPE is disclosed in the prompt: it is part of every DataBench record, and
without it a correct answer fails on form (e.g. "36.9 years" for a boolean).

DISPERSION
----------
rl/dispersion.py scores with rl/reward.py, which handles scalar keys only:
$DEEP_DATA_ROOT/generated/databench/test links the 359 boolean/category/number questions (list
questions left out); number keys declare `absolute_tolerance` 0.005, an approximation of the
two-decimal truncation rule. rl/reward.py compares strings exactly, so there a boolean must be
True/False (JSON true/false) rather than yes/no. A reference_grader.py link lets harness/runner.py
run on that path too.

Run:
  venv/bin/python scripts/fetch_databench.py                # fetch + build + verify
  venv/bin/python scripts/fetch_databench.py --verify-only
"""
from __future__ import annotations

import argparse
import ast
import collections
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

REPO_ID = "cardiffnlp/databench"
REVISION = "e75d53add267d2f9cfa32efd65ad77f0807adfad"      # main, 2025-08-06
EVALUATOR = "https://github.com/jorses/databench_eval @ 65f92cfad4f986f5496170f6bbcee48323f56c2f"
LICENCE = ("MIT (HF dataset card); each table keeps its source's terms -- Kaggle datasets, "
           "World Bank, Open Food Facts (ODbL), IAB -- listed per table in the card")
TEST = ("066_IBM_HR", "067_TripAdvisor", "068_WorldBank_Awards", "069_Taxonomy", "070_OpenFoodFacts",
        "071_COL", "072_Admissions", "073_Med_Cost", "074_Lift", "075_Mortality", "076_NBA",
        "077_Gestational", "078_Fires", "079_Coffee", "080_Books")
DOMAIN = {"066_IBM_HR": "Business", "067_TripAdvisor": "Travel and Locations",
          "068_WorldBank_Awards": "Business", "069_Taxonomy": "Health", "070_OpenFoodFacts": "Business",
          "071_COL": "Travel and Locations", "072_Admissions": "Social Networks and Surveys",
          "073_Med_Cost": "Health", "074_Lift": "Sports and Entertainment", "075_Mortality": "Health",
          "076_NBA": "Sports and Entertainment", "077_Gestational": "Health",
          "078_Fires": "Social Networks and Surveys", "079_Coffee": "Business", "080_Books": "Business"}
SCALAR = ("boolean", "category", "number")
CACHE = data_root() / "cache" / "databench_hf"
DEFAULT_ROOT = data_root() / "benchmarks_ext" / "databench"
GENERATED = data_root() / "generated" / "databench" / "test"

GRADER = r'''"""DataBench grading, ported in behaviour from the official evaluator,
databench_eval.Evaluator.default_compare (github.com/jorses/databench_eval @ 65f92cf).

The official evaluator compares str(answer) with the gold STRING, by the question's type:
  boolean          true/yes/y vs false/no/n, case-insensitive
  category         equal after stripping [ ] ' " and spaces, or parse to the same date
  number           keep digits . and -, then trunc(100 x) equal  (truncation, not rounding)
  list[category]   split on commas, strip, same length and same set (or same dates)
  list[number]     split on commas, same length and same set after truncation to 2 decimals
and a null gold (nan/None/"") matches only a null answer. Deliberately NOT "improved", so scores
are comparable to the SemEval-2025 Task 8 leaderboard. The gold string is the key's `truth`.

`absolute_tolerance` in a key's config is NOT read here; it exists for rl/reward.py (dispersion).
"""
from __future__ import annotations

import math
import warnings

STRIP_CHARS = "[]'\" "
NULLS = [None, "nan", "", " ", float("nan"), "np.nan", "None"]


def _dates(items):
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [pd.to_datetime(x).date() for x in items]


def compare(value, truth, semantic: str) -> bool:
    semantic = semantic.strip()
    v0, t0 = str(value).strip(STRIP_CHARS), str(truth).strip(STRIP_CHARS)
    if v0 in NULLS and t0 in NULLS:
        return True
    if v0 in NULLS or t0 in NULLS:
        return False
    if semantic == "boolean":
        yes, no = ("true", "yes", "y"), ("false", "no", "n")
        v, t = v0.lower(), t0.lower()
        return (v in yes and t in yes) or (v in no and t in no)
    if semantic == "category":
        if v0 == t0:
            return True
        try:
            (vd,), (td,) = _dates([v0]), _dates([t0])
            return vd == td
        except (ValueError, TypeError, OverflowError):
            return v0 == t0
    if semantic == "number":
        try:
            vc = "".join(c for c in str(value) if c.isdigit() or c in ".-")
            tc = "".join(c for c in str(truth) if c.isdigit() or c in ".-")
            return math.trunc(float(vc) * 100) == math.trunc(float(tc) * 100)
        except Exception:
            return False
    if semantic == "list[category]":
        try:
            vl = [x.strip(STRIP_CHARS) for x in str(value).strip("[]").split(",")]
            tl = [x.strip(STRIP_CHARS) for x in str(truth).strip("[]").split(",")]
            vl = [x if x not in NULLS else "" for x in vl]
            tl = [x if x not in NULLS else "" for x in tl]
            if len(vl) != len(tl):
                return False
            try:
                return set(_dates(vl)) == set(_dates(tl))
            except (ValueError, TypeError, OverflowError):
                return set(vl) == set(tl)
        except Exception:
            return False
    if semantic == "list[number]":
        try:
            def nums(s):
                return sorted(float("".join(c for c in x.strip() if c.isdigit() or c in ".-"))
                              for x in str(s).strip("[]").split(",") if x.strip())
            vl = [int(x * 100) / 100 for x in nums(value)]
            tl = [int(x * 100) / 100 for x in nums(truth)]
            if len(vl) != len(tl):
                return False
            return set(vl) == set(tl)
        except Exception:
            return False
    raise ValueError(f"semantic not supported: {semantic}")


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "databench_official":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gt = config.get("ground_truth") or {}
    keys = (grader.get("config") or {}).get("keys") or {}
    obj = submission or {}
    if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    if not isinstance(obj, dict):
        obj = {}
    details, correct = [], 0
    for k, spec in keys.items():
        truth = spec.get("truth", gt.get(k))
        ok = k in obj and compare(obj[k], truth, spec["semantic"])
        correct += bool(ok)
        details.append({"key": k, "semantic": spec["semantic"], "correct": bool(ok),
                        "expected": truth, "got": obj.get(k)})
    n = max(len(keys), 1)
    return {"eval_id": config.get("id"), "grader_type": "databench_official",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n,
            "details": details}


def grade(submission: dict, cfg: dict) -> dict:
    return evaluate(cfg, submission)
'''

TASK = """You are given a table of real-world data and a question about it.
Answer the question by computing the result from the full table.

File in data_files/:
  {fname}  ({rows:,} rows x {cols} columns; read it with pandas.read_parquet)

Question: {question}

Answer type: {kind}.{note}
Return your answer as JSON: {{"answer": {{"{pid}": {slot}}}}}"""

KIND = {"boolean": ("boolean (True or False)", "<true or false>", ""),
        "number": ("a single number", "<number>",
                   " Give it unrounded: numbers are compared to two decimal places."),
        "category": ("a single category -- one value written as it appears in the data, e.g. a "
                     "name, label or date", '"<value>"', ""),
        "list[category]": ("a list of categories -- values written as they appear in the data",
                           '["<value>", ...]', ""),
        "list[number]": ("a list of numbers", "[<number>, ...]",
                         " Give them unrounded: numbers are compared to two decimal places.")}


# --------------------------------------------------------------------------- fetch
def fetch() -> Path:
    """The 15 test tables and their QA files, anonymously, at the pinned revision."""
    need = [f"data/{d}/{f}" for d in TEST for f in ("all.parquet", "qa.parquet")] + ["README.md"]
    if all((CACHE / p).exists() for p in need):
        return CACHE
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=need,
                      local_dir=str(CACHE), token=False, max_workers=2)
    readme = (CACHE / "README.md").read_text()
    test_block = readme.split("split: test", 1)[1].split("---", 1)[0]
    listed = [ln.split("data/")[1].split("/")[0] for ln in test_block.splitlines() if "data/" in ln]
    assert tuple(listed) == TEST, f"the card's semeval test split changed: {listed}"
    return CACHE


def parse_gold(ans, semantic: str):
    """The gold string as a JSON value: the form an oracle submits and rl/reward.py reads."""
    s = str(ans)
    if semantic == "boolean":
        t = s.strip("[]'\" ").lower()
        return "True" if t in ("true", "yes", "y") else "False" if t in ("false", "no", "n") else s
    if semantic == "number":
        c = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
        try:
            f = float(c)
        except ValueError:
            return s
        if not math.isfinite(f):
            return s
        return int(f) if f.is_integer() and "." not in c else f
    if semantic.startswith("list"):
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple)) else s
        except (ValueError, SyntaxError):
            return s
    return s


def link(src: Path, dst: Path) -> None:
    """522 problems share 15 tables: hard-link (else copy), and make it read-only -- a shared inode
    is what scripts/protect_benchmark_data.sh guards, since a cell writing to data_files/ under
    Sandbox(link_data=True) would otherwise corrupt every problem on that table."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode & ~0o222)


def build(root: Path) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    fetch()
    (root / "problems").mkdir(parents=True, exist_ok=True)
    (root / "reference_grader.py").write_text(GRADER)
    made, types = 0, collections.Counter()
    for ds in TEST:
        table = CACHE / "data" / ds / "all.parquet"
        meta = pq.ParquetFile(table).metadata
        qa = pd.read_parquet(CACHE / "data" / ds / "qa.parquet")
        for i, r in enumerate(qa.itertuples(index=False)):
            pid = f"db-{ds}-q{i:02d}"
            sem = str(r.type).strip()
            fname = f"{ds}.parquet"
            pdir = root / "problems" / pid
            link(table, pdir / "data_files" / fname)
            kind, slot, note = KIND[sem]
            key = {"semantic": sem, "truth": str(r.answer)}
            g = parse_gold(r.answer, sem)
            if sem == "number" and isinstance(g, (int, float)):
                key["absolute_tolerance"] = 0.005      # rl/reward.py only; see module docstring
            cfg = {
                "id": pid,
                "task": TASK.format(fname=fname, rows=meta.num_rows, cols=meta.num_columns,
                                    question=str(r.question).strip(), kind=kind, note=note,
                                    pid=pid, slot=slot),
                "data_files": [fname],
                "ground_truth": {pid: g},
                "grader": {"type": "databench_official", "config": {"keys": {pid: key}}},
                "_provenance": {
                    "benchmark": "DataBench (SemEval-2025 Task 8, test split)",
                    "paper": "Osés Grijalba et al., LREC-COLING 2024; SemEval-2025 Task 8",
                    "source": f"https://huggingface.co/datasets/{REPO_ID}", "version": REVISION,
                    "evaluator": EVALUATOR, "licence": LICENCE,
                    "dataset": ds, "domain": DOMAIN[ds], "question_index": i, "semantic": sem,
                    "table_rows": meta.num_rows, "table_cols": meta.num_columns,
                    # DataBench-lite's answer on the 20-row sample -- a floor, never shown to agents.
                    "sample_answer": str(r.sample_answer),
                },
            }
            (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1, default=str))
            made += 1
            types[sem] += 1
    link_generated(root)
    return {"n": made, "by_semantic": dict(types)}


def link_generated(root: Path) -> int:
    """generated/databench/test: symlinks to the scalar-answer problems, for rl/dispersion.py."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    for p in GENERATED.iterdir():
        if p.is_symlink():
            p.unlink()
    n = 0
    for p in sorted((root / "problems").iterdir()):
        cfg = json.loads((p / "eval_config.json").read_text())
        (k, g), = cfg["ground_truth"].items()
        sem = cfg["_provenance"]["semantic"]
        if sem in SCALAR and not (sem == "number" and isinstance(g, str)):
            (GENERATED / p.name).symlink_to(p)
            n += 1
    (GENERATED / "reference_grader.py").symlink_to(root / "reference_grader.py")
    return n


# --------------------------------------------------------------------------- verify
def selftest(g) -> None:
    """The ported comparator against the official one's behaviour, case by case."""
    c = g.compare
    assert c(True, "True", "boolean") and c("yes", "True", "boolean") and not c("False", "True", "boolean")
    assert c("Travel_Rarely", "Travel_Rarely", "category") and not c("travel_rarely", "Travel_Rarely", "category")
    assert c("2018-01-05", "2018-01-05 00:00:00", "category")            # same date
    assert c(36.923, "36.92", "number") and c("36.92 years", "36.92", "number")
    assert not c(36.9, "36.92", "number") and not c(36.93, "36.92", "number")
    assert c(0.29, "0.29", "number") and not c(0.2900001, "0.29", "number")  # truncation, as published
    assert c(["b", "a"], "['a', 'b']", "list[category]") and not c(["a"], "['a', 'b']", "list[category]")
    assert c([2.759, 1.5], "[1.5, 2.75]", "list[number]") and not c([1.5], "[1.5, 2.75]", "list[number]")
    assert c([2.01], "[2.01]", "list[number]")      # int(2.01*100) == 200 on both sides, so they agree
    assert c("", "nan", "number") and not c("", "3", "number") and not c(None, "True", "boolean")
    cfg = {"id": "t", "ground_truth": {"t": 3.5},
           "grader": {"type": "databench_official", "config": {"keys": {"t": {"semantic": "number", "truth": "3.5"}}}}}
    assert g.evaluate(cfg, {"answer": {"t": 3.509}})["passed"] and not g.evaluate(cfg, {"answer": {}})["passed"]


def verify(root: Path) -> dict:
    probs = sorted(p for p in (root / "problems").iterdir() if (p / "eval_config.json").exists())
    bad = [f for p in probs for f in (p / "data_files").iterdir() if f.suffix != ".parquet"]
    assert not bad, f"non-parquet files reached data_files/: {bad[:5]}"

    sys.path.insert(0, str(PROJECT_ROOT / "harness"))
    import runner  # noqa: E402
    from rl.dispersion import instance_tol  # noqa: E402
    from rl.reward import StepReward  # noqa: E402
    assert [p.name for p in runner.problems(root)] == [p.name for p in probs]
    g = runner.load_grader(probs[0])
    assert g.__file__ == str(root / "reference_grader.py"), g.__file__
    selftest(g)

    cfgs = [json.loads((p / "eval_config.json").read_text()) for p in probs]
    oracle_fail, null_pass = [], []
    for p, cfg in zip(probs, cfgs):
        assert cfg["id"] == p.name and (p / "data_files" / cfg["data_files"][0]).is_file()
        if not g.evaluate(cfg, {"answer": cfg["ground_truth"]})["passed"]:
            oracle_fail.append(p.name)
        null = {k: "" if isinstance(v, str) else 0 for k, v in cfg["ground_truth"].items()}
        if g.evaluate(cfg, {"answer": null})["passed"]:
            null_pass.append(p.name)
    assert not oracle_fail, f"gold answers fail their own grader: {oracle_fail}"

    linked = sorted(p for p in GENERATED.iterdir() if (p / "eval_config.json").exists())
    assert runner.load_grader(linked[0]).__file__.endswith("reference_grader.py")
    disagree = []
    for p in linked:
        cfg = json.loads((p / "eval_config.json").read_text())
        sr = StepReward(tolerance=instance_tol(cfg, 0.025))
        r = sr.score(cfg, {"answer": cfg["ground_truth"]})
        assert r.passed and r.reward == 1.0, (p.name, r)
        null = {k: "" if isinstance(v, str) else 0 for k, v in cfg["ground_truth"].items()}
        if sr.score(cfg, {"answer": null}).passed != g.evaluate(cfg, {"answer": null})["passed"]:
            disagree.append(p.name)

    report = {"n_problems": len(probs), "n_dispersion_view": len(linked),
              "by_semantic": dict(collections.Counter(c["_provenance"]["semantic"] for c in cfgs)),
              "by_dataset": dict(collections.Counter(c["_provenance"]["dataset"] for c in cfgs)),
              "null_agent_passes_gold_is_zero_or_null": null_pass,
              "dispersion_null_disagreements": disagree,
              "baselines": baselines(g, cfgs)}
    print(json.dumps(report, indent=1))
    print(f"verify OK: {len(probs)} problems load through harness/runner.py, every gold answer "
          f"passes its own grader, grader self-test passes, {len(linked)} scalar problems score "
          f"1.0 under rl/reward.py in {GENERATED}")
    return report


def baselines(g, cfgs: list[dict]) -> dict:
    """Pass rates of answers that ignore the table.

    `modal_by_type` answers the most frequent gold of the question's type, chosen IN-SAMPLE over
    the test split -- optimistic, the conservative way to state a floor. `lite` answers what the
    20-row sample gives (DataBench-lite's gold): what reading a few rows, not the table, earns.
    """
    by_sem = collections.defaultdict(list)
    for c in cfgs:
        by_sem[c["_provenance"]["semantic"]].append(next(iter(c["grader"]["config"]["keys"].values()))["truth"])
    modal = {s: collections.Counter(v).most_common(1)[0][0] for s, v in by_sem.items()}
    rules = {
        "null": lambda c, v: "" if isinstance(v, str) else 0,
        "always_True": lambda c, v: True,
        "modal_by_type": lambda c, v: modal[c["_provenance"]["semantic"]],
        "lite_sample_answer": lambda c, v: c["_provenance"]["sample_answer"],
    }
    out = {}
    for name, fn in rules.items():
        per = collections.Counter()
        tot = 0
        for c in cfgs:
            ok = g.evaluate(c, {"answer": {k: fn(c, v) for k, v in c["ground_truth"].items()}})["passed"]
            tot += ok
            per[c["_provenance"]["semantic"]] += ok
        out[name] = {"pass_rate": round(tot / len(cfgs), 4),
                     "by_semantic": {s: f"{per[s]}/{len(by_sem[s])}" for s in by_sem}}
    out["modal_answers"] = modal
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="benchmark root (default $DEEP_DATA_ROOT/benchmarks_ext/databench)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        m = build(a.root)
        print(f"built {m['n']} problems in {a.root}: {m['by_semantic']}")
    verify(a.root)
