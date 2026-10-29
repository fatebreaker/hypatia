"""Adapt BioDSBench (Python tasks) into harness-compatible problems (healthcare / biomedical data-science coding).

SOURCE    huggingface.co/datasets/zifeng-ai/BioDSBench (Wang, Danek, Yang, Chen, Sun; "Can Large Language Models Replace
          Data Scientists in Biomedical Research?", arXiv 2410.21591; Nature Biomedical Engineering 2026, "Making large
          language models reliable data science programming copilots for biomedical research"). Code repo
          github.com/RyanWangZf/BioDSBench (moved to RyanWangZf/BioDSA) -- NEITHER ships the test runner, so the
          protocol below is built from the paper's rule and the dataset's fields.
LICENSE   MIT (dataset card). Data: cBioPortal / UCSC Xena study tables shipped in the dataset's tarball.
PINNED    dataset revision REV below (python_tasks_with_class.jsonl + data_files/raw_patient_data_for_python_tasks.tar.gz).
TASK      a biomedical analysis request (survival curves, mutation frequencies, CNA status, ...) over 1 study's tables;
          the answer is a PROGRAM that defines the requested variables. Graded by the task's own unit tests.

n: 118 Python tasks at REV (the paper says 128; the released file has 118) -> 107 KEPT, 11 DROPPED (DROP below):
   7 whose TESTS import PyComplexHeatmap and 1 that asks for a `ridgeplot` HTML figure -- packages absent from the
     harness sandbox interpreter (the shared venv is read-only by policy); restorable with a sandbox interpreter that
     has them ($GBP_SANDBOX_PYTHON) and `--keep-missing-packages`;
   3 whose reference solution fails its OWN tests on the released data: 28481359_4/_7/_8 read
     /workdir/gene_{mutation_dna,expression_rna}_sub.csv and index genes as COLUMNS, while the released files -- and
     tasks 28481359_0/_1, which read the very same paths and pass -- have genes as ROWS. Unsatisfiable as released.
ORACLE: the released reference passes its tests on 105/118. Of the 107 kept, 2 references are PATCHED for the oracle
   only (tests untouched; REF_PATCH below): 28472509_4 uses pd/KaplanMeierFitter without importing them;
   37699004_1 draws an incidental plotly figure the task never asks for. After that the oracle passes 107/107.

DESIGN (grading an agent in a notebook harness by the benchmark's own tests)
  The agent explores and iterates in the sandbox as usual, then writes its FINAL self-contained program to
  solution.py in its working directory and answers {"<key>": "solution.py"}. harness/runner.py attaches that file to
  the submission (opt-in: the grader declares SUBMISSION_FILES; no other grader does, so nothing else changes), and
  the grader runs  prefix + solution.py + the task's tests  in a FRESH process with the study's tables in
  data_files/ -- the single-shot "code must pass all the testing cases" protocol, applied to the agent's final code
  rather than to its notebook state (exploration cells that leave the right variables behind do not count).
  Inline code as the answer value is also accepted (the oracle uses it; so can a harness without the opt-in).
DATA      each study's tables converted as the upstream workdir had them: raw TSV -> <table>.csv, pandas
  read_csv(sep="\\t") -> to_csv(index=False), ALL columns kept (the published schema previews drop all-empty columns,
  but 28985567_3 reads an all-empty column: keeping them is what the tests require; verified). `/workdir` and
  `./workdir` are rewritten to `data_files` in prefix/tests/task text; nothing else in a test is changed.
PROMPT    "vanilla" setting: study description, table list, the prefix code (already executed) and the query. The
  dataset's cot_instructions (the paper's CoT setting) are kept in _provenance, not shown; --cot builds
  benchmarks/biodsbench_cot with them shown.

PUBLISHED NUMBERS (arXiv 2410.21591 abstract, over the paper's 293 Python+R tasks and six LLMs, NOT split by model or
  language on the source page): chain-of-thought 56.6% vs 35.3% vanilla code accuracy; self-reflection 45.5% vs 34.3%.
  Paper text: Pass@1 (temperature 0) "0.40-0.80" on Easy, "0.15-0.40" Medium, "0.05-0.15" Hard Python tasks.
  Per-model Pass@1 values are in figures only -- UNVERIFIED here. Our n=107 and agentic protocol differ from both.

Run:
  venv/bin/python scripts/fetch_biodsbench.py                 # build + verify (oracle passes, empty/garbage fail)
  venv/bin/python scripts/fetch_biodsbench.py --verify-only
Evaluate (from harness/):
  ../venv/bin/python runner.py --agent llm:MODEL --problems ../benchmarks/biodsbench --workers 8 --outdir OUT
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

HF = "zifeng-ai/BioDSBench"
REV = "e59af82ee9461db78ed399544ec8520afeb02ce5"
FILES = ("python_tasks_with_class.jsonl", "python_task_table_schemas.jsonl",
         "data_files/raw_patient_data_for_python_tasks.tar.gz")
CACHE = data_root() / "cache" / "biodsbench" / REV
ROOT = PROJECT_ROOT / "benchmarks" / "biodsbench"
GRADER = Path(__file__).resolve().parent / "biodsbench_reference_grader.py"

MISSING_PKG = {
    "PyComplexHeatmap": ["28472509_8", "28472509_9", "32864625_4", "32864625_5", "25303977_6", "28985567_4",
                         "34819518_5"],
    "ridgeplot": ["28481359_5"],
}
DROP = {
    **{u: f"needs `{pkg}`, absent from the sandbox interpreter (its tests or requested output use it)"
       for pkg, us in MISSING_PKG.items() for u in us},
    **{u: "reference fails its own tests on the released data: indexes genes as columns of "
          "gene_*_sub.csv, which the released files (and tasks 28481359_0/_1 on the same paths) have as rows"
       for u in ("28481359_4", "28481359_7", "28481359_8")},
}
REF_PATCH = {   # oracle-only fixes to the REFERENCE solution; tests are never patched
    "28472509_4": [("import matplotlib.pyplot as plt\n",
                    "import pandas as pd\nfrom lifelines import KaplanMeierFitter\nimport matplotlib.pyplot as plt\n")],
    "37699004_1": [("import plotly.express as px\n", ""),
                   ("fig = px.pie(stage_counts, values='Count', names='Stage', title='Stage Categories Percentage "
                    "Ratio for Average Onset Patients')\n", ""),
                   ('fig.write_image("stage_categories_percentage_ratio.png")\n', ""),
                   ("fig.show()\n", "")],
}
WORKDIR = re.compile(r"(?<![\w/.])\.?/workdir")


def rw(text: str | None) -> str:
    return WORKDIR.sub("data_files", text or "")


def fetch() -> Path:
    for f in FILES:
        out = CACHE / f
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{HF}/resolve/{REV}/{f}", tmp)
            tmp.rename(out)
    ex = CACHE / "extracted"
    if not (ex / "datasets").is_dir():
        with tarfile.open(CACHE / FILES[2]) as t:
            t.extractall(ex)
    return ex / "datasets"


def convert(rows: list[dict], raw: Path) -> dict[str, list[str]]:
    import pandas as pd
    cfgs = {p.parent.name: p for p in raw.glob("*/*_dataset.json")}
    tables = {}
    for s in sorted({r["study_ids"] for r in rows}):
        cfg = json.loads(next(r for r in rows if r["study_ids"] == s)["study_data_configs"])
        subdirs = [d for d in cfgs[s].parent.iterdir() if d.is_dir()]
        assert len(subdirs) == 1, (s, subdirs)
        out = ROOT / "data" / s
        out.mkdir(parents=True, exist_ok=True)
        tables[s] = []
        for fname, name, fmt in cfg["tables"]:
            assert fmt == "tsv", (s, fname, fmt)
            dst = out / f"{name}.csv"
            if not dst.exists():
                tmp = out / f"{name}.csv.part"
                pd.read_csv(subdirs[0] / fname, sep="\t", low_memory=False).to_csv(tmp, index=False)
                os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                tmp.rename(dst)
            tables[s].append(f"{name}.csv")
    return tables


TASK = """You are given tables from a published biomedical study and a data-science coding task.

Study: {title}
Dataset: {dataset}
Tables (CSV, under data_files/):
  {tables}
{prefix_block}
Task:
{query}
{cot_block}
What to deliver. Explore the data and develop your code in the notebook as usual. Your FINAL deliverable is a
self-contained Python program saved as the file solution.py in your working directory, for example with
    SOLUTION = r'''<your code>'''
    open("solution.py", "w").write(SOLUTION)
It is graded by hidden unit tests (assert statements over the variables the task asks for), run in a FRESH Python
process as:  {order}  -- so solution.py must itself do everything the task needs (imports, loading data from
data_files/, defining every requested variable with exactly the requested name). Nothing else from your notebook
session carries over. Plots are allowed (non-interactive backend) but are not graded unless the task says so.
Available packages include pandas, numpy, scipy, statsmodels, scikit-learn, lifelines, matplotlib and seaborn.
Check it before answering, e.g. exec(open("solution.py").read()) in a fresh namespace.

Return your answer as JSON: {{"answer": {{"{key}": "solution.py"}}}}"""


def build(cot: bool, keep_missing: bool) -> tuple[int, list]:
    raw = fetch()
    rows = [json.loads(line) for line in open(CACHE / FILES[0])]
    assert len(rows) == 118, len(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GRADER, ROOT / "reference_grader.py")
    tables = convert(rows, raw)
    if (ROOT / "problems").exists():
        shutil.rmtree(ROOT / "problems")
    drop = {u: why for u, why in DROP.items()
            if not (keep_missing and u in sum(MISSING_PKG.values(), []))}
    kept = []
    for r in rows:
        uid = r["unique_question_ids"]
        if uid in drop:
            continue
        pid, s = f"bds-{uid}", r["study_ids"]
        key = f"{pid}-q1"
        cfg_s = json.loads(r["study_data_configs"])
        prefix = rw(r["code_histories"])
        ref = rw(r["reference_answer"])
        for old, new in REF_PATCH.get(uid, []):
            assert old in ref, (uid, old)
            ref = ref.replace(old, new)
        pdir = ROOT / "problems" / pid
        (pdir / "data_files").mkdir(parents=True)
        for t in tables[s]:
            os.link(ROOT / "data" / s / t, pdir / "data_files" / t)
        prefix_block = ("\nThe following code has ALREADY been run (prefix); your solution runs after it, in the same "
                        f"namespace:\n```python\n{prefix.strip()}\n```\n") if prefix.strip() else ""
        cot_block = f"\nStep-by-step plan:\n{r['cot_instructions']}\n" if cot and r.get("cot_instructions") else ""
        order = "prefix, then solution.py, then the tests" if prefix.strip() else "solution.py, then the tests"
        cfg = {
            "id": pid,
            "task": TASK.format(title=r["study_title"], dataset=cfg_s.get("dataset_name", ""),
                                tables="\n  ".join(f"data_files/{t}" for t in tables[s]), prefix_block=prefix_block,
                                query=rw(r["queries"]).strip(), cot_block=cot_block, order=order, key=key),
            "data_files": tables[s],
            "ground_truth": {key: ref},             # the reference PROGRAM: what the oracle submits (inline code)
            "grader": {"type": "biodsbench_tests", "config": {"keys": {key: {
                "prefix": prefix, "tests": rw(r["test_cases"]), "study": s}}}},
            "_provenance": {
                "benchmark": "BioDSBench (python)", "dataset": f"{HF}@{REV}", "unique_question_id": uid,
                "study_id": s, "study_title": r["study_title"], "dataset_url": r["dataset_url"],
                "analysis_types": r["analysis_types"], "study_types": r["study_types"],
                "reference_patched": uid in REF_PATCH, "cot_instructions": r.get("cot_instructions"),
                "paper": "arXiv:2410.21591",
            },
        }
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        kept.append(uid)
    (ROOT / "manifest.json").write_text(json.dumps({
        "dataset": f"{HF}@{REV}", "n_source": len(rows), "n_kept": len(kept),
        "dropped": drop, "reference_patched": sorted(REF_PATCH)}, indent=1))
    return len(kept), sorted(drop)


def load_grader():
    spec = importlib.util.spec_from_file_location("biods_grader", ROOT / "reference_grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(workers: int) -> None:
    g = load_grader()
    cfgs = [json.loads(p.read_text()) for p in sorted((ROOT / "problems").glob("*/eval_config.json"))]

    def one(cfg):
        key = next(iter(cfg["ground_truth"]))
        ok = g.evaluate(cfg, {"answer": dict(cfg["ground_truth"])})
        empty = g.evaluate(cfg, {"answer": {key: ""}})                       # what the harness null agent sends
        exit0 = g.evaluate(cfg, {"answer": {key: "import sys\nsys.exit(0)\n"}})   # must not skip the tests
        viafile = g.evaluate(cfg, {"answer": {key: "solution.py"}, "_files": {"solution.py": cfg["ground_truth"][key]}})
        return cfg["id"], ok, empty["passed"], exit0["passed"], viafile["passed"]

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, cfgs))
    bad = [(i, (ok["details"][0].get("failed_stage"), ok["details"][0].get("stderr_tail", "")[-200:]))
           for i, ok, *_ in res if not ok["passed"]]
    trivially = [i for i, _, e, *_ in res if e]
    assert not bad, f"oracle fails {len(bad)}: {bad[:5]}"
    assert not trivially, f"an EMPTY solution passes (prefix alone satisfies the tests): {trivially}"
    assert not any(x for *_, x, _ in res), "a solution that exits early passed"
    assert all(v for *_, v in res), "the solution.py (attached file) route disagrees with inline"
    print(f"[biodsbench] {len(cfgs)} problems: oracle (reference) passes {len(cfgs)}/{len(cfgs)} inline and via "
          f"solution.py; empty solution fails all; sys.exit(0) fails all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--cot", action="store_true", help="show the dataset's cot_instructions (the paper's CoT setting)")
    ap.add_argument("--keep-missing-packages", action="store_true",
                    help="keep the PyComplexHeatmap/ridgeplot tasks (needs a sandbox interpreter that has them)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cot:                   # a separate benchmark, never over the vanilla one
        ROOT = PROJECT_ROOT / "benchmarks" / "biodsbench_cot"
    if not a.verify_only:
        n, dropped = build(a.cot, a.keep_missing_packages)
        print(f"[biodsbench] built {n} problems under {ROOT}; dropped {len(dropped)}: {dropped}")
    verify(a.workers)
