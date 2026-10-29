"""EHRSQL-2024 reliability scores RS(0) / RS(5) / RS(10) / RS(N), exactly as the official scoring.py prints them.

The harness grades each problem to an item score in {+1, 0, -1} (grade["reliability"], see
scripts/ehrsql2024_reference_grader.py); this aggregates them with the official penalize(). RS(10) is the shared task's
primary metric; RS(0) equals the harness pass rate.

  venv/bin/python scripts/ehrsql2024_score.py RUNNER_OUTDIR [--problems benchmarks/ehrsql2024]
  venv/bin/python scripts/ehrsql2024_score.py --prediction-json pred.json [--problems ...]   # official file format
  venv/bin/python scripts/ehrsql2024_score.py --abstain-all [--problems ...]                  # the ABSTAIN-ALL baseline

A problem with NO submission (the agent never emitted an answer, or the harness errored) has no official analogue --
the official scorer requires a prediction for every id. It is counted as an ABSTENTION by default (the system gave the
user no answer) and the count is printed; --missing error scores it -1 instead. Both totals are printed when any exist.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_grader(bench: Path):
    spec = importlib.util.spec_from_file_location("ehrsql_grader", bench / "reference_grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?", help="runner.py --outdir of an EHRSQL-2024 run")
    ap.add_argument("--problems", default=str(ROOT / "benchmarks" / "ehrsql2024"))
    ap.add_argument("--prediction-json", default=None, help="official format: {ehrsql_id: SQL or 'null'}")
    ap.add_argument("--abstain-all", action="store_true")
    ap.add_argument("--missing", choices=("abstain", "error"), default="abstain")
    ap.add_argument("--json", default=None, help="also write the scores here")
    a = ap.parse_args()

    bench = Path(a.problems).resolve()
    if bench.name == "problems":
        bench = bench.parent
    g = load_grader(bench)
    cfgs = {}
    for p in sorted((bench / "problems").glob("*/eval_config.json")):
        c = json.loads(p.read_text())
        cfgs[c["id"]] = c

    def answerable(cfg):
        return next(iter(cfg["ground_truth"].values())) != "null"

    items: dict[str, int] = {}          # problem id -> reliability
    missing: list[str] = []
    abstained: set[str] = set()
    if a.abstain_all or a.prediction_json:
        preds = json.loads(Path(a.prediction_json).read_text()) if a.prediction_json else {}
        for pid, cfg in cfgs.items():
            key, spec = next(iter(cfg["grader"]["config"]["keys"].items()))
            sql = "null" if a.abstain_all else preds.get(spec["ehrsql_id"])
            if sql is None:
                missing.append(pid)
                continue
            r = g.evaluate(cfg, {"answer": {key: sql}})
            items[pid] = r["reliability"]
            abstained |= {pid} if r["details"][0]["abstained"] else set()
    else:
        if not a.outdir:
            ap.error("give a runner outdir, --prediction-json or --abstain-all")
        for f in sorted(Path(a.outdir).glob("*.json")):
            if f.name == "summary.json":
                continue
            rec = json.loads(f.read_text())
            pid = rec.get("id")
            if pid not in cfgs:
                continue
            grade = rec.get("grade") or {}
            if "reliability" in grade:
                items[pid] = grade["reliability"]
                if grade.get("details") and grade["details"][0].get("abstained"):
                    abstained.add(pid)
            else:
                missing.append(pid)
        not_run = sorted(set(cfgs) - set(items) - set(missing))
        if not_run:
            print(f"NOTE: {len(not_run)} of {len(cfgs)} problems have no result file in {a.outdir} "
                  f"(partial run); scores below are over the {len(items) + len(missing)} that do.")
            cfgs = {k: v for k, v in cfgs.items() if k not in set(not_run)}

    def totals(miss_as: str) -> dict:
        scores = []
        for pid, cfg in cfgs.items():
            if pid in items:
                scores.append(items[pid])
            else:   # missing submission
                scores.append(-1 if miss_as == "error" else (0 if answerable(cfg) else 1))
        return g.reliability_scores(scores) if scores else {}

    ans = [p for p in cfgs if answerable(cfgs[p])]
    una = [p for p in cfgs if not answerable(cfgs[p])]
    res = totals(a.missing)
    out = {
        "benchmark": bench.name, "n": len(cfgs), "n_answerable": len(ans), "n_unanswerable": len(una),
        "n_missing_submission": len(missing), "missing_counted_as": a.missing,
        **{k: round(float(v), 2) for k, v in res.items()},
        "answerable_correct": sum(items.get(p) == 1 for p in ans),
        "answerable_abstained": sum(p in abstained for p in ans),
        "answerable_wrong": sum(items.get(p) == -1 for p in ans),
        "unanswerable_abstained": sum(items.get(p) == 1 for p in una),
        "unanswerable_answered": sum(items.get(p) == -1 for p in una),
    }
    if missing:
        alt = totals("error" if a.missing == "abstain" else "abstain")
        out["if_missing_counted_as_" + ("error" if a.missing == "abstain" else "abstain")] = {
            k: round(float(v), 2) for k, v in alt.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
