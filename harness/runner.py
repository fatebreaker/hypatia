#!/usr/bin/env python
"""Run agents against GeneBench-Pro public problems and grade them.

Grading imports the vendored reference_grader.py rather than reimplementing it,
so scores are exactly what OpenAI's grader produces. Do not "fix" that file.

Usage:
  python runner.py --agent oracle                # harness self-test, expect 10/10
  python runner.py --agent null                  # floor, expect ~0/10
  python runner.py --agent llm:gpt-5.6 --workers 4
  python runner.py --agent llm:Qwen/Qwen3-8B     # with GBP_BASE_URL=http://node:8000/v1

Results land in results/<agent>/<timestamp>/: per-problem JSON (submission,
grade, full transcript) plus summary.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PKG = BASE / "benchmarks" / "genebench-pro-public"

sys.path.insert(0, str(HERE))
from agents import make_agent          # noqa: E402
from sandbox import Sandbox            # noqa: E402


def load_grader(problem_dir: Path | None = None):
    """Import the benchmark's own grader so our scores match theirs.

    Resolved PER BENCHMARK, not globally: each benchmark root ships its own
    reference_grader.py (GeneBench-Pro grades numeric tolerance, BixBench
    grades multiple choice), so the right one is the nearest one above the
    problem. Generated instances live outside any benchmark root and fall back
    to the GeneBench grader, which is what they are graded by.
    """
    root = PKG
    if problem_dir is not None:
        for cand in (problem_dir, *problem_dir.parents):
            if (cand / "reference_grader.py").is_file():
                root = cand
                break
    spec = importlib.util.spec_from_file_location(
        f"reference_grader_{root.name.replace('-', '_')}", root / "reference_grader.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def problems(root: Path | None = None) -> list[Path]:
    """Problem dirs under `root`, which may be a benchmark package (with a
    problems/ subdir) or a flat directory of generated instances."""
    base = Path(root) if root else PKG
    if (base / "problems").is_dir():
        base = base / "problems"
    return sorted(p for p in base.iterdir() if (p / "eval_config.json").exists())


SUBMISSION_FILE_MAX_CHARS = 200_000


def _attach_submission_files(grader, submission, sb: Sandbox) -> None:
    """OPT-IN (2026-09-30, code-generation benchmarks such as benchmarks/biodsbench).

    A grader module that declares `SUBMISSION_FILES = ("solution.py", ...)` is graded on files the agent left in its
    working directory, not only on the JSON answer: each named file that exists is attached to the submission as
    submission["_files"][name] = text, BEFORE the sandbox is cleaned up, so the grader sees it and the per-problem
    record keeps it (regradable offline). No grader declared this before, so every existing benchmark is unaffected.
    """
    want = getattr(grader, "SUBMISSION_FILES", None)
    if not want or not isinstance(submission, dict):
        return
    files = {}
    for name in want:
        p = Path(sb.workdir) / name
        if p.is_file():
            files[name] = p.read_text(errors="replace")[:SUBMISSION_FILE_MAX_CHARS]
    submission["_files"] = files


def run_one(pdir: Path, agent_spec: str, outdir: Path, timeout: int) -> dict:
    cfg = json.loads((pdir / "eval_config.json").read_text())
    # GBP_RESUME=1 (2026-10-01; opt-in): reuse a problem's earlier result in this outdir -- same agent spec, graded, no
    # harness error -- so an eval cut short (a preempted pod) restarts where it stopped instead of from problem one.
    if os.environ.get("GBP_RESUME") == "1":
        prev = outdir / f"{cfg['id']}.json"
        if prev.exists():
            try:
                old = json.loads(prev.read_text())
                g = old.get("grade")
                if isinstance(g, str):
                    import ast
                    g = ast.literal_eval(g)
                if old.get("agent") == agent_spec and not old.get("error") and isinstance(g, dict) and "passed" in g:
                    old["grade"] = g
                    print(f"  [resumed] {cfg['id']}", flush=True)
                    return old
            except Exception:                              # noqa: BLE001 -- unreadable: run it again
                pass
    grader = load_grader(pdir)
    agent = make_agent(agent_spec)
    sb = Sandbox(data_dir=pdir / "data_files", timeout=timeout)
    rec: dict = {"id": cfg["id"], "agent": agent_spec}

    try:
        submission, transcript = agent.solve(cfg, sb)
        _attach_submission_files(grader, submission, sb)
        rec["submission"] = submission
        rec["n_cells"] = sb.attempts        # blocks submitted, incl. ones that raised
        rec["n_cell_failures"] = sb.failures
        if submission is None:
            rec["grade"] = {"passed": False, "score": 0.0,
                            "details": [{"status": "no_answer_emitted"}]}
        else:
            rec["grade"] = grader.evaluate(cfg, submission)
        rec["transcript"] = transcript
    except Exception:
        rec["error"] = traceback.format_exc()
        # Without this the transcript is lost on exception and the run looks
        # like "0 turns, never started" when it actually did a lot of work.
        rec.setdefault("n_cells", sb.attempts)
        rec.setdefault("n_cell_failures", sb.failures)
        rec["grade"] = {"passed": False, "score": 0.0,
                        "details": [{"status": "harness_error"}]}
    finally:
        sb.cleanup()

    (outdir / f"{cfg['id']}.json").write_text(json.dumps(rec, indent=2, default=str))
    mark = "PASS" if rec["grade"]["passed"] else "fail"
    print(f"  [{mark}] {cfg['id']:<48} score={rec['grade']['score']:.3f}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="oracle", help="oracle | null | llm:<model>")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300, help="per-cell seconds")
    ap.add_argument("--only", default=None, help="substring filter on problem id")
    ap.add_argument("--problems", default=None,
                    help="problem set dir (default: the public GeneBench-Pro package)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if not PKG.exists():   # reference_grader.py ships with the package
        print(f"ERROR: benchmark package not found at {PKG}", file=sys.stderr)
        return 2

    probs = problems(args.problems)
    if args.only:
        probs = [p for p in probs if args.only in p.name]
    if not probs:
        print("ERROR: no problems matched", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) if args.outdir else BASE / "results" / args.agent.replace(":", "_") / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"agent={args.agent}  problems={len(probs)}  workers={args.workers}")
    print(f"out={outdir}\n")

    if args.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(lambda p: run_one(p, args.agent, outdir, args.timeout), probs))
    else:
        recs = [run_one(p, args.agent, outdir, args.timeout) for p in probs]

    n = len(recs)
    passed = sum(r["grade"]["passed"] for r in recs)
    mean = sum(r["grade"]["score"] for r in recs) / n
    cells = sum(r.get("n_cells", 0) for r in recs)
    errs = [r["id"] for r in recs if r.get("error")]
    # An agent that answers without running code is guessing, however it scored.
    is_reference_agent = args.agent in ("oracle", "null")
    guessed = [] if is_reference_agent else [
        r["id"] for r in recs
        if r.get("n_cells", 0) == 0 and r.get("submission") is not None]

    # An answer produced without running any code is a guess. Report the raw
    # grade for transparency, but the adjusted figures are the honest ones.
    gset = set(guessed)
    adj_passed = sum(r["grade"]["passed"] for r in recs if r["id"] not in gset)
    adj_mean = sum(0.0 if r["id"] in gset else r["grade"]["score"] for r in recs) / n

    summary = {
        "agent": args.agent,
        "timestamp": stamp,
        "n_problems": n,
        "n_passed": passed,
        "pass_rate": passed / n,
        "mean_score": mean,
        "adjusted_pass_rate": adj_passed / n,
        "adjusted_mean_score": adj_mean,
        "n_cells_total": cells,
        "n_answered_without_code": len(guessed),
        "n_errors": len(errs),
        "per_problem": {r["id"]: {"passed": r["grade"]["passed"],
                                  "score": r["grade"]["score"],
                                  "n_cells": r.get("n_cells", 0),
                                  "n_cell_failures": r.get("n_cell_failures", 0),
                                  "error": bool(r.get("error"))} for r in recs},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*64}")
    print(f"pass rate : {passed}/{n} = {passed/n:.1%}")
    print(f"mean score: {mean:.3f}")
    print(f"cells run : {cells}   answered-without-code: {len(guessed)}   errors: {len(errs)}")
    if guessed:
        print(f"ADJUSTED  : {adj_passed}/{n} = {adj_passed/n:.1%}   mean {adj_mean:.3f}"
              f"   (guesses zeroed)")
    print(f"{'='*64}")

    # These tasks cannot be solved without computing on the data. Three earlier
    # runs reported a plausible-looking pass rate while executing ZERO code --
    # the score was measuring the harness, not the model. Refuse to exit clean
    # in that state so a meaningless number can never be quoted as a result.
    if not is_reference_agent and cells == 0:
        print("\n*** HARNESS FAILURE: 0 code cells executed across all problems.")
        print("*** The agent answered without touching the data; this score is")
        print("*** NOT a capability measurement. Check finish_reason, the token")
        print("*** budget vs --max-model-len, and code-block extraction.")
        if errs:
            print(f"*** problems that errored: {', '.join(errs)}")
        return 3
    if not is_reference_agent and guessed:
        print(f"\n*** WARNING: {len(guessed)} problem(s) answered with no code run:")
        print(f"***   {', '.join(guessed)}")
        print("*** Those answers are guesses; exclude them before reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
