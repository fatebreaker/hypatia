#!/usr/bin/env python
"""Shared engine for executed demonstration traces.

Factored out after the popgen/wf builders were written twice: the first pair
templated conversations without running anything -- placeholder tool responses,
a hand-written system prompt, and a final answer copied from `ground_truth`
rather than parsed from output. 220 such traces trained a model to 0.000 on the
real problem and 0.003 on its own instances, because what it actually learned
was that tool output carries no information.

Every family now shares this engine, so those defects cannot be reintroduced
one file at a time. The contract it enforces:

  * each cell is EXECUTED in the harness's own Sandbox against the instance's
    real data files, and the real stdout is the tool response;
  * SYSTEM and build_prompt are imported from `harness/agents` -- the exact
    objects the eval path uses, file listing included, so training and
    inference see the same conversation shape;
  * the answer is PARSED from executed output, never recomputed in Python;
  * the parsed answer is graded with the INSTANCE'S OWN grader config, handling
    composite graders (an exact-match string plus tolerance-checked floats),
    and a trace that fails its own instance is discarded.

Sandbox re-executes every prior cell on each call, so cost is quadratic in the
number of cells: keep expensive fits in ONE cell and print the final answer
from that same cell where possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

from agents import SYSTEM, build_prompt          # noqa: E402
from sandbox import Sandbox                      # noqa: E402


def grade(cfg: dict, answer: dict) -> bool:
    """Apply the instance's own grader config.

    Not StepReward: that assumes every graded field is numeric and dies on
    float("B") for families whose answer includes a categorical.
    """
    g = cfg.get("grader") or {}
    c = g.get("config") or {}
    gt = cfg["ground_truth"]
    ex = c.get("exact_match_keys") or {}
    num = dict(c.get("keys") or {})
    num.update(c.get("numeric_keys") or {})
    if not ex and not num:                      # plain multi-numeric
        num = {k: {"absolute_tolerance": 0.0} for k in gt}
    for k in ex:
        if str(answer.get(k)) != str(gt[k]):
            return False
    for k, spec in num.items():
        if k not in gt:
            continue
        tol = float(spec.get("absolute_tolerance", 0.0))
        try:
            if abs(float(answer[k]) - float(gt[k])) > tol:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def make(cfg: dict, inst_dir: Path, moves, parse, reasoning: str,
         timeout: int = 300) -> dict | None:
    # Sandbox copies data_dir's CONTENTS into `data_files/`, so it must be
    # given the instance's data_files -- handing it the instance directory
    # yields data_files/data_files/... and every cell dies on FileNotFoundError.
    sb = Sandbox(data_dir=inst_dir / "data_files", timeout=timeout)
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
        for prose, code in moves:
            res = sb.run(code)
            if res.returncode != 0 or res.timed_out:
                return None
            msgs.append({"role": "assistant",
                         "content": f"{prose}\n\n```python\n{code}\n```"})
            msgs.append({"role": "user", "content": res.as_feedback()})

        answer = parse(msgs[-1]["content"])
        if answer is None or not grade(cfg, answer):
            return None
        msgs.append({"role": "assistant", "content": "```json\n" + json.dumps(
            {"answer": answer, "reasoning": reasoning}, indent=2) + "\n```"})
        return {"id": cfg["id"], "messages": msgs, "reward": 1.0,
                "n_cells": sb.attempts}
    finally:
        sb.cleanup()


def run_cli(corpus: str, moves, parse, reasoning: str, timeout: int = 300) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=corpus)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--min-keep", type=int, default=10)
    args = ap.parse_args()

    traces, bad = [], 0
    for p in sorted(Path(args.corpus).glob("train/*/eval_config.json")):
        if len(traces) >= args.n + args.val:
            break
        t = make(json.loads(p.read_text()), p.parent, moves, parse, reasoning,
                 timeout)
        if t is None:
            bad += 1
            if bad <= 3:
                print(f"  discarded {p.parent.name}", flush=True)
            continue
        traces.append(t)
        if len(traces) % 25 == 0:
            print(f"  {len(traces)} traces ({bad} discarded)", flush=True)
    if len(traces) < args.min_keep:
        raise SystemExit(f"only {len(traces)} usable traces "
                         f"({bad} discarded) -- the demonstration is wrong, "
                         f"not the instances")
    out = Path(args.out)
    v = min(args.val, max(1, len(traces) // 8))
    Path(f"{out}_train.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in traces[:-v]))
    Path(f"{out}_val.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in traces[-v:]))
    print(f"wrote {len(traces)-v} train / {v} val to {out}_*.jsonl "
          f"({bad} discarded)")
    return 0
