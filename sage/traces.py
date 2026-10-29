"""Executed demonstration traces of a family's own reference code (generalises rl/traces_astro.py).

The contract every trace builder in rl/ converged on after templated traces trained a model
to 0.000 (rl/trace_common.py's docstring), kept here once for every family:

  * each cell is EXECUTED in the harness's Sandbox against the instance's own data files,
    and the real stdout is the tool response;
  * SYSTEM and build_prompt are harness/agents' own, so training and inference see the
    same conversation shape;
  * the cells run the family's reference analysis -- `Cell.uses` embeds its functions'
    source verbatim -- so the demonstration and the ground truth are the same code;
  * the answer is PARSED from the last cell's `FINAL k1 = v1 k2 = v2` line, never copied
    from the ground truth, and graded with the instance's own grader; a trace that fails
    it is discarded (never ship a wrong-answer trace).

A family opts in through `Family.demo(cfg) -> [Cell]`.

  venv/bin/python -m sage traces --family enzyme_mm --gen ROOT --n 20 --out PREFIX
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sage import ROOT

sys.path.insert(0, str(ROOT / "harness"))

NUM = r"(-?[\d.]+(?:[eE][-+]?\d+)?)"


def final_re(keys) -> re.Pattern:
    return re.compile(r"FINAL\s+" + r"\s+".join(rf"{k}\s*=\s*{NUM}" for k in keys))


def cell_code(cell) -> str:
    """The code a cell executes: preamble, then the verbatim source of `uses`, then `code`."""
    if not cell.uses and not cell.preamble:
        return cell.code
    src = "\n\n".join(inspect.getsource(fn).rstrip() for fn in cell.uses)
    head = (cell.preamble + "\n\n") if cell.preamble else ""
    return head + (src + "\n\n\n" if src else "") + cell.code


def trace(fam, cfg: dict, inst_dir: Path) -> dict | None:
    from agents import SYSTEM, build_prompt            # harness/agents.py
    from sandbox import Sandbox                        # harness/sandbox.py
    from rl.trace_common import grade
    cells = fam.demo(cfg)
    if not cells:
        raise ValueError(f"{fam.name} has no demo hook")
    sb = Sandbox(data_dir=Path(inst_dir) / "data_files", timeout=fam.demo_timeout)
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
        outs: list[str] = []
        for cell in cells:
            code = cell_code(cell)
            res = sb.run(code)
            if res.returncode != 0 or res.timed_out:
                return None
            text = cell.prose(outs) if callable(cell.prose) else cell.prose
            msgs.append({"role": "assistant", "content": f"{text}\n\n```python\n{code}\n```"})
            msgs.append({"role": "user", "content": res.as_feedback()})
            outs.append(res.stdout)
        hit = final_re(fam.keys).search(msgs[-1]["content"])
        if not hit:
            return None
        ans = {k: float(v) for k, v in zip(fam.keys, hit.groups())}
        if not grade(cfg, ans):
            return None                                    # never ship a wrong-answer trace
        msgs.append({"role": "assistant", "content": "```json\n" + json.dumps({"answer": ans}, indent=2) + "\n```"})
        return {"id": cfg["id"], "messages": msgs, "reward": 1.0, "n_cells": sb.attempts, "_family": fam.name}
    finally:
        sb.cleanup()


def build(fam, gen_root: Path, n: int, val: int, out: str, split: str = "train", workers: int = 1) -> dict:
    """First n+val successful traces over <gen_root>/<family>/<split>/, in instance order."""
    paths = sorted((Path(gen_root) / fam.name / split).glob("*/eval_config.json"))
    traces, bad, w = [], [], max(1, workers)
    with ThreadPoolExecutor(w) as ex:
        for i0 in range(0, len(paths), w):
            if len(traces) >= n + val:
                break
            batch = paths[i0:i0 + w]
            for p, t in zip(batch, ex.map(lambda q: trace(fam, json.loads(q.read_text()), q.parent), batch)):
                if t is None:
                    bad.append(p.parent.name)
                elif len(traces) < n + val:
                    traces.append(t)
    nv = min(val, max(len(traces) - 1, 0))
    tr, va = traces[:len(traces) - nv], traces[len(traces) - nv:]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr))
    Path(f"{out}_val.jsonl").write_text("".join(json.dumps(t) + "\n" for t in va))
    return {"family": fam.name, "train": len(tr), "val": len(va), "discarded": bad}
