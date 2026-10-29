#!/usr/bin/env python
"""What a constant answer earns on the causal RL pool, fitted on train and scored on held-out test.

The manuscript (docs/nmi/manuscript.tex l.43, l.155) says the second domain is one where "no
constant answer can score, because every instance has its own effect". That holds for the
real CauSciBench analyses; on the GENERATED pool a policy can condition on what the prompt
states. Three data-ignoring policies, each constant chosen to maximise mean reward on the
train split and priced on the test split with rl/reward.py:

  pooled     one constant for every instance
  design     one constant per design (confounded / iv / did / rdd / att)
  context    one constant per context template -- the unit description the prompt names
             ("farmers in a development project", "municipal elections", ...)

Read-only over $DEEP_DATA_ROOT/generated/causal_<design>/{train,test}.

  venv/bin/python -m sage.tests.causal_constant_floor [--root DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DESIGNS = ("att", "confounded", "did", "iv", "rdd")
CONTEXT = re.compile(r"(?:Observational data on|Data on|Panel data on) \d+ ([^.]+?)(?: observed over|\.|;)")


def load(root: Path, split: str) -> list[dict]:
    return [json.loads(Path(f).read_text()) for d in DESIGNS
            for f in sorted(glob.glob(str(root / f"causal_{d}" / split / f"causal_{d}_*" / "eval_config.json")))]


def context(c: dict) -> str:
    m = CONTEXT.search(c["task"])
    return c["_provenance"]["family"] + " | " + (m.group(1) if m else "?")


def best_constant(r, train: list[dict], max_candidates: int = 300) -> float:
    cands = sorted({round(c["ground_truth"]["effect"], 6) for c in train})
    cands = cands[::max(1, len(cands) // max_candidates)]
    return max(cands, key=lambda v: st.mean(r.score(c, {"effect": v}).reward for c in train))


def price(r, v: float, test: list[dict]) -> tuple[float, float]:
    res = [r.score(c, {"effect": v}) for c in test]
    return st.mean(x.reward for x in res), st.mean(x.passed for x in res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="generated root (default $DEEP_DATA_ROOT/generated)")
    args = ap.parse_args()
    from paths import data_root
    from rl.reward import StepReward
    r = StepReward()
    root = Path(args.root) if args.root else data_root() / "generated"
    tr, te = load(root, "train"), load(root, "test")
    print(f"causal pool: {len(tr)} train, {len(te)} test instances")
    v = best_constant(r, tr)
    m, p = price(r, v, te)
    print(f"pooled   one constant {v:.4f}: held-out reward {m:.3f}, pass {p:.3f}")
    for label, key in (("design", lambda c: c["_provenance"]["family"]), ("context", context)):
        g_tr, g_te = defaultdict(list), defaultdict(list)
        for c in tr:
            g_tr[key(c)].append(c)
        for c in te:
            g_te[key(c)].append(c)
        tot_m = tot_p = n = 0
        for k in sorted(g_te):
            if k not in g_tr:
                continue
            v = best_constant(r, g_tr[k])
            m, p = price(r, v, g_te[k])
            tot_m, tot_p, n = tot_m + m * len(g_te[k]), tot_p + p * len(g_te[k]), n + len(g_te[k])
            if label == "context":
                print(f"   {k[:66]:<66} n_test {len(g_te[k]):>3}  const {v:>9.4f}  reward {m:.3f} pass {p:.3f}")
        print(f"{label:<8} one constant per {label}: held-out reward {tot_m / n:.3f}, pass {tot_p / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
