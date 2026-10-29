#!/usr/bin/env python
"""Is a policy computing the answer, or exploiting the answer distribution?

Two families here FAIL `rl/reward.py --audit`: a policy that emits the split
mean and never reads the data passes 25 % of scRNA instances and 30 % of
`rare_burden` ones (§26). So a high solve rate on those families is not by
itself evidence of analysis, and the two checks that settle it are the ones in
this script.

  tolerance sweep   re-grade the SAME saved rollouts at progressively tighter
                    tolerances. A distribution-exploiting policy decays with
                    the constant-predictor baseline; one that computes the
                    answer barely moves. Reporting this curve is strictly
                    better than reporting any single tolerance, because it
                    shows the result does not depend on where the threshold
                    was set.

  fidelity          correlate the policy's answer against PER-INSTANCE truth.
                    A mean-guesser sits at 0 by construction, whatever its pass
                    rate. This is the check that cannot be gamed by a generous
                    grader.

Answers come from the harness's own `extract_answer` when a run predates
`dispersion.py` recording them, so what is analysed is what was graded.

  venv/bin/python scripts/answer_fidelity.py --family scrna_1k --split test \\
      --runs dr2_fs=SFT dr4_s150=RL@150
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from paths import data_root  # noqa: E402


def _answer(rollout: dict) -> dict | None:
    if rollout.get("answer") is not None:
        return rollout["answer"]
    from agents import extract_answer
    for m in reversed(rollout.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        got = extract_answer(m.get("content") or "")
        if isinstance(got, dict):
            return got.get("answer", got)
    return None


def _cfgs(family: str, split: str) -> dict[str, dict]:
    root = data_root() / "generated" / family / split
    return {p.parent.name: json.loads(p.read_text())
            for p in map(Path, glob.glob(str(root / "*/eval_config.json")))}


def load(run: str, cfgs: dict) -> tuple[list, dict]:
    """Returns (rollout pairs, per-instance answers).

    Pass rates are per ROLLOUT -- that is what a solve rate means. Fidelity is
    per INSTANCE, averaging the k rollouts first: the question is whether the
    policy's answer tracks the instance, and correlating raw rollouts folds in
    within-instance sampling noise, which drags the coefficient down without
    saying anything about distribution-exploitation. The two statistics differ
    materially (0.369 vs 0.617 for RL@150 on scRNA), so they must not be
    confused -- RESULTS §24c reports the per-instance one.
    """
    d = json.loads((data_root() / "rl" / f"{run}.json").read_text())
    pairs, per_inst = [], {}
    for iid, rolls in d["groups"].items():
        if iid not in cfgs:
            continue
        gt = cfgs[iid]["ground_truth"]
        answers = [_answer(r) for r in rolls]
        for a in answers:
            pairs.append((a, gt))
        per_inst[iid] = (answers, gt)
    return pairs, per_inst


def passes(pairs, keys, tols, scale) -> float:
    ok = 0
    for a, gt in pairs:
        if not isinstance(a, dict):
            continue
        try:
            ok += all(abs(float(a[k]) - float(gt[k])) <= tols[k] * scale for k in keys)
        except (KeyError, TypeError, ValueError):
            pass
    return ok / max(len(pairs), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True, help="name=label")
    ap.add_argument("--scales", default="1,0.6,0.4,0.2,0.1")
    args = ap.parse_args()

    cfgs = _cfgs(args.family, args.split)
    if not cfgs:
        raise SystemExit(f"no instances under generated/{args.family}/{args.split}")
    any_cfg = next(iter(cfgs.values()))
    keys = sorted(any_cfg["ground_truth"])
    kc = ((any_cfg.get("grader") or {}).get("config") or {}).get("keys") or {}
    tols = {k: float((kc.get(k) or {}).get("absolute_tolerance", 0.05)) for k in keys}

    truth = {k: [float(c["ground_truth"][k]) for c in cfgs.values()] for k in keys}
    mean = {k: st.fmean(truth[k]) for k in keys}
    print(f"family={args.family}/{args.split}  n={len(cfgs)}  keys={keys}")
    for k in keys:
        sd = st.pstdev(truth[k])
        ratio = tols[k] / sd if sd else float("inf")
        print(f"  {k}: tol={tols[k]:g}  sd={sd:.4f}  tol/sd={ratio:.3f}  "
              f"-> single-key floor {2*0.5*(1+math.erf(ratio/math.sqrt(2)))-1:.3f}")

    arms = []
    for spec in args.runs:
        name, _, label = spec.partition("=")
        pairs, per_inst = load(name, cfgs)
        arms.append((label or name, pairs, per_inst))

    scales = [float(x) for x in args.scales.split(",")]
    # Constant baseline over the SAME instances the arms were evaluated on --
    # the eval used 40 of the split's instances, so scoring the constant against
    # all of them would compare two different instance sets.
    const = [(mean, gt) for _, gt in arms[0][1]]
    print(f"\n{'tolerance x':>12s} {'constant':>10s} " +
          " ".join(f"{lab:>10s}" for lab, _, _ in arms))
    for sc in scales:
        row = [passes(const, keys, tols, sc)] + [passes(p, keys, tols, sc)
                                                  for _, p, _ in arms]
        print(f"{sc:12.2f} " + " ".join(f"{v:10.3f}" for v in row))

    print(f"\nfidelity: correlation with PER-INSTANCE truth "
          f"(a mean-guesser scores 0 whatever its pass rate)")
    print(f"{'arm':>12s} " + " ".join(f"{'corr:'+k:>18s} {'|err|':>8s}" for k in keys))
    for lab, _pairs, per_inst in arms:
        cells = []
        for k in keys:
            xs = []
            for answers, gt in per_inst.values():
                vals = [float(a[k]) for a in answers
                        if isinstance(a, dict) and isinstance(a.get(k), (int, float))
                        and math.isfinite(float(a[k]))]
                if vals:
                    xs.append((st.fmean(vals), float(gt[k])))
            if len(xs) > 2:
                p = np.array([x for x, _ in xs]); t = np.array([y for _, y in xs])
                good = np.isfinite(p) & np.isfinite(t)
                r = float(np.corrcoef(p[good], t[good])[0, 1]) if good.sum() > 2 else float("nan")
                cells.append(f"{r:18.3f} {np.abs(p[good]-t[good]).mean():8.3f}")
            else:
                cells.append(f"{'n/a':>18s} {'n/a':>8s}")
        print(f"{lab:>12s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
