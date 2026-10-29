#!/usr/bin/env python3
"""Compare two arms of a general-agent run, with a paired bootstrap over instances.

    venv/bin/python scripts/gen_compare.py DIR [--fam FAM] [--a ARM] [--b ARM]
    venv/bin/python scripts/gen_compare.py results/pods/general_agent_v8 --a selfver8 --b control

rl/dispersion.py writes {"groups": {instance_id: [rollout, ...]}} and a rollout carries "reward" and "passed" --
NOT "score", which is harness/runner.py's key for external benchmarks. Reading the wrong one returns silent zeros
(it cost two false alarms on 2026-09-17), so this file is the one place that knows the difference.

The bootstrap resamples INSTANCES, not rollouts: the k rollouts of one instance share a generated dataset and are not
independent, and treating them as such narrows every interval by roughly sqrt(k).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def arm(path: Path) -> dict[str, list[float]]:
    """instance id -> the rewards of its rollouts."""
    d = json.loads(path.read_text())
    return {inst: [r["reward"] for r in rolls] for inst, rolls in d["groups"].items()}


def passes(path: Path) -> dict[str, list[bool]]:
    d = json.loads(path.read_text())
    return {inst: [bool(r["passed"]) for r in rolls] for inst, rolls in d["groups"].items()}


def compare(a: dict[str, list[float]], b: dict[str, list[float]], n_boot: int = 20000, seed: int = 20260918):
    """Mean(a) - mean(b) with a paired bootstrap over whichever unit is actually replicated.

    The resampling unit is the INSTANCE: the k rollouts of one instance share a generated dataset and are not
    independent, so resampling rollouts would narrow every interval by roughly sqrt(k).

    But a real-data family has exactly ONE instance -- there is one SPECT file, one theophylline study -- and
    resampling a single instance returns that instance every time. The interval collapses to zero width and EVERY
    difference reads as significant: pk_real_theoph came back -0.052 [-0.052,-0.052]* from 16 rollouts. There the
    rollouts ARE the replication (independent draws of the model on one fixed dataset), so resample those instead and
    say which unit was used, because the two answer different questions: across instances generalises to new datasets
    of this kind, across rollouts only to new attempts at this one.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    rng = np.random.default_rng(seed)
    if len(shared) == 1:
        av = np.array(a[shared[0]], dtype=float)
        bv = np.array(b[shared[0]], dtype=float)
        diff = float(av.mean() - bv.mean())
        # unpaired: the arms' rollouts are not matched to each other, only to the shared dataset
        ai = rng.integers(0, len(av), size=(n_boot, len(av)))
        bi = rng.integers(0, len(bv), size=(n_boot, len(bv)))
        boots = av[ai].mean(1) - bv[bi].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return dict(n=1, unit="rollouts", a=float(av.mean()), b=float(bv.mean()), diff=diff,
                    lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))
    am = np.array([np.mean(a[i]) for i in shared])
    bm = np.array([np.mean(b[i]) for i in shared])
    diff = float(am.mean() - bm.mean())
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    boots = am[idx].mean(1) - bm[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(shared), unit="instances", a=float(am.mean()), b=float(bm.mean()), diff=diff,
                lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--a", default=None, help="treatment arm tag (default: every non-control arm)")
    ap.add_argument("--b", default="control")
    ap.add_argument("--fam", default=None)
    # A new arm is usually measured into its OWN directory (that is how a re-measurement avoids being skipped as
    # already-present), while its control sits in the directory the first sweep wrote. Without this the comparison
    # cannot be made at all: "no 'control' arm (have selfver8)".
    ap.add_argument("--b-dir", default=None, help="directory holding the baseline arm, if not the same one")
    args = ap.parse_args()

    for d in args.dirs:
        p = Path(d) if Path(d).is_absolute() else ROOT / d
        if not p.is_dir():
            print(f"{d}: not a directory"); continue
        fams = defaultdict(dict)
        for f in sorted(p.glob("*.json")):
            tag, _, fam = f.stem.partition("_")
            fams[fam][tag] = f
        print(f"\n== {p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p} ==")
        for fam, arms in sorted(fams.items()):
            if args.fam and fam != args.fam:
                continue
            bpath = arms.get(args.b)
            if args.b_dir:
                bd = Path(args.b_dir) if Path(args.b_dir).is_absolute() else ROOT / args.b_dir
                cand = bd / f"{args.b}_{fam}.json"
                bpath = cand if cand.exists() else bpath
            if bpath is None:
                print(f"  {fam}: no '{args.b}' arm (have {', '.join(sorted(arms))})"); continue
            base = arm(bpath)
            for tag in sorted(arms):
                if tag == args.b or (args.a and tag != args.a):
                    continue
                r = compare(arm(arms[tag]), base)
                if r is None:
                    print(f"  {fam:24s} {tag:10s} no shared instances"); continue
                pa = passes(arms[tag]); pb = passes(bpath)
                na = sum(sum(v) for v in pa.values()); da = sum(len(v) for v in pa.values())
                nb = sum(sum(v) for v in pb.values()); db = sum(len(v) for v in pb.values())
                star = "*" if r["sig"] else " "
                # A reward gain with no PASS on either side is partial credit moving, not a problem getting solved --
                # eco_size reads +0.050* at 32B with 0/96 passes in both arms. Flag it rather than let it read as one.
                note = "  [partial credit only: no pass in either arm]" if na == 0 and nb == 0 else ""
                # Reward and solve rate can point in OPPOSITE directions, and when they do the reward is the
                # misleading one: on causal_confounded at 60 instances the loop read reward +0.031 while passing
                # 14/240 against the control's 39/240 (Fisher p=0.0004) -- partial credit rose as correct answers
                # were lost. That reversal is invisible in the reward column, so it gets said out loud here.
                if da and db:
                    _dp = na / da - nb / db
                    if r["diff"] * _dp < 0 and abs(r["diff"]) > 0.005:
                        note += f"  [!! REWARD AND SOLVE RATE DISAGREE: reward {r['diff']:+.3f}, pass rate {_dp:+.3f}]"
                print(f"  {fam:24s} {tag:10s} {r['a']:.3f} vs {r['b']:.3f}  "
                      f"diff {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{star}  "
                      f"pass {na}/{da} vs {nb}/{db}  ({r['n']} {r['unit']}){note}")


if __name__ == "__main__":
    main()
