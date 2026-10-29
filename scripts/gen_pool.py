#!/usr/bin/env python3
"""Pool repeated draws of the same arm into one estimate.

    venv/bin/python scripts/gen_pool.py FAM DIR [DIR ...] [--a selfver] [--b control]

Repeated draws of a family re-run the SAME instances (dispersion.py takes `dirs[:n]` from a fixed sorted list), so a
second draw does not add instances -- it adds independent rollouts on the instances already there. Pooling therefore
keeps the INSTANCE as the resampling unit and merges each instance's rollouts across draws. Averaging the per-draw
differences instead would throw away the fact that later draws are as informative as the first.

Why it matters: the first draw is systematically the one that reads highest, because it is the one that prompted the
replication. causal_did went +0.232 -> +0.162 -> +0.108 across three draws and pools to +0.167 [+0.098,+0.235];
surv_cox went +0.537 -> +0.396 and pools to +0.466 [+0.396,+0.539]. Quote the pool, never the draw that started it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _arm(d: Path, fam: str, tag: str):
    p = d / f"{tag}_{fam}.json"
    if not p.exists():
        return None
    return {i: [r["reward"] for r in v] for i, v in json.loads(p.read_text())["groups"].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fam")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--a", default="selfver")
    ap.add_argument("--b", default="control")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    A: dict[str, list[float]] = {}
    B: dict[str, list[float]] = {}
    used = []
    for d in args.dirs:
        p = Path(d) if Path(d).is_absolute() else ROOT / d
        a, b = _arm(p, args.fam, args.a), _arm(p, args.fam, args.b)
        if a is None or b is None:
            print(f"  skip {d}: missing {args.a} or {args.b} for {args.fam}")
            continue
        used.append(d)
        for i, v in a.items():
            A.setdefault(i, []).extend(v)
        for i, v in b.items():
            B.setdefault(i, []).extend(v)
    shared = sorted(set(A) & set(B))
    if not shared:
        print("nothing to pool")
        return 2
    am = np.array([np.mean(A[i]) for i in shared])
    bm = np.array([np.mean(B[i]) for i in shared])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(shared), size=(args.boot, len(shared)))
    boots = am[idx].mean(1) - bm[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    diff = float(am.mean() - bm.mean())
    star = "*" if lo > 0 or hi < 0 else ""
    print(f"{args.fam}: {args.a} {am.mean():.3f} vs {args.b} {bm.mean():.3f}  "
          f"diff {diff:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    print(f"  pooled over {len(used)} draw(s): {len(shared)} instances, "
          f"{sum(len(A[i]) for i in shared)} rollouts per arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
