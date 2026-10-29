#!/usr/bin/env python
"""Collapse a real-data corpus to one instance per underlying record set.

Simulated families draw a fresh instance from every seed, so N seeds give N
distinct problems. A real-data family cannot: `gwas_loci` draws a real trait
from a finite pool, with replacement, so a 900-seed mint contains the same
trait many times over. Those duplicates share an answer, and a train/test split
that puts two of them on opposite sides is leakage of exactly the kind that
produced the retracted results in RESULTS.md 18/19/23.

Keeps the instance with the most live traps per trait, ties broken by id so the
choice is deterministic.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--key", default="trait")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--splits", default="train=40,rl=72,val=40,test=80",
                    help="name=count,... ; leftovers go to the last split")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    best: dict[str, tuple] = {}
    tol = None
    for f in sorted(src.glob("*/eval_config.json")):
        c = json.loads(f.read_text())
        p = c["_provenance"]
        k = p[args.key]
        live = sum(1 for n, v in p["ablations"].items()
                   if n != "reference" and v["deviation"] > 1.0)
        cand = (live, c["id"], f.parent)
        if k not in best or (live, ) > (best[k][0], ):
            best[k] = cand

    prefix = args.prefix or (src.name.replace("_raw", ""))
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Shuffle before splitting so splits are not ordered by trait name -- an
    # alphabetical split would put every "protein levels" trait in one bucket.
    import random
    keys = sorted(best)
    random.Random(args.seed).shuffle(keys)

    spec = []
    for part in args.splits.split(","):
        name, _, cnt = part.partition("=")
        spec.append((name.strip(), int(cnt)))
    bounds, at = [], 0
    for j, (name, cnt) in enumerate(spec):
        end = len(keys) if j == len(spec) - 1 else min(at + cnt, len(keys))
        bounds.append((name, keys[at:end]))
        at = end

    live_hist: Counter = Counter()
    split_sizes = {}
    seen_keys: dict[str, str] = {}
    for name, ks in bounds:
        split_sizes[name] = len(ks)
        for i, k in enumerate(ks):
            live, _id, d = best[k]
            # One instance per trait means splits cannot share a record set --
            # assert it rather than trust it, because a leak here is invisible
            # and would inflate every held-out number this corpus produces.
            assert k not in seen_keys, f"{k!r} in both {seen_keys[k]} and {name}"
            seen_keys[k] = name
            pid = f"{prefix}_{name}_{i:05d}"
            shutil.copytree(d, dst / name / pid)
            c = json.loads((dst / name / pid / "eval_config.json").read_text())
            c["id"] = pid
            c["_provenance"]["n_live_traps"] = live
            c["_provenance"]["split"] = name
            (dst / name / pid / "eval_config.json").write_text(json.dumps(c, indent=2))
            live_hist[live] += 1

    (dst / "_manifest.json").write_text(json.dumps(
        {"family": prefix, "n": len(best), "deduped_from": len(
            list(src.glob("*/eval_config.json"))),
         "unique_key": args.key, "seed": args.seed, "splits": split_sizes,
         "live_trap_histogram": {str(k): v for k, v in sorted(live_hist.items())}},
        indent=2))
    print("  splits:", split_sizes)
    print(f"{len(list(src.glob('*/eval_config.json')))} instances -> "
          f"{len(best)} unique by {args.key!r}")
    print("  live traps per instance:",
          {k: live_hist[k] for k in sorted(live_hist, reverse=True)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
