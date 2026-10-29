#!/usr/bin/env python
"""Leave-one-family-out corpora: does the agent learn PRIMITIVES or PROBLEMS?

Every capability result so far is per-problem. cis-MVMR and scRNA are solved
because a faithful generator was hand-built for each; the other eight public
problems sit at exactly 0.000. If that is all there is, capability scales
linearly with engineering time -- about a day per problem -- and "dry lab in a
domain" is out of reach.

But the ten case studies reuse a small set of analysis primitives: a QC gate
before estimation, calibration/deconvolution, reweighting for selection, latent
state reconstruction, confounder-vs-mediator discipline, and orientation/label
conventions. If training on several families teaches those primitives rather
than those problems, a held-out family should be partly solvable WITHOUT ever
having generated data for it.

That is the experiment this builds. For held-out family F:

    train on every OTHER family's demonstrations
    evaluate on F's REAL benchmark problem, and on F's own generated test set

Three outcomes, all informative:

  transfers          the agent learned primitives. Coverage stops being the
                     binding constraint and a new problem no longer needs a
                     bespoke generator.
  transfers partly   primitives help but are not sufficient; the honest scaling
                     story is bespoke generators with a warm start.
  does not transfer  capability is per-problem. The deliverable is the PIPELINE
                     -- published method in, specialist model out, about a day
                     -- not a general agent.

The bespoke single-family SFT for the same held-out family is the reference
point: LOFO is only interesting relative to what a purpose-built corpus buys.

  venv/bin/python rl/build_lofo.py --holdout wf --out data/sft/lofo_wf
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

SFT = data_root() / "sft"

# Benchmark families with BOTH a generated corpus and a real public problem.
# `burden` and `gwas` are ours rather than the benchmark's, but they exercise
# the same primitives and are included as training signal -- never as holdouts,
# since there is no real problem to score them against.
BENCH = {
    "cis":    ["traces_cismvmr_fixed_train.jsonl"],
    "scrna":  ["scrna_newprompt.jsonl"],
    "popgen": ["popgen_train.jsonl"],
    "wf":     ["wf_train.jsonl"],
}
EXTRA = {
    "burden": ["traces_burden_train.jsonl"],
    "gwas":   ["fewshot_gwas_train.jsonl"],
}


def load(names: list[str]) -> list[dict]:
    out = []
    for n in names:
        p = SFT / n
        if not p.exists():
            raise SystemExit(f"missing trace file {p}")
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", required=True, choices=sorted(BENCH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-extra", action="store_true",
                    help="add burden/gwas traces as extra primitive coverage")
    ap.add_argument("--cap", type=int, default=260,
                    help="per-family cap, so one large corpus cannot dominate")
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, comp = [], {}
    src = dict(BENCH)
    if args.include_extra:
        src.update(EXTRA)
    for fam, files in src.items():
        if fam == args.holdout:
            continue
        t = load(files)
        # Cap per family: cis-MVMR alone has 800 traces against popgen's 220,
        # and an unbalanced mix would teach one problem rather than the shared
        # primitives this experiment is about.
        rng.shuffle(t)
        t = t[: args.cap]
        for r in t:
            r["_family"] = fam
        rows += t
        comp[fam] = len(t)

    rng.shuffle(rows)
    val = rows[: args.val]
    train = rows[args.val:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    Path(f"{out}_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    print(f"holdout={args.holdout}  train={len(train)}  val={len(val)}")
    print(f"  composition: {comp}")
    print(f"  wrote {out}_train.jsonl / {out}_val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
