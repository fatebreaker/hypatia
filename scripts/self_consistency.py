#!/usr/bin/env python
"""Self-consistency and pass@k baselines from saved dispersion runs.

RESULTS §24 claims RL roughly doubles the held-out solve rate, and lists "no
best-of-n baseline" as a caveat. That caveat matters: drawing k samples from
the SFT policy costs no training at all, and if simple resampling captures the
same gain then the RL result is a much smaller claim than it looks.

Three numbers, and the distinction between them is the whole point:

  pass@1          mean over rollouts. What §24 reports.
  pass@k          did ANY of the k rollouts pass. An ORACLE bound -- it needs
                  the grader to pick the winner, which no deployment has. Quote
                  it as a ceiling, never as a method.
  self-cons@k     pick the answer the k rollouts most agree with, THEN grade
                  the one that was picked. Deployable, because agreement is
                  computed without knowing the truth. This is the honest
                  comparator for a pass@1 claim.

Answers are recovered from the saved transcripts with the harness's own
`extract_answer`, so what is analysed here is exactly what was graded. The run
asserts that recomputed per-rollout passes match the stored ones; if the parse
drifted, the numbers below would be quietly measuring something else.

Run:
  venv/bin/python scripts/self_consistency.py \\
      --runs dr2_fs=SFT dr2_s10=RL@10 dr2_s30=RL@30 dr2_s50=RL@50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from paths import data_root                       # noqa: E402
from rl.reward import StepReward                  # noqa: E402


def _answer(rollout: dict) -> dict | None:
    if rollout.get("answer") is not None:
        return rollout["answer"]
    msgs = rollout.get("messages") or []
    from agents import extract_answer
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        got = extract_answer(m.get("content") or "")
        if isinstance(got, dict):
            return got.get("answer", got)
    return None


def _cfg(family: str, split: str, iid: str) -> dict:
    p = data_root() / "generated" / family / split / iid / "eval_config.json"
    return json.loads(p.read_text())


def _dist(a: dict, b: dict, keys: list[str], scales: list[float]) -> float:
    """Distance in tolerance units; inf if either answer is unusable."""
    try:
        va = [float(a[k]) for k in keys]
        vb = [float(b[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return max(abs(x - y) / s for x, y, s in zip(va, vb, scales))


def _select(answers: list[dict | None], keys: list[str],
            scales: list[float]) -> dict | None:
    """The answer the group most agrees with, chosen without the truth.

    Ties on cluster size go to the tighter cluster (smallest total distance),
    which is deterministic and does not peek at the ground truth.
    """
    cand = [a for a in answers if isinstance(a, dict)]
    if not cand:
        return None
    best, best_key = None, None
    for a in cand:
        d = [_dist(a, b, keys, scales) for b in cand]
        size = sum(1 for x in d if x <= 1.0)
        tight = sum(x for x in d if np.isfinite(x))
        key = (-size, tight)
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


def analyse(name: str, label: str) -> dict:
    d = json.loads((data_root() / "rl" / f"{name}.json").read_text())
    family, split, k = d["family"], d["split"], d["k"]
    scorer = StepReward()

    per_inst = {}
    mismatches = 0
    for iid, rolls in d["groups"].items():
        cfg = _cfg(family, split, iid)
        keys = StepReward._fields(cfg)
        scales = scorer._scales(cfg, keys)

        answers = [_answer(r) for r in rolls]
        passes = []
        for r, a in zip(rolls, answers):
            got = scorer.score(cfg, {"answer": a} if a is not None else None).passed
            mismatches += int(bool(got) != bool(r["passed"]))
            passes.append(bool(r["passed"]))       # trust the run's own grade

        sel = _select(answers, keys, scales)
        sel_pass = scorer.score(cfg, {"answer": sel} if sel else None).passed
        per_inst[iid] = {"p1": float(np.mean(passes)),
                         "pk": float(any(passes)),
                         "sc": float(bool(sel_pass))}

    return {"label": label, "name": name, "family": family, "split": split,
            "k": k, "per_inst": per_inst, "parse_mismatches": mismatches,
            "n": len(per_inst)}


def _boot(vals: dict[str, dict], field: str, ref: dict[str, dict] | None,
          rng: np.random.Generator, b: int = 10_000) -> tuple:
    """Cluster bootstrap over INSTANCES, paired against `ref` when given."""
    ids = sorted(vals)
    x = np.array([vals[i][field] for i in ids])
    r = np.array([ref[i][field] for i in ids]) if ref else None
    idx = rng.integers(0, len(ids), size=(b, len(ids)))
    draws = x[idx].mean(1) - (r[idx].mean(1) if r is not None else 0.0)
    point = x.mean() - (r.mean() if r is not None else 0.0)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=label, first is the reference arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    arms = []
    for spec in args.runs:
        name, _, label = spec.partition("=")
        arms.append(analyse(name, label or name))

    base = arms[0]
    bad = sum(a["parse_mismatches"] for a in arms)
    print(f"family={base['family']}  split={base['split']}  k={base['k']}  "
          f"n={base['n']} instances")
    if bad:
        print(f"  WARNING: {bad} rollouts re-graded differently from the saved "
              f"run -- answer extraction has drifted, treat with suspicion")
    else:
        print("  answer extraction reproduces every saved per-rollout grade")

    print(f"\n{'arm':10s} {'pass@1':>18s} {'self-cons@k':>18s} {'pass@k (oracle)':>18s}")
    for a in arms:
        cells = []
        for f in ("p1", "sc", "pk"):
            p, lo, hi = _boot(a["per_inst"], f, None, rng)
            cells.append(f"{p:.3f} [{p+lo-p:.3f},{hi:.3f}]" if False
                         else f"{p:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{a['label']:10s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")

    print(f"\npaired vs {base['label']} (bootstrap over instances, 95 % CI)")
    print(f"{'arm':10s} {'d pass@1':>22s} {'d self-cons@k':>22s}")
    for a in arms[1:]:
        row = []
        for f in ("p1", "sc"):
            p, lo, hi = _boot(a["per_inst"], f, base["per_inst"], rng)
            star = "*" if lo > 0 or hi < 0 else " "
            row.append(f"{p:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
        print(f"{a['label']:10s} {row[0]:>22s} {row[1]:>22s}")

    print(f"\nTHE comparison: does resampling the {base['label']} policy k times "
          f"reach the RL arm's pass@1?")
    p_sc, lo, hi = _boot(base["per_inst"], "sc", None, rng)
    for a in arms[1:]:
        p1, _, _ = _boot(a["per_inst"], "p1", None, rng)
        verdict = "NO -- RL is ahead" if p1 > hi else (
            "YES -- baseline covers it" if p1 >= lo else "baseline exceeds RL")
        print(f"  {base['label']} self-cons@{base['k']} {p_sc:.3f} "
              f"[{lo:.3f},{hi:.3f}]  vs  {a['label']} pass@1 {p1:.3f}   -> {verdict}")
    print("\n(pass@k is an oracle ceiling: it needs the grader to pick the "
          "winner. Never quote it as a deployable method.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
