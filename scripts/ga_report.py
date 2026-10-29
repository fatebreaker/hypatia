#!/usr/bin/env python3
"""Every general-agent number, derived from disk in one place.

    venv/bin/python scripts/ga_report.py

The programme's results are spread over ~20 result directories written on different nights at different powers, and
the section that will quote them must not be written from numbers retyped out of a transcript. This prints the whole
evidence base in one pass, in the form the text will use, so each claim can be traced to files.

Three rules are enforced here because each cost something to learn:
  1. `rl/dispersion.py` records carry `reward`/`passed` -- never `score`, which is `harness/runner.py`'s key. Reading
     the wrong one returns silent zeros.
  2. A family is only informative if the untrained control sometimes succeeds; 21 of 27 sit at zero and cannot
     distinguish a working method from a broken one.
  3. PASS COUNTS, not reward. Reward said +0.031 (a null) on causal_confounded while the solve rate fell by two
     thirds; the two are near-uncorrelated where the trap is WHICH analysis was run rather than how much of it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "pods"
BOOT, SEED = 20000, 20260918


def rollouts(path: Path):
    """instance -> rollouts, from a dispersion.py result file."""
    return json.loads(path.read_text())["groups"]


def arm(path: Path):
    g = rollouts(path)
    rew = {i: [r["reward"] for r in v] for i, v in g.items()}
    ps = [bool(r["passed"]) for v in g.values() for r in v]
    return rew, ps


def paired(a: dict, b: dict):
    sh = sorted(set(a) & set(b))
    if not sh:
        return None
    if len(sh) == 1:
        # A real-data family has ONE instance, and resampling one instance returns it every time -- the interval
        # collapses to zero width and everything reads significant (real SPECT came back +0.173 [+0.173,+0.173]*).
        # There the ROLLOUTS are the replication, so resample those. Same fix as scripts/gen_compare.py; it was
        # reproduced here by copying the instance-level bootstrap without its guard.
        av = np.array(a[sh[0]], dtype=float)
        bv = np.array(b[sh[0]], dtype=float)
        rng = np.random.default_rng(SEED)
        ai = rng.integers(0, len(av), size=(BOOT, len(av)))
        bi = rng.integers(0, len(bv), size=(BOOT, len(bv)))
        boots = av[ai].mean(1) - bv[bi].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(av.mean() - bv.mean()), float(lo), float(hi), 1
    am = np.array([np.mean(a[i]) for i in sh])
    bm = np.array([np.mean(b[i]) for i in sh])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sh), size=(BOOT, len(sh)))
    boots = am[idx].mean(1) - bm[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(am.mean() - bm.mean()), float(lo), float(hi), len(sh)


def compare(dtreat: Path, ttag: str, dctl: Path, ctag: str, fam: str):
    ft, fc = dtreat / f"{ttag}_{fam}.json", dctl / f"{ctag}_{fam}.json"
    if not (ft.exists() and fc.exists()):
        return None
    ra, pa = arm(ft)
    rb, pb = arm(fc)
    st = paired(ra, rb)
    if st is None:
        return None
    d, lo, hi, n = st
    _, pv = fisher_exact([[sum(pa), len(pa) - sum(pa)], [sum(pb), len(pb) - sum(pb)]])
    return dict(diff=d, lo=lo, hi=hi, n=n, pa=sum(pa), na=len(pa), pb=sum(pb), nb=len(pb), p=pv)


def line(label: str, r: dict | None) -> str:
    if r is None:
        return f"  {label:34s} (not run)"
    star = "*" if r["lo"] > 0 or r["hi"] < 0 else " "
    flag = ""
    if r["na"] and r["nb"]:
        dp = r["pa"] / r["na"] - r["pb"] / r["nb"]
        if r["diff"] * dp < 0 and abs(r["diff"]) > 0.005:
            flag = "  [!! reward and solve rate disagree]"
    if r["pa"] == 0 and r["pb"] == 0:
        flag += "  [partial credit only]"
    return (f"  {label:34s} {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{star}  "
            f"{r['pa']:3d}/{r['na']:<4d} vs {r['pb']:3d}/{r['nb']:<4d}  Fisher p={r['p']:.3g}{flag}")


def main() -> None:
    print("=" * 100)
    print("FOOTHOLD MAP -- a family can only test the method if the untrained model sometimes succeeds")
    print("=" * 100)
    best = defaultdict(dict)
    for d in sorted(RES.glob("general_agent*")):
        if not d.is_dir() or any(k in d.name for k in ("qwen14", "qwen32", "mistral", "olmo")):
            continue
        for f in d.glob("*.json"):
            tag, _, fam = f.stem.partition("_")
            if tag not in ("control", "selfver"):
                continue
            try:
                _, ps = arm(f)
            except Exception:
                continue
            cur = best[fam].get(tag)
            if cur is None or len(ps) > cur[1]:
                best[fam][tag] = (sum(ps), len(ps), f)
    foot = []
    for fam in sorted(best):
        c = best[fam].get("control")
        if not c:
            continue
        mark = "FOOTHOLD" if c[0] >= 5 else ("thin" if c[0] else "zero -- untestable")
        if c[0] >= 5:
            foot.append(fam)
        print(f"  {fam:24s} control {c[0]:3d}/{c[1]:<4d}  {mark}")
    print(f"\n  testable families: {', '.join(foot)}")

    print("\n" + "=" * 100)
    print("THE LADDER at 60 instances (240 rollouts per arm), against each family's own control")
    print("=" * 100)
    for fam, d in (("surv_cox", RES / "general_agent_ladder60"),
                   ("causal_did", RES / "general_agent_ladder60_causal_did"),
                   ("hc_dx", RES / "general_agent_ladder60_hc_dx"),
                   ("causal_confounded", RES / "general_agent_cc_pow"),
                   ("causal_iv", RES / "general_agent_iv60")):
        if not d.is_dir():
            continue
        print(f"\n{fam}  ({d.name})")
        for tag, nm in (("selfver0", "plan-only"), ("selfver8", "v8 state it"),
                        ("selfver9", "v9 state + falsify"), ("selfver", "loop build it"),
                        ("selfver7", "v7 loop + caution"), ("selfver10", "v10 falsify + caution")):
            print(line(nm, compare(d, tag, d, "control", fam)))

    print("\n" + "=" * 100)
    print("SCALE GRID -- the loop against its own control, per model size")
    print("=" * 100)
    for fam in ("surv_cox", "causal_did", "hc_dx", "causal_confounded"):
        print(f"\n{fam}")
        for nm, dirs in (("8B ", ["general_agent", "general_agent_ladder60", "general_agent_cc_pow"]),
                         ("14B", ["general_agent_qwen14", "general_agent_qwen14_did", "general_agent_qwen14_cc"]),
                         ("32B", ["general_agent_qwen32", "general_agent_qwen32_did", "general_agent_qwen32_cc"])):
            r = None
            for dn in dirs:
                d = RES / dn
                if d.is_dir():
                    r = compare(d, "selfver", d, "control", fam) or r
            print(line(nm, r))

    print("\n" + "=" * 100)
    print("CROSS-MODEL and REAL DATA")
    print("=" * 100)
    for nm, d, fam in (("Mistral-7B surv_cox", "general_agent_mistral", "surv_cox"),
                       ("Olmo-3-7B surv_cox", "general_agent_olmo", "surv_cox"),
                       ("real SPECT", "general_agent_k64", "hc_dx_real_spect"),
                       ("real theophylline", "general_agent_k64", "pk_real_theoph")):
        p = RES / d
        print(line(nm, compare(p, "selfver", p, "control", fam) if p.is_dir() else None))


if __name__ == "__main__":
    main()
