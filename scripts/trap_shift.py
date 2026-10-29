#!/usr/bin/env python3
"""Does an arm change WHICH wrong analysis the agent reaches for, even when it passes nothing?

Eight unseen families sit at exactly 0.0% pass at 8B (2026-09-20), so pass rate cannot measure transfer there.
But `attribution` records the documented trap each failed rollout fell into, and the programme's established
mechanism on causal_confounded is precisely a redistribution of traps (post_treatment_control 169->83 while
reference 14->94). This compares the trap distribution of an arm against control with a chi-square test on the
FAILED rollouts only (so a pass-rate difference cannot masquerade as a trap shift), and reports which traps
thinned and which thickened.

    venv/bin/python scripts/trap_shift.py --family astro_transit --arm general --control control [--dir general_agent_trapshift2]
    venv/bin/python scripts/trap_shift.py --self-test

Files are gated the same way ga_table.py gates them: a file with >=10% never-reached-the-server rollouts is refused.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ga_table import parse_spec, _DEAD  # noqa: E402


def load(pattern: str, model: str):
    """Pooled trap counts over every clean file matching the glob whose agent spec is on `model`."""
    counts = collections.Counter(); n = passed = 0; files = []
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        if os.path.basename(f).startswith("."):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "groups" not in d or "agent" not in d:
            continue
        m, _arm = parse_spec(d["agent"])
        if m != model:
            continue
        rolls = [r for v in d["groups"].values() for r in v]
        dead = sum(any(s in str(r.get("error_msg") or "") for s in _DEAD) for r in rolls)
        if rolls and dead / len(rolls) >= 0.10:
            print(f"  !! EXCLUDED {os.path.relpath(f, ROOT)}: {dead}/{len(rolls)} never reached the server", file=sys.stderr)
            continue
        files.append(os.path.relpath(f, ROOT))
        for r in rolls:
            n += 1
            if r.get("passed"):
                passed += 1
            else:
                counts[str(r.get("attribution"))] += 1
    return counts, n, passed, files


def chi2_pvalue(a: collections.Counter, b: collections.Counter):
    from scipy.stats import chi2_contingency
    keys = sorted(set(a) | set(b))
    table = [[a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys]]
    # drop traps absent from both (cannot happen after the union) and guard tiny tables
    if len(keys) < 2 or sum(table[0]) == 0 or sum(table[1]) == 0:
        return float("nan"), keys
    stat, p, dof, _ = chi2_contingency(table)
    return p, keys


def compare(family: str, arm: str, control: str, dir_glob: str, model: str, quiet=False):
    ca, na, pa, fa = load(f"results/pods/{dir_glob}/{arm}_{family}.json", model)
    cc, nc, pc, fc = load(f"results/pods/{dir_glob}/{control}_{family}.json", model)
    if not na or not nc:
        if not quiet:
            print(f"{family}: nothing to compare (arm n={na}, control n={nc})")
        return None
    p, keys = chi2_pvalue(ca, cc)
    if not quiet:
        print(f"\n{family}  [{model}]  arm={arm} n={na} pass={pa}   control={control} n={nc} pass={pc}")
        print(f"  trap distribution among FAILED rollouts, chi-square p = {p:.3g}")
        print(f"  {'trap':32s} {'control':>10s} {'arm':>10s}   shift")
        fa_tot = max(sum(ca.values()), 1); fc_tot = max(sum(cc.values()), 1)
        for k in sorted(keys, key=lambda k: -(cc.get(k, 0) + ca.get(k, 0))):
            sc, sa = 100 * cc.get(k, 0) / fc_tot, 100 * ca.get(k, 0) / fa_tot
            print(f"  {k:32s} {sc:9.1f}% {sa:9.1f}%   {sa - sc:+6.1f}")
        print(f"  files: {len(fc)} control, {len(fa)} arm")
    return p, ca, cc


def self_test() -> int:
    ok = True
    def chk(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    # the known shift: v12 vs control on causal_confounded (post_treatment_control thins, hard)
    r = compare("causal_confounded", "selfver12", "control", "general_agent*", "Qwen3-8B", quiet=True)
    chk("v12 vs control on causal_confounded loads", r is not None)
    if r:
        p, ca, cc = r
        fa, fc = sum(ca.values()), sum(cc.values())
        chk("known shift is significant (p < 1e-6)", p < 1e-6)
        chk("post_treatment_control share falls under v12",
            ca["post_treatment_control"] / fa < cc["post_treatment_control"] / fc)
    # THE METRIC IS PLACEBO-SENSITIVE. v20 (REPLICATE scaffold, no causal content) shifts the trap distribution
    # against control at p<0.01 even though it passes LESS than control: any prompt change, helpful or harmful,
    # moves which trap the failures land in. So an arm-vs-control shift is NOT evidence of transfer; the only
    # admissible comparison is arm vs a matched placebo (the queued trap-shift items carry a selfver20 arm for
    # exactly this). This test pins the fact so nobody reads an arm-vs-control shift as a result again.
    r2 = compare("causal_confounded", "selfver20", "control", "general_agent*", "Qwen3-8B", quiet=True)
    chk("v20 placebo vs control loads", r2 is not None)
    if r2:
        chk("placebo ITSELF shifts traps vs control (p<0.01) -> compare arms against the placebo, not control", r2[0] < 0.01)
    # dead-server files are refused: the quarantined gen_d2 files are dot-prefixed and must be skipped
    c, n, _, files = load("results/pods/general_agent_gen_d2/*_surv_tvc.json", "Qwen3-8B")
    chk("quarantined dead files are not loaded", n == 0 and not files)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--arm", default="general"); ap.add_argument("--control", default="control")
    ap.add_argument("--dir", default="general_agent*", help="results/pods directory glob")
    ap.add_argument("--model", default="Qwen3-8B"); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.family:
        ap.error("--family is required")
    compare(a.family, a.arm, a.control, a.dir, a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
