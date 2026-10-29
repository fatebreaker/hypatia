#!/usr/bin/env python
"""Items 366/367/369: does causal_confounded's content effect need the supplied method menu?

The registered contrast is GAP vs GAP, never a blind arm against a plain control:

    gap_plain = :general - selfver20   on causal_confounded        (menu present)
    gap_blind = :general - selfver20   on causal_confounded_blind  (menu removed)

Same 60 instances, byte-identical CSVs, identical reference answers; the menu is the only difference. The two gaps
are compared with a 2x2 test of independence on the four arm counts (Breslow-Day style via the interaction odds
ratio), because the question is whether the EFFECT differs, not whether either arm does.

Every count comes from ga_table.score(), which refuses a file with >=10% rollouts that never reached the server.
Run with no arguments once items 366 and 367 have landed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ga_table import score
from scipy.stats import fisher_exact, chi2_contingency

PAIRS = [
    ("8B",  "general_agent_blindmenu8_plain", "causal_confounded",
            "general_agent_blindmenu8",       "causal_confounded_blind"),
    ("14B", None,                              None,
            "general_agent_blindmenu14",      "causal_confounded_blind"),
]
ROOT = "results/pods"


def arm(d, a, fam):
    p = os.path.join(ROOT, d, f"{a}_{fam}.json")
    if not os.path.isfile(p):
        return None
    try:
        return score(p)
    except ValueError as e:
        print(f"  !! EXCLUDED {a}_{fam}: {e}")
        return None


def draw(d, fam):
    if d is None:
        return None
    out = {a: arm(d, a, fam) for a in ("general", "selfver20", "control")}
    return None if any(v is None for v in out.values()) else out


def pct(kn):
    k, n = kn
    return 100.0 * k / n if n else float("nan")


for label, pd_, pf, bd, bf in PAIRS:
    print(f"\n=== {label} ===")
    plain, blind = draw(pd_, pf), draw(bd, bf)
    for name, dd in (("menu present", plain), ("menu removed", blind)):
        if dd is None:
            print(f"  {name}: not complete yet")
            continue
        print(f"  {name}:  " + "  ".join(f"{a} {pct(dd[a]):.1f}%" for a in ("general", "selfver20", "control")))
        g, s = dd["general"], dd["selfver20"]
        print(f"     general vs placebo p = {fisher_exact([[g[0], g[1]-g[0]], [s[0], s[1]-s[0]]])[1]:.3g}"
              f"   gap = {pct(g) - pct(s):+.1f} points")
    if plain and blind:
        gp, sp = plain["general"], plain["selfver20"]
        gb, sb = blind["general"], blind["selfver20"]
        tab = [[gp[0], gp[1]-gp[0]], [sp[0], sp[1]-sp[0]], [gb[0], gb[1]-gb[0]], [sb[0], sb[1]-sb[0]]]
        p_int = chi2_contingency([[gp[0], gp[1]-gp[0], gb[0], gb[1]-gb[0]],
                                  [sp[0], sp[1]-sp[0], sb[0], sb[1]-sb[0]]])[1]
        print(f"  GAP CHANGE: {pct(gp)-pct(sp):+.1f} (menu) -> {pct(gb)-pct(sb):+.1f} (no menu);"
              f" interaction p = {p_int:.3g}")
        print("  READ: gap survives -> the causal-order content acts on the DATA, claim stands as written.")
        print("        gap collapses toward the placebo -> the effect is menu-steering and the manuscript's"
              " causal_confounded claim must be re-scoped.")
