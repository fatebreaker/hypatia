#!/usr/bin/env python
"""Compare two arms, and REFUSE by default if they come from different draws.

WHY THIS EXISTS. Cross-draw comparison is the single most recurrent error in this programme:

  * the falsify result (withdrawn 2026-09-21): 11.9% against a POOLED surv_tvc control of 6.5%, where that draw's
    own control was 9.2%. The whole effect was the control mismatch.
  * "the spec harms causal_iv at 14B, p=0.001": :general pooled across draws against a pooled control.
  * "v26 partially repairs the harm, 32% of the trap gap" (mine, 2026-09-21 11:00, withdrawn at 12:05): v26 from
    one draw against :general from two others. With the same draw's own :general arm, p = 0.41.

The last of those was written FOUR HOURS after I recorded the rule against it. A rule in a document does not
prevent the error; a function that refuses to compute it does. This is the same move that fixed dead-server
contamination -- put the refusal in the scorer (`ga_table.score`), not in the analyst's memory.

  from draw_compare import arm, compare
  a = arm("general_agent_gate14", "selfver26", "causal_iv")
  b = arm("general_agent_gate14", "general",   "causal_iv")
  compare(a, b)                       # fine -- same draw
  c = arm("general_agent_iv14_attr", "general", "causal_iv")
  compare(a, c)                       # raises
  compare(a, c, across_draws="control stability measured at p=0.94; context only, no p-value cited")
"""
from __future__ import annotations
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score, resolved_spec
from scipy.stats import fisher_exact

ROOT = "results/pods"


class Arm:
    def __init__(self, draw: str, name: str, family: str):
        self.draw, self.name, self.family = draw, name, family
        path = os.path.join(ROOT, draw, f"{name}_{family}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = json.load(open(path))                        # parsed ONCE; score() takes the dict (see ga_table)
        self.k, self.n = score(path, d)                  # raises on >=10% dead rollouts
        self.attrib = collections.Counter()
        for rolls in d["groups"].values():
            for r in rolls:
                self.attrib[r.get("attribution") or "none"] += 1
        self.terminated = self.n - self.attrib["none"]
        # the RESOLVED instruction, not the alias -- see ga_table.resolved_spec for why this is printed
        self.model, self.spec = resolved_spec(d.get("agent", ""))

    @property
    def rate(self):
        return 100.0 * self.k / self.n if self.n else float("nan")

    def __repr__(self):
        return (f"{self.name}@{self.draw}[{self.family}] {self.k}/{self.n} = {self.rate:.1f}% "
                f"({self.model} {self.spec})")


def arm(draw, name, family):
    return Arm(draw, name, family)


def compare(a: Arm, b: Arm, trap: str | None = None, across_draws: str | None = None):
    """Fisher exact on pass rates, or on `trap` among TERMINATED rollouts.

    Refuses if the two arms are from different draws unless `across_draws` gives a written reason, which is
    echoed into the result so it travels with the number.
    """
    if a.family != b.family and not across_draws:
        raise ValueError(f"different families ({a.family} vs {b.family}) -- pass across_draws=... if deliberate")
    if a.draw != b.draw and not across_draws:
        raise ValueError(
            f"CROSS-DRAW COMPARISON REFUSED: {a.name} is from {a.draw}, {b.name} is from {b.draw}.\n"
            f"  A draw's own control is the only one its arms have earned a p-value against. If you truly mean to\n"
            f"  compare across draws, pass across_draws='<why this is sound>' and the reason will be printed with\n"
            f"  the number. See the module docstring for three claims this rule has already withdrawn."
        )
    if trap:
        ka, na = a.attrib[trap], a.terminated
        kb, nb = b.attrib[trap], b.terminated
        gap = abs(100*a.attrib["none"]/a.n - 100*b.attrib["none"]/b.n)
        quality = "clean" if gap < 5 else ("suspect" if gap < 10 else "CONFOUNDED -- do not read")
    else:
        ka, na, kb, nb = a.k, a.n, b.k, b.n
        gap, quality = 0.0, "n/a"
    p = fisher_exact([[ka, na-ka], [kb, nb-kb]])[1]
    out = {"a": repr(a), "b": repr(b), "metric": trap or "passed",
           "a_pct": 100*ka/na, "b_pct": 100*kb/nb, "p": p,
           "termination_gap_pt": gap, "quality": quality}
    if across_draws:
        out["ACROSS_DRAWS"] = across_draws
    return out


if __name__ == "__main__":
    a = arm("general_agent_gate14", "selfver26", "causal_iv")
    b = arm("general_agent_gate14", "general", "causal_iv")
    print("same draw:", compare(a, b, trap="ols_ignores_instrument"))
    try:
        compare(a, arm("general_agent_iv14_attr", "general", "causal_iv"), trap="ols_ignores_instrument")
    except ValueError as e:
        print("\nrefused as intended:\n", str(e).split("\n")[0])
