#!/usr/bin/env python3
"""Score v21's ROUTING separately from its outcome.

v21 (the router) prints one line -- `ROUTE order because <reason>` or `ROUTE plain because <reason>` -- before
building its simulator, and the whole point of printing it is that routing accuracy and pass rate can then be
measured independently: a router that routes correctly but still fails tells you the branch is at fault; one that
fails because it routes everything one way tells you the reading step is at fault. Collapsing the two into a pass
rate would repeat the mistake the self-check analysis made (a score with no attribution).

The expected route per family is the PREDICTION registered in queue items 247/248 before any v21 result existed:
`order` where the task asks for an adjusted effect from one cross-section (causal_confounded, causal_att), `plain`
everywhere else. If those expectations are ever revised after seeing results, say so where they are used.

  scripts/route_score.py results/pods/general_agent_v21/selfver21_<family>.json [...]
  scripts/route_score.py --self-test
"""
import json
import re
import sys

# tolerant of "ROUTE order", "ROUTE: plain", "**ROUTE** order", lowercase; first match in the transcript wins,
# because the instruction says to print the line before the code and later restatements may quote both words.
ROUTE_RE = re.compile(r"ROUTE\b[:*\s]*\**\s*(order|plain)\b", re.IGNORECASE)

EXPECTED = {"causal_confounded": "order", "causal_att": "order"}   # everything else: plain


def route_of(rollout) -> str:
    for m in rollout.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        hit = ROUTE_RE.search(str(m.get("content", "")))
        if hit:
            return hit.group(1).lower()
    return "NONE"


def score_file(path: str) -> None:
    d = json.load(open(path))
    fam = d.get("family", "?")
    want = EXPECTED.get(fam, "plain")
    n = {"order": [0, 0], "plain": [0, 0], "NONE": [0, 0]}   # route -> [rollouts, passes]
    for rollouts in (d.get("groups") or {}).values():
        for r in rollouts:
            k = route_of(r)
            n[k][0] += 1
            n[k][1] += bool(r.get("passed"))
    tot = sum(v[0] for v in n.values())
    acc = n[want][0] / tot if tot else 0.0
    print(f"\n{fam}  ({path})")
    print(f"  expected route: {want}   routing accuracy: {acc:.1%}")
    for k in ("order", "plain", "NONE"):
        r, p = n[k]
        if r:
            print(f"    ROUTE {k:<6} {r:>4} rollouts   pass {p}/{r} = {p / r:.1%}")


def self_test() -> int:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    mk = lambda txt: {"messages": [{"role": "assistant", "content": txt}]}          # noqa: E731
    chk("plain form", route_of(mk("ROUTE order because the task asks...")) == "order")
    chk("colon+bold", route_of(mk("**ROUTE**: plain because censoring")) == "plain")
    chk("lowercase", route_of(mk("route Plain because rates")) == "plain")
    chk("first match wins", route_of(mk("ROUTE order because X... later: ROUTE plain")) == "order")
    chk("absent -> NONE", route_of(mk("I will write a simulator with BLOCK 1")) == "NONE")
    chk("no messages -> NONE", route_of({}) == "NONE")
    # a REAL pre-v21 file must yield 100% NONE -- if this fails, the regex is matching something it should not
    d = json.load(open("results/pods/general_agent_v20/selfver20_causal_confounded.json"))
    routes = {route_of(r) for rollouts in d["groups"].values() for r in rollouts}
    chk("v20 file (no router) is all NONE", routes == {"NONE"})
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    for p in sys.argv[1:]:
        score_file(p)
