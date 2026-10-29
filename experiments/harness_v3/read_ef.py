#!/usr/bin/env python
"""EF reading -- THE EXECUTION-FORCING HARNESS, NO TRAINING (docs/RL_PLAN.md 2026-09-28 14:40 UTC, registered before any
build or run; the analysis behind it: results/pods/mechanism/MECHANISM.md). The untrained Qwen3-8B scored in the v3
evaluation harness with ONE change -- an answer is accepted only if its numbers were printed by the agent's own executed
code (experiments/harness_v3/exec_forcing.py via screen_v3ef.sh -> <root>/V3EF) -- paired by instance with the untrained
8B in the plain v3 harness (<root>/V3) and with PR s1-s3 at step 100 (<root>/V3__<PR run>-s{1,2,3}_s100). Same TEST draw
(17 families, 12x2), HO6, DESIGN causal_rdd.

    venv/bin/python experiments/harness_v3/read_ef.py [--root results/pods/harness_v3] [--base V3] [--ef V3EF]
                                                      [--perms 20000] [--seed 0] [--out DIR]

A NEW FILE: read_v3.py / analyze.py are imported for their helpers (cell_pass, paired = the sign-flip permutation, HO6,
contrast) and never edited. Family pass = mean over instances of the per-instance pass fraction (analyze.cell_pass);
p = one-sided sign-flip over instances (analyze.paired), in the direction of the rule.

REGISTERED READINGS (each evaluated on its own; INCOMPLETE until the families it needs are on disk in both cells):
  EXEC-EXPLAINS-SURV   EF surv_cox >= 40 (base 20.8). Not firing = the execution account of surv_cox is wrong.
  RESIDUAL-CONFOUNDED  EF causal_confounded <= 27 (PR s1/s2 66.7 / 70.8): the >= 20-point residual to PR s1/s2 is what
                       RL adds beyond execution.
  NO-SPILL             the mean change EF - base over the 14 TEST families other than causal_confounded, surv_cox and
                       surv_tvc lies within +-3 points. "Mean" = the unweighted mean of the 14 per-family changes (as
                       MECHANISM.md s.5 averages them); the twin-weighted mean (analyze.py's TEST weights) is printed
                       beside it.
BESIDE (never deciding): the share of PR's mean gain that EF recovers without training, on HO6 and on TEST-all
(twin-weighted, as read_v3_q8.py's TEST-all):  share = d(EF vs base) / mean_s d(PR_s vs base), all paired by instance;
if EF recovers most of it, the transferable part of C1 is a harness property (execution discipline) that needs no RL.
Also paired EF vs each PR seed, the per-family table, and the EF mechanics (refusals per rollout, how answers were
accepted, no-answer / forced / cells rates vs the base) from rollout["v2"]["ef"].
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_v3 as R  # noqa: E402  (puts ROOT, harness/ and experiments/harness_v2 on sys.path)

A = R.A
HO6 = R.HO6
DESIGN_FAM = "causal_rdd"
PR_RUN = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s{seed}"
PR_SEEDS = (1, 2, 3)
PR_STEP = 100
SURV, CONF, TVC = "surv_cox", "causal_confounded", "surv_tvc"
SURV_BAR, CONF_BAR, SPILL_BAR = 40.0, 27.0, 3.0
COMPETENT = (CONF, SURV, TVC)
FORCE_PREFIX = "STOP. You have"
FEEDBACK_HEAD = "Your answer is not backed by printed output"


def fmt(x, nd=1, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def uniform_sets(sets: dict) -> dict:
    """The same families with every TEST family its own group, so analyze.paired weights them equally."""
    u = copy.deepcopy(sets)
    for f in u["test"]:
        u["test"][f] = dict(u["test"][f], group=f)
    return u


def fam(res, sets, f, s="test"):
    return None if res is None else A.cell_pass(res, sets, s, [f])[0]


def complete(res, sets, members, s="test", n=None, k=None) -> list:
    """Families of `members` missing or short (fewer than the draw's n instances x k rollouts) in `res`."""
    if res is None:
        return list(members)
    d = json.loads(A.FAMJ.read_text())["draws"][s]
    n, k = n or int(d["n"]), k or int(d["k"])
    bad = []
    for f in members:
        inst = res["families"].get((s, f))
        if not inst or len(inst) < n or any(len(v) < k for v in inst.values()):
            bad.append(f)
    return bad


def pair(a, b, sets, perms, rng, members, s="test"):
    """Paired Delta (points) of a vs b over `members`: {'d', 'p_gt', 'p_lt', 'n_fams'} or None."""
    if a is None or b is None:
        return None
    x = A.paired(a, b, sets, s, members, perms, rng)
    if x is None:
        return None
    return {"d": x[0], "p_gt": x[2], "p_lt": x[3], "n_fams": x[6]}


def mechanics(res) -> dict:
    """EF bookkeeping and behaviour over the TEST rollouts of one configuration."""
    rs = [r for (s, f), rows in res["rollouts"].items() if s == "test" for r in rows]
    n = len(rs)
    if not n:
        return {"n": 0}
    efl = [((r.get("v2") or {}).get("ef")) for r in rs]
    has = [e for e in efl if isinstance(e, dict)]
    acc: dict = {}
    for e in has:
        acc[str(e.get("accepted"))] = acc.get(str(e.get("accepted")), 0) + 1

    def forced(r):
        return any(m.get("role") == "user" and str(m.get("content") or "").startswith(FORCE_PREFIX)
                   for m in (r.get("messages") or []))

    def fb_turns(r):
        return sum(1 for m in (r.get("messages") or []) if m.get("role") == "user"
                   and str(m.get("content") or "").startswith(FEEDBACK_HEAD))
    rej = [int(e.get("rejected") or 0) for e in has]
    return {"n": n, "with_ef_log": len(has),
            "rejected_mean": float(np.mean(rej)) if rej else None,
            "rollouts_with_rejection": sum(1 for x in rej if x > 0),
            "rejected_total": int(sum(rej)),
            "feedback_turns_total": int(sum(fb_turns(r) for r in rs)),
            "accepted": acc, "na": int(sum(int(e.get("na") or 0) for e in has)),
            "gate_errors": int(sum(int(e.get("gate_errors") or 0) for e in has)),
            "no_answer": sum(1 for r in rs if not r.get("answered")),
            "errored": sum(1 for r in rs if str(r.get("attribution") or "").startswith("error:")),
            "forced": sum(1 for r in rs if forced(r)),
            "cells_mean": float(np.mean([r.get("n_cells") or 0 for r in rs])),
            "zero_cell_answers": sum(1 for r in rs if r.get("answered") and not (r.get("n_cells") or 0)),
            "passed": sum(1 for r in rs if r.get("passed"))}


def ef_provenance(root: Path, name: str, sets) -> list:
    """Problems with the EF directory's provenance: config.json and every family JSON must record exec_forcing."""
    bad = []
    cj = root / name / "config.json"
    if not cj.is_file():
        return [f"{cj} missing"]
    c = json.loads(cj.read_text())
    if not (c.get("harness") or {}).get("exec_forcing"):
        bad.append("config.json does not record exec_forcing -- NOT an EF directory")
    if c.get("model", "Qwen3-8B") != "Qwen3-8B" or c.get("ckpt"):
        bad.append(f"config.json model={c.get('model')} ckpt={c.get('ckpt')!r} (EF is the untrained 8B)")
    h = dict(c.get("harness") or {})
    h.pop("exec_forcing", None)
    want = {"kernel": True, "library": "", "siblings": 0, "system": "science", "echo_last": "1"}
    off = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    if off:
        bad.append(f"config.json harness differs from the v3 harness: {off}")
    for s in ("design", "test"):
        for f in sets[s]:
            p = root / name / s / f"{f}.json"
            if p.is_file() and not (json.loads(p.read_text()).get("harness_v2") or {}).get("exec_forcing"):
                bad.append(f"{s}/{f}.json does not record exec_forcing")
    return bad


def reading(cells: dict) -> dict:
    """cells: {'surv': x|None, 'conf': x|None, 'spill': d|None} -> {label: FIRES / DOES-NOT-FIRE / INCOMPLETE}."""
    out = {}
    out["EXEC-EXPLAINS-SURV"] = "INCOMPLETE" if cells.get("surv") is None else \
        ("FIRES" if cells["surv"] >= SURV_BAR else "DOES-NOT-FIRE")
    out["RESIDUAL-CONFOUNDED"] = "INCOMPLETE" if cells.get("conf") is None else \
        ("FIRES" if cells["conf"] <= CONF_BAR else "DOES-NOT-FIRE")
    out["NO-SPILL"] = "INCOMPLETE" if cells.get("spill") is None else \
        ("FIRES" if abs(cells["spill"]) <= SPILL_BAR else "DOES-NOT-FIRE")
    return out


MEANING = {("EXEC-EXPLAINS-SURV", "FIRES"): "execution alone lifts surv_cox to >= 40: the execution account holds there",
           ("EXEC-EXPLAINS-SURV", "DOES-NOT-FIRE"): "surv_cox does not reach 40 under forced execution: the execution "
                                                    "account of surv_cox is wrong",
           ("RESIDUAL-CONFOUNDED", "FIRES"): "execution alone stays <= 27 on causal_confounded: the residual to PR s1/s2 "
                                             "is what RL adds beyond execution",
           ("RESIDUAL-CONFOUNDED", "DOES-NOT-FIRE"): "execution alone exceeds 27 on causal_confounded: part of what was "
                                                     "attributed to RL beyond execution is execution",
           ("NO-SPILL", "FIRES"): "forcing execution does not move the 14 zero-competence families (+-3)",
           ("NO-SPILL", "DOES-NOT-FIRE"): "forcing execution moves the 14 zero-competence families by more than 3 points"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--ef", default="V3EF")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where EF_READING.md/.json go (default: <root>/<ef>, if it exists)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    sets = A.load_families()
    usets = uniform_sets(sets)
    rng = np.random.default_rng(a.seed)
    test_all = A.cell_members(sets, "test")
    others = [f for f in test_all if f not in COMPETENT]

    def load(name):
        return A.load_config(root, name, sets) if (root / name).is_dir() else None
    base, ef = load(a.base), load(a.ef)
    prn = {s: f"{a.base}__{PR_RUN.format(seed=s)}_s{PR_STEP}" for s in PR_SEEDS}
    pr = {s: load(n) for s, n in prn.items()}
    warn = []
    if ef is not None:
        warn += [f"EF PROVENANCE: {x}" for x in ef_provenance(root, a.ef, sets)]
    miss = {"base": complete(base, sets, test_all), "ef": complete(ef, sets, test_all)}
    miss.update({f"pr_s{s}": complete(pr[s], sets, test_all) for s in PR_SEEDS})
    L = ["# EF reading -- execution-forcing harness, no training (docs/RL_PLAN.md 2026-09-28 14:40 UTC)\n",
         f"root `{root}`; EF `{a.ef}` vs base `{a.base}` (untrained 8B, plain v3 harness) and PR s1-s3 @{PR_STEP}; paired "
         f"by instance; p one-sided sign-flip ({a.perms} perms). Rules: read_ef.py docstring.\n"]
    if ef is None:
        L.append(f"**{root / a.ef} does not exist yet (queue item 0a_0091_EF_1_exec_forcing_base8b) -- PENDING.**\n")
    for k, v in miss.items():
        if v and not (k == "ef" and ef is None):
            L.append(f"* {k}: {len(v)} TEST families missing or short: {', '.join(v)}")

    # ------------------------------------------------------------------ registered readings
    ok_s = ef is not None and base is not None and SURV not in miss["ef"] and SURV not in miss["base"]
    ok_c = ef is not None and base is not None and CONF not in miss["ef"] and CONF not in miss["base"]
    ok_o = ef is not None and base is not None and not (set(others) & (set(miss["ef"]) | set(miss["base"])))
    surv, conf = fam(ef, sets, SURV), fam(ef, sets, CONF)
    d_surv = pair(ef, base, sets, a.perms, rng, [SURV])
    d_conf = pair(ef, base, sets, a.perms, rng, [CONF])
    spill_u = pair(ef, base, usets, a.perms, rng, others)
    spill_w = pair(ef, base, sets, a.perms, rng, others)
    resid = {s: pair(pr[s], ef, sets, a.perms, rng, [CONF]) for s in PR_SEEDS}
    cells = {"surv": surv if ok_s else None, "conf": conf if ok_c else None,
             "spill": spill_u["d"] if (ok_o and spill_u) else None}
    prov = {"surv": surv, "conf": conf, "spill": spill_u["d"] if spill_u else None}
    verdict = reading(cells)
    provisional = reading(prov)
    L.append("\n## Registered readings\n")
    L.append("| reading | rule | EF | base | EF - base (p one-sided) | result |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| EXEC-EXPLAINS-SURV | surv_cox >= {SURV_BAR:.0f} | {fmt(surv)} | {fmt(fam(base, sets, SURV))} | "
             + (f"{d_surv['d']:+.1f} (p>0 {d_surv['p_gt']:.4f})" if d_surv else "-") + f" | **{verdict['EXEC-EXPLAINS-SURV']}** |")
    L.append(f"| RESIDUAL-CONFOUNDED | causal_confounded <= {CONF_BAR:.0f} | {fmt(conf)} | {fmt(fam(base, sets, CONF))} | "
             + (f"{d_conf['d']:+.1f} (p>0 {d_conf['p_gt']:.4f})" if d_conf else "-") + f" | **{verdict['RESIDUAL-CONFOUNDED']}** |")
    L.append(f"| NO-SPILL | mean change on the 14 others within +-{SPILL_BAR:.0f} | "
             + (f"{fmt(spill_u['d'], sign=True)} (unweighted, {spill_u['n_fams']}/14 fams; p>0 {spill_u['p_gt']:.4f}, "
                f"p<0 {spill_u['p_lt']:.4f})" if spill_u else "-")
             + " | - | " + (f"twin-weighted {spill_w['d']:+.1f} (p>0 {spill_w['p_gt']:.4f}, p<0 {spill_w['p_lt']:.4f})"
                            if spill_w else "-") + f" | **{verdict['NO-SPILL']}** |")
    for k, v in verdict.items():
        if v in ("FIRES", "DOES-NOT-FIRE"):
            L.append(f"* {k} {v}: {MEANING[(k, v)]}")
        elif provisional[k] != "INCOMPLETE":
            L.append(f"* {k} INCOMPLETE -- PROVISIONAL on the rows present (NOT the registered reading): {provisional[k]}")
    L.append("* residual PR - EF on causal_confounded: " + "; ".join(
        f"s{s} {fmt(fam(pr[s], sets, CONF))} ({resid[s]['d']:+.1f}, p>0 {resid[s]['p_gt']:.4f})" if resid[s] else f"s{s} -"
        for s in PR_SEEDS))

    # ------------------------------------------------------------------ beside: share of PR's gain
    L.append("\n## Beside (never deciding): the share of PR's mean gain that EF recovers without training\n")
    L.append("| set | base | EF (d vs base, p>0) | PR s1 / s2 / s3 @100 (d vs base) | PR mean gain | EF share | EF - PR s1 / s2 / s3 (p<0) |")
    L.append("|---|---|---|---|---|---|---|")
    beside = {}
    for label, members in (("HO6", HO6), ("TEST-all", test_all)):
        b = A.cell_pass(base, sets, "test", members)[0] if base is not None else None
        e = A.cell_pass(ef, sets, "test", members)[0] if ef is not None else None
        de = pair(ef, base, sets, a.perms, rng, members)
        dp = {s: pair(pr[s], base, sets, a.perms, rng, members) for s in PR_SEEDS}
        ep = {s: pair(ef, pr[s], sets, a.perms, rng, members) for s in PR_SEEDS}
        gains = [dp[s]["d"] for s in PR_SEEDS if dp[s]]
        g = float(np.mean(gains)) if len(gains) == len(PR_SEEDS) else None
        share = (de["d"] / g) if (de and g) else None
        full = not any(set(members) & set(miss[k]) for k in miss)
        beside[label] = {"base": b, "ef": e, "d_ef": de, "pr": {s: A.cell_pass(pr[s], sets, "test", members)[0]
                                                                if pr[s] is not None else None for s in PR_SEEDS},
                         "d_pr": dp, "pr_mean_gain": g, "share": share, "ef_minus_pr": ep, "complete": full}
        de_s = "-" if not de else f"{de['d']:+.1f}, p={de['p_gt']:.4f}"
        L.append(f"| {label} | {fmt(b)} | {fmt(e)} ({de_s}) | "
                 + " / ".join(f"{fmt(beside[label]['pr'][s])} ({fmt(dp[s]['d'], sign=True) if dp[s] else '-'})"
                              for s in PR_SEEDS)
                 + f" | {fmt(g, sign=True)} | {'-' if share is None else f'{100 * share:.0f}%'} | "
                 + " / ".join(f"{ep[s]['d']:+.1f} (p={ep[s]['p_lt']:.4f})" if ep[s] else "-" for s in PR_SEEDS)
                 + ("" if full else " (PARTIAL)") + " |")

    # ------------------------------------------------------------------ per family
    L.append("\n## Per family (pass %, 12x2; DESIGN causal_rdd 16x2)\n")
    L.append("| family | base | EF | EF - base (p>0) | PR s1 / s2 / s3 | EF refusals / rollout |")
    L.append("|---|---|---|---|---|---|")
    per = {}
    for s, members in (("design", [DESIGN_FAM]), ("test", test_all)):
        for f in members:
            b, e = fam(base, sets, f, s), fam(ef, sets, f, s)
            d = pair(ef, base, sets, a.perms, rng, [f], s)
            prs = [fam(pr[x], sets, f, s) for x in PR_SEEDS]
            rows = (ef or {}).get("rollouts", {}).get((s, f)) or []
            rj = [int((((r.get("v2") or {}).get("ef")) or {}).get("rejected") or 0) for r in rows]
            per[f"{s}/{f}"] = {"base": b, "ef": e, "d": d, "pr": prs, "refusals": float(np.mean(rj)) if rj else None}
            L.append(f"| {f}{' (DESIGN)' if s == 'design' else (' (HO6)' if f in HO6 else '')} | {fmt(b)} | {fmt(e)} | "
                     + (f"{d['d']:+.1f} ({d['p_gt']:.3f})" if d else "-") + " | "
                     + " / ".join(fmt(x) for x in prs) + f" | {fmt(per[f'{s}/{f}']['refusals'], 2)} |")

    # ------------------------------------------------------------------ mechanics
    mech = {"ef": mechanics(ef) if ef is not None else None, "base": mechanics(base) if base is not None else None}
    L.append("\n## EF mechanics (TEST rollouts)\n")
    if mech["ef"] and mech["ef"].get("n"):
        me, mb = mech["ef"], mech["base"] or {}
        L.append(f"* rollouts {me['n']}; with an EF log {me['with_ef_log']}; refusals total {me['rejected_total']} "
                 f"(feedback turns in the stored messages {me['feedback_turns_total']}), mean {fmt(me['rejected_mean'], 2)} "
                 f"per rollout, rollouts with >= 1 refusal {me['rollouts_with_rejection']}")
        L.append(f"* how the final answer was accepted: {json.dumps(me['accepted'])}; rule n/a {me['na']}; gate errors "
                 f"{me['gate_errors']}")
        L.append(f"* no answer: EF {me['no_answer']} vs base {mb.get('no_answer', '-')}; forced by the turn limit: EF "
                 f"{me['forced']} vs base {mb.get('forced', '-')}; cells/rollout EF {fmt(me['cells_mean'], 2)} vs base "
                 f"{fmt(mb.get('cells_mean'), 2)}; answers with no executed cell EF {me['zero_cell_answers']} vs base "
                 f"{mb.get('zero_cell_answers', '-')}; errored EF {me['errored']} vs base {mb.get('errored', '-')}")
        if me["with_ef_log"] < me["n"] - me["errored"]:
            warn.append(f"{me['n'] - me['with_ef_log']} EF rollouts have no EF log (and only {me['errored']} errored)")
        if me["gate_errors"]:
            warn.append(f"{me['gate_errors']} gate errors (fail-open: those answers were accepted as in V3)")
    else:
        L.append("* (no EF rollouts yet)")
    status = "READ" if all(v != "INCOMPLETE" for v in verdict.values()) else "INCOMPLETE"
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(2, f"**Status: {status}** -- " + ", ".join(f"{k} {v}" for k, v in verdict.items()) + "\n")
    summary = {"root": str(root), "base": a.base, "ef": a.ef, "pr": prn, "status": status, "readings": verdict,
               "provisional": provisional, "cells": cells, "surv": {"ef": surv, "d": d_surv},
               "conf": {"ef": conf, "d": d_conf, "residual_pr_minus_ef": resid},
               "spill": {"unweighted": spill_u, "twin_weighted": spill_w, "families": others},
               "beside": beside, "per_family": per, "mechanics": mech, "missing": miss, "warnings": warn}
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out) if a.out else ((root / a.ef) if (root / a.ef).is_dir() else None)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "EF_READING.md").write_text(text)
        (out / "EF_READING.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
