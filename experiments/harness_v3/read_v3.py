#!/usr/bin/env python
"""HARNESS V3 reading (docs/RL_PLAN.md s.15, registered 2026-09-26 before any v3 run). Every checkpoint -- the v3 PR and
PR-SHAM runs AND the matched P14 runs -- is scored in the SAME v3 evaluation harness (experiments/harness_v3/screen_v3.sh:
KERNEL + science + echo, NO library) and paired by instance with each other and with the untrained 14B in that harness.

    venv/bin/python experiments/harness_v3/read_v3.py [--root results/pods/harness_v3_q14] [--base V3]
                                                      [--ckpt-root $DEEP_DATA_ROOT/ckpt] [--perms 20000]

Arms (from the run name): pr (-hv3pr<lam>-), sham (-hv3prsham<lam>-), plain (-hv3pr0-, optional), p14 (-hv2lb3d-).
HO6 = unweighted mean pass of the six RL_PLAN held-out families (TEST, 12x2). Contrasts are PAIRED by instance
(analyze.py's sign-flip permutation, one-sided in the direction of the rule).

Registered decision (read at steps 50 and 100 only; the SAME step for both bars):
  VALID(arm, s)   TRAIN(arm, s) >= min(base TRAIN + 20, 95)            (the P14 rule, base = the v3 14B base)
  CONTRASTS       dP14(f, s)  = HO6(PR_f, s) - HO6(P14_f, s)          f in {causal_iv, arr_kinetics}
                  dSHAM(s)    = HO6(PR_causal_iv, s) - HO6(SHAM_causal_iv, s)
  COST            an external of PR_f at step 100 below the v3 base with McNemar p < 0.05 (blocks promotion of that arm)
  causal_iv:
    PROCESS-WINS  some s in {50, 100}: VALID(PR, s) and dP14 >= +5 and dSHAM >= +5, and no COST
                  -> 2 more PR seeds on causal_iv + 1 more SHAM seed (replication; one seed is not a finding)
    EXTRA-SIGNAL  not PROCESS-WINS; some s: dP14 >= +5 while dSHAM < +2 at both steps and HO6(SHAM) - HO6(P14) >= +3:
                  the gain is extra reward / live groups, not the process content
    KILL          dSHAM < +2 at both steps AND dP14 < +2 at both steps: the process reward adds nothing
    KILL-SHAM     dSHAM < +2 at both steps (otherwise): process content no better than a randomly assigned bonus
    NO-FOOTHOLD   PR never VALID (not a transfer test; reported, never a null)
    WEAK          anything else
  arr_kinetics (no SHAM arm of its own; the causal_iv SHAM contrast is the control):
    PROMOTE       some s: VALID(PR, s) and dP14 >= +5, no COST, AND causal_iv read PROCESS-WINS
    UNCONTROLLED  dP14 >= +5 at a VALID step but causal_iv did not read PROCESS-WINS
    DROP          dP14 < +2 at both steps       NO-FOOTHOLD / WEAK as above
  KILL of the whole v3 process reward: KILL or KILL-SHAM on causal_iv AND arr_kinetics DROP.
  Labels beside the decision (never deciding): DO-LESS -- HO6 eval cells/rollout of PR_f < 0.7 x the base's; LOCAL --
  the training process score rose >= +0.2 (mean p, steps 1-10 vs 91-100) while the process score on the HO6 eval rollouts
  did not rise >= +0.05 over the base's (the detectors were satisfied on the training family only).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "experiments" / "harness_v2"))
import analyze as A  # noqa: E402

HO6 = ["causal_confounded", "causal_att", "surv_cox", "enzyme_mm", "hc_dx", "osc_damped"]
FAMS = ("causal_iv", "arr_kinetics")
READ_STEPS = (50, 100)
BAR, KILL_BAR, SHAM_OVER_P14, VALID_GAIN, VALID_CAP = 5.0, 2.0, 3.0, 20.0, 95.0
CELLS_FLOOR, LOCAL_RISE, LOCAL_EVAL = 0.7, 0.2, 0.05
RUN_RE = re.compile(r"^(?P<base>[A-Za-z0-9]+)__(?P<run>rl-czero-(?P<fam>[a-z0-9_]+?)-q14base-.*-s(?P<seed>\d+))_s(?P<step>\d+)$")


def arm_of(run: str) -> str | None:
    if "-hv3prsham" in run:
        return "sham"
    if "-hv3prcols" in run:
        return "cols"
    m = re.search(r"-hv3pr([0-9.]+)-", run)
    if m:
        return "plain" if float(m.group(1)) == 0 else "pr"
    if "-hv2lb3d-" in run:
        return "p14"
    return None


def ho6(res, sets):
    return A.cell_pass(res, sets, "test", HO6)[0]


def contrast(a, b, sets, perms, rng, members=HO6, s="test"):
    """(Delta, one-sided p for Delta > 0, one-sided p for Delta < 0, n families) of a vs b, paired, or None."""
    d = A.paired(a, b, sets, s, members, perms, rng)
    return None if d is None else (d[0], d[2], d[3], d[6])


def cells_mean(res, members=HO6) -> float | None:
    xs = [r.get("n_cells") or 0 for (s, f), rs in res["rollouts"].items() if s == "test" and f in members for r in rs]
    return float(np.mean(xs)) if xs else None


def process_on_eval(root: Path, cfgname: str, members=HO6) -> dict | None:
    """Process-signal rates on the HO6 EVAL rollouts of one configuration (the mechanism readout)."""
    from paths import data_root
    from process_signals import SIGNALS, ProcessScorer
    G = data_root() / "generated"
    ps = ProcessScorer()
    vals = {k: [] for k in SIGNALS}
    scores = []
    for f in members:
        p = root / cfgname / "test" / f"{f}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        for inst, rs in (d.get("groups") or {}).items():
            idir = G / f / d.get("split", "test") / inst
            if not (idir / "eval_config.json").is_file():
                continue
            cfg = json.loads((idir / "eval_config.json").read_text())
            for r in rs:
                if not r.get("messages") or not (r.get("n_cells") or 0):
                    scores.append(0.0)
                    continue
                s = ps.signals(r["messages"], cfg, idir / "data_files", answer=r.get("answer"),
                               n_prefix=int(r.get("n_prefix") or 0))
                scores.append(s.score)
                for k in SIGNALS:
                    if s.fired[k] is not None:
                        vals[k].append(s.fired[k])
    if not scores:
        return None
    out = {"p": float(np.mean(scores)), "n": len(scores)}
    out.update({k: (float(np.mean(v)) if v else None) for k, v in vals.items()})
    return out


def training_curve(ck: Path) -> dict | None:
    f = ck / "process_history.jsonl"
    if not f.is_file():
        return None
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if not rows:
        return None
    early = [r["mean_p"] for r in rows if 1 <= r["step"] <= 10]
    late = [r["mean_p"] for r in rows if 91 <= r["step"] <= 100]
    alive_a = [r["groups_alive_answer"] for r in rows if r["step"] > 50]
    alive_t = [r["groups_alive_total"] for r in rows if r["step"] > 50]
    return {"steps": len(rows), "p_early": statistics.fmean(early) if early else None,
            "p_late": statistics.fmean(late) if late else None,
            "alive_answer_after50": statistics.fmean(alive_a) if alive_a else None,
            "alive_total_after50": statistics.fmean(alive_t) if alive_t else None,
            "rates_last": {k: rows[-1].get(f"rate_{k}") for k in ("inspect", "strata", "check", "spec", "grounded")}}


def valid(res, base, sets, fam) -> tuple[bool | None, float | None, float | None]:
    t = A.cell_pass(res, sets, "design", [fam])[0]
    t0 = A.cell_pass(base, sets, "design", [fam])[0]
    if t is None or t0 is None:
        return None, t, t0
    return t >= min(t0 + VALID_GAIN, VALID_CAP), t, t0


def cost(res, base) -> list:
    out = []
    for b in A.BENCHES:
        if b in res["ext"] and b in base["ext"]:
            g, l, p, n = A.mcnemar(res["ext"][b], base["ext"][b])
            acc = 100.0 * np.mean(list(res["ext"][b].values()))
            acc0 = 100.0 * np.mean([base["ext"][b][k] for k in res["ext"][b] if k in base["ext"][b]])
            out.append({"bench": b, "acc": acc, "delta": acc - acc0, "gained": g, "lost": l, "p": p,
                        "cost": (acc < acc0 and p < A.COST_EXT_P)})
    return out


def decide_iv(c: dict) -> tuple[str, list]:
    """c[s] = {'valid', 'dP14', 'dSHAM', 'sham_over_p14'} for s in READ_STEPS; c['cost'] = bool."""
    why = []
    have = [s for s in READ_STEPS if s in c and c[s].get("dP14") is not None and c[s].get("dSHAM") is not None]
    if len(have) < len(READ_STEPS):
        return "INCOMPLETE", [f"steps with PR, P14 and SHAM on disk: {have}"]
    why.append("VALID " + ", ".join(f"s{s}={c[s]['valid']}" for s in READ_STEPS))
    why.append("dP14 " + ", ".join(f"s{s} {c[s]['dP14']:+.1f}" for s in READ_STEPS)
               + "; dSHAM " + ", ".join(f"s{s} {c[s]['dSHAM']:+.1f}" for s in READ_STEPS))
    win = [s for s in READ_STEPS if c[s]["valid"] and c[s]["dP14"] >= BAR and c[s]["dSHAM"] >= BAR]
    if win and not c.get("cost"):
        return "PROCESS-WINS", why + [f"both bars at step(s) {win}"]
    if win and c.get("cost"):
        why.append(f"bars met at {win} but an external COST blocks promotion")
    sham_low = all(c[s]["dSHAM"] < KILL_BAR for s in READ_STEPS)
    p14_low = all(c[s]["dP14"] < KILL_BAR for s in READ_STEPS)
    if not any(c[s]["valid"] for s in READ_STEPS):
        return "NO-FOOTHOLD", why + ["PR never reached base TRAIN + 20: not a transfer test"]
    if sham_low and any(c[s]["dP14"] >= BAR and (c[s].get("sham_over_p14") or -99) >= SHAM_OVER_P14
                        for s in READ_STEPS):
        return "EXTRA-SIGNAL", why + ["PR > P14 but PR ~ SHAM and SHAM > P14: extra reward, not process content"]
    if sham_low and p14_low:
        return "KILL", why + ["no gain over answer-only RL and none over the sham"]
    if sham_low:
        return "KILL-SHAM", why + ["the process content is no better than a randomly assigned bonus"]
    return "WEAK", why + ["between the bars: reported, not replicated"]


def decide_arr(c: dict, iv_decision: str) -> tuple[str, list]:
    why = []
    have = [s for s in READ_STEPS if s in c and c[s].get("dP14") is not None]
    if len(have) < len(READ_STEPS):
        return "INCOMPLETE", [f"steps with PR and P14 on disk: {have}"]
    why.append("VALID " + ", ".join(f"s{s}={c[s]['valid']}" for s in READ_STEPS)
               + "; dP14 " + ", ".join(f"s{s} {c[s]['dP14']:+.1f}" for s in READ_STEPS))
    win = [s for s in READ_STEPS if c[s]["valid"] and c[s]["dP14"] >= BAR]
    if win and not c.get("cost"):
        if iv_decision == "PROCESS-WINS":
            return "PROMOTE", why + [f"dP14 >= +{BAR:.0f} at {win}; causal_iv's SHAM control passed"]
        return "UNCONTROLLED", why + [f"dP14 >= +{BAR:.0f} at {win}, but causal_iv read {iv_decision}: no SHAM control"]
    if not any(c[s]["valid"] for s in READ_STEPS):
        return "NO-FOOTHOLD", why
    if all(c[s]["dP14"] < KILL_BAR for s in READ_STEPS):
        return "DROP", why + [f"dP14 < +{KILL_BAR:.0f} at both steps"]
    return "WEAK", why + (["an external COST blocks promotion"] if win else [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3_q14"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-process", action="store_true", help="skip the process-signal readout on eval rollouts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    if a.ckpt_root:
        ck_root = Path(a.ckpt_root)
    else:
        from paths import data_root
        ck_root = data_root() / "ckpt"
    sets = A.load_families()
    rng = np.random.default_rng(a.seed)
    if not (root / a.base).is_dir():
        print(f"no base {root}/{a.base} yet (item HV3_0) -- nothing to read")
        return 1
    base = A.load_config(root, a.base, sets)
    found: dict = {}
    for p in sorted(root.iterdir()):
        m = RUN_RE.match(p.name)
        if p.is_dir() and m and m["base"] == a.base and arm_of(m["run"]):
            found.setdefault((m["fam"], arm_of(m["run"])), {"run": m["run"], "steps": {}})["steps"][int(m["step"])] = p.name
    b_ho = ho6(base, sets)
    b_cells = cells_mean(base)
    b_proc = None if a.no_process else process_on_eval(root, a.base)
    lines = ["# Harness v3 reading (docs/RL_PLAN.md s.15)\n",
             f"root `{root}`; base `{a.base}` = untrained Qwen3-14B in the v3 harness; every checkpoint scored in the same "
             f"harness, paired by instance; p one-sided sign-flip ({a.perms} perms). Rule: read_v3.py docstring.\n",
             f"Base: HO6 {b_ho if b_ho is not None else float('nan'):.1f}; cells/rollout on HO6 {b_cells or 0:.2f}; "
             f"process on HO6 {json.dumps(b_proc) if b_proc else '-'}\n"]
    summary: dict = {"base": {"ho6": b_ho, "cells": b_cells, "process": b_proc}, "arms": {}}
    loaded: dict = {}
    for (fam, arm), info in sorted(found.items()):
        for s, name in info["steps"].items():
            loaded[(fam, arm, s)] = A.load_config(root, name, sets)
    table = ["| family | arm | step | TRAIN (base) | VALID | HO6 | dHO6 vs base | TEST-all d | cells/rollout (x base) | "
             "process p on HO6 | externals |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (fam, arm, s), res in sorted(loaded.items()):
        v, t, t0 = valid(res, base, sets, fam)
        h = ho6(res, sets)
        d = contrast(res, base, sets, a.perms, rng)
        ta = contrast(res, base, sets, a.perms, rng, members=A.cell_members(sets, "test"))
        cm = cells_mean(res)
        pe = None if a.no_process else process_on_eval(root, found[(fam, arm)]["steps"][s])
        ext = cost(res, base)
        summary["arms"].setdefault(f"{fam}:{arm}", {})[s] = {
            "run": found[(fam, arm)]["run"], "train": t, "train_base": t0, "valid": v, "ho6": h,
            "d_base": d and d[0], "test_all_d": ta and ta[0], "cells": cm, "process": pe, "ext": ext,
            "ho6_fams": {f: A.cell_pass(res, sets, "test", [f])[0] for f in HO6}}
        table.append(f"| {fam} | {arm} | {s}{'' if s in READ_STEPS else ' (not read)'} | "
                     f"{t if t is None else round(t, 1)} ({t0 if t0 is None else round(t0, 1)}) | {v} | "
                     f"{h if h is None else round(h, 1)} | {'-' if d is None else f'{d[0]:+.1f} (p={d[1]:.3f})'} | "
                     f"{'-' if ta is None else f'{ta[0]:+.1f}'} | "
                     f"{'-' if cm is None else f'{cm:.2f} ({cm / b_cells:.2f}x)' if b_cells else cm} | "
                     f"{'-' if not pe else round(pe['p'], 3)} | "
                     + ("; ".join(f"{e['bench']} {e['delta']:+.1f} (p={e['p']:.3f}{', COST' if e['cost'] else ''})"
                                  for e in ext) or "-") + " |")
    lines += table
    decisions = {}
    for fam in FAMS:
        c = {}
        for s in READ_STEPS:
            pr, p14, sh = loaded.get((fam, "pr", s)), loaded.get((fam, "p14", s)), loaded.get((fam, "sham", s))
            if pr is None:
                continue
            row = {"valid": valid(pr, base, sets, fam)[0]}
            x = contrast(pr, p14, sets, a.perms, rng) if p14 is not None else None
            row["dP14"], row["p_dP14"] = (x[0], x[1]) if x else (None, None)
            if fam == "causal_iv":
                y = contrast(pr, sh, sets, a.perms, rng) if sh is not None else None
                row["dSHAM"], row["p_dSHAM"] = (y[0], y[1]) if y else (None, None)
                z = contrast(sh, p14, sets, a.perms, rng) if (sh is not None and p14 is not None) else None
                row["sham_over_p14"] = z[0] if z else None
            c[s] = row
        pr100 = loaded.get((fam, "pr", 100))
        c["cost"] = any(e["cost"] for e in cost(pr100, base)) if pr100 is not None else False
        if fam == "causal_iv":
            decisions[fam] = decide_iv(c)
        else:
            decisions[fam] = decide_arr(c, decisions.get("causal_iv", ("INCOMPLETE", []))[0])
        labels = []
        for s in READ_STEPS:
            arm = summary["arms"].get(f"{fam}:pr", {}).get(s)
            if arm and arm["cells"] is not None and b_cells and arm["cells"] < CELLS_FLOOR * b_cells:
                labels.append(f"DO-LESS at s{s} ({arm['cells']:.2f} < {CELLS_FLOOR} x {b_cells:.2f})")
        run = found.get((fam, "pr"), {}).get("run")
        curve = training_curve(ck_root / run) if run else None
        arm100 = summary["arms"].get(f"{fam}:pr", {}).get(100)
        if curve and curve["p_early"] is not None and curve["p_late"] is not None and arm100 and arm100.get("process") \
                and b_proc and curve["p_late"] - curve["p_early"] >= LOCAL_RISE \
                and arm100["process"]["p"] - b_proc["p"] < LOCAL_EVAL:
            labels.append(f"LOCAL (training p {curve['p_early']:.2f} -> {curve['p_late']:.2f}; HO6 eval p "
                          f"{arm100['process']['p']:.2f} vs base {b_proc['p']:.2f})")
        summary.setdefault("contrasts", {})[fam] = c
        summary.setdefault("curves", {})[fam] = curve
        summary.setdefault("labels", {})[fam] = labels
        lines.append(f"\n## {fam}: **{decisions[fam][0]}**")
        lines += [f"* {w}" for w in decisions[fam][1]] + [f"* label: {x}" for x in labels]
        if curve:
            lines.append(f"* training process curve: {json.dumps(curve)}")
    whole = ("KILL" if decisions.get("causal_iv", ("",))[0] in ("KILL", "KILL-SHAM")
             and decisions.get("arr_kinetics", ("",))[0] == "DROP" else None)
    if whole:
        lines.append("\n**The v3 process reward is KILLED as registered (causal_iv KILL/KILL-SHAM and arr_kinetics DROP).**")
    summary["decisions"] = {k: v[0] for k, v in decisions.items()}
    out = Path(a.out) if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    (out / "V3_READING.md").write_text("\n".join(lines) + "\n")
    (out / "V3_READING.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
