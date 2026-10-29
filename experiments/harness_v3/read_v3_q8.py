#!/usr/bin/env python
"""HARNESS V3 AT 8B reading (docs/RL_PLAN.md s.16, registered 2026-09-26 before any run). Qwen3-8B trained with the v3
recipe (rl/grpo_v3.py via experiments/harness_v3/run_v3.sh) on ONE family, causal_rdd, ONE seed, 150 steps, three arms:
PR (lambda 0.3), PLAIN (lambda 0 = answer-only in the SAME harness: the matched control) and SHAM (lambda 0.3, process
term permuted within each GRPO group). Every checkpoint and the untrained 8B are scored in the v3 evaluation harness
(experiments/harness_v3/screen_v3.sh, MODEL=Qwen3-8B, no MTAG -> results/pods/harness_v3/) and paired by instance.

    venv/bin/python experiments/harness_v3/read_v3_q8.py [--root results/pods/harness_v3] [--base V3]
        [--r0-root results/pods/rl_plan] [--ckpt-root $DEEP_DATA_ROOT/ckpt] [--perms 20000] [--no-process] [--out DIR]

A NEW FILE: read_v3.py (the 14B reading) is imported for its helpers (ho6, contrast, cells_mean, process_on_eval, cost,
arm_of, HO6) and never edited. Runs on partial rows: every cell says which HO6 families it is missing, and the verdict is
INCOMPLETE (with a clearly marked PROVISIONAL reading on the families present) until the six cells it needs are complete.

HO6(arm, s) = unweighted mean pass of the six RL_PLAN held-out families (TEST, 12x2). Contrasts are PAIRED by instance
(analyze.py's sign-flip permutation; p one-sided in the direction of the bar).
  dPLAIN(s) = HO6(pr, s) - HO6(plain, s)      dSHAM(s) = HO6(pr, s) - HO6(sham, s)
  FADE(arm) = HO6(arm, 150) - HO6(arm, 100)
REGISTERED VERDICT (read at 100 and 150; 50 is reported; 25/75/125 are curve only). Each label is evaluated on its own;
the verdict lists every label that fires, headed by the first of PROCESS-WINS-8B, EXTRA-SIGNAL, PROCESS-HOLDS, KILL:
  PROCESS-WINS-8B  some s in {100, 150}: dPLAIN(s) >= +5 AND dSHAM(s) >= +5 (both bars at the SAME step)
  EXTRA-SIGNAL     some s in {100, 150}: HO6(pr) >= HO6(plain) + 5 AND HO6(sham) >= HO6(plain) + 5 AND |dSHAM(s)| < 3
                   (the reward's variance, not its content)
  PROCESS-HOLDS    FADE(pr) >= -2 AND FADE(plain) <= -3 AND dPLAIN(150) >= +5
  KILL             dPLAIN(100) <= +2 AND dPLAIN(150) <= +2
  (none)           NONE -- between the bars
  MIXED            PROCESS-WINS-8B at one step and EXTRA-SIGNAL at the other (both printed; no process claim until
                   replicated)
Beside the verdict, never deciding: COST (an external of an arm at step 150 down >= 5 points vs the 8B V3 base with exact
McNemar p < 0.05); HARNESS-EFFECT (HO6(plain, s) against the R0 seed range at s -- a different harness and draw, so a
label, not a contrast); NO-FOOTHOLD (an arm's TRAIN never reaches min(base TRAIN + 20, 95) at 50/100/150: its contrasts
are not a transfer test); DO-LESS (an arm's HO6 cells/rollout < 0.7 x the base's).
R0 REFERENCE (reported only): base_echo and rl-czero-causal_rdd-base-echo-rc-lr1e-5-s{1..4} HO6 at 50/100/150 from
results/pods/rl_plan (the plain harness, 24x4, rollout key "passed").
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
sys.path.insert(0, str(HERE))
import read_v3 as R  # noqa: E402  (its import puts ROOT, harness/ and experiments/harness_v2 on sys.path)

A = R.A
HO6 = R.HO6
FAM = "causal_rdd"
SEED = 1
ARMS = ("pr", "plain", "sham")
RUNS = {"pr": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s1",
        "plain": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0-lr1e-5-s1",
        "sham": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3prsham0.3-lr1e-5-s1"}
READ_STEPS = (50, 100, 150)
DECIDE_STEPS = (100, 150)
CURVE_STEPS = (25, 75, 125)
BAR, KILL_BAR, SAME_BAR = 5.0, 2.0, 3.0
FADE_PR_MIN, FADE_PLAIN_MAX, HOLD_GAP = -2.0, -3.0, 5.0
COST_PTS, COST_P = 5.0, 0.05
VALID_GAIN, VALID_CAP, CELLS_FLOOR = 20.0, 95.0, 0.7
MODEL = "Qwen3-8B"
R0_RUNS = [f"rl-czero-causal_rdd-base-echo-rc-lr1e-5-s{s}" for s in (1, 2, 3, 4)]
RUN_RE8 = re.compile(r"^(?P<base>[A-Za-z0-9]+)__(?P<run>rl-czero-(?P<fam>[a-z0-9_]+?)-base-echo-.*-hv3.*-s(?P<seed>\d+))"
                     r"_s(?P<step>\d+)$")


def fmt(x, nd=1, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ho6_detail(res, sets) -> dict:
    fams = {f: A.cell_pass(res, sets, "test", [f])[0] for f in HO6}
    present = [f for f in HO6 if fams[f] is not None]
    return {"ho6": R.ho6(res, sets), "n": len(present), "fams": fams,
            "missing": [f for f in HO6 if fams[f] is None], "complete": len(present) == len(HO6)}


def pair(a, b, sets, perms, rng, members=HO6):
    """Paired Delta of a vs b (points), one-sided p for > 0 and < 0, the families both have, and what either misses."""
    if a is None or b is None:
        return None
    x = R.contrast(a, b, sets, perms, rng, members=members)
    if x is None:
        return None
    fa = {f for f in members if ("test", f) in a["families"]}
    fb = {f for f in members if ("test", f) in b["families"]}
    return {"d": x[0], "p_gt": x[1], "p_lt": x[2], "n_fams": x[3],
            "missing": sorted(set(members) - (fa & fb)), "complete": fa >= set(members) and fb >= set(members)}


def ext_rows(res, base) -> list:
    rows = R.cost(res, base) if base is not None else []
    for e in rows:
        e["cost"] = bool(e["delta"] <= -COST_PTS and e["p"] < COST_P)
    return rows


def training_curve(ck: Path) -> dict | None:
    f, h = ck / "process_history.jsonl", ck / "history.jsonl"
    if not f.is_file():
        return None
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if not rows:
        return None

    def mean(key, lo, hi, src=rows):
        xs = [r[key] for r in src if lo <= r.get("step", -1) <= hi and r.get(key) is not None]
        return statistics.fmean(xs) if xs else None
    hist = [json.loads(x) for x in h.read_text().splitlines() if x.strip()] if h.is_file() else []
    return {"last_step": rows[-1].get("step"),
            "p_1_10": mean("mean_p", 1, 10), "p_91_100": mean("mean_p", 91, 100), "p_141_150": mean("mean_p", 141, 150),
            "alive_answer_51_150": mean("groups_alive_answer", 51, 150),
            "alive_total_51_150": mean("groups_alive_total", 51, 150),
            "pass_1_10": mean("pass_rate", 1, 10, hist), "pass_141_150": mean("pass_rate", 141, 150, hist),
            "cells_last": rows[-1].get("cells_per_rollout"),
            "rates_last": {k: rows[-1].get(f"rate_{k}") for k in ("inspect", "strata", "check", "spec", "grounded")}}


def r0_reference(r0_root: Path) -> dict:
    """HO6 of base_echo and the four R0 causal_rdd seeds at 50/100/150 (the plain harness; rollout key 'passed')."""
    def one(d: Path):
        fams = {}
        for f in HO6:
            p = d / f"{f}.json"
            if p.is_file():
                inst, _ = A.load_family(p)
                v = [np.mean(x) for x in inst.values() if x]
                if v:
                    fams[f] = 100.0 * float(np.mean(v))
        return {"ho6": float(np.mean(list(fams.values()))) if fams else None, "fams": fams,
                "missing": [f for f in HO6 if f not in fams]}
    out = {"base_echo": one(r0_root / "base_echo"), "seeds": {}, "range": {}}
    for run in R0_RUNS:
        out["seeds"][run] = {s: one(r0_root / run / f"step{s}") for s in READ_STEPS}
    for s in READ_STEPS:
        xs = [out["seeds"][run][s]["ho6"] for run in R0_RUNS if out["seeds"][run][s]["ho6"] is not None
              and not out["seeds"][run][s]["missing"]]
        out["range"][s] = (min(xs), max(xs), len(xs)) if xs else None
    return out


def decide(c: dict) -> tuple[str, list, list]:
    """c = {'dPLAIN': {s: d}, 'dSHAM': {s: d}, 'dSP': {s: d}, 'FADE': {'pr': d, 'plain': d}} (points; None if unread).
    -> (verdict, labels that fired, why)."""
    need = [("dPLAIN", s) for s in DECIDE_STEPS] + [("dSHAM", s) for s in DECIDE_STEPS] + \
           [("dSP", s) for s in DECIDE_STEPS] + [("FADE", "pr"), ("FADE", "plain")]
    miss = [f"{k}({s})" for k, s in need if c.get(k, {}).get(s) is None]
    if miss:
        return "INCOMPLETE", [], [f"not yet computable: {', '.join(miss)}"]
    dp, ds, dsp, fade = c["dPLAIN"], c["dSHAM"], c["dSP"], c["FADE"]
    why = ["dPLAIN " + ", ".join(f"s{s} {dp[s]:+.1f}" for s in DECIDE_STEPS)
           + "; dSHAM " + ", ".join(f"s{s} {ds[s]:+.1f}" for s in DECIDE_STEPS)
           + "; SHAM-PLAIN " + ", ".join(f"s{s} {dsp[s]:+.1f}" for s in DECIDE_STEPS)
           + f"; FADE pr {fade['pr']:+.1f}, plain {fade['plain']:+.1f}"]
    wins = [s for s in DECIDE_STEPS if dp[s] >= BAR and ds[s] >= BAR]
    extra = [s for s in DECIDE_STEPS if dp[s] >= BAR and dsp[s] >= BAR and abs(ds[s]) < SAME_BAR]
    holds = fade["pr"] >= FADE_PR_MIN and fade["plain"] <= FADE_PLAIN_MAX and dp[150] >= HOLD_GAP
    kill = all(dp[s] <= KILL_BAR for s in DECIDE_STEPS)
    labels = []
    if wins:
        labels.append(f"PROCESS-WINS-8B (step {'/'.join(map(str, wins))})")
    if extra:
        labels.append(f"EXTRA-SIGNAL (step {'/'.join(map(str, extra))})")
    if holds:
        labels.append("PROCESS-HOLDS")
    if kill:
        labels.append("KILL")
    if wins and extra:
        return "MIXED", labels, why + [f"PROCESS-WINS-8B at {wins} but EXTRA-SIGNAL at {extra}: no process claim "
                                       "until replicated"]
    if wins:
        return "PROCESS-WINS-8B", labels, why + [f"both bars (+{BAR:.0f}) at step(s) {wins}"]
    if extra:
        return "EXTRA-SIGNAL", labels, why + [f"PR ~ SHAM >= PLAIN + {BAR:.0f} at {extra}: the reward's variance, "
                                              "not its content"]
    if holds:
        return "PROCESS-HOLDS", labels, why + ["PR does not fade while PLAIN does, PR >= PLAIN + 5 at 150; the SHAM "
                                               "contrast is between the bars"]
    if kill:
        return "KILL", labels, why + [f"PR <= PLAIN + {KILL_BAR:.0f} at both 100 and 150"]
    return "NONE", labels, why + ["between the bars: reported, not replicated"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--r0-root", default=str(ROOT / "results" / "pods" / "rl_plan"))
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-process", action="store_true", help="skip the process-signal readout on HO6 eval rollouts")
    ap.add_argument("--out", default=None, help="where V3Q8_READING.md/.json go (default: --root, if it exists)")
    a = ap.parse_args()
    root = Path(a.root)
    if a.ckpt_root:
        ck_root = Path(a.ckpt_root)
    else:
        from paths import data_root
        ck_root = data_root() / "ckpt"
    sets = A.load_families()
    rng = np.random.default_rng(a.seed)
    have_base = (root / a.base).is_dir()
    base = A.load_config(root, a.base, sets) if have_base else None
    L = ["# Harness v3 at 8B reading (docs/RL_PLAN.md s.16)\n",
         f"root `{root}`; base `{a.base}` = untrained Qwen3-8B in the v3 harness; paired by instance; p one-sided "
         f"sign-flip ({a.perms} perms). Rule: read_v3_q8.py docstring / RL_PLAN s.16.\n"]
    warn = []
    if have_base:
        cj = root / a.base / "config.json"
        m = json.loads(cj.read_text()).get("model") if cj.is_file() else None
        if m not in (None, MODEL):
            warn.append(f"base config.json says model {m}, not {MODEL}")
    else:
        L.append(f"**no base {root}/{a.base} yet (item HV3_17)** -- base-relative columns are empty.\n")
    # ------------------------------------------------------------------ cells
    found, other = {}, []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            mm = RUN_RE8.match(p.name)
            if not (p.is_dir() and mm and mm["base"] == a.base):
                continue
            arm = next((k for k, v in RUNS.items() if v == mm["run"]), None)
            if arm is None:
                other.append(p.name)
                continue
            found[(arm, int(mm["step"]))] = p.name
    loaded = {k: A.load_config(root, v, sets) for k, v in found.items()}
    for (arm, s), name in found.items():
        cj = root / name / "config.json"
        m = json.loads(cj.read_text()).get("model") if cj.is_file() else None
        if m not in (None, MODEL):
            warn.append(f"{name}: config.json says model {m}, not {MODEL}")
    b_det = ho6_detail(base, sets) if base is not None else None
    b_train = A.cell_pass(base, sets, "design", [FAM])[0] if base is not None else None
    b_test = A.cell_pass(base, sets, "test", A.cell_members(sets, "test")) if base is not None else (None, 0)
    b_cells = R.cells_mean(base) if base is not None else None
    b_proc = R.process_on_eval(root, a.base) if (base is not None and not a.no_process) else None
    summary: dict = {"root": str(root), "base_name": a.base, "runs": RUNS, "warnings": warn,
                     "base": {"present": have_base, "train": b_train, "ho6": b_det, "test_all": b_test[0],
                              "test_all_n": b_test[1], "cells": b_cells, "process": b_proc,
                              "ext": {k: 100.0 * float(np.mean(list(v.values())))
                                      for k, v in (base or {}).get("ext", {}).items()}},
                     "cells": {}, "unregistered_dirs": other}
    if base is not None:
        L.append(f"Base: TRAIN ({FAM}) {fmt(b_train)}; HO6 {fmt(b_det['ho6'])} ({b_det['n']}/6"
                 f"{'; missing ' + ', '.join(b_det['missing']) if b_det['missing'] else ''}); TEST-all "
                 f"{fmt(b_test[0])} ({b_test[1]}/17); HO6 cells/rollout {fmt(b_cells, 2)}; process on HO6 "
                 f"{fmt(b_proc['p'], 3) if b_proc else '-'}; externals "
                 + (", ".join(f"{k} {v:.1f}" for k, v in summary["base"]["ext"].items()) or "-") + "\n")
    table = ["| arm | step | TRAIN (base) | HO6 [fams] | dHO6 vs base (p>0) | TEST-all (d) | cells/rollout (x base) | "
             "process p HO6 | externals (d, McNemar p) | HO6 missing |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, s) in sorted(loaded, key=lambda k: (ARMS.index(k[0]), k[1])):
        res = loaded[(arm, s)]
        det = ho6_detail(res, sets)
        train = A.cell_pass(res, sets, "design", [FAM])[0]
        d = pair(res, base, sets, a.perms, rng)
        ta = A.cell_pass(res, sets, "test", A.cell_members(sets, "test"))
        tad = pair(res, base, sets, a.perms, rng, members=A.cell_members(sets, "test"))
        cm = R.cells_mean(res)
        pe = None if a.no_process else R.process_on_eval(root, found[(arm, s)])
        ext = ext_rows(res, base)
        summary["cells"].setdefault(arm, {})[s] = {
            "dir": found[(arm, s)], "train": train, "ho6": det, "d_base": d, "test_all": ta[0], "test_all_n": ta[1],
            "test_all_d": tad and tad["d"], "cells": cm, "cells_x_base": (cm / b_cells) if (cm and b_cells) else None,
            "process": pe, "ext": ext}
        tag = "" if s in READ_STEPS else " (curve)"
        dstr = "-" if d is None else f"{d['d']:+.1f} (p={d['p_gt']:.3f})"
        table.append(
            f"| {arm} | {s}{tag} | {fmt(train)} ({fmt(b_train)}) | {fmt(det['ho6'])} [{det['n']}/6] | {dstr} | "
            f"{fmt(ta[0])} ({'-' if tad is None else fmt(tad['d'], sign=True)}; {ta[1]}/17) | "
            f"{fmt(cm, 2)} ({fmt(cm / b_cells, 2) + 'x' if (cm and b_cells) else '-'}) | "
            f"{fmt(pe['p'], 3) if pe else '-'} | "
            + ("; ".join(f"{e['bench']} {e['delta']:+.1f} (+{e['gained']}/-{e['lost']}, p={e['p']:.3f}"
                         f"{', COST' if e['cost'] else ''})" for e in ext) or "-")
            + f" | {', '.join(det['missing']) or '-'} |")
    L += table
    if other:
        L.append(f"\nNot in the registered reading (other names under this base): {', '.join(other)}")
    # ------------------------------------------------------------------ contrasts
    c = {"dPLAIN": {}, "dSHAM": {}, "dSP": {}, "FADE": {}}
    detail = {"dPLAIN": {}, "dSHAM": {}, "dSP": {}, "FADE": {}}
    prov = {"dPLAIN": {}, "dSHAM": {}, "dSP": {}, "FADE": {}}
    for s in READ_STEPS:
        for key, (x, y) in {"dPLAIN": ("pr", "plain"), "dSHAM": ("pr", "sham"), "dSP": ("sham", "plain")}.items():
            r = pair(loaded.get((x, s)), loaded.get((y, s)), sets, a.perms, rng)
            detail[key][s] = r
            if r is not None:
                prov[key][s] = r["d"]
                if r["complete"]:
                    c[key][s] = r["d"]
    for arm in ARMS:
        r = pair(loaded.get((arm, 150)), loaded.get((arm, 100)), sets, a.perms, rng)
        detail["FADE"][arm] = r
        if r is not None:
            prov["FADE"][arm] = r["d"]
            if r["complete"]:
                c["FADE"][arm] = r["d"]
    L.append("\n## Contrasts (paired by instance; HO6)\n")
    L.append("| step | dPLAIN = PR - PLAIN (p>0) | dSHAM = PR - SHAM (p>0) | SHAM - PLAIN (p>0) |")
    L.append("|---|---|---|---|")

    def cellstr(r):
        if r is None:
            return "-"
        s = f"{r['d']:+.1f} (p={r['p_gt']:.3f})"
        return s if r["complete"] else s + f" PARTIAL {r['n_fams']}/6, missing {', '.join(r['missing'])}"
    for s in READ_STEPS:
        L.append(f"| {s}{'' if s in DECIDE_STEPS else ' (reported)'} | {cellstr(detail['dPLAIN'][s])} | "
                 f"{cellstr(detail['dSHAM'][s])} | {cellstr(detail['dSP'][s])} |")
    L.append("\nFADE = HO6(150) - HO6(100), paired: " + "; ".join(
        f"{arm} {cellstr(detail['FADE'][arm]) if detail['FADE'][arm] else '-'}" for arm in ARMS))
    verdict, labels, why = decide(c)
    summary["contrasts"] = detail
    summary["verdict"] = verdict
    summary["labels"] = labels
    summary["why"] = why
    L.append(f"\n## Verdict: **{verdict}**")
    L += [f"* {w}" for w in why]
    if labels:
        L.append("* labels that fire: " + ", ".join(labels))
    if verdict == "INCOMPLETE":
        pv, pl, pw = decide(prov)
        summary["provisional"] = {"verdict": pv, "labels": pl, "why": pw}
        if pv != "INCOMPLETE":
            L.append(f"* PROVISIONAL (partial rows, on the HO6 families both cells have -- NOT the registered verdict): "
                     f"{pv}; {'; '.join(pw)}")
    # ------------------------------------------------------------------ beside the verdict
    beside = []
    for arm in ARMS:
        e150 = summary["cells"].get(arm, {}).get(150, {}).get("ext") or []
        costs = [e for e in e150 if e["cost"]]
        if costs:
            beside.append(f"COST {arm}@150: " + ", ".join(f"{e['bench']} {e['delta']:+.1f} (p={e['p']:.3g})"
                                                            for e in costs))
        if b_train is not None:
            tr = {s: summary["cells"].get(arm, {}).get(s, {}).get("train") for s in READ_STEPS}
            if all(tr[s] is not None for s in READ_STEPS) and \
                    not any(tr[s] >= min(b_train + VALID_GAIN, VALID_CAP) for s in READ_STEPS):
                beside.append(f"NO-FOOTHOLD {arm}: TRAIN {', '.join(fmt(tr[s]) for s in READ_STEPS)} never reaches "
                              f"{min(b_train + VALID_GAIN, VALID_CAP):.1f}: its contrasts are not a transfer test")
        for s in READ_STEPS:
            cell = summary["cells"].get(arm, {}).get(s)
            if cell and cell["cells"] is not None and b_cells and cell["cells"] < CELLS_FLOOR * b_cells:
                beside.append(f"DO-LESS {arm}@{s} ({cell['cells']:.2f} < {CELLS_FLOOR} x {b_cells:.2f})")
    r0 = r0_reference(Path(a.r0_root))
    summary["r0"] = r0
    harness = {}
    for s in READ_STEPS:
        cell = summary["cells"].get("plain", {}).get(s)
        rg = r0["range"].get(s)
        if cell and cell["ho6"]["ho6"] is not None and rg:
            v = cell["ho6"]["ho6"]
            where = "ABOVE" if v > rg[1] else "BELOW" if v < rg[0] else "WITHIN"
            harness[s] = where
            beside.append(f"HARNESS-EFFECT s{s}: PLAIN-v3 HO6 {v:.1f}{'' if cell['ho6']['complete'] else ' (partial)'} "
                          f"is {where} R0's seed range [{rg[0]:.1f}, {rg[1]:.1f}] ({rg[2]} seeds)")
    summary["harness_effect"] = harness
    summary["beside"] = beside
    L.append("\n## Beside the verdict (never deciding)")
    L += [f"* {x}" for x in beside] or ["* (none)"]
    # ------------------------------------------------------------------ training curves
    L.append("\n## Training curves (process_history.jsonl / history.jsonl)")
    summary["curves"] = {}
    for arm in ARMS:
        cv = training_curve(ck_root / RUNS[arm])
        summary["curves"][arm] = cv
        L.append(f"* {arm}: " + (json.dumps(cv) if cv else f"no process log under {ck_root / RUNS[arm]} yet"))
    # ------------------------------------------------------------------ R0 reference
    L.append("\n## R0 reference (plain harness, 24x4; reported only, never a contrast)")
    be = r0["base_echo"]
    L.append(f"* base_echo HO6 {fmt(be['ho6'])}" + (f" (missing {', '.join(be['missing'])})" if be["missing"] else ""))
    for run in R0_RUNS:
        L.append(f"* {run}: " + ", ".join(
            f"s{s} {fmt(r0['seeds'][run][s]['ho6'])}"
            + (f" (missing {', '.join(r0['seeds'][run][s]['missing'])})" if r0['seeds'][run][s]['missing'] else "")
            for s in READ_STEPS))
    L.append("* R0 range: " + ", ".join(f"s{s} [{rg[0]:.1f}, {rg[1]:.1f}]" if rg else f"s{s} -"
                                          for s, rg in r0["range"].items()))
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out) if a.out else (root if root.is_dir() else None)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "V3Q8_READING.md").write_text(text)
        (out / "V3Q8_READING.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
