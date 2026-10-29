#!/usr/bin/env python
"""BR -- BRANCH-AT-FOOTHOLD reading (docs/RL_PLAN.md, entry 2026-09-28 05:20 UTC 'SECOND OPINION ... ADOPTED': design and
reading registered there before any build or run; operationalised in the builder's entry 'BR + GR: BUILT, TESTED, QUEUED').

Three Qwen3-8B arms continue PR seed 1's step-25 adapter (rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s1/
step25: its TRAIN foothold) for 75 steps to step 100 -- same seed (21), fresh Adam, the s.16 recipe and harness otherwise
(items HV3_92/93/94) -- and each step-100 checkpoint is scored by screen_v3.sh exactly as PR s1 @100 was (HV3_95/96/97 ->
results/pods/harness_v3/V3__<run>_s100/):

  BR_PR     lambda 0.3              rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s21
  BR_PLAIN  lambda 0 (answer only)  rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0-lr1e-5-s21
  BR_SHAM   lambda 0.3, permuted    rl-czero-causal_rdd-base-echo-sysscience-rc-hv3prsham0.3-lr1e-5-s21

    venv/bin/python experiments/harness_v3/read_br.py [--root results/pods/harness_v3] [--base V3] [--step 100]
        [--runs pr=NAME,plain=NAME,sham=NAME] [--ckpt-root $DEEP_DATA_ROOT/ckpt] [--perms 20000] [--no-process]
        [--out results/pods/br]

REGISTERED (verbatim): "Read at continuation step 75 (= step 100 of s1), HO6 + TEST-all, paired sign-flip p. Validity:
every arm TRAIN >= 60 and cells/rollout <= 2x BR_PR's, else UNREADABLE for that arm.
    CONTENT        BR_PR >= BR_PLAIN + 5 AND >= BR_SHAM + 5 on HO6, both p <= 0.05
    FOOTHOLD-ONLY  |BR_PR - BR_PLAIN| <= 3 on HO6
    otherwise      INCONCLUSIVE"
OPERATIONALISATION (the builder's; no bar changed):
  TRAIN          the step-100 evaluation's causal_rdd DESIGN pass (16x2) -- the "TRAIN" column of V3Q8_READING.md;
  cells/rollout  mean executed cells over the HO6 evaluation rollouts (read_v3.cells_mean) -- the column the consultant
                 quoted (SHAM s1 @100 19.47 vs PR s1 6.36);
  contrasts      paired by instance over the six HO6 families (read_v3_q8.pair -> analyze.paired: the family-weighted
                 sign-flip permutation, p one-sided for Delta > 0), exactly as s.16's dPLAIN / dSHAM;
  unreadable     a label that needs an UNREADABLE arm cannot fire: BR_PR UNREADABLE -> the verdict is UNREADABLE (the
                 foothold did not hold, so nothing branches from it); CONTENT needs PR, PLAIN and SHAM readable;
                 FOOTHOLD-ONLY needs PR and PLAIN readable; otherwise INCONCLUSIVE, naming the unreadable arm(s).
                 (CONTENT and FOOTHOLD-ONLY exclude each other: +5 vs <= 3.)
  complete       the verdict needs all three step-100 cells with all six HO6 families and the causal_rdd DESIGN row;
                 until then INCOMPLETE, with a PROVISIONAL line on what is present.
BESIDE THE VERDICT (never deciding): TEST-all contrasts (paired, 17 families); each arm vs the untrained 8B (V3);
PR s1 @100 (the unbranched run: the same foothold, seed 1, its own Adam) as a reference cell; the process score on the HO6
evaluation rollouts (the mediator question the consultant raised: PR s1 and SHAM s1 did not differ on it); PROVENANCE of each
run dir (br_seed.json: source and sha256; first logged step == 26; no step below 25); training curves (steps 26-35 and
91-100: pass rate, cells/rollout, process p, groups alive).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_v3 as R  # noqa: E402  (its import puts ROOT, harness/ and experiments/harness_v2 on sys.path)
import read_v3_q8 as Q  # noqa: E402

A = R.A
HO6 = R.HO6
FAM = "causal_rdd"
SEED = 21
STEP = 100
FROM = 25
ARMS = ("pr", "plain", "sham")
RUNS = {"pr": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s21",
        "plain": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0-lr1e-5-s21",
        "sham": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3prsham0.3-lr1e-5-s21"}
SOURCE = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s1"      # the foothold's run (step25)
REFERENCE = {"pr_s1": (SOURCE, 100)}                                             # beside only
TRAIN_MIN, CELLS_MAX_X = 60.0, 2.0
BAR, P_BAR, SAME = 5.0, 0.05, 3.0
EPS = 1e-9


def validity(cells: dict) -> dict:
    """cells[arm] = {'train': float|None, 'cells': float|None} -> {arm: (True|False|None, why)}; None = not computable."""
    out = {}
    pr_cells = (cells.get("pr") or {}).get("cells")
    for arm in ARMS:
        c = cells.get(arm)
        if not c or c.get("train") is None or c.get("cells") is None or pr_cells is None:
            out[arm] = (None, "not computable yet (step-100 TRAIN row, HO6 cells or BR_PR's cells missing)")
            continue
        why = []
        if c["train"] < TRAIN_MIN - EPS:
            why.append(f"TRAIN {c['train']:.1f} < {TRAIN_MIN:.0f}")
        if c["cells"] > CELLS_MAX_X * pr_cells + EPS:
            why.append(f"cells/rollout {c['cells']:.2f} > {CELLS_MAX_X:.0f} x BR_PR's {pr_cells:.2f}")
        out[arm] = (not why, "; ".join(why) if why else
                    f"TRAIN {c['train']:.1f} >= {TRAIN_MIN:.0f}, cells {c['cells']:.2f} <= {CELLS_MAX_X * pr_cells:.2f}")
    return out


def decide(valid: dict, dplain: dict | None, dsham: dict | None) -> tuple[str, list, list]:
    """valid[arm] = (bool|None, why); dplain / dsham = read_v3_q8.pair(...) dicts ({'d', 'p_gt', 'complete', ...}) or None.
    -> (verdict, labels, why). The registered rule (module docstring) with the builder's operationalisation."""
    why = [f"validity {arm.upper()}: " + ("READABLE" if v[0] else "UNREADABLE" if v[0] is False else "PENDING")
           + f" ({v[1]})" for arm, v in valid.items()]
    miss = [arm for arm in ARMS if valid.get(arm, (None,))[0] is None]
    if miss or dplain is None or dsham is None or not dplain.get("complete") or not dsham.get("complete"):
        more = []
        if dplain is None or not dplain.get("complete"):
            more.append("PR - PLAIN on HO6 incomplete")
        if dsham is None or not dsham.get("complete"):
            more.append("PR - SHAM on HO6 incomplete")
        return "INCOMPLETE", [], why + [f"not yet computable: {', '.join(miss + more)}"]
    labels = [f"UNREADABLE-{arm.upper()}" for arm in ARMS if valid[arm][0] is False]
    why.append(f"HO6: PR - PLAIN {dplain['d']:+.1f} (p={dplain['p_gt']:.4f}); PR - SHAM {dsham['d']:+.1f} "
               f"(p={dsham['p_gt']:.4f})")
    if valid["pr"][0] is False:
        return "UNREADABLE", labels, why + ["BR_PR itself fails validity: nothing branches from the foothold"]
    ok_plain, ok_sham = valid["plain"][0], valid["sham"][0]
    content = (ok_plain and ok_sham and dplain["d"] >= BAR - EPS and dplain["p_gt"] <= P_BAR + EPS
               and dsham["d"] >= BAR - EPS and dsham["p_gt"] <= P_BAR + EPS)
    foothold = ok_plain and abs(dplain["d"]) <= SAME + EPS
    if content:
        return "CONTENT", labels + ["CONTENT"], why + [f"PR >= PLAIN + {BAR:.0f} and >= SHAM + {BAR:.0f} on HO6, both "
                                                         f"p <= {P_BAR}: from the SAME foothold, what the process reward "
                                                         "pays for adds held-out transfer"]
    if foothold:
        return "FOOTHOLD-ONLY", labels + ["FOOTHOLD-ONLY"], why + [f"|PR - PLAIN| <= {SAME:.0f} on HO6: once the foothold "
                                                                    "exists, answer-only RL transfers as much -- the "
                                                                    "process term's role was the foothold"]
    blocked = []
    if not ok_plain:
        blocked.append("BR_PLAIN is UNREADABLE: neither CONTENT nor FOOTHOLD-ONLY can fire")
    elif not ok_sham and dplain["d"] >= BAR - EPS:
        blocked.append("BR_SHAM is UNREADABLE: CONTENT cannot fire")
    return "INCONCLUSIVE", labels, why + (blocked or ["between the bars"])


def arm_cell(res, base, sets, perms, rng) -> dict:
    det = Q.ho6_detail(res, sets)
    ta = A.cell_pass(res, sets, "test", A.cell_members(sets, "test"))
    return {"train": A.cell_pass(res, sets, "design", [FAM])[0], "ho6": det, "test_all": ta[0], "test_all_n": ta[1],
            "cells": R.cells_mean(res),
            "vs_base": Q.pair(res, base, sets, perms, rng) if base is not None else None,
            "test_all_vs_base": (Q.pair(res, base, sets, perms, rng, members=A.cell_members(sets, "test"))
                                 if base is not None else None)}


def provenance(ck: Path) -> dict:
    """What the run dir says about its start: br_seed.json, the steps present, the first logged step."""
    out = {"dir": str(ck), "exists": ck.is_dir()}
    if not ck.is_dir():
        return out
    sj = ck / "br_seed.json"
    out["br_seed"] = json.loads(sj.read_text()) if sj.is_file() else None
    steps = sorted(int(p.name[4:]) for p in ck.glob("step*") if p.name[4:].isdigit())
    out["steps"] = steps
    hist = ck / "history.jsonl"
    rows = [json.loads(x) for x in hist.read_text().splitlines() if x.strip()] if hist.is_file() else []
    out["first_logged_step"] = rows[0].get("step") if rows else None
    out["last_logged_step"] = rows[-1].get("step") if rows else None
    out["resume"] = (ck / "resume").is_dir()
    problems = []
    if out["br_seed"] is None:
        problems.append("no br_seed.json")
    elif out["br_seed"].get("from_step") != FROM or not str(out["br_seed"].get("source", "")).endswith(f"{SOURCE}/step{FROM}"):
        problems.append(f"br_seed.json names another source/step: {out['br_seed'].get('source')} {out['br_seed'].get('from_step')}")
    if steps and min(steps) != FROM:
        problems.append(f"lowest stepN is {min(steps)}, not {FROM}")
    if out["first_logged_step"] is not None and out["first_logged_step"] != FROM + 1:
        problems.append(f"history starts at step {out['first_logged_step']}, not {FROM + 1}")
    out["problems"] = problems
    return out


def curve(ck: Path) -> dict | None:
    f, h = ck / "process_history.jsonl", ck / "history.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []
    hist = [json.loads(x) for x in h.read_text().splitlines() if x.strip()] if h.is_file() else []
    if not rows and not hist:
        return None

    def mean(src, key, lo, hi):
        xs = [r[key] for r in src if lo <= r.get("step", -1) <= hi and r.get(key) is not None]
        return statistics.fmean(xs) if xs else None
    lo = (FROM + 1, FROM + 10)
    return {"last_step": (hist or rows)[-1].get("step"),
            "pass_26_35": mean(hist, "pass_rate", *lo), "pass_91_100": mean(hist, "pass_rate", 91, 100),
            "cells_26_35": mean(rows, "cells_per_rollout", *lo), "cells_91_100": mean(rows, "cells_per_rollout", 91, 100),
            "p_26_35": mean(rows, "mean_p", *lo), "p_91_100": mean(rows, "mean_p", 91, 100),
            "alive_answer_76_100": mean(rows, "groups_alive_answer", 76, 100),
            "alive_total_76_100": mean(rows, "groups_alive_total", 76, 100)}


def parse_runs(s: str | None) -> dict:
    runs = dict(RUNS)
    if s:
        for kv in s.split(","):
            k, _, v = kv.partition("=")
            if k not in ARMS or not v:
                raise SystemExit(f"--runs: expected pr=NAME,plain=NAME,sham=NAME, got {kv!r}")
            runs[k] = v
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--runs", default=None, help="override run names (tests): pr=NAME,plain=NAME,sham=NAME")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-process", action="store_true", help="skip the process-signal readout on HO6 eval rollouts")
    ap.add_argument("--out", default=str(ROOT / "results" / "pods" / "br"),
                    help="where BR_READING.md/.json go (a NEW directory; never an existing results dir's files)")
    a = ap.parse_args(argv)
    root, runs = Path(a.root), parse_runs(a.runs)
    if a.ckpt_root:
        ck_root = Path(a.ckpt_root)
    else:
        from paths import data_root
        ck_root = data_root() / "ckpt"
    sets = A.load_families()
    rng = np.random.default_rng(a.seed)
    base = A.load_config(root, a.base, sets) if (root / a.base).is_dir() else None
    names = {arm: f"{a.base}__{runs[arm]}_s{a.step}" for arm in ARMS}
    loaded = {arm: A.load_config(root, names[arm], sets) for arm in ARMS if (root / names[arm]).is_dir()}
    warn = []
    for arm, name in names.items():
        cj = root / name / "config.json"
        m = json.loads(cj.read_text()).get("model") if cj.is_file() else None
        if m not in (None, Q.MODEL):
            warn.append(f"{name}: config.json says model {m}, not {Q.MODEL}")
    cells = {arm: arm_cell(res, base, sets, a.perms, rng) for arm, res in loaded.items()}
    if not a.no_process:
        for arm in loaded:
            cells[arm]["process"] = R.process_on_eval(root, names[arm])
    valid = validity(cells)
    dplain = Q.pair(loaded.get("pr"), loaded.get("plain"), sets, a.perms, rng)
    dsham = Q.pair(loaded.get("pr"), loaded.get("sham"), sets, a.perms, rng)
    dsp = Q.pair(loaded.get("sham"), loaded.get("plain"), sets, a.perms, rng)
    tmem = A.cell_members(sets, "test")
    t_plain = Q.pair(loaded.get("pr"), loaded.get("plain"), sets, a.perms, rng, members=tmem)
    t_sham = Q.pair(loaded.get("pr"), loaded.get("sham"), sets, a.perms, rng, members=tmem)
    verdict, labels, why = decide(valid, dplain, dsham)
    L = ["# BR -- branch-at-foothold reading (docs/RL_PLAN.md 2026-09-28 05:20 UTC)\n",
         f"root `{root}`; step {a.step} (= continuation step {a.step - FROM} from {SOURCE}/step{FROM}); paired by instance; "
         f"p one-sided sign-flip ({a.perms} perms). Rule and operationalisation: read_br.py docstring.\n"]
    b_train = A.cell_pass(base, sets, "design", [FAM])[0] if base is not None else None
    b_det = Q.ho6_detail(base, sets) if base is not None else None
    b_cells = R.cells_mean(base) if base is not None else None
    if base is not None:
        L.append(f"Untrained 8B (V3): TRAIN {Q.fmt(b_train)}; HO6 {Q.fmt(b_det['ho6'])} ({b_det['n']}/6); HO6 cells/rollout "
                 f"{Q.fmt(b_cells, 2)}\n")
    else:
        L.append(f"**no base {root}/{a.base}** -- the vs-base columns are empty\n")
    L += ["| arm | run | TRAIN (>= 60) | HO6 [fams] | vs base (p>0) | TEST-all (vs base) | HO6 cells/rollout (<= 2x PR) | "
          "process p HO6 | validity |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        c = cells.get(arm)
        v = valid[arm]
        if c is None:
            L.append(f"| BR_{arm.upper()} | `{runs[arm]}` | - | - | - | - | - | - | missing: {root / names[arm]} |")
            continue
        vb, tb = c["vs_base"], c["test_all_vs_base"]
        pe = c.get("process")
        vbs = "-" if vb is None else "%+.1f (p=%.3f)" % (vb["d"], vb["p_gt"])
        tbs = "-" if tb is None else "%+.1f" % tb["d"]
        vs = "READABLE" if v[0] else "UNREADABLE" if v[0] is False else "PENDING"
        L.append(f"| BR_{arm.upper()} | `{runs[arm]}` | {Q.fmt(c['train'])} | {Q.fmt(c['ho6']['ho6'])} [{c['ho6']['n']}/6] | "
                 f"{vbs} | {Q.fmt(c['test_all'])} ({tbs}; {c['test_all_n']}/17) | {Q.fmt(c['cells'], 2)} | "
                 f"{Q.fmt(pe['p'], 3) if pe else '-'} | {vs} |")

    def cs(r):
        if r is None:
            return "-"
        s = f"{r['d']:+.1f} (p={r['p_gt']:.4f})"
        return s if r["complete"] else s + f" PARTIAL {r['n_fams']} fams, missing {', '.join(r['missing'])}"
    L += ["\n## Contrasts at step %d (paired by instance)\n" % a.step,
          "| contrast | HO6 (deciding) | TEST-all (beside) |", "|---|---|---|",
          f"| BR_PR - BR_PLAIN | {cs(dplain)} | {cs(t_plain)} |",
          f"| BR_PR - BR_SHAM | {cs(dsham)} | {cs(t_sham)} |",
          f"| BR_SHAM - BR_PLAIN | {cs(dsp)} | - |"]
    L.append(f"\n## Verdict: **{verdict}**")
    L += [f"* {w}" for w in why]
    if labels:
        L.append("* labels: " + ", ".join(labels))
    summary = {"root": str(root), "step": a.step, "runs": runs, "cells": cells, "validity": valid,
               "contrasts": {"pr_plain": dplain, "pr_sham": dsham, "sham_plain": dsp,
                             "test_all": {"pr_plain": t_plain, "pr_sham": t_sham}},
               "verdict": verdict, "labels": labels, "why": why, "warnings": warn}
    if verdict == "INCOMPLETE" and dplain is not None:
        pv = {arm: (valid[arm][0] if valid[arm][0] is not None else True, valid[arm][1]) for arm in ARMS}
        pvd, pl, pw = decide(pv, {**dplain, "complete": True},
                             {**dsham, "complete": True} if dsham is not None else {"d": float("nan"), "p_gt": 1.0,
                                                                                     "complete": True})
        summary["provisional"] = {"verdict": pvd, "labels": pl}
        L.append(f"* PROVISIONAL (partial rows / pending validity counted as readable -- NOT the registered verdict): {pvd}")
    # ------------------------------------------------------------------ beside
    L.append("\n## Beside the verdict (never deciding)")
    ref = {}
    for key, (run, s) in REFERENCE.items():
        name = f"{a.base}__{run}_s{s}"
        if (root / name).is_dir():
            res = A.load_config(root, name, sets)
            ref[key] = arm_cell(res, base, sets, a.perms, rng)
            r = ref[key]
            L.append(f"* reference {run} @ {s} (unbranched: same foothold, seed 1, its own Adam): TRAIN {Q.fmt(r['train'])}, "
                     f"HO6 {Q.fmt(r['ho6']['ho6'])}, TEST-all {Q.fmt(r['test_all'])}, cells {Q.fmt(r['cells'], 2)}")
            if "pr" in loaded:
                x = Q.pair(loaded["pr"], res, sets, a.perms, rng)
                L.append(f"* BR_PR - PR s1 @ {s} on HO6 (two draws of the same continuation recipe; the seed/Adam effect): "
                         f"{cs(x)}")
                ref[key + "_vs_br_pr"] = x
    summary["reference"] = ref
    prov, curves = {}, {}
    for arm in ARMS:
        prov[arm] = provenance(ck_root / runs[arm])
        curves[arm] = curve(ck_root / runs[arm])
        p = prov[arm]
        if not p["exists"]:
            L.append(f"* provenance BR_{arm.upper()}: no run dir {p['dir']} yet")
        else:
            L.append(f"* provenance BR_{arm.upper()}: steps {p['steps']}, history {p['first_logged_step']}..{p['last_logged_step']}, "
                     f"seed sha {((p['br_seed'] or {}).get('adapter_sha256') or '-')[:16]}"
                     + (f" -- PROBLEMS: {'; '.join(p['problems'])}" if p["problems"] else " -- OK (branched at step 25)"))
        if curves[arm]:
            L.append(f"* training BR_{arm.upper()}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in curves[arm].items()})}")
    summary["provenance"], summary["curves"] = prov, curves
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BR_READING.md").write_text(text)
    (out / "BR_READING.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
