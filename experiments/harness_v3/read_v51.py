#!/usr/bin/env python
"""HARNESS v5.1 READINGS -- PHASE 1' (the untrained 8B, harness only) and PHASE 2' (RL inside v5.1 vs H-LRGR), as registered
(docs/RL_PLAN.md 2026-09-29 21:10 UTC, "HARNESS v5.1 REGISTERED"; the choices the registration leaves open are listed in the
"HARNESS v5.1 BUILT" entry and marked [OPEN] below).

    venv/bin/python experiments/harness_v3/read_v51.py [--phase 1|2|both] [--root results/pods/harness_v3] [--perms 20000]
        [--seed 0] [--out DIR] [--oracle [--oracle-k 8] [--workers 8]]

A NEW FILE: read_v5.py (the v5 Phase-1 reader), read_ef2.py / read_ef.py / analyze.py are imported for their helpers and never
edited.

PHASE 1' -- cells, paired by instance (TEST: the 17 sealed families, 12x2, TEST-all twin-weighted) and by problem (externals):
  v3    <root>/V3        (HV3_17)                 externals <root>/V3/ext
  v4    <root>/V3EFS     (EFS_1, the soft gate)   externals <root>/V3EFSX/ext  (EFS_2)
  v5.1  <root>/V3V51     (V51_1)                  externals <root>/V3V51X/ext  (V51_2)
  REGISTERED: "V51_1 (sealed TEST, 17 families) and V51_2 (externals) with the v5 Phase-1 bars unchanged: sealed TEST-all >=
  v4 + 2 (paired, p <= 0.05) AND no real-data benchmark down > 3 points vs v4." -> read_v5's rule with v5.1 in v5's place:
    V51-ADOPT       iff TEST-all(v5.1) >= TEST-all(v4) + 2 (one-sided sign-flip p <= 0.05) AND no external (QRData, DABench,
                    BixBench) down by more than 3 points vs v4
    V51-EQUIVALENT  iff not ADOPT but TEST-all(v5.1) >= TEST-all(v3) + 3 (p <= 0.05) and no external down > 3 vs v3
                    [OPEN: v5's second bar, kept because "the v5 Phase-1 bars unchanged"; the 21:10 text states only ADOPT's]
    V51-NULL        otherwise
  STATUS = the label once every cell is complete (every TEST family at 12x2 in v3 / v4 / v5.1; every problem of every benchmark
  with summary.json in the three external cells), else INCOMPLETE with the provisional label. BESIDE (never deciding): HO6; the
  per-family table; the v5 mechanics (read_v5.mechanics); the LOOK-FIRST mechanics (rollouts that looked, by detector and
  how, look-first refusals, rollouts refused and never looked, the first look's turn, inotify-overflow cells); the QRData
  causal split; provenance (every v5.1 cell records program_first_v51 with look_first on -- never program_first_v5 or a gate --
  over the plain v3 harness, untrained 8B); the oracle general rate of the final solve() programs (--oracle, read_v5.oracle).

PHASE 2' -- REGISTERED: "RL inside v5.1 on causal_rdd, mu 0.3, dynamic sampling (draw 12), LR 3e-6, seed 1, 100 steps
(HV3_142). Comparator: H-LRGR (HV3_140 ...). Bar: v5.1-RL >= HV3_140 + 3 on sealed TEST-all AND on HO6 at step 100 (paired
one-sided p <= 0.05); ALIVE reported beside. Also reported: v5.1-RL's labelled sibling pass rate over training ... and its
inspect rate."
    V51RL-WINS  iff TEST-all(v5.1-RL @100) - TEST-all(HV3_140 @100) >= +3 with one-sided sign-flip p <= 0.05 AND the same on
                HO6; else V51RL-NO. INCOMPLETE until both step-100 cells have every TEST family at 12x2.
  [OPEN] the harness of the registered read: the V3 harness -- <root>/V3__<RN_V51>_s100 (HV3_143) vs
  <root>/V3__<RN_LRGR>_s100 (HV3_140's own step-100 v3 eval, NOT YET BUILT as an item) -- the one harness both runs are
  scored in (as the v5 Phase-2 read was the v3 read); v5.1-RL scored IN v5.1 (<root>/V3V51__<RN_V51>_s100, HV3_144) against
  the same comparator is printed beside with the same statistics (--p2-harness v51 makes it the decisive one).
  BESIDE: ALIVE (the arm-B dynamic-sampling reading registered 2026-09-29 02:30 / 03:00: clause 1 mean
  groups_alive_drawn_scaled over steps 51-100 >= 3; clause 2 mean history.jsonl nll over steps 51-100 >= 0.05; clause 3 mean
  mean_r_answer_drawn over steps 91-100 >= 0.30) for both runs; over v5.1-RL's training, by step window: the LABELLED sibling
  pass rate (v51_labelled_mean_drawn: the signal (b) provides), the old v5 portability (v5_portability_mean_drawn), the
  INSPECT rate (rate_inspect on the kept batch, v51_inspect_rate_drawn on the drawn one), look-first refusals and the looked
  rate per drawn rollout, the answer pass (pass_rate_drawn); both step-100 cells vs the untrained base (<root>/V3).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_v5 as RV  # noqa: E402  (read_ef2 / read_ef / analyze helpers, the v5 mechanics and oracle; never edited)

R2, E, A = RV.R2, RV.E, RV.A
HO6, BENCHES, fmt = RV.HO6, RV.BENCHES, RV.fmt
ADOPT_BAR, ADOPT_P, EQ_BAR, EQ_P, DOWN_BAR = RV.ADOPT_BAR, RV.ADOPT_P, RV.EQ_BAR, RV.EQ_P, RV.DOWN_BAR
P2_BAR, P2_P = 3.0, 0.05
EPS = 1e-9
HARNESS = "program_first_v51"
GATES = RV.GATES
RN_V51 = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3v51lp0.3-ds12-lr3e-6-s1"     # HV3_142
RN_LRGR = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3gr-ds12-lr3e-6-s1"         # HV3_140 (H-LRGR)
STEP = 100
ALIVE_GROUPS, ALIVE_NLL, ALIVE_ANSWER = 3.0, 0.05, 0.30
WINDOWS = ((1, 10), (11, 25), (26, 50), (51, 75), (76, 100), (91, 100))


# ----------------------------------------------------------------------------------------------------------- phase 1'
def verdict1(t_v4, t_v3, x_v4: dict, x_v3: dict, complete: bool) -> dict:
    """read_v5.verdict with v5.1 in v5's place (same bars, same legs)."""
    legs = {}
    a_test = None if t_v4 is None else (t_v4["d"] >= ADOPT_BAR - EPS and t_v4["p_gt"] <= ADOPT_P)
    legs["ADOPT.TEST"] = {"ok": a_test, "why": "no paired rows" if t_v4 is None else
                          f"TEST-all v5.1 - v4 {t_v4['d']:+.1f} (sign-flip p>0 {t_v4['p_gt']:.4f}) "
                          + ("is" if a_test else "is not") + f" >= +{ADOPT_BAR:.0f} with p <= {ADOPT_P}"}
    for b in BENCHES:
        o = RV.ext_ok(x_v4.get(b))
        legs[f"ADOPT.{b}"] = {"ok": o, "why": "no paired rows" if o is None else
                              f"{b} v5.1 - v4 {x_v4[b]['d']:+.1f} (+{x_v4[b]['b01']}/-{x_v4[b]['b10']}) "
                              + ("is not" if o else "IS") + f" down by more than {DOWN_BAR:.0f}"}
    e_test = None if t_v3 is None else (t_v3["d"] >= EQ_BAR - EPS and t_v3["p_gt"] <= EQ_P)
    legs["EQUIV.TEST"] = {"ok": e_test, "why": "no paired rows" if t_v3 is None else
                          f"TEST-all v5.1 - v3 {t_v3['d']:+.1f} (sign-flip p>0 {t_v3['p_gt']:.4f}) "
                          + ("is" if e_test else "is not") + f" >= +{EQ_BAR:.0f} with p <= {EQ_P}"}
    for b in BENCHES:
        o = RV.ext_ok(x_v3.get(b))
        legs[f"EQUIV.{b}"] = {"ok": o, "why": "no paired rows" if o is None else
                              f"{b} v5.1 - v3 {x_v3[b]['d']:+.1f} (+{x_v3[b]['b01']}/-{x_v3[b]['b10']}) "
                              + ("is not" if o else "IS") + f" down by more than {DOWN_BAR:.0f}"}
    adopt = all(legs[k]["ok"] is True for k in legs if k.startswith("ADOPT."))
    equiv = all(legs[k]["ok"] is True for k in legs if k.startswith("EQUIV."))
    label = "V51-ADOPT" if adopt else ("V51-EQUIVALENT" if equiv else "V51-NULL")
    return {"status": label if complete else "INCOMPLETE", "label": label, "final": bool(complete), "legs": legs,
            "failed": [k for k, v in legs.items() if v["ok"] is False]}


def look_mech(rows: list) -> dict:
    """The look-first bookkeeping over one cell's rows (synthetic rollouts or external problem records)."""
    logs = [RV.v5_of(r) for r in rows]
    lf = [x for x in logs if x and x.get("look_first")]
    if not rows:
        return {"n": 0}
    looked = [x for x in lf if x.get("looked")]
    turns = [int(x["look_turn"]) for x in looked if x.get("look_turn") is not None]
    return {"n": len(rows), "with_look_log": len(lf), "looked": len(looked),
            "looked_rate": 100.0 * len(looked) / len(lf) if lf else None,
            "how": dict(Counter(str(x.get("look_how")) for x in looked)),
            "detector": dict(Counter(str(x.get("look_detector")) for x in lf)),
            "refusals_total": sum(int(x.get("look_refusals") or 0) for x in lf),
            "rows_refused": sum(1 for x in lf if int(x.get("look_refusals") or 0) > 0),
            "never_looked_refused": sum(1 for x in lf if not x.get("looked") and int(x.get("look_refusals") or 0) > 0),
            "look_turn_mean": float(np.mean(turns)) if turns else None,
            "overflow_cells": sum(int(x.get("look_overflow") or 0) for x in lf),
            "look_errors": sum(1 for x in lf if x.get("look_error") and x.get("look_detector") == "inotify")}


def look_line(tag: str, m: dict | None) -> str:
    if not m or not m.get("n"):
        return f"* {tag}: (no rows)"
    return (f"* {tag}: rows {m['n']} (look-first log {m['with_look_log']}); looked {m['looked']} ({fmt(m['looked_rate'])}%; "
            f"how {m['how']}; detector {m['detector']}); look-first refusals {m['refusals_total']} in {m['rows_refused']} rows, "
            f"{m['never_looked_refused']} of them never looked; first look at turn {fmt(m['look_turn_mean'], 2)} (mean); "
            f"inotify-overflow cells {m['overflow_cells']}")


def v51_provenance(root: Path, name: str, sets, plain_harness: dict | None, ext: bool, ckpt: bool = False) -> list:
    """Problems with a v5.1 cell's provenance (empty list = clean)."""
    c = R2.load_cfg(root / name)
    if c is None:
        return [f"{root / name}/config.json missing"]
    bad = []
    h = dict(c.get("harness") or {})
    for g in GATES:
        if g in h:
            bad.append(f"config.json records {g} (an execution gate) -- not a v5.1 directory")
            h.pop(g, None)
    if "program_first_v5" in h:
        bad.append("config.json records program_first_v5 (v5, not v5.1)")
        h.pop("program_first_v5", None)
    rec = h.pop(HARNESS, None) or {}
    if rec.get("harness") != HARNESS or not rec.get("on") or not (rec.get("look_first") or {}).get("on"):
        bad.append("config.json does not record program_first_v51 with look_first on -- NOT a v5.1 directory")
    elif rec.get("k_siblings") != 3 or rec.get("max_revisions") != 2 or rec.get("signature") != "solve(data_dir, task)":
        bad.append(f"program_first_v51 recorded with K={rec.get('k_siblings')} / revisions {rec.get('max_revisions')} / "
                   f"signature {rec.get('signature')!r}, not the registered 3 / 2 / solve(data_dir, task)")
    if not ckpt and (c.get("model", "Qwen3-8B") != "Qwen3-8B" or c.get("ckpt")):
        bad.append(f"config.json model={c.get('model')} ckpt={c.get('ckpt')!r} (Phase 1' is read on the untrained 8B)")
    want = dict(plain_harness) if plain_harness else dict(R2.V3_HARNESS, system="science")
    off = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    if off:
        bad.append(f"config.json harness differs from the plain v3 cell's: {off}")
    if not ext:
        for s in ("design", "test"):
            for f in sets[s]:
                p = root / name / s / f"{f}.json"
                if p.is_file():
                    hv = json.loads(p.read_text()).get("harness_v2") or {}
                    if (hv.get(HARNESS) or {}).get("harness") != HARNESS or any(g in hv for g in GATES) \
                            or "program_first_v5" in hv:
                        bad.append(f"{s}/{f}.json does not record program_first_v51 (only)")
    else:
        for b in BENCHES:
            hj = root / name / "ext" / b / "harness_v2.json"
            if hj.is_file():
                hv = json.loads(hj.read_text())
                if (hv.get(HARNESS) or {}).get("harness") != HARNESS or any(g in hv for g in GATES) or "program_first_v5" in hv:
                    bad.append(f"ext/{b}/harness_v2.json does not record program_first_v51 (only)")
    return bad


def phase1(a, root: Path, broot: Path, sets, test_all, load, out_dir) -> tuple:
    v3, v4, v51 = load(a.base), load(a.v4), load(a.v51)
    miss = {"v3": E.complete(v3, sets, test_all), "v4": E.complete(v4, sets, test_all), "v5.1": E.complete(v51, sets, test_all)}
    plain_h = (R2.load_cfg(root / a.base) or {}).get("harness")
    nprob = R2.problem_counts(broot)
    xdir = {"v3": root / a.base / "ext", "v4": root / a.v4x / "ext", "v5.1": root / a.v51x / "ext"}
    recs = {k: {b: R2.load_ext(d / b) for b in BENCHES} for k, d in xdir.items()}
    xcomp = {k: {b: (xdir[k] / b / "summary.json").is_file() and len(recs[k][b]) == nprob[b] for b in BENCHES} for k in xdir}
    passed = {k: {b: R2.ext_passed(recs[k][b]) for b in BENCHES} for k in xdir}

    def mc(x, y, b, ids=None):
        return R2.mcnemar(passed[x][b], passed[y][b], ids) if recs[x][b] and recs[y][b] else None
    x_v4 = {b: mc("v5.1", "v4", b) for b in BENCHES}
    x_v3 = {b: mc("v5.1", "v3", b) for b in BENCHES}
    rng = np.random.default_rng(a.seed)                               # the registered reading's own rng
    t_v4 = E.pair(v51, v4, sets, a.perms, rng, test_all)
    t_v3 = E.pair(v51, v3, sets, a.perms, rng, test_all)
    complete = (all(x is not None for x in (v3, v4, v51)) and not any(miss.values())
                and all(xcomp[k][b] for k in xdir for b in BENCHES))
    V = verdict1(t_v4, t_v3, x_v4, x_v3, complete)
    rng_b = np.random.default_rng(a.seed + 1)
    cp = {k: (A.cell_pass(v, sets, "test", test_all)[0] if v is not None else None) for k, v in (("v3", v3), ("v4", v4), ("v5.1", v51))}
    ho = {k: (A.cell_pass(v, sets, "test", HO6)[0] if v is not None else None) for k, v in (("v3", v3), ("v4", v4), ("v5.1", v51))}
    beside = {"test_all": {**cp, "v51_minus_v4": t_v4, "v51_minus_v3": t_v3},
              "ho6": {**ho, "v51_minus_v4": E.pair(v51, v4, sets, a.perms, rng_b, HO6),
                      "v51_minus_v3": E.pair(v51, v3, sets, a.perms, rng_b, HO6)}}
    cau = R2.qrdata_causal(broot)
    split = {lab: {"n_ids": len({k for k, v in cau.items() if v == want}),
                   "v51_minus_v4": mc("v5.1", "v4", "qrdata", {k for k, v in cau.items() if v == want}),
                   "v51_minus_v3": mc("v5.1", "v3", "qrdata", {k for k, v in cau.items() if v == want})}
             for lab, want in (("causal", True), ("non-causal", False))}
    rows_syn = RV.syn_rows(v51)
    mech = {"synthetic": RV.syn_mech(v51), "externals": {b: RV.ext_mech(recs["v5.1"][b]) for b in BENCHES}}
    look = {"synthetic": look_mech(rows_syn), "externals": {b: look_mech(list(recs["v5.1"][b].values())) for b in BENCHES}}
    per = {}
    for f in test_all:
        m = RV.syn_mech(v51, [f])
        lm = look_mech(RV.syn_rows(v51, [f]))
        per[f] = {"v3": E.fam(v3, sets, f), "v4": E.fam(v4, sets, f), "v5.1": E.fam(v51, sets, f),
                  "v51_minus_v4": E.pair(v51, v4, sets, a.perms, rng_b, [f]),
                  "valid_rate": (m or {}).get("valid_rate"), "looked_rate": lm.get("looked_rate"),
                  "param_family": f in RV.PARAM_FAMILIES}
    warn = []
    if v51 is not None:
        warn += [f"PROVENANCE {a.v51}: {x}" for x in v51_provenance(root, a.v51, sets, plain_h, ext=False)]
    if (root / a.v51x).is_dir():
        warn += [f"PROVENANCE {a.v51x}: {x}" for x in v51_provenance(root, a.v51x, sets, plain_h, ext=True)]
    for tag, m in [("TEST", look["synthetic"])] + [(b, look["externals"][b]) for b in BENCHES]:
        if m and m.get("n") and m["with_look_log"] < m["n"]:
            warn.append(f"{tag}: {m['n'] - m['with_look_log']} v5.1 rows carry no look-first log")
        if m and m.get("look_errors"):
            warn.append(f"{tag}: {m['look_errors']} rows had a look-detector error")
    orc = None
    if a.oracle:
        if v51 is None or out_dir is None:
            warn.append("--oracle: no v5.1 TEST cell -- nothing to re-run")
        else:
            gen = Path(a.gen_root) if a.gen_root else None
            if gen is None:
                sys.path.insert(0, str(ROOT))
                from paths import data_root
                gen = data_root() / "generated"
            orc = RV.oracle(v51, sets, out_dir, a.oracle_k, a.workers, gen)
    L = ["## PHASE 1' -- the untrained 8B in v5.1 vs v4 and v3 (registered 2026-09-29 21:10 UTC)\n",
         f"cells: v3 `{a.base}` (+ ext), v4 `{a.v4}` / `{a.v4x}/ext`, v5.1 `{a.v51}` / `{a.v51x}/ext`; paired by instance "
         f"(TEST, one-sided sign-flip, {a.perms} perms) and by problem (externals; exact McNemar beside).\n",
         f"**Phase 1': {V['status']}**" + ("" if V["final"] else f" (provisional on the rows present: {V['label']})") + "\n"]
    for n, what in ((a.v51, "V51_1 (TEST)"), (a.v51x, "V51_2 (externals)")):
        if not (root / n).is_dir():
            L.append(f"* {root / n} does not exist yet ({what}) -- PENDING.")
    for k, v in miss.items():
        if v and not (k == "v5.1" and v51 is None):
            L.append(f"* TEST {k}: {len(v)} families missing or short: {', '.join(v)}")
    for k in xdir:
        part = [f"{b} {len(recs[k][b])}/{nprob[b]}" for b in BENCHES if not xcomp[k][b]]
        if part:
            L.append(f"* externals {k}: INCOMPLETE -- " + ", ".join(part))
    L += ["\n| leg | holds | detail |", "|---|---|---|"]
    for k, lg in V["legs"].items():
        L.append(f"| {k} | {({True: 'yes', False: '**NO**', None: '-'})[lg['ok']]} | {lg['why']} |")
    L.append("\n[OPEN] EQUIV.* is v5's second bar, kept because the registration says the v5 Phase-1 bars are unchanged.")
    L += ["\n### Beside: TEST-all / HO6 (never deciding)\n", "| set | v3 | v4 | v5.1 | v5.1 - v4 | v5.1 - v3 |", "|---|---|---|---|---|---|"]
    for lab, key in (("TEST-all (twin-weighted)", "test_all"), ("HO6", "ho6")):
        b = beside[key]
        L.append(f"| {lab} | {fmt(b['v3'])} | {fmt(b['v4'])} | {fmt(b['v5.1'])} | {RV.ptxt(b['v51_minus_v4'])} | "
                 f"{RV.ptxt(b['v51_minus_v3'])} |")
    L += ["\n### Externals\n", "| benchmark | v3 (n) | v4 (n) | v5.1 (n) | v5.1 - v4 | v5.1 - v3 |", "|---|---|---|---|---|---|"]
    for b in BENCHES:
        L.append(f"| {b} | " + " | ".join(RV.rate_txt(passed[k][b]) for k in ("v3", "v4", "v5.1"))
                 + f" | {RV.mtxt(x_v4[b])} | {RV.mtxt(x_v3[b])} |")
    for lab, s in split.items():
        L.append(f"* QRData {lab} ({s['n_ids']}): v5.1 - v4 {RV.mtxt(s['v51_minus_v4'])}; v5.1 - v3 {RV.mtxt(s['v51_minus_v3'])}")
    L.append("\n### Mechanics (v5.1 cells): the program-first protocol and LOOK FIRST\n")
    L.append(RV.mech_line("TEST", mech["synthetic"]))
    L.append(look_line("TEST look-first", look["synthetic"]))
    for b in BENCHES:
        L.append(RV.mech_line(b, mech["externals"][b]))
        L.append(look_line(f"{b} look-first", look["externals"][b]))
    L += ["\n### Per TEST family (pass %, 12x2)\n",
          "| family | v3 | v4 | v5.1 | v5.1 - v4 (p>0) | looked | valid solve |", "|---|---|---|---|---|---|---|"]
    for f in test_all:
        x = per[f]
        L.append(f"| {f}{' (HO6)' if f in HO6 else ''}{' [task-param]' if x['param_family'] else ''} | {fmt(x['v3'])} | "
                 f"{fmt(x['v4'])} | {fmt(x['v5.1'])} | "
                 + (f"{x['v51_minus_v4']['d']:+.1f} ({x['v51_minus_v4']['p_gt']:.3f})" if x["v51_minus_v4"] else "-") + " | "
                 + (f"{x['looked_rate']:.1f}%" if x["looked_rate"] is not None else "-") + " | "
                 + (f"{x['valid_rate']:.1f}%" if x["valid_rate"] is not None else "-") + " |")
    L.append("\n### Oracle general rate of the final solve() programs (beside)\n")
    L.append("* not computed (--oracle on a CPU allocation)." if orc is None else
             f"* {orc['general']} of {orc['programs']} final programs are GENERAL: {fmt(orc['rate'])}%.")
    obj = {"status": V["status"], "label": V["label"], "final": V["final"], "legs": V["legs"], "failed": V["failed"],
           "values": {"test_v51_minus_v4": t_v4, "test_v51_minus_v3": t_v3, "ext_v51_minus_v4": x_v4, "ext_v51_minus_v3": x_v3},
           "complete": complete, "missing": miss, "ext_completeness": xcomp,
           "ext_n": {k: {b: len(recs[k][b]) for b in BENCHES} for k in xdir}, "beside": beside, "qrdata_split": split,
           "mechanics": mech, "look": look, "per_family": per, "oracle": orc, "warnings": warn}
    return L, obj


# ----------------------------------------------------------------------------------------------------------- phase 2'
def read_jsonl_steps(p: Path) -> dict:
    """{step: the LAST line logged for that step} (a resumed run may log a step twice)."""
    out = {}
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and isinstance(x.get("step"), int):
            out[x["step"]] = x
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def alive(run_dir: Path) -> dict:
    """The arm-B (dynamic sampling) ALIVE reading, registered 2026-09-29 02:30 / 03:00 UTC."""
    ph, h = read_jsonl_steps(run_dir / "process_history.jsonl"), read_jsonl_steps(run_dir / "history.jsonl")
    c1 = _mean(ph[s].get("groups_alive_drawn_scaled") for s in range(51, 101) if s in ph)
    c2 = _mean(h[s].get("nll") for s in range(51, 101) if s in h)
    c3 = _mean(ph[s].get("mean_r_answer_drawn") for s in range(91, 101) if s in ph)
    done = all(s in ph for s in range(1, STEP + 1)) and all(s in h for s in range(1, STEP + 1))
    legs = {"groups": (c1, None if c1 is None else c1 >= ALIVE_GROUPS - EPS),
            "nll": (c2, None if c2 is None else c2 >= ALIVE_NLL - EPS),
            "answer": (c3, None if c3 is None else c3 >= ALIVE_ANSWER - EPS)}
    ok = all(v[1] is True for v in legs.values())
    return {"dir": str(run_dir), "steps_logged": len(ph), "last_step": max(ph) if ph else None, "complete": done,
            "legs": {k: {"value": v[0], "ok": v[1]} for k, v in legs.items()},
            "status": ("ALIVE" if ok else "NOT ALIVE") if done else "INCOMPLETE"}


def training_signals(run_dir: Path) -> dict:
    """v5.1-RL's training signals by step window (drawn batch unless noted)."""
    ph = read_jsonl_steps(run_dir / "process_history.jsonl")
    out = {}
    for lo, hi in WINDOWS:
        xs = [ph[s] for s in range(lo, hi + 1) if s in ph]

        def per_rollout(key):
            num = sum(float(x.get(key) or 0) for x in xs)
            den = sum(float(x.get("v5_rollouts_drawn") or 0) for x in xs)
            return num / den if den else None
        out[f"{lo}-{hi}"] = {"steps": len(xs),
                             "labelled_mean_drawn": _mean(x.get("v51_labelled_mean_drawn") for x in xs),
                             "portability_old_drawn": _mean(x.get("v5_portability_mean_drawn") for x in xs),
                             "rate_inspect_kept": _mean(x.get("rate_inspect") for x in xs),
                             "inspect_rate_drawn": _mean(x.get("v51_inspect_rate_drawn") for x in xs),
                             "look_refusals_per_rollout": per_rollout("v51_look_refusals_drawn"),
                             "looked_rate": per_rollout("v51_looked_drawn"),
                             "pass_rate_drawn": _mean(x.get("pass_rate_drawn") for x in xs),
                             "mean_r_answer_drawn": _mean(x.get("mean_r_answer_drawn") for x in xs)}
    return out


def verdict2(t: dict | None, ho: dict | None, complete: bool) -> dict:
    legs = {}
    for k, x in (("TEST-all", t), ("HO6", ho)):
        ok = None if x is None else (x["d"] >= P2_BAR - EPS and x["p_gt"] <= P2_P)
        legs[k] = {"ok": ok, "why": "no paired rows" if x is None else
                   f"{k} v5.1-RL - HV3_140 {x['d']:+.1f} (sign-flip p>0 {x['p_gt']:.4f}) " + ("is" if ok else "is not")
                   + f" >= +{P2_BAR:.0f} with p <= {P2_P}"}
    label = "V51RL-WINS" if all(v["ok"] is True for v in legs.values()) else "V51RL-NO"
    return {"status": label if complete else "INCOMPLETE", "label": label, "final": bool(complete), "legs": legs}


def phase2(a, root: Path, sets, test_all, load, ckpt_root: Path) -> tuple:
    cells = {"v51rl_v3": f"V3__{a.rn_v51}_s{STEP}", "lrgr_v3": f"V3__{a.rn_lrgr}_s{STEP}",
             "v51rl_v51": f"V3V51__{a.rn_v51}_s{STEP}", "base": a.base}
    res = {k: load(n) for k, n in cells.items()}
    dec, dec_cmp = ("v51rl_v51", "lrgr_v3") if a.p2_harness == "v51" else ("v51rl_v3", "lrgr_v3")
    bes = "v51rl_v3" if a.p2_harness == "v51" else "v51rl_v51"
    miss = {k: E.complete(res[k], sets, test_all) for k in cells}
    complete = res[dec] is not None and res[dec_cmp] is not None and not miss[dec] and not miss[dec_cmp]
    rng = np.random.default_rng(a.seed)
    t = E.pair(res[dec], res[dec_cmp], sets, a.perms, rng, test_all)
    ho = E.pair(res[dec], res[dec_cmp], sets, a.perms, rng, HO6)
    V = verdict2(t, ho, complete)
    rng_b = np.random.default_rng(a.seed + 1)
    b_t = E.pair(res[bes], res[dec_cmp], sets, a.perms, rng_b, test_all)
    b_ho = E.pair(res[bes], res[dec_cmp], sets, a.perms, rng_b, HO6)
    Vb = verdict2(b_t, b_ho, res[bes] is not None and not miss[bes] and res[dec_cmp] is not None and not miss[dec_cmp])
    vs_base = {k: {"test_all": E.pair(res[k], res["base"], sets, a.perms, rng_b, test_all),
                   "ho6": E.pair(res[k], res["base"], sets, a.perms, rng_b, HO6)} for k in ("v51rl_v3", "v51rl_v51", "lrgr_v3")}
    score = {k: {"test_all": (A.cell_pass(v, sets, "test", test_all)[0] if v is not None else None),
                 "ho6": (A.cell_pass(v, sets, "test", HO6)[0] if v is not None else None)} for k, v in res.items()}
    al = {"v51rl": alive(ckpt_root / a.rn_v51), "lrgr": alive(ckpt_root / a.rn_lrgr)}
    sig = training_signals(ckpt_root / a.rn_v51)
    warn = []
    if res["v51rl_v51"] is not None:
        warn += [f"PROVENANCE {cells['v51rl_v51']}: {x}" for x in
                 v51_provenance(root, cells["v51rl_v51"], sets, (R2.load_cfg(root / a.base) or {}).get("harness"), ext=False,
                                ckpt=True)]
    hj = ckpt_root / a.rn_v51 / "harness_v51.json"
    if (ckpt_root / a.rn_v51).is_dir() and not hj.is_file():
        warn.append(f"{ckpt_root / a.rn_v51} has no harness_v51.json -- not a v5.1 run")
    L = [f"## PHASE 2' -- RL inside v5.1 (HV3_142, `{a.rn_v51}`) vs H-LRGR (HV3_140, `{a.rn_lrgr}`) at step {STEP}\n",
         f"Decisive read: the {'v5.1' if a.p2_harness == 'v51' else 'v3'} harness -- `{cells[dec]}` vs `{cells[dec_cmp]}` "
         f"([OPEN] the registration names no harness; default v3, the one harness both runs are scored in).\n",
         f"**Phase 2': {V['status']}**" + ("" if V["final"] else f" (provisional on the rows present: {V['label']})") + "\n"]
    for k, n in cells.items():
        if res[k] is None:
            L.append(f"* `{n}` does not exist yet -- PENDING" + (" (HV3_140's step-100 v3 eval: NO ITEM BUILT YET)"
                                                              if k == "lrgr_v3" else ""))
        elif miss[k]:
            L.append(f"* `{n}`: {len(miss[k])} TEST families missing or short: {', '.join(miss[k])}")
    L += ["\n| leg | holds | detail |", "|---|---|---|"]
    for k, lg in V["legs"].items():
        L.append(f"| {k} | {({True: 'yes', False: '**NO**', None: '-'})[lg['ok']]} | {lg['why']} |")
    L.append(f"\nBeside (the other harness): v5.1-RL `{cells[bes]}` vs `{cells[dec_cmp]}`: {Vb['status']}"
             + ("" if Vb["final"] else f" (provisional {Vb['label']})") + "; "
             + "; ".join(f"{k} {lg['why']}" for k, lg in Vb["legs"].items()))
    L += ["\n### Scores (pass %) and vs the untrained base\n", "| cell | TEST-all | HO6 | TEST-all - base | HO6 - base |",
          "|---|---|---|---|---|"]
    for k in ("v51rl_v3", "v51rl_v51", "lrgr_v3", "base"):
        vb = vs_base.get(k) or {}
        L.append(f"| `{cells[k]}` | {fmt(score[k]['test_all'])} | {fmt(score[k]['ho6'])} | {RV.ptxt(vb.get('test_all'))} | "
                 f"{RV.ptxt(vb.get('ho6'))} |")
    L += ["\n### ALIVE (beside; arm-B reading: groups_alive_drawn_scaled 51-100 >= 3, nll 51-100 >= 0.05, "
          "mean_r_answer_drawn 91-100 >= 0.30)\n", "| run | status | groups | nll | answer | steps logged |", "|---|---|---|---|---|---|"]
    for k, x in al.items():
        lg = x["legs"]
        L.append(f"| {'v5.1-RL (HV3_142)' if k == 'v51rl' else 'H-LRGR (HV3_140)'} | {x['status']} | "
                 + " | ".join(f"{fmt(lg[c]['value'], 3)}{'' if lg[c]['ok'] is None else (' ok' if lg[c]['ok'] else ' FAIL')}"
                              for c in ("groups", "nll", "answer")) + f" | {x['steps_logged']} |")
    L += ["\n### v5.1-RL training signals by step window (drawn batch unless noted)\n",
          "| steps | n | LABELLED sibling pass | old portability | inspect (kept) | inspect (drawn) | look refusals / rollout "
          "| looked | answer pass |", "|---|---|---|---|---|---|---|---|---|"]
    for w, x in sig.items():
        L.append(f"| {w} | {x['steps']} | {fmt(x['labelled_mean_drawn'], 3)} | {fmt(x['portability_old_drawn'], 3)} | "
                 f"{fmt(x['rate_inspect_kept'], 3)} | {fmt(x['inspect_rate_drawn'], 3)} | {fmt(x['look_refusals_per_rollout'], 3)} | "
                 f"{fmt(x['looked_rate'], 3)} | {fmt(x['pass_rate_drawn'], 3)} |")
    obj = {"status": V["status"], "label": V["label"], "final": V["final"], "legs": V["legs"], "decisive_harness": a.p2_harness,
           "cells": cells, "missing": miss, "values": {"test_all": t, "ho6": ho}, "beside_other_harness": Vb,
           "beside_values": {"test_all": b_t, "ho6": b_ho}, "scores": score, "vs_base": vs_base, "alive": al,
           "training_signals": sig, "warnings": warn}
    return L, obj


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("1", "2", "both"), default="both")
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--bench-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--ckpt-root", default=None, help="the RL runs' checkpoint root (default: paths.data_root()/ckpt)")
    ap.add_argument("--base", default="V3")
    ap.add_argument("--v4", default="V3EFS")
    ap.add_argument("--v4x", default="V3EFSX")
    ap.add_argument("--v51", default="V3V51")
    ap.add_argument("--v51x", default="V3V51X")
    ap.add_argument("--rn-v51", default=RN_V51)
    ap.add_argument("--rn-lrgr", default=RN_LRGR)
    ap.add_argument("--p2-harness", choices=("v3", "v51"), default="v3")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where V51_READING.md/.json go (default: <root>/V3V51, else <root>/V3V51X, "
                                                "else <root>/V3__<RN_V51>_s100, if one exists; else nothing is written)")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--oracle-k", type=int, default=RV.ORACLE_K)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-root", default=None)
    a = ap.parse_args(argv)
    root, broot = Path(a.root), Path(a.bench_root)
    if a.ckpt_root:
        ckpt_root = Path(a.ckpt_root)
    else:
        sys.path.insert(0, str(ROOT))
        from paths import data_root
        ckpt_root = data_root() / "ckpt"
    sets = A.load_families()
    test_all = A.cell_members(sets, "test")

    def load(name):
        if not (root / name).is_dir():
            return None
        r = A.load_config(root, name, sets)
        r["_dir"] = root / name
        return r
    out = Path(a.out) if a.out else next((root / n for n in (a.v51, a.v51x, f"V3__{a.rn_v51}_s{STEP}") if (root / n).is_dir()),
                                         None)
    L = ["# HARNESS v5.1 readings (docs/RL_PLAN.md 2026-09-29 21:10 UTC; rules: read_v51.py docstring)\n"]
    obj = {"root": str(root)}
    summary = []
    if a.phase in ("1", "both"):
        l1, o1 = phase1(a, root, broot, sets, test_all, load, out)
        L += l1
        obj["phase1"] = o1
        summary.append(f"Phase 1' {o1['status']}" + ("" if o1["final"] else f" (provisional {o1['label']})"))
    if a.phase in ("2", "both"):
        l2, o2 = phase2(a, root, sets, test_all, load, ckpt_root)
        L += ["\n"] + l2
        obj["phase2"] = o2
        summary.append(f"Phase 2' {o2['status']}" + ("" if o2["final"] else f" (provisional {o2['label']})"))
    warn = (obj.get("phase1") or {}).get("warnings", []) + (obj.get("phase2") or {}).get("warnings", [])
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(1, "**" + " | ".join(summary) + "**\n")
    text = "\n".join(L) + "\n"
    print(text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "V51_READING.md").write_text(text)
        (out / "V51_READING.json").write_text(json.dumps(obj, indent=1, default=str))
    print("V51 SUMMARY: " + "; ".join(summary) + (f"; written {out / 'V51_READING.md'}" if out is not None else "; nothing written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
