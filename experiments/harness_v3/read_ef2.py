#!/usr/bin/env python
"""EF2 reading -- does the execution-forcing gate generalise as a harness property? (docs/RL_PLAN.md 2026-09-28 23:00 UTC, "EF2
REGISTERED (before any build or run)"). Same gate as EF (experiments/harness_v3/exec_forcing.py, unchanged rule), paired by
instance, one-sided sign-flip p (analyze.paired, as read_ef.py) -- exact McNemar for the externals (the sign test on the
discordant problems, i.e. the sign-flip test for pass/fail pairs; one-sided in the rule's direction, the two-sided
ext_compare.py value printed beside).

    venv/bin/python experiments/harness_v3/read_ef2.py [--arms ef14 efx efpr] [--root8 results/pods/harness_v3]
        [--root14 results/pods/harness_v3_q14] [--perms 20000] [--seed 0] [--out DIR]

A NEW FILE: read_ef.py / read_v3.py / analyze.py are imported for their helpers (cell_pass, paired, complete, mechanics,
HO6, the NO-SPILL family set) and never edited; read_ef.py keeps the EF reading reproducible.

REGISTERED READINGS (each INCOMPLETE until every family / problem it needs is on disk in both cells; a provisional value
on the rows present is printed, marked):
  EF14  untrained Qwen3-14B + gate (<root14>/V3EF, item EF_2) vs the 14B plain base (<root14>/V3).
        GATE-SCALES iff TEST-all(EF14) - TEST-all(base) >= +3 points with one-sided p <= 0.05 AND NO-SPILL: the mean change
        over the 14 families other than causal_confounded, surv_cox and surv_tvc (EF's NO-SPILL set; "mean" = the
        unweighted mean of the 14 per-family changes, as EF's NO-SPILL; the twin-weighted mean beside) within +-3.
        Otherwise NOT-SCALES (the failing leg named). TEST-all = the twin-weighted TEST mean (read_ef.py / read_v3_q8.py).
  EFX   untrained 8B + gate on the externals (<root8>/V3EFX/ext, item EF_3) vs the 8B plain v3 base (<root8>/V3/ext).
        GATE-REAL iff QRData pass(EFX) - pass(base) >= +3 points with exact McNemar one-sided p <= 0.05 AND DABench not
        down by more than 3 points (EFX - base >= -3). Otherwise NOT-REAL (the failing leg named). BixBench beside; the
        QRData causal / non-causal split beside (keywords from each problem's _provenance containing "causal", the
        scratchpad qrdata_split.py rule).
  EFPR  PR s1 @100 + gate (<root8>/V3EF__<PR s1>_s100, item EF_4) vs PR s1 @100 plain (<root8>/V3__<PR s1>_s100).
        GATE-ADDS iff TEST-all(EFPR) - TEST-all(PR) >= +2 points with one-sided p <= 0.05; GATE-REDUNDANT iff
        |TEST-all diff| <= 1; otherwise NEITHER.
BESIDE (never deciding): HO6 for EF14 / EFPR, the per-family tables, the design family (EFPR), EF mechanics (refusals,
accept modes, no-answer / cells vs the plain cell), for EFX the gate's coverage per benchmark (offers the rule applied to,
problems gated, why it did not apply), and provenance (the EF2 directories must record exec_forcing with the SAME rule,
feedback and file hashes as EF_1's V3EF; EFX must also record the external formats).
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_ef as E  # noqa: E402  (read_v3 helpers + EF's pair / complete / mechanics; never edited)

A = E.A
HO6 = E.HO6
PR_RUN = E.PR_RUN.format(seed=1)
PR_STEP = 100
PR_PLAIN = f"V3__{PR_RUN}_s{PR_STEP}"
EFPR_DIR = f"V3EF__{PR_RUN}_s{PR_STEP}"
SCALE_BAR, SCALE_P, SPILL_BAR = 3.0, 0.05, 3.0
REAL_BAR, REAL_P, DA_DOWN = 3.0, 0.05, 3.0
ADDS_BAR, ADDS_P, REDUNDANT_BAR = 2.0, 0.05, 1.0
BENCHES = ("qrdata", "dabench", "bixbench")
V3_HARNESS = {"kernel": True, "library": "", "siblings": 0, "echo_last": "1"}
fmt = E.fmt


# ----------------------------------------------------------------------------------------------------------- helpers
def gate_of(cfgjson: dict) -> dict:
    """The EF gate record of a config.json without the external-format part (compared with EF_1's)."""
    g = copy.deepcopy((cfgjson.get("harness") or {}).get("exec_forcing") or {})
    g.pop("ext", None)
    return g


def load_cfg(d: Path) -> dict | None:
    p = d / "config.json"
    return json.loads(p.read_text()) if p.is_file() else None


def provenance(root: Path, name: str, sets, model: str, ckpt_tail: str | None, ref_gate: dict | None,
               plain_harness: dict | None, ext: bool = False, stages=("design", "test")) -> list:
    """Problems with an EF2 directory's provenance (empty list = clean)."""
    bad = []
    c = load_cfg(root / name)
    if c is None:
        return [f"{root / name}/config.json missing"]
    h = dict(c.get("harness") or {})
    ef = h.pop("exec_forcing", None) or {}
    if not ef.get("on"):
        bad.append("config.json does not record exec_forcing -- NOT an EF directory")
    if ext and not ef.get("ext"):
        bad.append("config.json does not record the external answer formats (exec_forcing.ext)")
    if not ext and ef.get("ext"):
        bad.append("config.json records the external-format extension on a synthetic read")
    if ref_gate is not None and gate_of(c) != ref_gate:
        bad.append("the gate record (rule / feedback / sha256) differs from EF_1's V3EF -- not the same gate")
    if c.get("model", "Qwen3-8B") != model:
        bad.append(f"config.json model={c.get('model')} (expected {model})")
    ck = c.get("ckpt") or ""
    if ckpt_tail is None and ck:
        bad.append(f"config.json ckpt={ck!r} (expected the untrained base)")
    if ckpt_tail is not None and (not ck.rstrip("/").endswith(ckpt_tail) or str(c.get("step")) != str(PR_STEP)):
        bad.append(f"config.json ckpt={ck!r} step={c.get('step')!r} (expected .../{ckpt_tail} step {PR_STEP})")
    want = dict(plain_harness) if plain_harness else dict(V3_HARNESS, system="science")
    off = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    if off:
        bad.append(f"config.json harness differs from the plain cell's: {off}")
    if not ext:
        for s in stages:
            for f in sets[s]:
                p = root / name / s / f"{f}.json"
                if p.is_file() and not (json.loads(p.read_text()).get("harness_v2") or {}).get("exec_forcing"):
                    bad.append(f"{s}/{f}.json does not record exec_forcing")
    return bad


def spill_set(sets) -> list:
    return [f for f in A.cell_members(sets, "test") if f not in E.COMPETENT]


def verdict_ef14(d_all: dict | None, spill: dict | None) -> tuple[str, list]:
    """(label, reasons) from the TEST-all contrast and the unweighted NO-SPILL mean."""
    if d_all is None or spill is None:
        return "INCOMPLETE", ["no paired rows"]
    why = []
    up = d_all["d"] >= SCALE_BAR - 1e-9 and d_all["p_gt"] <= SCALE_P
    if not up:
        why.append(f"TEST-all {d_all['d']:+.1f} (p>0 {d_all['p_gt']:.4f}) is not >= +{SCALE_BAR:.0f} with p <= {SCALE_P}")
    flat = abs(spill["d"]) <= SPILL_BAR + 1e-9
    if not flat:
        why.append(f"NO-SPILL fails: the 14 families moved {spill['d']:+.1f} (> +-{SPILL_BAR:.0f})")
    return ("GATE-SCALES" if (up and flat) else "NOT-SCALES"), why


def verdict_efpr(d_all: dict | None) -> tuple[str, list]:
    if d_all is None:
        return "INCOMPLETE", ["no paired rows"]
    d = d_all["d"]
    if d >= ADDS_BAR - 1e-9 and d_all["p_gt"] <= ADDS_P:
        return "GATE-ADDS", [f"TEST-all {d:+.1f} >= +{ADDS_BAR:.0f}, p>0 {d_all['p_gt']:.4f} <= {ADDS_P}"]
    if abs(d) <= REDUNDANT_BAR + 1e-9:
        return "GATE-REDUNDANT", [f"|TEST-all diff| {abs(d):.1f} <= {REDUNDANT_BAR:.0f}"]
    return "NEITHER", [f"TEST-all {d:+.1f} (p>0 {d_all['p_gt']:.4f}, p<0 {d_all['p_lt']:.4f}): neither >= +{ADDS_BAR:.0f} "
                       f"with p <= {ADDS_P} nor within +-{REDUNDANT_BAR:.0f}"]


def mcnemar(ef: dict, base: dict, ids=None) -> dict | None:
    """Exact McNemar on the problems in both: b01 = EF passes where the base fails, b10 the reverse."""
    common = sorted(k for k in ef if k in base and (ids is None or k in ids))
    if not common:
        return None
    b01 = sum(1 for k in common if ef[k] and not base[k])
    b10 = sum(1 for k in common if base[k] and not ef[k])
    m = b01 + b10
    if m:
        p_gt = sum(math.comb(m, i) for i in range(b01, m + 1)) / 2 ** m
        p_lt = sum(math.comb(m, i) for i in range(0, b01 + 1)) / 2 ** m
        p2 = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(b01, b10) + 1)) / 2 ** m)
    else:
        p_gt = p_lt = p2 = 1.0
    n = len(common)
    ep, bp = sum(ef[k] for k in common), sum(base[k] for k in common)
    return {"n": n, "ef_pass": ep, "base_pass": bp, "ef": 100.0 * ep / n, "base": 100.0 * bp / n,
            "d": 100.0 * (b01 - b10) / n, "b01": b01, "b10": b10, "p_gt": float(p_gt), "p_lt": float(p_lt), "p2": float(p2)}


def verdict_efx(q: dict | None, da: dict | None) -> tuple[str, list]:
    if q is None or da is None:
        return "INCOMPLETE", ["QRData and DABench both needed"]
    why = []
    up = q["d"] >= REAL_BAR - 1e-9 and q["p_gt"] <= REAL_P
    if not up:
        why.append(f"QRData {q['d']:+.1f} (+{q['b01']}/-{q['b10']}, McNemar p>0 {q['p_gt']:.4f}) is not >= +{REAL_BAR:.0f} "
                   f"with p <= {REAL_P}")
    safe = da["d"] >= -DA_DOWN - 1e-9
    if not safe:
        why.append(f"DABench is down {da['d']:+.1f} (> {DA_DOWN:.0f} points)")
    return ("GATE-REAL" if (up and safe) else "NOT-REAL"), why


def problem_counts(bench_root: Path) -> dict:
    return {b: len(list((bench_root / b / "problems").glob("*/eval_config.json"))) for b in BENCHES}


def load_ext(d: Path) -> dict:
    """{problem id: record} of one runner output directory (summary.json / harness_v2.json excluded)."""
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name in ("summary.json", "harness_v2.json"):
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and "id" in r and "grade" in r:
            out[r["id"]] = r
    return out


def ext_passed(recs: dict) -> dict:
    return {k: bool((r.get("grade") or {}).get("passed")) for k, r in recs.items()}


def qrdata_causal(bench_root: Path) -> dict:
    """{problem id: is causal} -- any _provenance keyword containing 'causal' (scratchpad qrdata_split.py)."""
    out = {}
    for cfg in sorted((bench_root / "qrdata" / "problems").glob("*/eval_config.json")):
        c = json.loads(cfg.read_text())
        ks = (c.get("_provenance") or {}).get("keywords") or []
        if isinstance(ks, str):
            try:
                ks = ast.literal_eval(ks)
            except (ValueError, SyntaxError):
                ks = [ks]
        out[c["id"]] = any("causal" in str(x).lower() for x in ks)
    return out


def ext_mechanics(ef: dict, base: dict) -> dict:
    """The gate's bookkeeping and coverage over one benchmark's EFX records, and the plain cell's behaviour beside."""
    n = len(ef)
    if not n:
        return {"n": 0}
    logs = [r.get("ef") for r in ef.values()]
    has = [x for x in logs if isinstance(x, dict)]
    acc, na_why = Counter(), Counter()
    offers = gated_offers = gated_probs = final_gated = 0
    for x in has:
        acc[str(x.get("accepted"))] += 1
        ents = x.get("ext") or []
        offers += len(ents)
        g = [e for e in ents if e.get("fired") is not None]
        gated_offers += len(g)
        gated_probs += bool(g)
        acc_ents = [e for e in ents if e.get("accepted")]
        if acc_ents and acc_ents[-1].get("fired") is not None:
            final_gated += 1
        for e in ents:
            for w in (e.get("na") or {}).values():
                w = str(w)
                na_why["interval/range/inequality option" if any(t in w for t in ("interval", "range", "inequality"))
                       else ("text option" if w.startswith("option") else w)] += 1

    def beh(recs):
        rs = list(recs.values())
        return {"no_answer": sum(1 for r in rs if r.get("submission") is None),
                "cells_mean": float(np.mean([r.get("n_cells") or 0 for r in rs])) if rs else None,
                "answered_without_code": sum(1 for r in rs if r.get("submission") is not None and not (r.get("n_cells") or 0)),
                "errored": sum(1 for r in rs if r.get("error"))}
    rej = [int(x.get("rejected") or 0) for x in has]
    return {"n": n, "with_ef_log": len(has), "offered": int(sum(int(x.get("offered") or 0) for x in has)),
            "rejected_total": int(sum(rej)), "rejected_mean": float(np.mean(rej)) if rej else None,
            "problems_with_refusal": sum(1 for v in rej if v > 0), "accepted": dict(acc),
            "na_offers": int(sum(int(x.get("na") or 0) for x in has)),
            "gate_errors": int(sum(int(x.get("gate_errors") or 0) for x in has)),
            "ext_offers": offers, "gated_offers": gated_offers, "problems_gated": gated_probs,
            "final_answer_gated": final_gated, "na_keys_by_reason": dict(na_why.most_common()),
            "ef": beh(ef), "base": beh(base)}


def write(out: Path | None, stem: str, text: str, obj: dict):
    if out is None:
        return None
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.md").write_text(text)
    (out / f"{stem}.json").write_text(json.dumps(obj, indent=1, default=str))
    return out / f"{stem}.md"


def ptxt(x, key="p_gt"):
    return "-" if not x else f"{x['d']:+.1f} (p>0 {x['p_gt']:.4f}, p<0 {x['p_lt']:.4f})"


# ----------------------------------------------------------------------------------------------------------- arms
def read_synthetic(arm: str, root: Path, base_name: str, ef_name: str, sets, perms: int, rng, model: str,
                   ckpt_tail, ref_gate, plain_harness, extra_cells: dict) -> tuple[list, dict]:
    """EF14 / EFPR: the registered TEST-all contrast (and NO-SPILL for EF14), HO6 and the per-family table beside."""
    usets = E.uniform_sets(sets)
    test_all = A.cell_members(sets, "test")
    others = spill_set(sets)

    def load(name):
        return A.load_config(root, name, sets) if (root / name).is_dir() else None
    base, ef = load(base_name), load(ef_name)
    miss = {"plain": E.complete(base, sets, test_all), "ef": E.complete(ef, sets, test_all)}
    full = base is not None and ef is not None and not miss["plain"] and not miss["ef"]
    d_all = E.pair(ef, base, sets, perms, rng, test_all)
    d_ho6 = E.pair(ef, base, sets, perms, rng, HO6)
    spill_u = E.pair(ef, base, usets, perms, rng, others)
    spill_w = E.pair(ef, base, sets, perms, rng, others)
    if arm == "ef14":
        label, why = verdict_ef14(d_all, spill_u)
    else:
        label, why = verdict_efpr(d_all)
    status = label if full else "INCOMPLETE"
    warn = []
    if ef is not None:
        warn += [f"PROVENANCE: {x}" for x in provenance(root, ef_name, sets, model, ckpt_tail, ref_gate, plain_harness)]
    title = {"ef14": "EF14 -- untrained Qwen3-14B + the execution-forcing gate vs the 14B plain base",
             "efpr": f"EFPR -- PR s1 @{PR_STEP} + the execution-forcing gate vs PR s1 @{PR_STEP} plain"}[arm]
    L = [f"# {title} (docs/RL_PLAN.md 2026-09-28 23:00 UTC, EF2)\n",
         f"root `{root}`; gate `{ef_name}` vs plain `{base_name}`; paired by instance; p one-sided sign-flip ({perms} perms). "
         f"Rules: read_ef2.py docstring.\n"]
    if ef is None:
        L.append(f"**{root / ef_name} does not exist yet (queue item {'0a_0095_EF_2_ef14' if arm == 'ef14' else '0a_0095_EF_4_efpr_s1'}) "
                 f"-- PENDING.**\n")
    for k, v in miss.items():
        if v and not (k == "ef" and ef is None):
            L.append(f"* {k}: {len(v)} TEST families missing or short: {', '.join(v)}")
    L.append("\n## Registered reading\n")
    b_all = A.cell_pass(base, sets, "test", test_all)[0] if base is not None else None
    e_all = A.cell_pass(ef, sets, "test", test_all)[0] if ef is not None else None
    if arm == "ef14":
        L.append("| reading | rule | gate | plain | gate - plain | NO-SPILL (14 families, unweighted) | result |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| GATE-SCALES | TEST-all >= plain + {SCALE_BAR:.0f} (p <= {SCALE_P}) and the 14 within +-{SPILL_BAR:.0f} | "
                 f"{fmt(e_all)} | {fmt(b_all)} | {ptxt(d_all)} | "
                 + (f"{spill_u['d']:+.1f} ({spill_u['n_fams']}/14 fams; twin-weighted "
                    f"{spill_w['d']:+.1f})" if spill_u else "-") + f" | **{status}** |")
    else:
        L.append("| reading | rule | PR + gate | PR plain | diff | result |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| GATE-ADDS / GATE-REDUNDANT | ADDS: TEST-all >= +{ADDS_BAR:.0f} (p <= {ADDS_P}); REDUNDANT: |diff| <= "
                 f"{REDUNDANT_BAR:.0f} | {fmt(e_all)} | {fmt(b_all)} | {ptxt(d_all)} | **{status}** |")
    if label != "INCOMPLETE":
        for w in why:
            L.append(f"* {'' if full else 'PROVISIONAL on the rows present (not the registered reading): '}{label}: {w}")
    L.append("\n## Beside (never deciding)\n")
    L.append(f"* HO6: gate {fmt(A.cell_pass(ef, sets, 'test', HO6)[0] if ef is not None else None)} vs plain "
             f"{fmt(A.cell_pass(base, sets, 'test', HO6)[0] if base is not None else None)}: {ptxt(d_ho6)}")
    for lab, (cell_root, cell_name) in extra_cells.items():
        cc = A.load_config(cell_root, cell_name, sets) if (cell_root / cell_name).is_dir() else None
        if cc is None or ef is None:
            L.append(f"* gate vs {lab} (`{cell_name}`): -")
            continue
        x = E.pair(ef, cc, sets, perms, rng, test_all)
        L.append(f"* gate vs {lab} (`{cell_name}`), TEST-all: {fmt(A.cell_pass(cc, sets, 'test', test_all)[0])} -> "
                 f"{fmt(e_all)}: {ptxt(x)}")
    zc = None
    if arm == "ef14" and base is not None:
        nz = {f: fam for f in others if (fam := E.fam(base, sets, f)) not in (None, 0.0)}
        L.append(f"* the 14 NO-SPILL families (EF's registered set) at 14B: the plain base passes something on {len(nz)} of "
                 f"them" + (": " + ", ".join(f"{f} {v:.1f}" for f, v in nz.items()) if nz else " (all zero-competence)"))
        zero = [f for f in others if f not in nz]
        zc = E.pair(ef, base, usets, perms, rng, zero) if zero else None
        L.append(f"* beside only: the unweighted mean change over the {len(zero)} of them the 14B plain base passes NOTHING "
                 f"on (the literal zero-competence subset at 14B): {ptxt(zc)}")
    L.append("\n| family | plain | gate | gate - plain (p>0) | gate refusals / rollout |")
    L.append("|---|---|---|---|---|")
    per = {}
    stages = [("test", test_all)] + ([("design", [f for f in sets["design"] if ef is not None
                                                  and ("design", f) in ef["families"]])] if arm == "efpr" else [])
    for s, members in stages:
        for f in members:
            b, e = E.fam(base, sets, f, s), E.fam(ef, sets, f, s)
            d = E.pair(ef, base, sets, perms, rng, [f], s)
            rows = (ef or {}).get("rollouts", {}).get((s, f)) or []
            rj = [int((((r.get("v2") or {}).get("ef")) or {}).get("rejected") or 0) for r in rows]
            per[f"{s}/{f}"] = {"plain": b, "gate": e, "d": d, "refusals": float(np.mean(rj)) if rj else None}
            tag = " (DESIGN)" if s == "design" else (" (HO6)" if f in HO6 else (" (NO-SPILL set)" if f in others else ""))
            L.append(f"| {f}{tag} | {fmt(b)} | {fmt(e)} | " + (f"{d['d']:+.1f} ({d['p_gt']:.3f})" if d else "-")
                     + f" | {fmt(per[f'{s}/{f}']['refusals'], 2)} |")
    mech = {"gate": E.mechanics(ef) if ef is not None else None, "plain": E.mechanics(base) if base is not None else None}
    L.append("\n## EF mechanics (TEST rollouts)\n")
    me, mb = mech["gate"] or {}, mech["plain"] or {}
    if me.get("n"):
        L.append(f"* rollouts {me['n']}; with an EF log {me['with_ef_log']}; refusals total {me['rejected_total']}, mean "
                 f"{fmt(me['rejected_mean'], 2)} / rollout, rollouts with >= 1 refusal {me['rollouts_with_rejection']}")
        L.append(f"* accepted: {json.dumps(me['accepted'])}; rule n/a {me['na']}; gate errors {me['gate_errors']}")
        L.append(f"* no answer: gate {me['no_answer']} vs plain {mb.get('no_answer', '-')}; forced: gate {me['forced']} vs "
                 f"plain {mb.get('forced', '-')}; cells / rollout {fmt(me['cells_mean'], 2)} vs {fmt(mb.get('cells_mean'), 2)}; "
                 f"answers with no executed cell {me['zero_cell_answers']} vs {mb.get('zero_cell_answers', '-')}")
        if me["with_ef_log"] < me["n"] - me["errored"]:
            warn.append(f"{me['n'] - me['with_ef_log']} gate rollouts have no EF log (and only {me['errored']} errored)")
        if me["gate_errors"]:
            warn.append(f"{me['gate_errors']} gate errors (fail-open: accepted as in the plain harness)")
    else:
        L.append("* (no gate rollouts yet)")
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(2, f"**Status: {status}**" + ("" if full or label == "INCOMPLETE" else f" (provisional: {label})") + "\n")
    obj = {"arm": arm, "root": str(root), "plain": base_name, "gate": ef_name, "status": status, "label": label,
           "why": why, "complete": full, "test_all": {"plain": b_all, "gate": e_all, "d": d_all},
           "ho6": d_ho6, "spill": {"unweighted": spill_u, "twin_weighted": spill_w, "families": others,
                                   "zero_competence_subset": zc},
           "per_family": per, "mechanics": mech, "missing": miss, "warnings": warn}
    return L, obj


def read_efx(root8: Path, bench_root: Path, base_name: str, efx_name: str, ref_gate, plain_harness, sets) -> tuple[list, dict]:
    nprob = problem_counts(bench_root)
    efd, bd = root8 / efx_name / "ext", root8 / base_name / "ext"
    res, comp, mech = {}, {}, {}
    for b in BENCHES:
        er, br = load_ext(efd / b), load_ext(bd / b)
        comp[b] = {"gate": (efd / b / "summary.json").is_file() and len(er) == nprob[b],
                   "plain": (bd / b / "summary.json").is_file() and len(br) == nprob[b],
                   "n_gate": len(er), "n_plain": len(br), "n_problems": nprob[b]}
        res[b] = mcnemar(ext_passed(er), ext_passed(br)) if er and br else None
        mech[b] = ext_mechanics(er, br)
    q = res["qrdata"] if comp["qrdata"]["gate"] and comp["qrdata"]["plain"] else None
    da = res["dabench"] if comp["dabench"]["gate"] and comp["dabench"]["plain"] else None
    label, why = verdict_efx(res["qrdata"], res["dabench"])
    full = q is not None and da is not None
    status = label if full else "INCOMPLETE"
    warn = []
    if (root8 / efx_name).is_dir():
        warn += [f"PROVENANCE: {x}" for x in provenance(root8, efx_name, sets, "Qwen3-8B", None, ref_gate, plain_harness,
                                                         ext=True)]
        for b in BENCHES:
            hj = efd / b / "harness_v2.json"
            if hj.is_file() and not (json.loads(hj.read_text()).get("exec_forcing") or {}).get("ext"):
                warn.append(f"PROVENANCE: ext/{b}/harness_v2.json does not record the external formats")
            if mech[b].get("n") and mech[b]["with_ef_log"] < mech[b]["n"] - mech[b]["ef"]["errored"]:
                warn.append(f"{b}: {mech[b]['n'] - mech[b]['with_ef_log']} problems carry no gate log")
            if mech[b].get("gate_errors"):
                warn.append(f"{b}: {mech[b]['gate_errors']} gate errors (fail-open)")
    L = ["# EFX -- untrained 8B + the execution-forcing gate on the external benchmarks (docs/RL_PLAN.md 2026-09-28 23:00 UTC, "
         "EF2)\n",
         f"gate `{root8 / efx_name}/ext` vs plain `{root8 / base_name}/ext`; paired by problem; exact McNemar (one-sided in "
         "the rule's direction; two-sided = ext_compare.py beside). Rules: read_ef2.py docstring.\n"]
    if not (root8 / efx_name).is_dir():
        L.append(f"**{root8 / efx_name} does not exist yet (queue item 0a_0095_EF_3_efx_base8b) -- PENDING.**\n")
    L.append("\n## Registered reading\n")
    L.append("| benchmark | n | plain | gate | gate - plain | +gained / -lost | McNemar p>0 / p<0 / two-sided | role |")
    L.append("|---|---|---|---|---|---|---|---|")
    role = {"qrdata": f"GATE-REAL: >= +{REAL_BAR:.0f}, p>0 <= {REAL_P}", "dabench": f"GATE-REAL: not down > {DA_DOWN:.0f}",
            "bixbench": "beside"}
    for b in BENCHES:
        x = res[b]
        c = comp[b]
        part = "" if (c["gate"] and c["plain"]) else f" (PARTIAL: gate {c['n_gate']}/{c['n_problems']}, plain {c['n_plain']})"
        L.append(f"| {b} | {x['n'] if x else '-'} | {fmt(x['base']) if x else '-'} | {fmt(x['ef']) if x else '-'} | "
                 f"{fmt(x['d'], sign=True) if x else '-'} | " + (f"+{x['b01']} / -{x['b10']}" if x else "-") + " | "
                 + (f"{x['p_gt']:.4f} / {x['p_lt']:.4f} / {x['p2']:.4f}" if x else "-") + f" | {role[b]}{part} |")
    L.append(f"\n**{status}**" + ("" if full or label == "INCOMPLETE" else f" (provisional on the rows present: {label})"))
    if label != "INCOMPLETE":
        for w in why:
            L.append(f"* {'' if full else 'PROVISIONAL: '}{w}")
    # QRData causal split
    cau = qrdata_causal(bench_root)
    er, br = ext_passed(load_ext(efd / "qrdata")), ext_passed(load_ext(bd / "qrdata"))
    split = {}
    L.append("\n## Beside: QRData causal / non-causal (keywords from each problem's _provenance)\n")
    L.append("| subset | n | plain | gate | gate - plain | +gained / -lost | McNemar p>0 / two-sided |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, want in (("causal", True), ("non-causal", False)):
        ids = {k for k, v in cau.items() if v == want}
        x = mcnemar(er, br, ids) if er and br else None
        split[lab] = x
        L.append(f"| {lab} ({len(ids)}) | {x['n'] if x else '-'} | {fmt(x['base']) if x else '-'} | {fmt(x['ef']) if x else '-'} | "
                 f"{fmt(x['d'], sign=True) if x else '-'} | " + (f"+{x['b01']} / -{x['b10']}" if x else "-") + " | "
                 + (f"{x['p_gt']:.4f} / {x['p2']:.4f}" if x else "-") + " |")
    L.append("\n## Beside: the gate's coverage and mechanics per benchmark\n")
    L.append("| benchmark | problems | offers (ext entries) | rule applied (offers) | problems gated | final answer gated | "
             "refusals (mean / problem; problems refused >= 1) | accepted as | rule n/a keys, by reason | no answer gate / plain | "
             "cells / problem gate / plain |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BENCHES:
        m = mech[b]
        if not m.get("n"):
            L.append(f"| {b} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {b} | {m['n']} | {m['ext_offers']} | {m['gated_offers']} ({100 * m['gated_offers'] / max(m['ext_offers'], 1):.0f}%) | "
                 f"{m['problems_gated']} ({100 * m['problems_gated'] / m['n']:.0f}%) | {m['final_answer_gated']} | "
                 f"{m['rejected_total']} ({fmt(m['rejected_mean'], 2)}; {m['problems_with_refusal']}) | {json.dumps(m['accepted'])} | "
                 f"{json.dumps(m['na_keys_by_reason'])} | {m['ef']['no_answer']} / {m['base']['no_answer']} | "
                 f"{fmt(m['ef']['cells_mean'], 2)} / {fmt(m['base']['cells_mean'], 2)} |")
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(2, f"**Status: {status}**" + ("" if full or label == "INCOMPLETE" else f" (provisional: {label})") + "\n")
    obj = {"arm": "efx", "root": str(root8), "plain": base_name, "gate": efx_name, "status": status, "label": label,
           "why": why, "complete": full, "benchmarks": res, "completeness": comp, "qrdata_split": split,
           "mechanics": mech, "warnings": warn}
    return L, obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["ef14", "efx", "efpr"], choices=["ef14", "efx", "efpr"])
    ap.add_argument("--root8", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--root14", default=str(ROOT / "results" / "pods" / "harness_v3_q14"))
    ap.add_argument("--bench-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--ef8", default="V3EF", help="EF_1's directory under --root8 (the reference gate record)")
    ap.add_argument("--ef14", default="V3EF")
    ap.add_argument("--efx", default="V3EFX")
    ap.add_argument("--efpr", default=EFPR_DIR)
    ap.add_argument("--pr", default=PR_PLAIN)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write every arm's reading here (default: each arm's own directory, if it exists)")
    a = ap.parse_args(argv)
    root8, root14, broot = Path(a.root8), Path(a.root14), Path(a.bench_root)
    sets = A.load_families()
    ref = load_cfg(root8 / a.ef8)
    ref_gate = gate_of(ref) if ref else None
    out = Path(a.out) if a.out else None
    summary = {}
    for arm in a.arms:
        rng = np.random.default_rng(a.seed)
        if arm == "ef14":
            plain = load_cfg(root14 / a.base)
            L, obj = read_synthetic("ef14", root14, a.base, a.ef14, sets, a.perms, rng, "Qwen3-14B", None, ref_gate,
                                    (plain or {}).get("harness"), {})
            where = out or ((root14 / a.ef14) if (root14 / a.ef14).is_dir() else None)
            stem = "EF14_READING"
        elif arm == "efpr":
            plain = load_cfg(root8 / a.pr)
            L, obj = read_synthetic("efpr", root8, a.pr, a.efpr, sets, a.perms, rng, "Qwen3-8B", PR_RUN, ref_gate,
                                    (plain or {}).get("harness"),
                                    {"the untrained 8B plain": (root8, a.base), "the untrained 8B + gate (EF_1)": (root8, a.ef8)})
            where = out or ((root8 / a.efpr) if (root8 / a.efpr).is_dir() else None)
            stem = "EFPR_READING"
        else:
            plain = load_cfg(root8 / a.base)
            L, obj = read_efx(root8, broot, a.base, a.efx, ref_gate, (plain or {}).get("harness"), sets)
            where = out or ((root8 / a.efx) if (root8 / a.efx).is_dir() else None)
            stem = "EFX_READING"
        text = "\n".join(L) + "\n"
        print(text)
        p = write(where, stem, text, obj)
        obj["written"] = str(p) if p else None
        summary[arm] = {"status": obj["status"], "label": obj["label"], "why": obj["why"], "written": obj["written"]}
    print("EF2 SUMMARY: " + "; ".join(f"{k.upper()} {v['status']}" + ("" if v["status"] == v["label"] else f" (provisional "
                                                                                                         f"{v['label']})")
                                      for k, v in summary.items()))
    if out is not None:
        (out / "EF2_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
