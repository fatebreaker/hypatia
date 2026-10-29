#!/usr/bin/env python
"""HARNESS v4 reading -- the SOFT EXECUTION GATE (EFS) vs the plain v3 harness, the hard gate beside (docs/RL_PLAN.md
2026-09-29 04:40 UTC, "HARNESS v4 = v3 + SOFT EXECUTION GATE (EFS), REGISTERED before any build or run"). The untrained
Qwen3-8B, one run per cell, paired by instance (synthetic TEST, 17 families 12x2) and by problem (externals):
  plain  <root>/V3      (HV3_17: the v3 evaluation harness)          externals <root>/V3/ext
  hard   <root>/V3EF    (EF_1: + the execution-forcing gate)          externals <root>/V3EFX/ext   (EF_3)
  soft   <root>/V3EFS   (EFS_1: + the gate, soft fallback = v4)        externals <root>/V3EFSX/ext  (EFS_2)

    venv/bin/python experiments/harness_v3/read_efs.py [--root results/pods/harness_v3] [--base V3] [--ef V3EF]
        [--efs V3EFS] [--efx V3EFX] [--efsx V3EFSX] [--bench-root benchmarks] [--perms 20000] [--seed 0] [--out DIR]

A NEW FILE: read_ef2.py / read_ef.py / analyze.py are imported for their helpers (TEST-all twin-weighted cell_pass, the
paired sign-flip test, completeness, EF mechanics, the exact McNemar, the externals' loaders and mechanics, the QRData
causal split) and never edited.

REGISTERED RULE -- SOFT-GATE-ADOPT iff ALL FOUR legs hold, else NOT-ADOPT (the failing leg(s) named):
  TEST      TEST-all(soft) - TEST-all(plain) >= +3 points, one-sided sign-flip p (soft > plain) <= 0.05. TEST-all = the
            twin-weighted mean over the 17 sealed TEST families (analyze.py's TEST weights; read_ef.py / read_ef2.py).
  QRDATA    QRData(soft) - QRData(plain) >= +3 points, exact McNemar one-sided p (soft > plain) <= 0.05.
  DABENCH   DABench(soft) - DABench(plain) >= -3 points (not down by more than 3).
  BIXBENCH  BixBench(soft) - BixBench(plain) >= -3 points.
A leg is COMPLETE when all its rows are on disk in both cells (every TEST family at the draw's n x k; every problem of the
benchmark, summary.json written). STATUS = the label once all four legs are complete; NOT-ADOPT as soon as one COMPLETE leg
fails (a conjunction cannot recover; the pending legs are named); otherwise INCOMPLETE, with the provisional label on the
rows present printed and marked.
BESIDE (never deciding): HO6; the soft gate vs the HARD gate (V3EF / V3EFX) on TEST-all, HO6 and every benchmark; the
per-family table plain / hard / soft; no-answer counts per cell; how final answers were accepted; the soft-fallback count and
its pass rate (the grounded fallback's beside); "no answer although the model offered one" (must be 0 in the soft cells: the
soft fallback's invariant); the QRData causal / non-causal split; provenance (the soft cells must record exec_forcing_soft
-- never EF's exec_forcing -- with EF_1's gate record: same rule, feedback and exec_forcing.py sha).
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
import read_ef2 as R2  # noqa: E402  (read_ef / read_v3 / analyze helpers; never edited)

E = R2.E
A = R2.A
HO6 = E.HO6
fmt = E.fmt
BENCHES = R2.BENCHES
TEST_BAR, TEST_P, QR_BAR, QR_P, DOWN_BAR = 3.0, 0.05, 3.0, 0.05, 3.0
EPS = 1e-9
LEGS = ("TEST", "QRDATA", "DABENCH", "BIXBENCH")
LEG_BENCH = {"QRDATA": "qrdata", "DABENCH": "dabench", "BIXBENCH": "bixbench"}
HARNESS = "exec_forcing_soft"
SOFT, GROUNDED = "soft_fallback", "fallback"


# ----------------------------------------------------------------------------------------------------------- the rule
def leg_ok(leg: str, x: dict | None) -> tuple[bool | None, str]:
    """(holds, why) of one leg on a paired result (E.pair's {'d','p_gt',...} or R2.mcnemar's {'d','p_gt','b01','b10',...})."""
    if x is None:
        return None, "no paired rows"
    if leg == "TEST":
        ok = x["d"] >= TEST_BAR - EPS and x["p_gt"] <= TEST_P
        return ok, (f"TEST-all {x['d']:+.1f} (sign-flip p>0 {x['p_gt']:.4f}) "
                    + ("is" if ok else "is not") + f" >= +{TEST_BAR:.0f} with p <= {TEST_P}")
    if leg == "QRDATA":
        ok = x["d"] >= QR_BAR - EPS and x["p_gt"] <= QR_P
        return ok, (f"QRData {x['d']:+.1f} (+{x['b01']}/-{x['b10']}, McNemar p>0 {x['p_gt']:.4f}) "
                    + ("is" if ok else "is not") + f" >= +{QR_BAR:.0f} with p <= {QR_P}")
    ok = x["d"] >= -DOWN_BAR - EPS
    return ok, (f"{LEG_BENCH[leg]} {x['d']:+.1f} (+{x['b01']}/-{x['b10']}) "
                + ("is not" if ok else "IS") + f" down by more than {DOWN_BAR:.0f} points")


def verdict(vals: dict, complete: dict) -> dict:
    """vals {leg: paired result | None}, complete {leg: bool} -> {'status','label','legs','failed','pending'}."""
    legs = {}
    for leg in LEGS:
        ok, why = leg_ok(leg, vals.get(leg))
        legs[leg] = {"ok": ok, "why": why, "complete": bool(complete.get(leg))}
    if all(v["ok"] is True for v in legs.values()):
        label = "SOFT-GATE-ADOPT"
    elif any(v["ok"] is False for v in legs.values()):
        label = "NOT-ADOPT"
    else:
        label = "INCOMPLETE"
    failed = [k for k, v in legs.items() if v["ok"] is False]
    failed_complete = [k for k in failed if legs[k]["complete"]]
    pending = [k for k, v in legs.items() if not v["complete"]]
    if not pending:
        status = label
    elif failed_complete:
        status = "NOT-ADOPT"
    else:
        status = "INCOMPLETE"
    return {"status": status, "label": label, "legs": legs, "failed": failed, "failed_complete": failed_complete,
            "pending": pending, "final": status != "INCOMPLETE"}


# ----------------------------------------------------------------------------------------------------------- helpers
def acc_of(x) -> str | None:
    e = x.get("ef") if isinstance(x.get("ef"), dict) else ((x.get("v2") or {}).get("ef"))
    return (e or {}).get("accepted") if isinstance(e, dict) else None


def offered_of(x) -> int:
    e = x.get("ef") if isinstance(x.get("ef"), dict) else ((x.get("v2") or {}).get("ef"))
    return int((e or {}).get("offered") or 0) if isinstance(e, dict) else 0


def fallback_stats(rows: list, passed, answered, errored) -> dict:
    """Counts over one cell's rows (synthetic rollouts or external problem records): no answer, the grounded and the soft
    fallback (n, passed, pass %), and 'no answer although an answer was offered' (errored rows excluded)."""
    out = {"n": len(rows), "no_answer": sum(1 for r in rows if not answered(r)),
           "no_answer_after_offer": sum(1 for r in rows if not answered(r) and offered_of(r) > 0 and not errored(r)),
           "errored": sum(1 for r in rows if errored(r))}
    for lab in (GROUNDED, SOFT):
        sel = [r for r in rows if acc_of(r) == lab]
        k = sum(1 for r in sel if passed(r))
        out[lab] = {"n": len(sel), "passed": k, "rate": (100.0 * k / len(sel)) if sel else None}
    return out


def syn_rows(res, members=None) -> list:
    if res is None:
        return []
    return [r for (s, f), rows in res["rollouts"].items() if s == "test" and (members is None or f in members) for r in rows]


def syn_stats(res, members=None) -> dict | None:
    if res is None:
        return None
    return fallback_stats(syn_rows(res, members), lambda r: bool(r.get("passed")), lambda r: bool(r.get("answered")),
                          lambda r: str(r.get("attribution") or "").startswith("error:"))


def ext_stats(recs: dict) -> dict | None:
    if not recs:
        return None
    return fallback_stats(list(recs.values()), lambda r: bool((r.get("grade") or {}).get("passed")),
                          lambda r: r.get("submission") is not None, lambda r: bool(r.get("error")))


def soft_record(h: dict) -> dict:
    return (h or {}).get(HARNESS) or {}


def efs_provenance(root: Path, name: str, sets, ref_gate: dict | None, plain_harness: dict | None, ext: bool) -> list:
    """Problems with a soft cell's provenance (empty list = clean)."""
    c = R2.load_cfg(root / name)
    if c is None:
        return [f"{root / name}/config.json missing"]
    bad = []
    h = dict(c.get("harness") or {})
    if "exec_forcing" in h:
        bad.append("config.json records EF's exec_forcing (the HARD gate's key) -- not an EFS directory")
    rec = h.pop(HARNESS, None) or {}
    h.pop("exec_forcing", None)
    if rec.get("harness") != HARNESS or not rec.get("on"):
        bad.append("config.json does not record exec_forcing_soft -- NOT an EFS directory")
    gate = copy.deepcopy(rec.get("gate") or {})
    has_ext = bool(gate.pop("ext", None))
    if ext and not has_ext:
        bad.append("config.json does not record the external answer formats (exec_forcing_soft.gate.ext)")
    if not ext and has_ext:
        bad.append("config.json records the external-format extension on a synthetic read")
    if ref_gate is not None and gate != ref_gate:
        bad.append("the gate record (rule / feedback / exec_forcing.py sha) differs from EF_1's V3EF -- not the same gate")
    if c.get("model", "Qwen3-8B") != "Qwen3-8B" or c.get("ckpt"):
        bad.append(f"config.json model={c.get('model')} ckpt={c.get('ckpt')!r} (v4 is read on the untrained 8B)")
    want = dict(plain_harness) if plain_harness else dict(R2.V3_HARNESS, system="science")
    off = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    if off:
        bad.append(f"config.json harness differs from the plain cell's: {off}")
    if not ext:
        for s in ("design", "test"):
            for f in sets[s]:
                p = root / name / s / f"{f}.json"
                if p.is_file():
                    hv = json.loads(p.read_text()).get("harness_v2") or {}
                    if soft_record(hv).get("harness") != HARNESS or "exec_forcing" in hv:
                        bad.append(f"{s}/{f}.json does not record exec_forcing_soft (only)")
    else:
        for b in BENCHES:
            hj = root / name / "ext" / b / "harness_v2.json"
            if hj.is_file():
                hv = json.loads(hj.read_text())
                if soft_record(hv).get("harness") != HARNESS or "exec_forcing" in hv \
                        or not (soft_record(hv).get("gate") or {}).get("ext"):
                    bad.append(f"ext/{b}/harness_v2.json does not record exec_forcing_soft with the external formats")
    return bad


def ptxt(x):
    if not x:
        return "-"
    return f"{x['d']:+.1f} (p>0 {x['p_gt']:.4f}, p<0 {x['p_lt']:.4f})"


def mtxt(x):
    if not x:
        return "-"
    return f"{x['d']:+.1f} (+{x['b01']} / -{x['b10']}; p>0 {x['p_gt']:.4f}, p<0 {x['p_lt']:.4f}, two-sided {x['p2']:.4f})"


def rate_txt(p: dict) -> str:
    """A cell's own pass % over its own problems (the McNemar columns are over the problems both cells have)."""
    return f"{100.0 * sum(p.values()) / len(p):.1f} ({len(p)})" if p else "-"


def fb_txt(st: dict | None, lab: str) -> str:
    if not st:
        return "-"
    x = st[lab]
    return f"{x['n']}" + (f" ({x['passed']} pass, {x['rate']:.1f}%)" if x["n"] else "")


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--bench-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--base", default="V3", help="the plain v3 cell (TEST under it, externals under <base>/ext)")
    ap.add_argument("--ef", default="V3EF", help="the hard gate, TEST (EF_1; also the reference gate record)")
    ap.add_argument("--efs", default="V3EFS", help="the soft gate, TEST (EFS_1)")
    ap.add_argument("--efx", default="V3EFX", help="the hard gate, externals (EF_3)")
    ap.add_argument("--efsx", default="V3EFSX", help="the soft gate, externals (EFS_2)")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where EFS_READING.md/.json go (default: <root>/<efs>, else <root>/<efsx>, "
                                                "if either exists; else nothing is written)")
    a = ap.parse_args(argv)
    root, broot = Path(a.root), Path(a.bench_root)
    sets = A.load_families()
    test_all = A.cell_members(sets, "test")

    def load(name):
        return A.load_config(root, name, sets) if (root / name).is_dir() else None
    base, hard, soft = load(a.base), load(a.ef), load(a.efs)
    miss = {"plain": E.complete(base, sets, test_all), "hard": E.complete(hard, sets, test_all),
            "soft": E.complete(soft, sets, test_all)}
    ref = R2.load_cfg(root / a.ef)
    ref_gate = R2.gate_of(ref) if ref else None
    plain_cfg = R2.load_cfg(root / a.base)
    plain_h = (plain_cfg or {}).get("harness")

    # ------------------------------------------------------------------ externals
    nprob = R2.problem_counts(broot)
    xdir = {"plain": root / a.base / "ext", "hard": root / a.efx / "ext", "soft": root / a.efsx / "ext"}
    recs = {k: {b: R2.load_ext(d / b) for b in BENCHES} for k, d in xdir.items()}
    xcomp = {k: {b: (xdir[k] / b / "summary.json").is_file() and len(recs[k][b]) == nprob[b] for b in BENCHES}
             for k in xdir}
    passed = {k: {b: R2.ext_passed(recs[k][b]) for b in BENCHES} for k in xdir}
    vs_plain = {b: R2.mcnemar(passed["soft"][b], passed["plain"][b]) if recs["soft"][b] and recs["plain"][b] else None
                for b in BENCHES}
    hard_vs_plain = {b: R2.mcnemar(passed["hard"][b], passed["plain"][b]) if recs["hard"][b] and recs["plain"][b] else None
                     for b in BENCHES}
    vs_hard = {b: R2.mcnemar(passed["soft"][b], passed["hard"][b]) if recs["soft"][b] and recs["hard"][b] else None
               for b in BENCHES}

    # ------------------------------------------------------------------ the registered legs (own rng: beside cannot move them)
    rng = np.random.default_rng(a.seed)
    d_all = E.pair(soft, base, sets, a.perms, rng, test_all)
    vals = {"TEST": d_all, **{leg: vs_plain[b] for leg, b in LEG_BENCH.items()}}
    complete = {"TEST": base is not None and soft is not None and not miss["plain"] and not miss["soft"],
                **{leg: xcomp["soft"][b] and xcomp["plain"][b] for leg, b in LEG_BENCH.items()}}
    V = verdict(vals, complete)

    # ------------------------------------------------------------------ beside
    rng_b = np.random.default_rng(a.seed + 1)
    cp = {k: (A.cell_pass(v, sets, "test", test_all)[0] if v is not None else None) for k, v in
          (("plain", base), ("hard", hard), ("soft", soft))}
    ho = {k: (A.cell_pass(v, sets, "test", HO6)[0] if v is not None else None) for k, v in
          (("plain", base), ("hard", hard), ("soft", soft))}
    beside = {"test_all": {**cp, "soft_minus_plain": d_all, "soft_minus_hard": E.pair(soft, hard, sets, a.perms, rng_b, test_all),
                           "hard_minus_plain": E.pair(hard, base, sets, a.perms, rng_b, test_all)},
              "ho6": {**ho, "soft_minus_plain": E.pair(soft, base, sets, a.perms, rng_b, HO6),
                      "soft_minus_hard": E.pair(soft, hard, sets, a.perms, rng_b, HO6),
                      "hard_minus_plain": E.pair(hard, base, sets, a.perms, rng_b, HO6)}}
    cau = R2.qrdata_causal(broot)
    split = {}
    for lab, want in (("causal", True), ("non-causal", False)):
        ids = {k for k, v in cau.items() if v == want}
        split[lab] = {"n_ids": len(ids),
                      "soft_minus_plain": R2.mcnemar(passed["soft"]["qrdata"], passed["plain"]["qrdata"], ids)
                      if recs["soft"]["qrdata"] and recs["plain"]["qrdata"] else None,
                      "soft_minus_hard": R2.mcnemar(passed["soft"]["qrdata"], passed["hard"]["qrdata"], ids)
                      if recs["soft"]["qrdata"] and recs["hard"]["qrdata"] else None}
    fb = {"synthetic": {k: syn_stats(v) for k, v in (("plain", base), ("hard", hard), ("soft", soft))},
          "externals": {b: {k: ext_stats(recs[k][b]) for k in xdir} for b in BENCHES}}
    mech = {"synthetic": {"hard": E.mechanics(hard) if hard is not None else None,
                          "soft": E.mechanics(soft) if soft is not None else None,
                          "plain": E.mechanics(base) if base is not None else None},
            "externals": {b: {"soft": R2.ext_mechanics(recs["soft"][b], recs["plain"][b]),
                              "hard": R2.ext_mechanics(recs["hard"][b], recs["plain"][b])} for b in BENCHES}}

    # ------------------------------------------------------------------ provenance / warnings
    warn = []
    if soft is not None:
        warn += [f"PROVENANCE {a.efs}: {x}" for x in efs_provenance(root, a.efs, sets, ref_gate, plain_h, ext=False)]
    if (root / a.efsx).is_dir():
        warn += [f"PROVENANCE {a.efsx}: {x}" for x in efs_provenance(root, a.efsx, sets, ref_gate, plain_h, ext=True)]
    shas = {}
    for n in (a.efs, a.efsx):
        c = R2.load_cfg(root / n)
        if c:
            shas[n] = (soft_record(c.get("harness") or {}).get("sha256") or {})
    if len(shas) == 2 and shas[a.efs] != shas[a.efsx]:
        warn.append(f"the soft cells were written by different EFS builds: {json.dumps(shas)}")
    if ref_gate is None:
        warn.append(f"{root / a.ef}/config.json missing: the gate record cannot be compared with EF_1's")
    for tag, st in [("synthetic", fb["synthetic"]["soft"])] + [(b, fb["externals"][b]["soft"]) for b in BENCHES]:
        if st and st["no_answer_after_offer"]:
            warn.append(f"{tag}: {st['no_answer_after_offer']} soft-cell rows end with NO answer although the model offered "
                        f"one -- the soft fallback did not fire (not a v4 row?)")
    ms = mech["synthetic"]["soft"] or {}
    if ms.get("n") and ms["with_ef_log"] < ms["n"] - ms["errored"]:
        warn.append(f"{ms['n'] - ms['with_ef_log']} soft TEST rollouts carry no gate log (and only {ms['errored']} errored)")
    if ms.get("gate_errors"):
        warn.append(f"{ms['gate_errors']} gate errors in the soft TEST cell (fail-open)")
    for b in BENCHES:
        m = mech["externals"][b]["soft"]
        if m.get("n") and m["with_ef_log"] < m["n"] - m["ef"]["errored"]:
            warn.append(f"{b}: {m['n'] - m['with_ef_log']} soft problems carry no gate log")
        if m.get("gate_errors"):
            warn.append(f"{b}: {m['gate_errors']} gate errors in the soft cell (fail-open)")

    # ------------------------------------------------------------------ report
    L = ["# HARNESS v4 -- the soft execution gate (EFS) vs plain v3, the hard gate beside (docs/RL_PLAN.md 2026-09-29 04:40 UTC)\n",
         f"root `{root}`: plain `{a.base}` (+ `{a.base}/ext`), hard `{a.ef}` / `{a.efx}/ext`, soft `{a.efs}` / `{a.efsx}/ext`; "
         f"untrained Qwen3-8B; paired by instance (TEST, one-sided sign-flip, {a.perms} perms) and by problem (externals, exact "
         f"McNemar). Rule: read_efs.py docstring.\n"]
    for n, what in ((a.efs, "EFS_1 (TEST)"), (a.efsx, "EFS_2 (externals)")):
        if not (root / n).is_dir():
            L.append(f"**{root / n} does not exist yet ({what}) -- PENDING.**")
    for k, v in miss.items():
        if v and not (k == "soft" and soft is None) and not (k == "hard" and hard is None):
            L.append(f"* TEST {k}: {len(v)} families missing or short: {', '.join(v)}")
    for k in xdir:
        part = [f"{b} {len(recs[k][b])}/{nprob[b]}" for b in BENCHES if not xcomp[k][b] and recs[k][b]]
        if part:
            L.append(f"* externals {k}: PARTIAL -- " + ", ".join(part))
    L.append("\n## Registered reading (SOFT-GATE-ADOPT)\n")
    L.append("| leg | rule | plain | soft (v4) | soft - plain | complete | holds |")
    L.append("|---|---|---|---|---|---|---|")
    rule_txt = {"TEST": f"TEST-all >= plain + {TEST_BAR:.0f}, sign-flip p>0 <= {TEST_P}",
                "QRDATA": f"QRData >= plain + {QR_BAR:.0f}, McNemar p>0 <= {QR_P}",
                "DABENCH": f"DABench not down > {DOWN_BAR:.0f}", "BIXBENCH": f"BixBench not down > {DOWN_BAR:.0f}"}
    for leg in LEGS:
        x, lg = vals[leg], V["legs"][leg]
        if leg == "TEST":
            pl, so = fmt(cp["plain"]), fmt(cp["soft"])
            dd = ptxt(x)
        else:
            pl, so = (fmt(x["base"]), fmt(x["ef"])) if x else ("-", "-")
            dd = mtxt(x)
        n = ("" if leg == "TEST" else (f" (n {x['n']})" if x else ""))
        holds = {True: "yes", False: "**NO**", None: "-"}[lg["ok"]]
        L.append(f"| {leg} | {rule_txt[leg]} | {pl} | {so}{n} | {dd} | {'yes' if lg['complete'] else 'NO'} | {holds} |")
    L.append(f"\n**{V['status']}**" + ("" if V["final"] or V["label"] == "INCOMPLETE"
                                         else f" (provisional on the rows present: {V['label']})"))
    if V["status"] == "NOT-ADOPT" and V["pending"]:
        L.append(f"* decided by the complete failing leg(s) {', '.join(V['failed_complete'])}; still pending: "
                 f"{', '.join(V['pending'])}")
    for leg in LEGS:
        lg = V["legs"][leg]
        if lg["ok"] is False:
            L.append(f"* FAILING LEG {leg}{'' if lg['complete'] else ' (PROVISIONAL: rows incomplete)'}: {lg['why']}")
        elif lg["ok"] is True and not V["final"]:
            L.append(f"* {leg} holds on the rows present{'' if lg['complete'] else ' (PROVISIONAL)'}: {lg['why']}")
    if V["label"] == "SOFT-GATE-ADOPT" and V["final"]:
        L.append("* ADOPT: v4 becomes the default evaluation harness for new reads (old reads keep their harness); RL inside "
                 "v4 is registered together with EF_4's reading (the 04:40 entry).")

    L.append("\n## Beside: HO6 / TEST-all, soft vs plain and vs the hard gate (never deciding)\n")
    L.append("| set | plain | hard | soft | soft - plain | soft - hard | hard - plain |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, key in (("TEST-all (twin-weighted)", "test_all"), ("HO6", "ho6")):
        b = beside[key]
        L.append(f"| {lab} | {fmt(b['plain'])} | {fmt(b['hard'])} | {fmt(b['soft'])} | {ptxt(b['soft_minus_plain'])} | "
                 f"{ptxt(b['soft_minus_hard'])} | {ptxt(b['hard_minus_plain'])} |")
    L.append("\n## Beside: the externals, soft vs the hard gate (never deciding)\n")
    L.append("| benchmark | plain (n) | hard (n) | soft (n) | soft - hard (McNemar) | hard - plain (McNemar) |")
    L.append("|---|---|---|---|---|---|")
    for b in BENCHES:
        L.append(f"| {b} | " + " | ".join(rate_txt(passed[k][b]) for k in ("plain", "hard", "soft"))
                 + f" | {mtxt(vs_hard[b])} | {mtxt(hard_vs_plain[b])} |")
    L.append("\n| QRData subset | n | soft - plain (McNemar) | soft - hard (McNemar) |")
    L.append("|---|---|---|---|")
    for lab, s in split.items():
        L.append(f"| {lab} ({s['n_ids']}) | {s['soft_minus_plain']['n'] if s['soft_minus_plain'] else '-'} | "
                 f"{mtxt(s['soft_minus_plain'])} | {mtxt(s['soft_minus_hard'])} |")

    L.append("\n## Beside: no answer, and the fallbacks (never deciding)\n")
    L.append("| cell | rows | no answer | no answer although an answer was offered | grounded fallback (pass) | "
             "SOFT fallback (pass) |")
    L.append("|---|---|---|---|---|---|")
    rows_fb = [(f"TEST {k}", fb["synthetic"][k]) for k in ("plain", "hard", "soft")] \
        + [(f"{b} {k}", fb["externals"][b][k]) for b in BENCHES for k in ("plain", "hard", "soft")]
    for lab, st in rows_fb:
        if not st:
            L.append(f"| {lab} | - | - | - | - | - |")
            continue
        L.append(f"| {lab} | {st['n']} | {st['no_answer']} | {st['no_answer_after_offer']} | {fb_txt(st, GROUNDED)} | "
                 f"{fb_txt(st, SOFT)} |")

    L.append("\n## Per TEST family (pass %, 12x2)\n")
    L.append("| family | plain | hard | soft | soft - plain (p>0) | soft - hard | soft fallbacks (pass) | no answer hard / soft |")
    L.append("|---|---|---|---|---|---|---|---|")
    per = {}
    for f in test_all:
        b, h, s = E.fam(base, sets, f), E.fam(hard, sets, f), E.fam(soft, sets, f)
        d = E.pair(soft, base, sets, a.perms, rng_b, [f])
        dh = E.pair(soft, hard, sets, a.perms, rng_b, [f])
        st, sh = syn_stats(soft, [f]), syn_stats(hard, [f])
        per[f] = {"plain": b, "hard": h, "soft": s, "soft_minus_plain": d, "soft_minus_hard": dh,
                  "soft_fallback": (st or {}).get(SOFT), "no_answer": {"hard": (sh or {}).get("no_answer"),
                                                                       "soft": (st or {}).get("no_answer")}}
        L.append(f"| {f}{' (HO6)' if f in HO6 else ''} | {fmt(b)} | {fmt(h)} | {fmt(s)} | "
                 + (f"{d['d']:+.1f} ({d['p_gt']:.3f})" if d else "-") + " | " + (f"{dh['d']:+.1f}" if dh else "-") + " | "
                 + fb_txt(st, SOFT) + f" | {per[f]['no_answer']['hard'] if sh else '-'} / {per[f]['no_answer']['soft'] if st else '-'} |")

    L.append("\n## Gate mechanics\n")
    for k in ("hard", "soft"):
        m, mb = mech["synthetic"][k] or {}, mech["synthetic"]["plain"] or {}
        if m.get("n"):
            L.append(f"* TEST {k}: rollouts {m['n']} (gate log {m['with_ef_log']}); refusals {m['rejected_total']} "
                     f"({fmt(m['rejected_mean'], 2)} / rollout; {m['rollouts_with_rejection']} rollouts refused >= 1); accepted "
                     f"as {json.dumps(m['accepted'])}; no answer {m['no_answer']} (plain {mb.get('no_answer', '-')}); cells / "
                     f"rollout {fmt(m['cells_mean'], 2)} (plain {fmt(mb.get('cells_mean'), 2)})")
        else:
            L.append(f"* TEST {k}: (no rollouts yet)")
    for b in BENCHES:
        for k in ("hard", "soft"):
            m = mech["externals"][b][k]
            if m.get("n"):
                L.append(f"* {b} {k}: problems {m['n']}; offers {m['ext_offers']}, rule applied to {m['gated_offers']}; "
                         f"problems gated {m['problems_gated']}; refusals {m['rejected_total']}; accepted as "
                         f"{json.dumps(m['accepted'])}")
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(2, f"**Status: {V['status']}**" + ("" if V["final"] or V["label"] == "INCOMPLETE"
                                                 else f" (provisional: {V['label']})") + "\n")
    obj = {"root": str(root), "cells": {"plain": a.base, "hard": a.ef, "soft": a.efs, "hard_ext": a.efx, "soft_ext": a.efsx},
           "status": V["status"], "label": V["label"], "final": V["final"], "legs": V["legs"], "failed": V["failed"],
           "failed_complete": V["failed_complete"], "pending": V["pending"], "values": vals, "complete": complete,
           "beside": beside, "externals": {b: {"soft_minus_plain": vs_plain[b], "soft_minus_hard": vs_hard[b],
                                               "hard_minus_plain": hard_vs_plain[b],
                                               "completeness": {k: xcomp[k][b] for k in xdir},
                                               "n": {k: len(recs[k][b]) for k in xdir}, "n_problems": nprob[b]}
                                           for b in BENCHES},
           "qrdata_split": split, "fallbacks": fb, "per_family": per, "mechanics": mech, "missing": miss,
           "soft_sha256": shas, "warnings": warn}
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out) if a.out else ((root / a.efs) if (root / a.efs).is_dir()
                                     else ((root / a.efsx) if (root / a.efsx).is_dir() else None))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "EFS_READING.md").write_text(text)
        (out / "EFS_READING.json").write_text(json.dumps(obj, indent=1, default=str))
    print(f"EFS SUMMARY: {V['status']}" + ("" if V["final"] or V["label"] == "INCOMPLETE" else f" (provisional {V['label']})")
          + (f"; failing: {', '.join(V['failed'])}" if V["failed"] else "") + (f"; pending: {', '.join(V['pending'])}"
                                                                               if V["pending"] else "")
          + (f"; written {out / 'EFS_READING.md'}" if out is not None else "; nothing written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
