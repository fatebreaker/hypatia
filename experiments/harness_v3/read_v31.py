#!/usr/bin/env python
"""HARNESS V3.1 -- ELICIT-THEN-REWARD reading (docs/RL_PLAN.md s.18: registered 2026-09-26 16:35 UTC; prompt, run names and
this operationalisation registered by the builder's entry beneath it, before any run). Qwen3-8B on causal_rdd, one seed,
150 steps, the s.16 recipe with the system prompt science_check (rl/systems_extra.py: science + one sentence asking for a
robustness re-estimate):
  ELICIT-PR     rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0.3-lr1e-5-s1   (process reward, lambda 0.3)  HV3_54
  ELICIT-PLAIN  rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0-lr1e-5-s1     (answer-only, same prompt)     HV3_55
  ("sysscicheck2": both arms train with process DETECTOR v2, rl/process_signals_v31.py -- spec := reest, M3 check :=
   max(M3, reest); registered in the s.18 entry below 2026-09-26 19:15 UTC. No run under the old names ever started.)
  PR (s.16)     rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s1    (no instruction; HV3_18)
Rows: the v3.1 arms are scored by screen_v31.sh under science_check -> results/pods/harness_v31/V31__<run>_s<step>/, beside
the untrained 8B under the same prompt (V31/, HV3_62: the PRIMARY base). The s.16 PR arm and the s.16 base (V3/, HV3_17:
the SECONDARY base) are the science-prompt rows under results/pods/harness_v3/. Instances are the same draws in both roots.

    venv/bin/python experiments/harness_v3/read_v31.py [--root results/pods/harness_v31] [--v3-root results/pods/harness_v3]
        [--ckpt-root $DEEP_DATA_ROOT/ckpt] [--perms 20000] [--no-process] [--out DIR]

A NEW FILE: read_v3.py / read_v3_q8.py are imported for their helpers (ho6, contrast, cells_mean, process_on_eval, cost,
pair, ho6_detail, fmt) and never edited. Runs on partial rows (names what is missing; the verdict is then INCOMPLETE).

MEASURES per arm x step (50/100/150): TRAIN (causal_rdd DESIGN), HO6, TEST-all (17), TEST judgement / procedure, each with
its paired delta vs V31 (primary) and vs V3 (secondary); HO6 cells/rollout; the process score on the HO6 eval rollouts.

MECHANISM READOUT (training, <ckpt>/<run>/process.jsonl; the s.16 PR arm beside): firing rates of inspect / check / spec /
grounded as LOGGED (mean over the rollouts where the signal applies), reest, the v3 detectors' spec / check, the old
re-est proxy, pass and cells, by step block 1-10, 20-30, 45-55, 90-100, 140-150. The v3.1 arms log with DETECTOR v2
(rl/process_signals_v31.py): their logged spec IS reest and their logged check has M3 := max(M3, reest); the v3
detectors' own spec / check ride along in the spec detail ("[v3 spec x: ... | v3 check y]") and are printed beside. The
s.16 PR arm was logged by the v3 scorer (no reest; its training transcripts are not stored).
READABLE GATE (ELICIT-PR; AMENDED 2026-09-26 20:03 UTC, s.18 entry below 19:15, before any v3.1 run): reest pooled over
the rollouts of steps 41-50 >= 0.3 -> READABLE; else UNREADABLE (not null: the prompt is revised once and re-run);
PENDING until step 50 is logged; UNREADABLE also when the log carries no detector-v2 rows. (The registered gate -- check
AND spec >= 0.3 under the v3 detectors -- read 0.00 on the elicited step-0 transcripts: it could not see the behaviour.)
The same step-0 rates of the untrained 8B on causal_rdd under each prompt (V31 vs V3 DESIGN rollouts) are printed: what
the sentence elicits before any RL. BESIDE (never deciding): the v3 spec / check, and "re-est proxy" = spec fired or the v3
'where' counts >= 2 estimates -- an UPPER BOUND that also counts unchanged re-runs and two fits in one cell (the s.16 PR
arm: 0.22 over steps 41-50 with v3 spec / check 0.006; reest on its step-50 causal_rdd eval rollouts: 0.00).

TRANSFER (paired by instance, analyze.py's sign-flip, one-sided p; M in {HO6, TEST-all}; steps 100 and 150 decide, 50 is
reported; a contrast counts only when both cells have all of M's families):
  dPR(s, M) = M(ELICIT-PR, s) - M(PR, s)        dEP(s, M) = M(ELICIT-PR, s) - M(ELICIT-PLAIN, s)
  dIO(s, M) = M(ELICIT-PLAIN, s) - M(PR, s)
  ELICIT-WINS       some s in {100, 150} and M: dPR >= +5 with p <= 0.05 AND dEP >= +3 (same s, same M): the reward, not
                    the instruction, carries it
  INSTRUCTION-ONLY  some s in {100, 150} and M: dIO >= +5
  KILL              dPR <= +2 at both 100 and 150 on HO6 AND on TEST-all
  NONE              none of the above
  The verdict line is the first of ELICIT-WINS, INSTRUCTION-ONLY, KILL, NONE; every label that fires is listed. An
  UNREADABLE gate heads the verdict (the transfer rows are then reported, not read); INCOMPLETE until the six cells
  (ELICIT-PR, ELICIT-PLAIN, PR at 100 and 150) have all 17 TEST families.
  CAVEAT (printed): each arm is scored under its OWN training prompt (screen_any.sh's rule), so dPR carries the prompt at
  evaluation as well as the training difference. BESIDE (never deciding): V31 - V3 (the eval-prompt effect on the
  untrained 8B) and the difference-in-differences (ELICIT-PR - V31) - (PR - V3).
COST GUARD at 150 (beside): an external down >= 5 points vs V31 with exact McNemar p < 0.05; DABench code cells per
problem (x the base's); QRData NON-CAUSAL (and causal) paired delta, keywords from each problem's _provenance -- vs V31
and vs V3.
AUDIT (warnings): the system message stored in each row's rollouts must be the registered text of its prompt (V31 rows
science_check, V3 rows science); errored rollouts are counted per row -- a v3.1 checkpoint scored outside screen_v31.sh
is not a crash but every rollout error:KeyError 'science_check' with 0 cells (a silent 0%), named here.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_v3 as R  # noqa: E402  (puts ROOT, harness/ and experiments/harness_v2 on sys.path)
import read_v3_q8 as Q  # noqa: E402

A = R.A
HO6 = R.HO6
FAM = "causal_rdd"
RUNS = {"elicitpr": "rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0.3-lr1e-5-s1",
        "elicitplain": "rl-czero-causal_rdd-base-echo-sysscicheck2-rc-hv3pr0-lr1e-5-s1"}
OLD_RUNS = {"elicitpr": "rl-czero-causal_rdd-base-echo-sysscicheck-rc-hv3pr0.3-lr1e-5-s1",     # detector v1: never run
            "elicitplain": "rl-czero-causal_rdd-base-echo-sysscicheck-rc-hv3pr0-lr1e-5-s1"}
PR_RUN = Q.RUNS["pr"]
ARMS = ("elicitpr", "elicitplain", "pr")
LABEL = {"elicitpr": "ELICIT-PR", "elicitplain": "ELICIT-PLAIN", "pr": "PR (s.16)"}
BASE31, BASE3 = "V31", "V3"
SYSTEM31, SYSTEM3 = "science_check", "science"
READ_STEPS = (50, 100, 150)
DECIDE_STEPS = (100, 150)
BLOCKS = ((1, 10), (20, 30), (45, 55), (90, 100), (140, 150))
GATE_STEPS, GATE_MIN, GATE_SIGNALS = (41, 50), 0.3, ("reest",)
MECH_SIGNALS = ("inspect", "check", "spec", "grounded")
BESIDE = ("reest", "v3_spec", "v3_check", "proxy")
WIN_BAR, WIN_P, OVER_PLAIN, IO_BAR, KILL_BAR = 5.0, 0.05, 3.0, 5.0, 2.0
COST_PTS, COST_P = 5.0, 0.05
METRICS = ("HO6", "TEST-all")
MODEL = "Qwen3-8B"


# ----------------------------------------------------------------------------------------------------------- mechanism
def load_process(ck: Path) -> list:
    f = ck / "process.jsonl"
    if not f.is_file():
        return []
    out = []
    for x in f.read_text().splitlines():
        x = x.strip()
        if x:
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                continue      # a line being written while we read
    return out


def logged(r: dict) -> dict:
    """One process.jsonl row read under either detector: {'v2', 'reest', 'v3_spec', 'v3_check', 'v3_est'}."""
    from rl.process_signals_v31 import parse_logged
    lg = parse_logged((r.get("where") or {}).get("spec"))
    if not lg["v2"]:                                          # a v3-scorer row: its logged values ARE the v3 ones
        f = r.get("fired") or {}
        lg["v3_spec"], lg["v3_check"] = f.get("spec"), f.get("check")
    return lg


def reest(r: dict):
    """The detector-v2 signal of one logged rollout (1 / 0), or None when the row was not scored by detector v2 (the s.16
    arms, any v3-scorer log) or has no executed cell."""
    lg = logged(r)
    return None if not lg["v2"] or lg["reest"] is None else int(lg["reest"] >= 1)


def reest_proxy(r: dict):
    """The OLD detector-independent upper bound (beside, never deciding): 1 if the v3 spec fired or its 'where' counts
    >= 2 estimates of some response, 0 if fewer, None if unknown. It also counts an unchanged re-run and two fits in one
    cell -- which is why it cannot be the gate (19:15 entry; s.16 PR arm 0.22 with no re-estimate at all)."""
    lg = logged(r)
    if lg["v3_spec"] == 1:
        return 1
    return None if lg["v3_est"] is None else int(lg["v3_est"] >= 2)


def rates(rows: list, lo: int, hi: int) -> dict:
    """Signal firing rates over the rollouts of steps lo..hi (mean over the rollouts where the signal applies)."""
    sel = [r for r in rows if lo <= int(r.get("step", -1)) <= hi]
    out = {"n": len(sel), "steps": sorted({int(r["step"]) for r in sel})}
    for s in MECH_SIGNALS:
        v = [float(r["fired"][s]) for r in sel if (r.get("fired") or {}).get(s) is not None]
        out[s] = float(np.mean(v)) if v else None
        out[f"n_{s}"] = len(v)
    lgs = [logged(r) for r in sel]
    for k, fn in (("reest", reest), ("proxy", reest_proxy)):
        v = [x for x in (fn(r) for r in sel) if x is not None]
        out[k] = float(np.mean(v)) if v else None
        out[f"n_{k}"] = len(v)
    for k in ("v3_spec", "v3_check"):
        v = [x[k] for x in lgs if x[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    out["pass"] = float(np.mean([bool(r.get("passed")) for r in sel])) if sel else None
    out["cells"] = float(np.mean([r.get("cells") or 0 for r in sel])) if sel else None
    out["p"] = float(np.mean([r.get("p") or 0.0 for r in sel])) if sel else None
    return out


def gate(rows: list) -> dict:
    last = max((int(r.get("step", -1)) for r in rows), default=-1)
    g = rates(rows, *GATE_STEPS)
    vals = {s: g.get(s) for s in GATE_SIGNALS}
    if last < GATE_STEPS[1]:
        status = "PENDING"
    elif all(v is not None and v >= GATE_MIN for v in vals.values()):
        status = "READABLE"
    else:
        status = "UNREADABLE"
    return {"status": status, "last_step": last, "rates": vals, "n": g["n"], "n_reest": g.get("n_reest"),
            "reest": g.get("reest"), "v3": {"check": g.get("v3_check"), "spec": g.get("v3_spec")},
            "proxy": g.get("proxy"), "no_v2_rows": bool(g["n"]) and not g.get("n_reest")}


def mechanism(ck_root: Path) -> dict:
    out = {}
    for arm in ARMS:
        run = PR_RUN if arm == "pr" else RUNS[arm]
        rows = load_process(ck_root / run)
        out[arm] = {"run": run, "n": len(rows), "last_step": max((int(r.get("step", -1)) for r in rows), default=None),
                    "blocks": {f"{lo}-{hi}": rates(rows, lo, hi) for lo, hi in BLOCKS},
                    "gate": gate(rows) if rows else {"status": "PENDING", "last_step": None, "rates": {}, "n": 0}}
    return out


def process_rates_eval(root: Path, cfgname: str, split: str, fams) -> dict | None:
    """Process rates on stored EVAL rollouts of one configuration, scored by detector v2 (rl/process_signals_v31.py):
    the signals (spec = reest), plus the v3 detectors' spec / check beside (read_v3.process_on_eval for any set)."""
    from paths import data_root
    from process_signals import SIGNALS
    from rl.process_signals_v31 import ProcessScorerV31
    G = data_root() / "generated"
    ps = ProcessScorerV31()
    vals, scores = {k: [] for k in SIGNALS + ("reest", "v3_spec", "v3_check")}, []
    for f in fams:
        p = root / cfgname / split / f"{f}.json"
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
                vals["reest"].append(s.fired["spec"])
                for k in ("spec", "check"):
                    if s.detail["v3"][k] is not None:
                        vals["v3_" + k].append(s.detail["v3"][k])
    if not scores:
        return None
    out = {"p": float(np.mean(scores)), "n": len(scores)}
    out.update({k: (float(np.mean(v)) if v else None) for k, v in vals.items()})
    return out


def system_of_rows(root: Path, cfgname: str) -> str | None:
    """The system message actually stored in this configuration's rollouts (first rollout found), or None."""
    d = root / cfgname
    for sub in ("design", "test"):
        for p in sorted((d / sub).glob("*.json")) if (d / sub).is_dir() else []:
            try:
                g = json.loads(p.read_text()).get("groups") or {}
            except (OSError, json.JSONDecodeError):
                continue
            for rs in g.values():
                for r in rs:
                    m = r.get("messages") or []
                    if m and m[0].get("role") == "system":
                        return m[0].get("content")
    return None


# ----------------------------------------------------------------------------------------------------------- externals
def qrdata_causal_map() -> dict:
    cau = {}
    for cfg in sorted((ROOT / "benchmarks" / "qrdata" / "problems").glob("qr-*/eval_config.json")):
        ks = json.loads(cfg.read_text()).get("_provenance", {}).get("keywords") or []
        if isinstance(ks, str):
            try:
                ks = ast.literal_eval(ks)
            except (ValueError, SyntaxError):
                ks = [ks]
        cau[cfg.parent.name] = any("causal" in str(x).lower() for x in ks)
    return cau


def ext_cells(res, bench) -> float | None:
    d = (res or {}).get("ext_dir", {}).get(bench)
    if d is None or not (d / "summary.json").is_file():
        return None
    pp = json.loads((d / "summary.json").read_text()).get("per_problem") or {}
    xs = [v.get("n_cells") or 0 for v in pp.values()]
    return float(np.mean(xs)) if xs else None


def split_delta(res, base, cau: dict, want: bool) -> dict | None:
    a, b = (res or {}).get("ext", {}).get("qrdata"), (base or {}).get("ext", {}).get("qrdata")
    if not a or not b:
        return None
    ka = {k: v for k, v in a.items() if cau.get(k) is want}
    kb = {k: v for k, v in b.items() if cau.get(k) is want}
    g, l, p, n = A.mcnemar(ka, kb)
    if not n:
        return None
    return {"n": n, "delta": 100.0 * (g - l) / n, "gained": g, "lost": l, "p": p}


def guard(res, base, cau) -> dict:
    rows = Q.ext_rows(res, base) if (res and base) else []
    for e in rows:
        e["cost"] = bool(e["delta"] <= -COST_PTS and e["p"] < COST_P)
    dc, bc = ext_cells(res, "dabench"), ext_cells(base, "dabench")
    return {"ext": rows, "dabench_cells": dc, "dabench_cells_base": bc,
            "dabench_cells_x": (dc / bc) if (dc is not None and bc) else None,
            "qrdata_causal": split_delta(res, base, cau, True), "qrdata_noncausal": split_delta(res, base, cau, False)}


# ----------------------------------------------------------------------------------------------------------- transfer
def members(sets, metric: str) -> list:
    return list(HO6) if metric == "HO6" else A.cell_members(sets, "test")


def pair_m(a, b, sets, perms, rng, metric):
    return Q.pair(a, b, sets, perms, rng, members=members(sets, metric))


def decide(c: dict, gate_status: str) -> tuple[str, list, list]:
    """c[key][(s, M)] = {'d', 'p_gt', 'complete'} or None for key in dPR, dEP, dIO."""
    need = [(k, s, m) for k in ("dPR", "dEP", "dIO") for s in DECIDE_STEPS for m in METRICS]
    miss = [f"{k}({s},{m})" for k, s, m in need if not (c[k].get((s, m)) or {}).get("complete")]
    why = []
    if gate_status != "READABLE":
        why.append(f"READABLE gate: {gate_status}" + (" -- the arm is UNREADABLE, not null: revise the prompt once and "
                                                     "re-run; the transfer rows are reported, not read"
                                                     if gate_status == "UNREADABLE" else ""))
    if miss:
        return ("UNREADABLE" if gate_status == "UNREADABLE" else "INCOMPLETE"), [], \
            why + [f"not yet computable: {', '.join(miss)}"]
    g = lambda k, s, m: c[k][(s, m)]  # noqa: E731
    why.append("; ".join(f"{m}: dPR " + ", ".join(f"s{s} {g('dPR', s, m)['d']:+.1f} (p={g('dPR', s, m)['p_gt']:.3f})"
                                                  for s in DECIDE_STEPS)
                         + " | dEP " + ", ".join(f"s{s} {g('dEP', s, m)['d']:+.1f}" for s in DECIDE_STEPS)
                         + " | dIO " + ", ".join(f"s{s} {g('dIO', s, m)['d']:+.1f}" for s in DECIDE_STEPS)
                         for m in METRICS))
    wins = [(s, m) for s in DECIDE_STEPS for m in METRICS
            if g("dPR", s, m)["d"] >= WIN_BAR and g("dPR", s, m)["p_gt"] <= WIN_P and g("dEP", s, m)["d"] >= OVER_PLAIN]
    io = [(s, m) for s in DECIDE_STEPS for m in METRICS if g("dIO", s, m)["d"] >= IO_BAR]
    kill = all(g("dPR", s, m)["d"] <= KILL_BAR for s in DECIDE_STEPS for m in METRICS)
    labels = []
    if wins:
        labels.append("ELICIT-WINS (" + ", ".join(f"s{s} {m}" for s, m in wins) + ")")
    if io:
        labels.append("INSTRUCTION-ONLY (" + ", ".join(f"s{s} {m}" for s, m in io) + ")")
    if kill:
        labels.append("KILL")
    if gate_status == "UNREADABLE":
        return "UNREADABLE", labels, why
    if wins:
        return "ELICIT-WINS", labels, why + [f"dPR >= +{WIN_BAR:.0f} (p <= {WIN_P}) and dEP >= +{OVER_PLAIN:.0f} at "
                                             + ", ".join(f"s{s} {m}" for s, m in wins)]
    if io:
        return "INSTRUCTION-ONLY", labels, why + [f"ELICIT-PLAIN >= PR + {IO_BAR:.0f} at "
                                                  + ", ".join(f"s{s} {m}" for s, m in io)]
    if kill:
        return "KILL", labels, why + [f"ELICIT-PR <= PR + {KILL_BAR:.0f} at 100 and 150 on HO6 and TEST-all"]
    return "NONE", labels, why + ["between the bars: reported, not replicated"]


# ----------------------------------------------------------------------------------------------------------- main
def fmt(x, nd=1, sign=False):
    return Q.fmt(x, nd, sign)


def dstr(r):
    if r is None:
        return "-"
    s = f"{r['d']:+.1f} (p={r['p_gt']:.3f})"
    return s if r["complete"] else s + f" PARTIAL {r['n_fams']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v31"))
    ap.add_argument("--v3-root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-process", action="store_true", help="skip the process readouts on stored EVAL rollouts")
    ap.add_argument("--out", default=None, help="where V31_READING.md/.json go (default: --root, if it exists)")
    a = ap.parse_args(argv)
    root, root3 = Path(a.root), Path(a.v3_root)
    if a.ckpt_root:
        ck_root = Path(a.ckpt_root)
    else:
        from paths import data_root
        ck_root = data_root() / "ckpt"
    sets = A.load_families()
    rng = np.random.default_rng(a.seed)
    cau = qrdata_causal_map()
    warn = []

    def load(r: Path, name: str):
        if not (r / name).is_dir():
            return None
        cj = r / name / "config.json"
        m = json.loads(cj.read_text()).get("model") if cj.is_file() else None
        if m not in (None, MODEL):
            warn.append(f"{r / name}: config.json says model {m}, not {MODEL}")
        res = A.load_config(r, name, sets)
        # a v3.1 checkpoint scored OUTSIDE screen_v31.sh (no hook) does not crash: rl/dispersion.one_rollout records
        # every rollout as error:KeyError ('science_check') with 0 cells -- a silent 0%. Flag any errored rows.
        errs = [str(x.get("attribution")) + " " + str(x.get("error_msg") or "")
                for rs in res["rollouts"].values() for x in rs if str(x.get("attribution") or "").startswith("error")]
        if errs:
            key = sum("KeyError" in e and "science_check" in e for e in errs)
            warn.append(f"{name}: {len(errs)} errored rollouts" + (f", {key} of them KeyError 'science_check' -- "
                        "scored without the v3.1 hook: rescore through experiments/harness_v3/screen_v31.sh"
                        if key else ""))
        return res

    bases = {BASE31: load(root, BASE31), BASE3: load(root3, BASE3)}
    cells = {}
    for arm in ARMS:
        for s in READ_STEPS:
            if arm == "pr":
                cells[(arm, s)] = load(root3, f"{BASE3}__{PR_RUN}_s{s}")
            else:
                cells[(arm, s)] = load(root, f"{BASE31}__{RUNS[arm]}_s{s}")
    # the prompt actually stored in each root's rollouts (the audit that the V31 rows really ran under science_check)
    from rl import systems_extra as SX
    agents = SX.register()                     # the harness's own agents module, with the v3.1 keys added
    expected = {SYSTEM31: agents.SYSTEMS[SYSTEM31], SYSTEM3: agents.SYSTEMS[SYSTEM3]}
    prompt_check = {}
    for (r, name, key) in ([(root, BASE31, SYSTEM31), (root3, BASE3, SYSTEM3)]
                           + [(root, f"{BASE31}__{RUNS[arm]}_s{s}", SYSTEM31) for arm in RUNS for s in READ_STEPS]
                           + [(root3, f"{BASE3}__{PR_RUN}_s{s}", SYSTEM3) for s in READ_STEPS]):
        if not (r / name).is_dir():
            continue
        got = system_of_rows(r, name)
        ok = (got == expected[key]) if got is not None else None
        prompt_check[name] = {"expected": key, "ok": ok}
        if ok is False:
            warn.append(f"{name}: stored system message is NOT the {key} prompt")
    L = ["# Harness v3.1 ELICIT-THEN-REWARD reading (docs/RL_PLAN.md s.18)\n",
         f"v3.1 root `{root}` (base `{BASE31}` = untrained 8B under {SYSTEM31}: PRIMARY); v3 root `{root3}` (base `{BASE3}` "
         f"under {SYSTEM3}: secondary; the s.16 PR arm). Paired by instance; p one-sided sign-flip ({a.perms} perms). "
         "Rule: read_v31.py docstring / the RL_PLAN registration.\n",
         f"Prompt audit (stored system message == the registered text): "
         + (", ".join(f"{k} {'OK' if v['ok'] else ('MISMATCH' if v['ok'] is False else 'no messages')}"
                      for k, v in prompt_check.items()) or "no rows yet") + "\n"]
    for b, res in bases.items():
        if res is None:
            L.append(f"**no base {b} yet** ({'HV3_62 -> ' + str(root / b) if b == BASE31 else 'HV3_17 -> ' + str(root3 / b)})"
                     " -- its columns are empty.\n")
    # ------------------------------------------------------------------ table
    cls = sorted({v.get("class") for v in sets["test"].values() if v.get("class")})

    def metric(res, name):
        if res is None:
            return None, 0
        if name == "TRAIN":
            return A.cell_pass(res, sets, "design", [FAM])
        if name == "HO6":
            d = Q.ho6_detail(res, sets)
            return d["ho6"], d["n"]
        if name == "TEST-all":
            return A.cell_pass(res, sets, "test", A.cell_members(sets, "test"))
        return A.cell_pass(res, sets, "test", A.cell_members(sets, "test", name))

    def mem(name):
        if name == "TRAIN":
            return "design", [FAM]
        if name == "HO6":
            return "test", list(HO6)
        if name == "TEST-all":
            return "test", A.cell_members(sets, "test")
        return "test", A.cell_members(sets, "test", name)

    def vs(res, base, name):
        if res is None or base is None:
            return None
        s, m = mem(name)
        d = A.paired(res, base, sets, s, m, a.perms, rng)
        return None if d is None else {"d": d[0], "p_gt": d[2], "p_lt": d[3], "n_fams": d[6]}

    names = ["TRAIN", "HO6", "TEST-all"] + cls
    summary: dict = {"root": str(root), "v3_root": str(root3), "runs": {**RUNS, "pr": PR_RUN}, "warnings": warn,
                     "prompt_check": prompt_check, "bases": {}, "cells": {}}
    for b, res in bases.items():
        summary["bases"][b] = {n: metric(res, n)[0] for n in names} if res is not None else None
        if res is not None:
            summary["bases"][b]["cells"] = R.cells_mean(res)
    L.append("Bases: " + "; ".join(
        f"{b} " + (", ".join(f"{n} {fmt(summary['bases'][b][n])}" for n in names) if summary["bases"][b] else "-")
        for b in bases))
    if bases[BASE31] is not None and bases[BASE3] is not None:
        eff = {n: vs(bases[BASE31], bases[BASE3], n) for n in ("TRAIN", "HO6", "TEST-all")}
        summary["prompt_effect_base"] = eff
        L.append("Eval-prompt effect on the untrained 8B, V31 - V3 (beside, never deciding): "
                 + ", ".join(f"{n} {dstr(dict(v, complete=True)) if v else '-'}" for n, v in eff.items()))
    L.append("")
    L.append("| arm | step | " + " | ".join(f"{n} (d V31; d V3)" for n in names) + " | HO6 cells (x V31) | process p HO6 |")
    L.append("|---|---|" + "---|" * (len(names) + 2))
    for arm in ARMS:
        for s in READ_STEPS:
            res = cells[(arm, s)]
            if res is None:
                continue
            row = {"metrics": {}}
            parts = []
            for n in names:
                v, k = metric(res, n)
                d31, d3 = vs(res, bases[BASE31], n), vs(res, bases[BASE3], n)
                row["metrics"][n] = {"value": v, "n_fams": k, "d_v31": d31, "d_v3": d3}
                parts.append(f"{fmt(v)} [{k}] ({fmt(d31 and d31['d'], sign=True)}; {fmt(d3 and d3['d'], sign=True)})")
            cm = R.cells_mean(res)
            bc = summary["bases"][BASE31]["cells"] if summary["bases"].get(BASE31) else None
            proc = None
            if not a.no_process:
                proc = R.process_on_eval(root3 if arm == "pr" else root,
                                         f"{BASE3}__{PR_RUN}_s{s}" if arm == "pr" else f"{BASE31}__{RUNS[arm]}_s{s}")
            row.update(cells=cm, cells_x_v31=(cm / bc) if (cm and bc) else None, process=proc,
                       missing_ho6=Q.ho6_detail(res, sets)["missing"])
            summary["cells"].setdefault(arm, {})[s] = row
            L.append(f"| {LABEL[arm]} | {s} | " + " | ".join(parts)
                     + f" | {fmt(cm, 2)} ({fmt(cm / bc, 2) + 'x' if (cm and bc) else '-'}) | "
                     + f"{fmt(proc['p'], 3) if proc else '-'} |")
    # ------------------------------------------------------------------ mechanism
    mech = mechanism(ck_root)
    summary["mechanism"] = mech
    L.append("\n## Mechanism readout (training process.jsonl; rate over rollouts where the signal applies)\n")
    L.append("| arm | block | n | " + " | ".join(f"{x} (logged)" for x in MECH_SIGNALS)
             + " | reest | v3 spec (beside) | v3 check (beside) | re-est proxy (beside) | pass | cells | p |")
    L.append("|---|---|---|" + "---|" * (len(MECH_SIGNALS) + 7))
    for arm in ARMS:
        m = mech[arm]
        if not m["n"]:
            L.append(f"| {LABEL[arm]} | - | 0 | " + " | ".join("-" for _ in MECH_SIGNALS)
                     + f" | - | - | - | - | - | - | - |  (no process log under {ck_root / m['run']})")
            continue
        for blk, r in m["blocks"].items():
            L.append(f"| {LABEL[arm]} | {blk} | {r['n']} | " + " | ".join(fmt(r[x], 2) for x in MECH_SIGNALS)
                     + " | " + " | ".join(fmt(r[x], 2) for x in BESIDE)
                     + f" | {fmt(r['pass'], 2)} | {fmt(r['cells'], 2)} | {fmt(r['p'], 3)} |")
    gt = mech["elicitpr"]["gate"]
    summary["gate"] = gt
    L.append(f"\nREADABLE GATE (ELICIT-PR; reest >= {GATE_MIN} pooled over steps {GATE_STEPS[0]}-{GATE_STEPS[1]}, "
             f"detector v2; amended before any v3.1 run): **{gt['status']}** (last logged step {gt['last_step']}; "
             + ", ".join(f"{k} {fmt(v, 3)}" for k, v in (gt.get("rates") or {}).items())
             + f"; n={gt['n']}, with a detector-v2 value {gt.get('n_reest')})"
             + (" -- the log has NO detector-v2 rows: not a v3.1 detector-v2 run" if gt.get("no_v2_rows") else "")
             + f"; beside: v3 check {fmt((gt.get('v3') or {}).get('check'), 3)}, v3 spec "
             f"{fmt((gt.get('v3') or {}).get('spec'), 3)}, re-est proxy (an upper bound) {fmt(gt.get('proxy'), 3)}; the "
             f"s.16 PR arm, same window: v3 check {fmt((mech['pr']['gate'].get('v3') or {}).get('check'), 3)}, v3 spec "
             f"{fmt((mech['pr']['gate'].get('v3') or {}).get('spec'), 3)}, proxy {fmt(mech['pr']['gate'].get('proxy'), 3)}")
    gp = mech["elicitplain"]["gate"]
    L.append(f"ELICIT-PLAIN (the prompt without the reward), same window: "
             + ", ".join(f"{k} {fmt(v, 3)}" for k, v in (gp.get("rates") or {}).items()) + f" ({gp['status']} by the "
             "same bar; beside only)")
    if not a.no_process:
        step0 = {b: process_rates_eval(r, b, "design", [FAM]) for b, r in ((BASE31, root), (BASE3, root3))
                 if bases[b] is not None}
        summary["step0_design"] = step0
        L.append("Step 0 (untrained 8B, causal_rdd DESIGN eval rollouts, detector v2): " + ("; ".join(
            f"{b} " + ", ".join(f"{k} {fmt(v.get(k), 2)}" for k in ("inspect", "reest", "check", "v3_spec", "v3_check",
                                                                     "grounded")) + f" (n={v['n']})"
            for b, v in step0.items() if v) or "-"))
    # ------------------------------------------------------------------ contrasts
    c = {"dPR": {}, "dEP": {}, "dIO": {}}
    for s in READ_STEPS:
        for m in METRICS:
            c["dPR"][(s, m)] = pair_m(cells[("elicitpr", s)], cells[("pr", s)], sets, a.perms, rng, m)
            c["dEP"][(s, m)] = pair_m(cells[("elicitpr", s)], cells[("elicitplain", s)], sets, a.perms, rng, m)
            c["dIO"][(s, m)] = pair_m(cells[("elicitplain", s)], cells[("pr", s)], sets, a.perms, rng, m)
    L.append("\n## Contrasts (paired by instance)\n")
    L.append("| step | metric | dPR = ELICIT-PR - PR | dEP = ELICIT-PR - ELICIT-PLAIN | dIO = ELICIT-PLAIN - PR | "
             "DiD (ELICIT-PR - V31) - (PR - V3) |")
    L.append("|---|---|---|---|---|---|")
    did = {}
    for s in READ_STEPS:
        for m in METRICS:
            e, p = cells[("elicitpr", s)], cells[("pr", s)]
            x = vs(e, bases[BASE31], m) if e is not None else None
            y = vs(p, bases[BASE3], m) if p is not None else None
            did[(s, m)] = (x["d"] - y["d"]) if (x and y) else None
            L.append(f"| {s}{'' if s in DECIDE_STEPS else ' (reported)'} | {m} | {dstr(c['dPR'][(s, m)])} | "
                     f"{dstr(c['dEP'][(s, m)])} | {dstr(c['dIO'][(s, m)])} | {fmt(did[(s, m)], sign=True)} |")
    L.append("\nCAVEAT: each arm is scored under its OWN training prompt (screen_any.sh's rule); dPR carries the prompt at "
             "evaluation as well as the training difference -- V31 - V3 above and the DiD column separate them (beside).")
    verdict, labels, why = decide(c, gt["status"])
    summary["contrasts"] = {k: {f"{s}|{m}": v for (s, m), v in d.items()} for k, d in c.items()}
    summary["did"] = {f"{s}|{m}": v for (s, m), v in did.items()}
    summary.update(verdict=verdict, labels=labels, why=why)
    L.append(f"\n## Verdict: **{verdict}**")
    L += [f"* {w}" for w in why]
    if labels:
        L.append("* labels that fire: " + ", ".join(labels))
    # ------------------------------------------------------------------ cost guard
    L.append("\n## Cost guard at step 150 (beside, never deciding)")
    summary["guard"] = {}
    for arm in ARMS:
        res = cells[(arm, 150)]
        if res is None or not res.get("ext"):
            L.append(f"* {LABEL[arm]}@150: no externals yet")
            continue
        gd = {b: guard(res, bases[b], cau) for b in bases if bases[b] is not None and bases[b].get("ext")}
        summary["guard"][arm] = gd
        for b, g in gd.items():
            ext = "; ".join(f"{e['bench']} {e['delta']:+.1f} (+{e['gained']}/-{e['lost']}, p={e['p']:.3f}"
                            f"{', COST' if (e['cost'] and b == BASE31) else ''})" for e in g["ext"]) or "-"
            qn, qc = g["qrdata_noncausal"], g["qrdata_causal"]
            L.append(f"* {LABEL[arm]}@150 vs {b}: {ext}; DABench cells/problem {fmt(g['dabench_cells'], 2)} "
                     f"({fmt(g['dabench_cells_x'], 2)}x); QRData non-causal "
                     + (f"{qn['delta']:+.1f} (+{qn['gained']}/-{qn['lost']}, p={qn['p']:.3f}, n={qn['n']})" if qn else "-")
                     + "; causal "
                     + (f"{qc['delta']:+.1f} (+{qc['gained']}/-{qc['lost']}, p={qc['p']:.3f}, n={qc['n']})" if qc else "-"))
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out) if a.out else (root if root.is_dir() else None)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "V31_READING.md").write_text(text)
        (out / "V31_READING.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
