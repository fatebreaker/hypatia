#!/usr/bin/env python
"""CALIBRATION of the v3.1 re-estimate detector (rl/process_signals_v31.py, signal `reest`; docs/RL_PLAN.md s.18, the
registration entry below 2026-09-26 19:15 UTC). CPU only; reads stored rollouts, writes nothing unless --json.

    venv/bin/python experiments/harness_v3/reest_calib.py [--json OUT] [--no-families]

  labels     agreement with the hand labels in tests/fixtures/reest_labels.json (written before the detector ran)
  step 0     the untrained 8B on the causal_rdd DESIGN draw: V31 (science_check, HV3_62) vs V3 (science, HV3_17) --
             reest, and beside it the v3 detectors' spec / check and the reader's old re-est proxy (>= 2 estimates)
  s.16 arms  the same on the s.16 checkpoints' causal_rdd DESIGN eval rows (PR / PLAIN / SHAM, science prompt): the
             no-elicitation negative control (the training rollouts of the PR arm are not stored with messages, so
             reest cannot be computed on them; their logged v3 spec / check / proxy over steps 41-50 are printed beside)
  families   every family's V3 TEST row (untrained 8B, science prompt): reest vs v3 spec per family -- the detector must
             not be causal_rdd-specific; plus the count of rollouts where reest fired with < 2 successful estimates
             (must be 0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
from paths import data_root  # noqa: E402
from rl import process_signals_v31 as R  # noqa: E402

H3 = ROOT / "results" / "pods" / "harness_v3"
H31 = ROOT / "results" / "pods" / "harness_v31"
FIX = Path(__file__).resolve().parent / "tests" / "fixtures" / "reest_labels.json"
S16 = {"pr": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr1e-5-s1",
       "plain": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0-lr1e-5-s1",
       "sham": "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3prsham0.3-lr1e-5-s1"}


def score_row(path: Path, ps: R.ProcessScorerV31, tag: str) -> dict:
    """reest + the v3 values for every rollout of one stored row: {id: {...}}."""
    d = json.loads(path.read_text())
    G = data_root() / "generated" / d["family"] / d.get("split", "test")
    out = {}
    for inst, rs in sorted((d.get("groups") or {}).items()):
        idir = G / inst
        if not (idir / "eval_config.json").is_file():
            continue
        cfg = json.loads((idir / "eval_config.json").read_text())
        for j, r in enumerate(rs):
            rid = f"{tag}:{inst}:{j}"
            msgs = r.get("messages") or []
            if not msgs or not (r.get("n_cells") or 0):
                out[rid] = {"reest": None, "spec": None, "check": None, "proxy": None, "n_good": 0, "cells": 0,
                            "where": "no executed cell"}
                continue
            s = ps.signals(msgs, cfg, idir / "data_files", answer=r.get("answer"), n_prefix=int(r.get("n_prefix") or 0))
            lg = R.parse_logged(s.detail["spec"])
            cells, _ = R.executed_cells(msgs, int(r.get("n_prefix") or 0))
            A = R.ReestAnalyzer(ps.vocab(idir / "data_files", cfg)).run(cells)
            good, _ = R.run_status(cells, A)
            n_good = len({e.signature() for e in good})         # distinct specifications that ran
            out[rid] = {"reest": s.fired["spec"], "spec": s.detail["v3"]["spec"], "check": s.detail["v3"]["check"],
                        "proxy": int((lg["v3_est"] or 0) >= 2 or s.detail["v3"]["spec"] == 1),
                        "n_good": n_good, "cells": len(cells), "where": s.detail["reest"]["where"]}
    return out


def rate(rows: dict, k: str):
    v = [x[k] for x in rows.values() if x.get(k) is not None]
    return (sum(v) / len(v)) if v else None, len(v)


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(name: str, rows: dict) -> dict:
    """Rates over the rollouts with >= 1 executed cell (the process.jsonl / gate convention: a rollout with no executed
    cell has every signal None); reest over ALL rollouts (no cell = 0) beside."""
    out = {"n": len(rows)}
    for k in ("reest", "spec", "check", "proxy"):
        out[k], out[f"n_{k}"] = rate(rows, k)
    out["reest_all"] = sum(x["reest"] or 0 for x in rows.values()) / max(len(rows), 1)
    out["n_fired"] = sum(1 for x in rows.values() if x["reest"])
    print(f"  {name:36s} n={out['n']:4d} (with cells {out['n_reest']:3d})  reest {fmt(out['reest'])} "
          f"[{out['n_fired']}; all rollouts {fmt(out['reest_all'])}]  | v3 spec {fmt(out['spec'])}  v3 check "
          f"{fmt(out['check'])}  proxy(>=2 est) {fmt(out['proxy'])}")
    return out


def training_log(run: str, lo: int = 41, hi: int = 50) -> dict | None:
    f = data_root() / "ckpt" / run / "process.jsonl"
    if not f.is_file():
        return None
    rows = []
    for x in f.read_text().splitlines():
        try:
            r = json.loads(x)
        except json.JSONDecodeError:
            continue
        if lo <= int(r.get("step", -1)) <= hi:
            rows.append(r)
    sp = [r["fired"]["spec"] for r in rows if r["fired"].get("spec") is not None]
    ck = [r["fired"]["check"] for r in rows if r["fired"].get("check") is not None]
    px = [int((R.parse_logged((r.get("where") or {}).get("spec"))["v3_est"] or 0) >= 2) for r in rows]
    return {"n": len(rows), "spec": sum(sp) / len(sp) if sp else None, "check": sum(ck) / len(ck) if ck else None,
            "proxy": sum(px) / len(px) if px else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-families", action="store_true")
    a = ap.parse_args(argv)
    ps = R.ProcessScorerV31()
    out = {"detector": R.detector_record()}
    fx = json.loads(FIX.read_text())
    rows = {}
    for tag, rel in fx["rows"].items():
        rows.update(score_row(ROOT / rel, ps, tag))
    agree = [(x["id"], x["label"], int(rows[x["id"]]["reest"] or 0)) for x in fx["labels"] if x["id"] in rows]
    n_ok = sum(lbl == det for _, lbl, det in agree)
    out["labels"] = {"n": len(agree), "agree": n_ok,
                     "disagree": [{"id": i, "label": lbl, "detector": det} for i, lbl, det in agree if lbl != det]}
    print(f"LABELS: {n_ok}/{len(agree)} agree" + ("" if n_ok == len(agree) else f" -- {out['labels']['disagree']}"))
    print("STEP 0 (untrained 8B, causal_rdd DESIGN draw, 16 x 2):")
    out["step0"] = {}
    for tag in fx["rows"]:
        sub = {k: v for k, v in rows.items() if k.startswith(tag + ":")}
        out["step0"][tag] = summarize(f"{tag} ({'science_check' if tag == 'V31' else 'science'})", sub)
        out["step0"][tag]["fired"] = sorted(k for k, v in sub.items() if v["reest"])
    print("s.16 ARMS (science prompt, no elicitation; checkpoint eval rows on causal_rdd DESIGN):")
    out["s16"] = {}
    for arm, run in S16.items():
        for step in (50, 100, 150):
            p = H3 / f"V3__{run}_s{step}" / "design" / "causal_rdd.json"
            if p.is_file():
                out["s16"][f"{arm}@{step}"] = summarize(f"{arm.upper()} @{step}", score_row(p, ps, f"{arm}{step}"))
    tl = training_log(S16["pr"])
    out["pr_training_41_50"] = tl
    if tl:
        print(f"  PR TRAINING steps 41-50 (logged by the v3 scorer; no transcripts stored): n={tl['n']}  v3 spec "
              f"{fmt(tl['spec'])}  v3 check {fmt(tl['check'])}  proxy(>=2 est) {fmt(tl['proxy'])}")
    if not a.no_families:
        print("FAMILIES (untrained 8B, V3 TEST rows, science prompt):")
        out["families"], bad = {}, []
        for p in sorted((H3 / "V3" / "test").glob("*.json")):
            r = score_row(p, ps, p.stem)
            out["families"][p.stem] = summarize(p.stem, r)
            bad += [k for k, v in r.items() if v["reest"] and v["n_good"] < 2]
            both = [k for k, v in r.items() if v["spec"] == 1 and not v["reest"]]
            out["families"][p.stem]["v3_spec_not_reest"] = both
            out["families"][p.stem]["fired"] = sorted(k for k, v in r.items() if v["reest"])
        out["fired_with_lt2_estimates"] = bad
        print(f"  rollouts where reest fired with < 2 distinct successful specifications: {len(bad)}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
