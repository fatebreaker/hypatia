#!/usr/bin/env python
"""Process-signal base rates on STORED rollouts (docs/RL_PLAN.md s.15): how often each harness/process_signals.py signal
fires on real transcripts of a given policy, per family, and the same with the DECOY column set (the 'irrelevant
columns' sham, which fires ~never -- the reason the registered PROCESS-SHAM permutes the real term instead).

    venv/bin/python experiments/harness_v3/base_rates.py [--json OUT] [SET ...]

SETs (default: all): base_echo (untrained 8B, RL_PLAN 12 families), r0s3 (8B R0 seed 3 step 150, causal_rdd RL),
q14frozen (untrained 14B, frozen harness: causal_iv, arr_kinetics + the six held-out families), p14iv / p14arr (the P14
14B runs at step 100 on the same families). Rates are over APPLICABLE rollouts (None = not applicable is excluded);
'p' is the mean process score over all rollouts with messages.
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
from process_signals import SIGNALS, ProcessScorer, decoy_vocab  # noqa: E402

RL = ROOT / "results" / "pods" / "rl_plan"
Q14 = ROOT / "results" / "pods" / "harness_v2_q14"
HO6 = ["causal_confounded", "causal_att", "surv_cox", "enzyme_mm", "hc_dx", "osc_damped"]
P14 = "FROZEN__rl-czero-{}-q14base-echo-sysscience-rc-32k-hv2lb3d-lr1e-5-s1_s100"
SETS = {
    "base_echo": [RL / "base_echo" / f"{f}.json" for f in
                  ["causal_iv", "arr_kinetics", "causal_rdd", "causal_did", "popgen_pulse", "scrna_1k"] + HO6],
    "r0s3": [RL / "rl-czero-causal_rdd-base-echo-rc-lr1e-5-s3" / "step150" / f"{f}.json" for f in
             ["causal_iv", "arr_kinetics", "causal_rdd", "causal_did", "popgen_pulse", "scrna_1k"] + HO6],
    "q14frozen": [Q14 / "FROZEN" / "design" / "causal_iv.json", Q14 / "FROZEN" / "design" / "arr_kinetics.json"]
                 + [Q14 / "FROZEN" / "test" / f"{f}.json" for f in HO6],
    "p14iv": [Q14 / P14.format("causal_iv") / "design" / "causal_iv.json"]
             + [Q14 / P14.format("causal_iv") / "test" / f"{f}.json" for f in HO6],
    "p14arr": [Q14 / P14.format("arr_kinetics") / "design" / "arr_kinetics.json"]
              + [Q14 / P14.format("arr_kinetics") / "test" / f"{f}.json" for f in HO6],
}


def rates(path: Path, scorer: ProcessScorer) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    fam, split = d["family"], d.get("split", "test")
    G = data_root() / "generated" / fam / split
    vals = {k: [] for k in SIGNALS}
    ps, cells, passed = [], [], []
    for inst, rs in (d.get("groups") or {}).items():
        cfg = json.loads((G / inst / "eval_config.json").read_text())
        for r in rs:
            if not r.get("messages"):
                continue
            s = scorer.signals(r["messages"], cfg, G / inst / "data_files", answer=r.get("answer"),
                               n_prefix=int(r.get("n_prefix") or 0))
            ps.append(s.score)
            cells.append(int(r.get("n_cells") or 0))
            passed.append(bool(r.get("passed")))
            for k in SIGNALS:
                if s.fired[k] is not None:
                    vals[k].append(s.fired[k])
    n = len(ps)
    out = {"family": fam, "n": n, "pass": 100.0 * sum(passed) / max(n, 1), "cells": sum(cells) / max(n, 1),
           "p": sum(ps) / max(n, 1)}
    for k in SIGNALS:
        out[k] = (100.0 * sum(vals[k]) / len(vals[k])) if vals[k] else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", default=list(SETS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-decoy", action="store_true")
    a = ap.parse_args()
    real, decoy = ProcessScorer(), ProcessScorer(vocab_override=decoy_vocab)
    out = {}
    hdr = f"{'set':10s} {'family':18s} {'n':>4s} {'pass':>5s} {'cells':>5s} " + " ".join(f"{k:>8s}" for k in SIGNALS) \
        + f" {'p':>5s}" + ("" if a.no_decoy else f" {'p_decoy':>7s}")
    print(hdr)
    for sname in a.sets:
        for path in SETS[sname]:
            r = rates(path, real)
            if r is None:
                print(f"{sname:10s} {path} missing")
                continue
            if not a.no_decoy:
                r["p_decoy"] = rates(path, decoy)["p"]
            out.setdefault(sname, {})[r["family"]] = r
            f = lambda x: "     N/A" if x is None else f"{x:8.1f}"  # noqa: E731
            print(f"{sname:10s} {r['family']:18s} {r['n']:4d} {r['pass']:5.1f} {r['cells']:5.1f} "
                  + " ".join(f(r[k]) for k in SIGNALS) + f" {r['p']:5.3f}"
                  + ("" if a.no_decoy else f" {r['p_decoy']:7.3f}"))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
