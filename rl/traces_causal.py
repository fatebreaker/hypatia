#!/usr/bin/env python
"""Executed demonstration traces for the five causal-effect families (second domain).

Same contract as rl/traces_wf.py and rl/sft_traces.py: every cell is EXECUTED in the
harness's Sandbox against the instance's own data files, the tool response is the real
output, the system prompt and prompt builder are harness/agents' own, the answer is
PARSED from the executed output (never copied from ground truth), and a trace whose
parsed answer fails the instance's grader is discarded.

Each family's analysis is the reference analysis of generators/causal_effects.py,
written with pandas/statsmodels as an analyst in the sandbox would write it:

  confounded  inspect -> separate pre- from post-treatment variables -> OLS on the
              treatment plus the pre-treatment confounders only
  iv          inspect -> first-stage strength -> 2SLS with the covariates
  did         inspect -> group-by-period means -> two-way fixed effects
  rdd         inspect -> outcome near the cutoff -> local linear, triangular kernel,
              bandwidth 0.4 x half the running variable's range, a slope on each side
  att         inspect -> balance -> outcome model fitted on the untreated, imputed for
              the treated (the ATT the question asks for, not the ATE)

The variable roles come from the instance's _provenance (the model never sees it);
the prose explains how they follow from the dataset description.

  venv/bin/python rl/traces_causal.py --family iv --n 150 --out $DEEP_DATA_ROOT/sft/causal_iv
  venv/bin/python rl/traces_causal.py --family all --n 150 --out $DEEP_DATA_ROOT/sft/causal5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from agents import SYSTEM, build_prompt          # noqa: E402
from sandbox import Sandbox                      # noqa: E402

FINAL_RE = re.compile(r"FINAL\s+effect\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")
FAMILIES = ("confounded", "iv", "did", "rdd", "att")


def load_cell(csv: str) -> str:
    return (f'import pandas as pd, numpy as np\ndf = pd.read_csv("data_files/{csv}")\n'
            'print(df.shape)\nprint(df.head())\nprint(df.describe().T.round(3))')


def moves(fam: str, r: dict, csv: str) -> list[tuple[str, str]]:
    first = ("I start by loading the data and checking its size, columns and ranges.", load_cell(csv))
    if fam == "confounded":
        X, T, Y, M = r["confounders"], r["treatment"], r["outcome"], r["post_treatment"]
        return [first,
                (f"This is observational data, so treated and untreated units differ in their covariates. "
                 f"Before adjusting I check how the pre-treatment covariates differ by {T}. The description says "
                 f"{M} was measured after treatment, so it is a consequence of treatment, not a confounder: "
                 f"adjusting for it would block part of the effect.",
                 f'print(df.groupby("{T}")[{X + [M]}].mean().T.round(3))'),
                (f"The effect of {T} on {Y} is identified by adjusting for the pre-treatment confounders "
                 f"({', '.join(X)}) and nothing measured after treatment. I fit OLS with robust standard errors.",
                 'import statsmodels.formula.api as smf\n'
                 f'm = smf.ols("{Y} ~ {T} + {" + ".join(X)}", data=df).fit(cov_type="HC1")\n'
                 f'print(m.summary().tables[1])\nprint("FINAL effect =", round(m.params["{T}"], 6))')]
    if fam == "iv":
        T, Y, Z, W = r["treatment"], r["outcome"], r["instrument"], r["covariates"]
        return [first,
                (f"The description warns that unobserved factors affect both {T} and {Y}, so OLS of {Y} on {T} "
                 f"is biased. {Z} shifts {T} and plausibly affects {Y} only through it: it is the instrument. "
                 f"I check that it is strong in the first stage, conditional on the covariates.",
                 'import statsmodels.formula.api as smf\n'
                 f'fs = smf.ols("{T} ~ {Z} + {" + ".join(W)}", data=df).fit(cov_type="HC1")\n'
                 f'print(fs.params.round(4))\nprint("first-stage t on the instrument:", round(fs.tvalues["{Z}"], 2))'),
                (f"The first stage is strong. I estimate the effect by two-stage least squares, keeping the "
                 f"covariates ({', '.join(W)}) in both stages since the instrument is related to them.",
                 'from statsmodels.sandbox.regression.gmm import IV2SLS\n'
                 f'exog = df[{W}].assign(const=1.0)\n'
                 f'endog_x = pd.concat([df[["{T}"]], exog], axis=1)\n'
                 f'inst = pd.concat([df[["{Z}"]], exog], axis=1)\n'
                 f'iv = IV2SLS(df["{Y}"], endog_x, inst).fit()\n'
                 f'print(iv.params.round(4))\nprint("FINAL effect =", round(iv.params["{T}"], 6))')]
    if fam == "did":
        Y, U, P, G, PO = r["outcome"], r["unit"], r["period"], r["group"], r["post"]
        return [first,
                (f"Units are observed before and after the policy, and only the {G}=1 group is treated. "
                 f"Comparing treated and control after the change would mix the effect with permanent "
                 f"differences between the groups, and comparing the treated before and after would mix it "
                 f"with the common time trend. I look at the group-by-{P} means first.",
                 f'print(df.groupby(["{P}", "{G}"])["{Y}"].mean().unstack().round(3))'),
                (f"The pre-period gap between the groups is roughly stable, so difference-in-differences is "
                 f"appropriate. I fit two-way fixed effects (unit and {P}) with the treated-by-post indicator.",
                 'import statsmodels.formula.api as smf\n'
                 f'df["treat_post"] = df["{G}"] * df["{PO}"]\n'
                 f'm = smf.ols("{Y} ~ treat_post + C({U}) + C({P})", data=df).fit()\n'
                 f'print("FINAL effect =", round(m.params["treat_post"], 6))')]
    if fam == "rdd":
        Y, R, c = r["outcome"], r["running"], r["cutoff"]
        return [first,
                (f"Treatment switches on when {R} crosses {c:g}, so the effect is the jump in {Y} at the cutoff. "
                 f"A global fit over the whole range would confuse curvature in {Y} with the jump; I look at "
                 f"binned means near the cutoff first.",
                 f'bins = pd.cut(df["{R}"], 20)\nprint(df.groupby(bins, observed=True)["{Y}"].mean().round(3))'),
                (f"I estimate the jump locally: a linear fit on each side of the cutoff within a window of 0.4 "
                 f"times half the range of {R}, with triangular weights that decline away from the cutoff.",
                 'import statsmodels.api as sm\n'
                 f'x = df["{R}"] - {c!r}\nh = 0.4 * (df["{R}"].max() - df["{R}"].min()) / 2\n'
                 'keep = x.abs() <= h\nd = (x[keep] >= 0).astype(float)\n'
                 'X = pd.DataFrame({"const": 1.0, "d": d, "x": x[keep], "dx": d * x[keep]})\n'
                 f'w = 1 - x[keep].abs() / h\nm = sm.WLS(df.loc[keep, "{Y}"], X, weights=w).fit()\n'
                 'print("bandwidth:", round(h, 3), " observations:", int(keep.sum()))\n'
                 'print("FINAL effect =", round(m.params["d"], 6))')]
    if fam == "att":
        T, Y, X = r["treatment"], r["outcome"], r["covariates"]
        return [first,
                (f"The question asks for the effect on those who {T.replace('_', ' ')} -- the ATT, not the "
                 f"average over everyone. Treated and untreated units differ in their covariates, so I check "
                 f"balance first.",
                 f'print(df.groupby("{T}")[{X}].mean().T.round(3))'),
                (f"If the effect varies with the covariates, the ATT and the ATE differ. I fit an outcome model "
                 f"on the untreated units only, predict each treated unit's untreated outcome from its own "
                 f"covariates, and average the treated units' differences.",
                 'import statsmodels.formula.api as smf\n'
                 f'm0 = smf.ols("{Y} ~ {" + ".join(X)}", data=df[df["{T}"] == 0]).fit()\n'
                 f'tr = df[df["{T}"] == 1]\natt = (tr["{Y}"] - m0.predict(tr)).mean()\n'
                 'print("FINAL effect =", round(att, 6))')]
    raise ValueError(fam)


def make(cfg: dict, inst_dir: Path, fam: str) -> dict | None:
    r = cfg["_provenance"]["roles"]
    csv = Path(cfg["data_files"][0]).name
    sb = Sandbox(data_dir=inst_dir / "data_files", timeout=180)
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
        for prose, code in moves(fam, r, csv):
            res = sb.run(code)
            if res.returncode != 0 or res.timed_out:
                return None
            msgs.append({"role": "assistant", "content": f"{prose}\n\n```python\n{code}\n```"})
            msgs.append({"role": "user", "content": res.as_feedback()})
        hit = FINAL_RE.search(msgs[-1]["content"])
        if not hit:
            return None
        val = float(hit.group(1))
        tol = cfg["grader"]["config"]["keys"]["effect"]["absolute_tolerance"]
        if abs(val - cfg["ground_truth"]["effect"]) > tol:
            return None                                  # never ship a wrong-answer trace
        msgs.append({"role": "assistant", "content": "```json\n" + json.dumps(
            {"answer": {"effect": val}}, indent=2) + "\n```"})
        return {"id": cfg["id"], "messages": msgs, "reward": 1.0, "n_cells": sb.attempts,
                "_family": f"causal_{fam}"}
    finally:
        sb.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=list(FAMILIES) + ["all"])
    ap.add_argument("--n", type=int, default=150, help="traces per family")
    ap.add_argument("--val", type=int, default=10, help="validation traces per family")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fams = FAMILIES if args.family == "all" else (args.family,)
    tr_all, va_all = [], []
    for fam in fams:
        traces, bad = [], 0
        for p in sorted((data_root() / "generated" / f"causal_{fam}" / "train").glob("*/eval_config.json")):
            if len(traces) >= args.n + args.val:
                break
            t = make(json.loads(p.read_text()), p.parent, fam)
            if t is None:
                bad += 1
                if bad <= 3:
                    print(f"  discarded {p.parent.name}", flush=True)
                continue
            traces.append(t)
        print(f"{fam}: {len(traces)} traces ({bad} discarded)", flush=True)
        tr_all += traces[:-args.val]
        va_all += traces[-args.val:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr_all))
    Path(f"{out}_val.jsonl").write_text("".join(json.dumps(t) + "\n" for t in va_all))
    print(f"wrote {len(tr_all)} train / {len(va_all)} val to {out}_*.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
