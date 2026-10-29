"""Causal-effect estimation (second domain): the five designs of generators/causal_effects.py.

The legacy generator entangles three things in each gen_<design>(rng, k): the draw, the
reference estimator and the trap table. Here they are separated along the contract:

  draw       calls gen_<design>(rng, k) unchanged, so the random stream is consumed in the
             legacy order, and keeps only what the analyst gets: the table, its
             description, the question and the variable roles the description states
  reference  RE-ESTIMATES the effect from the table and the roles alone, with the legacy
             estimators (causal_effects.ols / tsls / local_linear), so the gold is provably
             a function of the data files; sage/tests/test_equivalence.py checks it equals
             the legacy gold to the last bit
  traps      the documented wrong analyses, recomputed the same way

One legacy quirk is kept deliberately: the ATT's standard error (which sets its tolerance)
is a bootstrap drawn from the DGP's own generator, so it is computed in draw() and carried
in the data, not re-derivable from the table. Recommended upstream refactor: split each
gen_<design> into draw_<design> and analyse_<design>, as astro_kepler.py does.

I1 is not checkable: gen_<design> does not return the latent effect (tau; for the ATT the
latent estimand is tau0 + tau1 E[x0 | T=1]). Exposing it is a one-line change per design.

Files are written in the legacy layout (CausalV1 below) so the paper's causal corpora are
reproduced byte for byte; the only differences from the canonical layout are formatting:
raw |effect - gold| deviation, gold_se and roles in provenance, no latent/steps blocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from generators import causal_effects as CE
from sage.core import Cell, Family, Sample, Trap

DESIGNS = tuple(sorted(CE.FAMILIES))            # att, confounded, did, iv, rdd: the seed-block order

STEPS = {
    "confounded": {1: "adjust for pre-treatment confounders", 2: "choose the adjustment set (no post-treatment "
                   "variables, no omitted confounder)"},
    "iv": {1: "use the instrument (endogenous treatment)", 2: "2SLS with the covariates, scaled by the first stage"},
    "did": {1: "difference-in-differences (both group and period contrasts)"},
    "rdd": {1: "local estimation at the cutoff (bandwidth, a slope on each side)"},
    "att": {1: "adjust for confounders", 2: "allow effect heterogeneity (outcome model)",
            3: "average over the treated (ATT, not ATE)"},
}
# Shared skill taxonomy (sage/skills.py). Causal steps are almost all design choices.
SKILLS = {
    "confounded": {1: "model-choice", 2: "model-choice"},
    "iv": {1: "model-choice", 2: "estimate"},
    "did": {1: "model-choice"},
    "rdd": {1: "model-choice"},
    "att": {1: "model-choice", 2: "model-choice", 3: "report"},
}
TOL_RULE = "max(0.10*|gold|, 0.5*SE)"
NOTE = "ground truth is the reference estimator on the sampled data, not the latent effect"
ANSWER = 'Return your answer as JSON: {"answer": {"effect": <value>}}'


# ----------------------------------------------------------------------------- analyses
def _cols(df, names):
    return np.column_stack([df[nm].to_numpy(float) for nm in names])


def analyse(design: str, df: pd.DataFrame, roles: dict, boot_se: float | None = None):
    """(gold, SE, {trap: (value, step)}) from the table and the roles alone -- the legacy
    gen_<design>'s estimator calls, in the legacy order, on the same columns."""
    ols, tsls, c1, ll = CE.ols, CE.tsls, CE.c1, CE.local_linear
    if design == "confounded":
        Xa = _cols(df, roles["confounders"])
        y, t, m = (df[roles[k]].to_numpy(float) for k in ("outcome", "treatment", "post_treatment"))
        b, se = ols(y, c1(t, Xa))
        gold, gse = float(b[1]), float(se[1])
        j = int(np.argmax(np.abs(ols(y, c1(t, Xa))[0][2:]) * Xa.std(0)))
        return gold, gse, {"no_adjustment": (float(y[t == 1].mean() - y[t == 0].mean()), 1),
                           "post_treatment_control": (float(ols(y, c1(t, Xa, m))[0][1]), 2),
                           "confounder_omitted": (float(ols(y, c1(t, np.delete(Xa, j, 1)))[0][1]), 2)}
    if design == "iv":
        y, t, z = (df[roles[k]].to_numpy(float) for k in ("outcome", "treatment", "instrument"))
        Wd = c1(*[df[nm].to_numpy(float) for nm in roles["covariates"]])
        gold, gse = tsls(y, t, z, Wd)
        rf = ols(y, np.column_stack([z, Wd]))[0][0]
        n = len(y)
        return gold, gse, {"ols_ignores_instrument": (float(ols(y, np.column_stack([t, Wd]))[0][0]), 1),
                           "reduced_form": (float(rf), 2),
                           "iv_without_covariates": (tsls(y, t, z, c1(np.zeros(n))[:, :1])[0], 2)}
    if design == "did":
        y = df[roles["outcome"]].to_numpy(float)
        g = df[roles["group"]].to_numpy(float)
        po = df[roles["post"]].to_numpy(float)
        U = pd.get_dummies(df[roles["unit"]], drop_first=True).to_numpy(float)
        Tt = pd.get_dummies(df[roles["period"]], drop_first=True).to_numpy(float)
        b, se = ols(y, c1(g * po, U, Tt))
        post_gap = y[(g == 1) & (po == 1)].mean() - y[(g == 0) & (po == 1)].mean()
        prepost = y[(g == 1) & (po == 1)].mean() - y[(g == 1) & (po == 0)].mean()
        return float(b[1]), float(se[1]), {"post_period_gap": (float(post_gap), 1),
                                           "treated_before_after": (float(prepost), 1)}
    if design == "rdd":
        cut = roles["cutoff"]
        rr, yy = df[roles["running"]].to_numpy(float), df[roles["outcome"]].to_numpy(float)
        h = 0.4 * (rr.max() - rr.min()) / 2
        gold, gse = ll(yy, rr, cut, h)
        d = (rr >= cut).astype(float)
        return gold, gse, {"global_linear": (float(ols(yy, c1(d, rr - cut))[0][1]), 1),
                           "difference_in_means": (float(yy[d == 1].mean() - yy[d == 0].mean()), 1),
                           "local_linear_half_bandwidth": (ll(yy, rr, cut, h / 2)[0], None),
                           "local_linear_wide_bandwidth": (ll(yy, rr, cut, 1.5 * h)[0], None)}
    if design == "att":
        y, t = df[roles["outcome"]].to_numpy(float), df[roles["treatment"]].to_numpy(float)
        Xa = _cols(df, roles["covariates"])
        b0 = ols(y[t == 0], c1(Xa[t == 0]))[0]
        b1 = ols(y[t == 1], c1(Xa[t == 1]))[0]
        att = float(np.mean(y[t == 1] - c1(Xa[t == 1]) @ b0))
        ate = float(np.mean(c1(Xa) @ b1 - c1(Xa) @ b0))
        return att, boot_se, {"no_adjustment": (float(y[t == 1].mean() - y[t == 0].mean()), 1),
                              "additive_ols": (float(ols(y, c1(t, Xa))[0][1]), 2),
                              "ate_not_att": (ate, 3)}
    raise ValueError(design)


# ----------------------------------------------------------------------------- the family
class CausalDesign(Family):
    domain = "causal"
    keys = ("effect",)
    max_tries = 12
    rng_policy = "stream"
    enforce = frozenset({"I1", "I2", "I3"})     # causal_effects.make's rule (I1 vacuous: no latent); I4 audited
    tol_rule = TOL_RULE
    note = NOTE
    demo_timeout = 180

    def __init__(self, design: str):
        self.design = design
        self.name = f"causal_{design}"
        self.steps = STEPS[design]
        self.skills = SKILLS[design]
        j = DESIGNS.index(design)                    # fixed per design, as the paper's corpora were minted
        self.seed_blocks = {"train": 7_000_000 + 100_000 * j, "test": 7_050_000 + 100_000 * j}
        self.layout = CAUSAL_V1

    def csv(self) -> str:
        return f"{self.design}_data.csv"

    def task_text(self, data, blind: bool = False) -> str:
        """CauSciBench's prompt (causal_effects.task_text, unchanged). Blind: the same prompt
        without its method menu, the one place it names designs. The description and the
        question stay: the data facts they state (a post column, a cutoff, unobserved
        confounding, what was measured after treatment, the ATT population) define the
        estimand, so blind mode cannot remove them."""
        text = CE.task_text(data["desc"], data["query"], self.csv())
        return self.blind_from_text(text) if blind else text

    @staticmethod
    def blind_from_text(task: str) -> str:
        menu = f"{CE.METHOD_MENU}\n\n"
        if menu not in task:
            raise ValueError("causal blind: the method menu is not in the task text (causal_effects.task_text changed?)")
        return task.replace(menu, "", 1)

    def task_phrasings(self, data, blind: bool = False) -> list[str]:
        """Two more templates of the same prompt (sage.variants validates them against it)."""
        csv, desc, query = self.csv(), data["desc"].strip(), data["query"].strip()
        menu = "" if blind else f"{CE.METHOD_MENU}\n\n"
        return [("A causal question about a tabular dataset needs answering; approach it as an expert in statistics and "
                 f"causal reasoning.\n\nThe question:\n```\n{query}\n```\n\nThe data are in data_files/{csv}, described "
                 f"as follows:\n```\n{desc}\n```\n\n{menu}Use the data to estimate the causal effect the question asks "
                 f"about. {ANSWER}"),
                (f"You are an expert in statistics and causal reasoning.\n\nDataset: data_files/{csv}\n\nWhat the "
                 f"dataset records:\n```\n{desc}\n```\n\n{menu}Question to answer from these data:\n```\n{query}\n```\n\n"
                 f"Estimate the causal effect from the data. {ANSWER}")]

    def draw(self, rng, k) -> Sample:
        df, desc, query, gold, gse, abl, roles = CE.FAMILIES[self.design](rng, k)
        data = {"df": df, "desc": desc, "query": query, "roles": roles,
                "boot_se": gse if self.design == "att" else None}
        return Sample(data, latent={})

    def _analysis(self, data):
        key = "_analysis"
        if key not in data:
            data[key] = analyse(self.design, data["df"], data["roles"], data["boot_se"])
        return data[key]

    def reference(self, data) -> dict:
        gold, gse, _ = self._analysis(data)
        return {"effect": gold, "_se": gse}

    def traps(self, data) -> dict:
        return {nm: Trap({"effect": v}, step) for nm, (v, step) in self._analysis(data)[2].items()}

    def tolerance(self, gold) -> dict:
        return {"effect": max(0.10 * abs(gold["effect"]), 0.5 * gold["_se"])}

    def strengthen(self, k, dead):
        return k * 1.4

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / self.csv(), index=False)
        return [self.csv()]

    def demo(self, cfg) -> list[Cell]:
        from rl import traces_causal as TC
        csv = Path(cfg["data_files"][0]).name
        return [Cell(p, c) for p, c in TC.moves(self.design, cfg["_provenance"]["roles"], csv)]


class CausalV1:
    """generators/causal_effects.py's own eval_config, field for field."""
    indent = 1

    def config(self, inst, files) -> dict:
        fam, data, a = inst.family, inst.sample.data, inst.a
        gold, tol = a.gold["effect"], a.tol["effect"]
        abl = {"reference": {"effect": gold, "fails_at_step": None, "deviation": abs(gold - gold)}}
        for r in a.rows:
            v = r.answer["effect"]
            abl[r.name] = {"effect": v, "fails_at_step": r.nominal, "deviation": abs(v - gold)}
        return {"id": inst.id, "task": fam.task_text(data), "data_files": [f"data_files/{n}" for n in files],
                "ground_truth": {"effect": gold},
                "grader": {"type": "multi_numeric_tolerance",
                           "config": {"keys": {"effect": {"absolute_tolerance": tol}}}},
                "_provenance": {"family": fam.name, "seed": inst.seed, "gold_se": a.gold["_se"],
                                "gen_tries": inst.tries, "roles": data["roles"], "tol_rule": TOL_RULE,
                                "note": NOTE, "ablations": abl}}

    def dumps(self, cfg) -> str:
        return json.dumps(cfg, indent=self.indent)


CAUSAL_V1 = CausalV1()
FAMILIES = [CausalDesign(d) for d in DESIGNS]
