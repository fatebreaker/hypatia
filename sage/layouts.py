"""eval_config.json layouts: how an assessed instance is written in GeneBench-Pro's format.

The canonical layout is the most complete of the three the project already ships -- the
one generators/astro_kepler.py writes -- so the astronomy families go through it unchanged
and byte-identical, and every new family gets it for free:

  ground_truth   the reference's answer keys (diagnostics such as "_se" are not graded)
  grader         multi_numeric_tolerance, one absolute tolerance per key
  _provenance    family, seed, gen_tries, tol_rule, note, latent (the hidden truth), steps
                 (names of the pipeline steps), [family extras], ablations

  ablations      "reference" first, then every trap and accepted variant in the family's
                 order: its answer keys, fails_at_step (None for an accepted variant and for
                 a conditional trap that coincides with the reference on this draw), and
                 deviation IN UNITS OF TOLERANCE, maxed over keys -- the one convention that
                 works for any number of keys on any scales (gwas_loci's and astro's)

Two legacy layouts are kept beside their adapters, only so that the paper's corpora are
reproduced byte for byte: `CausalV1` (sage/domains/causal.py: raw deviation, gold_se and
roles in provenance) and `EmitV1` (sage/domains/genomics.py: emit.py's indent=2 files with
the randomised config in provenance). They differ in formatting, never in validity logic.
"""
from __future__ import annotations

import json

import numpy as np


def grader(keys, tol: dict) -> dict:
    return {"type": "multi_numeric_tolerance",
            "config": {"keys": {k: {"absolute_tolerance": float(tol[k])} for k in keys}}}


def table(inst) -> dict:
    """The canonical ablation table: reference row first, then the family's rows."""
    keys = inst.family.keys
    gold = inst.a.gold
    t = {"reference": {**{k: gold[k] for k in keys}, "fails_at_step": None, "deviation": 0.0}}
    for r in inst.a.rows:
        t[r.name] = {**r.answer, "fails_at_step": r.step, "deviation": r.dev}
    return t


def latent_json(latent: dict) -> dict:
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in latent.items()}


class Canonical:
    indent = 1

    def config(self, inst, files: list[str]) -> dict:
        fam, a = inst.family, inst.a
        keys = fam.keys
        return {"id": inst.id, "task": fam.task_text(inst.sample.data),
                "data_files": [f"data_files/{n}" for n in files],
                "ground_truth": {k: float(a.gold[k]) for k in keys},
                "grader": grader(keys, a.tol),
                "_provenance": {"family": fam.name, "seed": inst.seed, "gen_tries": inst.tries,
                                "tol_rule": fam.tol_rule, "note": fam.note,
                                "latent": latent_json(inst.sample.latent),
                                "steps": fam.steps, **fam.extra_provenance(inst),
                                "ablations": table(inst)}}

    def dumps(self, cfg: dict) -> str:
        return json.dumps(cfg, indent=self.indent, default=float)


CANONICAL = Canonical()
