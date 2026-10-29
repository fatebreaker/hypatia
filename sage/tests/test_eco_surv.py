"""Smoke tests for the ecology and survival-analysis plug-ins (sage/domains/ecology.py, survival.py).

Fast (a few draws per family, in memory, nothing written): registration, the skill-tag and
blind-prompt conventions, the enforced invariants on the first seeds of each split, demonstration
cells that compile and stay within the length budget, and that each analysis function the
demonstrations embed is the one the reference calls. The full audit is
`python -m sage check --family F --n 60`.

  venv/bin/python -m sage.tests.test_eco_surv            (or: venv/bin/python -m pytest sage/tests/test_eco_surv.py)
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np

import sage  # noqa: F401
from sage import registry
from sage.engine import make
from sage.layouts import CANONICAL
from sage.traces import cell_code

FAMS = ("eco_size", "eco_mb", "surv_cox", "surv_tvc")
SKILLS = {"inspect", "qc", "normalise", "model-choice", "estimate", "uncertainty", "validate", "report"}
METHOD_WORDS = ("Approach:", "Huggins", "Horvitz", "conditional likelihood", "logit", "Cox", "Kaplan", "stratif",
                "Efron", "time-varying", "counting-process", "Lincoln", "M0")
MAX_DEMO_LINES = 120

warnings.filterwarnings("ignore")


def test_registered_with_conventions():
    for name in FAMS:
        f = registry.get(name)
        assert set(f.skills) == set(f.steps), name                      # one skill per step
        assert set(f.skills.values()) <= SKILLS, (name, f.skills)
        assert f.seed_blocks["train"] != f.seed_blocks["test"]
        assert f.domain in ("ecology", "survival")


def test_blind_prompts():
    for name in FAMS:
        f = registry.get(name)
        s = f.draw(np.random.default_rng(0), f.knobs(np.random.default_rng(0)))
        blind, full = f.task_text(s.data, blind=True), f.task_text(s.data)
        assert f.task_text(s.data) == f.task_text(s.data, blind=False)
        assert not [w for w in METHOD_WORDS if w in blind], (name, [w for w in METHOD_WORDS if w in blind])
        assert "Approach:" in full and len(full) > len(blind), name
        for k in f.keys:                                                # same question, same answer keys
            assert f'"{k}"' in blind and f'"{k}"' in full


def test_invariants_and_demos(n=2):
    for name in FAMS:
        f = registry.get(name)
        for split in ("train", "test"):
            for i in range(n):
                inst = make(f, f.seed_blocks[split] + i)
                assert not inst.failures(True), (name, split, i, inst.failures(True))
                assert all(math.isfinite(inst.a.gold[k]) for k in f.keys)
                assert all(r.nominal in f.steps or r.nominal is None for r in inst.a.rows)
                assert {r.nominal for r in inst.a.rows} - {None} == set(f.steps), name   # every step has a trap
        cfg = CANONICAL.config(inst, [])
        assert cfg["_provenance"]["skills"] == {str(s): v for s, v in f.skills.items()}
        codes = [cell_code(c) for c in f.demo(cfg)]
        for code in codes:
            compile(code, name, "exec")
        assert sum(len(c.splitlines()) for c in codes) <= MAX_DEMO_LINES, (name, sum(len(c.splitlines()) for c in codes))
        final = codes[-1][codes[-1].index('"FINAL '):]                   # the printed answer line, keys in order
        assert all(f"{k} = " in final for k in f.keys), name
        assert [final.index(f"{k} = ") for k in f.keys] == sorted(final.index(f"{k} = ") for k in f.keys), name


def test_demo_functions_are_the_reference():
    """The functions a demonstration embeds (Cell.uses) must be the module's own analysis
    functions, so the trace runs the code that defined the gold."""
    from sage.domains import ecology, survival
    for name, mod in (("eco_size", ecology), ("eco_mb", ecology), ("surv_cox", survival), ("surv_tvc", survival)):
        f = registry.get(name)
        inst = make(f, f.seed_blocks["train"])
        for cell in f.demo(CANONICAL.config(inst, [])):
            for fn in cell.uses:
                assert getattr(mod, fn.__name__) is fn, (name, fn.__name__)


def main() -> int:
    for fn in (test_registered_with_conventions, test_blind_prompts, test_invariants_and_demos,
               test_demo_functions_are_the_reference):
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
