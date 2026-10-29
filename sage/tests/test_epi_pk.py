"""Smoke tests for the epidemiology and pharmacokinetics plug-ins (sage/domains/epi.py, pk.py).

Fast (a few draws per family, in memory, nothing written): registration, the skill-tag and
blind-prompt conventions, the enforced invariants on the first seeds of each split, demonstration
cells that compile and stay within the length budget, and -- when its raw file is already present
-- the Theophylline real target. The full audit is `python -m sage check --family F --n 60`.

  venv/bin/python -m sage.tests.test_epi_pk            (or: venv/bin/python -m pytest sage/tests/test_epi_pk.py)
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np

import sage  # noqa: F401
from sage import registry
from sage.engine import make
from sage.traces import cell_code

FAMS = ("epi_rt", "epi_r0", "pk_oral")
SKILLS = {"inspect", "qc", "normalise", "model-choice", "estimate", "uncertainty", "validate", "report"}
METHOD_WORDS = ("Approach:", "offset", "segmented", "negative-binomial", "two-compartment", "AIC", "censored")
MAX_DEMO_LINES = 120

warnings.filterwarnings("ignore")


def test_registered_with_conventions():
    for name in FAMS:
        f = registry.get(name)
        assert set(f.skills) == set(f.steps), name                      # one skill per step
        assert set(f.skills.values()) <= SKILLS, (name, f.skills)
        assert f.seed_blocks["train"] != f.seed_blocks["test"]


def test_blind_prompts():
    for name in FAMS:
        f = registry.get(name)
        s = f.draw(np.random.default_rng(0), f.knobs(np.random.default_rng(0)))
        blind, full = f.task_text(s.data, blind=True), f.task_text(s.data)
        assert f.task_text(s.data) == f.task_text(s.data, blind=False)
        assert not [w for w in METHOD_WORDS if w in blind], name        # the method is not named
        assert "Approach:" in full, name
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
        from sage.layouts import CANONICAL
        cfg = CANONICAL.config(inst, [])
        cells = f.demo(cfg)
        codes = [cell_code(c) for c in cells]
        for code in codes:
            compile(code, name, "exec")
        assert sum(len(c.splitlines()) for c in codes) <= MAX_DEMO_LINES, name


def test_theoph_real_if_fetched():
    """Only when $DEEP_DATA_ROOT/raw/theoph/concentrations.csv exists (sage real fetches it)."""
    src = registry.real_source("theoph")()
    if not (src.raw / "concentrations.csv").exists():
        print("  (theoph raw file not fetched; skipped)")
        return
    from sage.real import assess_real
    r = assess_real(src, src.load("theoph"))
    assert r["reproduced"] and "two_compartment_forced" in r["live_traps"], r


def main() -> int:
    for fn in (test_registered_with_conventions, test_blind_prompts, test_invariants_and_demos,
               test_theoph_real_if_fetched):
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
