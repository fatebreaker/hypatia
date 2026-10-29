"""Smoke tests for the health care analytics plug-in (sage/domains/healthcare.py: hc_quality, hc_dx).

Fast (a few draws per family, in memory, nothing written except a temporary real-data file):
registration, the skill-tag and blind-prompt conventions, the enforced invariants on the first
seeds of each split, demonstration cells that compile and stay within the length budget, that each
analysis function the demonstrations embed is the one the reference calls, and that the real SPECT
source (rebuilt offline from its cell counts) reproduces the published estimate. The full audit is
`python -m sage check --family hc_quality,hc_dx --n 60`.

  venv/bin/python -m sage.tests.test_healthcare          (or: venv/bin/python -m pytest sage/tests/test_healthcare.py)
"""
from __future__ import annotations

import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

import sage  # noqa: F401
from sage import registry
from sage.engine import make
from sage.layouts import CANONICAL
from sage.traces import cell_code

FAMS = ("hc_quality", "hc_dx")
SKILLS = {"inspect", "qc", "normalise", "model-choice", "estimate", "uncertainty", "validate", "report"}
METHOD_WORDS = {"hc_quality": ("Approach:", "logistic", "random-intercept", "random intercept", "quadrature", "Gauss",
                               "indirect standardi", "fitted probabilit", "raw SMR", "shrink", "fixed effect", "funnel",
                               "hospital term"),
                "hc_dx": ("Approach:", "Begg", "Greenes", "inverse-probability", "inverse probability", "weight",
                          "Mann-Whitney", "impute", "missing, not negative", "express every result")}
MAX_DEMO_LINES = 120

warnings.filterwarnings("ignore")


def test_registered_with_conventions():
    for name in FAMS:
        f = registry.get(name)
        assert set(f.skills) == set(f.steps), name                      # one skill per step
        assert set(f.skills.values()) <= SKILLS, (name, f.skills)
        assert f.seed_blocks["train"] != f.seed_blocks["test"]
        assert f.domain == "healthcare"


def test_blind_prompts():
    for name in FAMS:
        f = registry.get(name)
        for seed in range(3):
            s = f.draw(np.random.default_rng(seed), f.knobs(np.random.default_rng(seed)))
            blind, full = f.task_text(s.data, blind=True), f.task_text(s.data)
            assert f.task_text(s.data) == f.task_text(s.data, blind=False)
            leaked = [w for w in METHOD_WORDS[name] if w in blind]
            assert not leaked, (name, leaked)
            assert "Approach:" in full and len(full) > len(blind), name
            for k in f.keys:                                            # same question, same answer keys
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
    from sage.domains import healthcare
    for name in FAMS:
        f = registry.get(name)
        inst = make(f, f.seed_blocks["train"])
        for cell in f.demo(CANONICAL.config(inst, [])):
            for fn in cell.uses:
                assert getattr(healthcare, fn.__name__) is fn, (name, fn.__name__)


def test_real_spect_offline():
    """SpectSource rebuilt from its cell counts (no network) reproduces the published estimate, and the
    complete-case trap reproduces the published complete-case values."""
    from sage.domains import healthcare as hc
    from sage.real import assess_real
    urls = hc.PVB_URLS
    hc.PVB_URLS = ()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = hc.SpectSource(raw=Path(tmp))
            t = src.load("spect")
            assert len(t.data["df"]) == 2688 and int(t.data["df"]["referred"].sum()) == 471
            ref = src.family.reference(t.data)
            for k, v in hc.SPECT_PUBLISHED.items():
                assert abs(ref[k] - v) < 1e-6, (k, ref[k], v)
            cc = src.family.traps(t.data)["referred_only"].answer
            assert round(cc["sensitivity"], 3) == 0.975 and round(cc["specificity"], 3) == 0.144, cc
            r = assess_real(src, t)
            assert r["status"] == "assessed" and r["reproduced"] and r["graded_keys"] == ["sensitivity", "specificity"]
            assert {"referred_only", "unreferred_as_negative"} <= set(r["live_traps"]), r["live_traps"]
    finally:
        hc.PVB_URLS = urls


def main() -> int:
    for fn in (test_registered_with_conventions, test_blind_prompts, test_invariants_and_demos,
               test_demo_functions_are_the_reference, test_real_spect_offline):
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
