"""Smoke tests for the signal-physics and chemical-kinetics plug-ins and the upgraded enzyme family
(sage/domains/oscillator.py, arrhenius.py, enzyme.py).

Fast (a few draws per family, in memory except one enzyme instance written to a temporary
directory): registration, the skill-tag and blind-prompt conventions, the enforced invariants on the
first seeds of each split, demonstration cells that compile, stay within the length budget and embed
the module's own analysis functions, and -- because the enzyme upgrade must not change its output --
enzyme_mm's default files against hashes of the pre-upgrade toy. The full audit is
`python -m sage check --family F --n 60` plus experiments/sage_domains/heldout_constant.py.

  venv/bin/python -m sage.tests.test_osc_arr_enz            (or: venv/bin/python -m pytest sage/tests/test_osc_arr_enz.py)
"""
from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

import sage  # noqa: F401
from sage import registry
from sage.engine import make, write_instance
from sage.layouts import CANONICAL
from sage.traces import cell_code

FAMS = ("osc_damped", "arr_kinetics", "enzyme_mm")
SKILLS = {"inspect", "qc", "normalise", "model-choice", "estimate", "uncertainty", "validate", "report"}
METHOD_WORDS = {
    "osc_damped": ("Approach:", "Lomb", "Scargle", "least squares", "least-squares", "soft-L1", "robust", "quadratic",
                   "FFT", "periodogram", "interpolat", "sqrt(", "f_d", "gamma"),
    "arr_kinetics": ("Approach:", "kelvin", "Kelvin", "1/T", "ln k", "weighted", "weights", "least squares",
                     "log scale", "leave out", "true = reading"),
    "enzyme_mm": ("Approach:", "Lambert", "integrated", "Lineweaver", "double-reciprocal", "least squares",
                  "initial rate", "hyperbola"),
}
MAX_DEMO_LINES = 120
# enzyme_mm_00000 of train and test as the pre-upgrade toy wrote them (numpy 2.2.6, scipy 1.15.3, pandas 2.3.3)
ENZYME_PINNED = {("train", 0): ("966484ec76b307f5226a81cf78553b639be994f95e0a40db7b21461e6f494036",
                                "e4bd3fb4ca500533a0990362bd56772a53a2c85d34e18af7d2449784d7058a29"),
                 ("train", 7): ("7ac27479b2ec5df9872898321b6e53c1cbbfbc9670600947ab594f403cd9206a",
                                "0382a7cdc114315bad13e1b4f3ce012fd566b7fdcc5df065e477dd56d66f6a11"),
                 ("test", 0): ("193ad3febb74e1bb85c2b4b5e3be6b87ab0eddcd50763674b4a29afdc09fa2a5",
                               "75dfd398b85ff1f4297d34700e0f27975c8f8ddce2eaadf2066d1a8f6050eda9")}
PINNED_STACK = ("2.2.6", "1.15.3")

warnings.filterwarnings("ignore")


def test_registered_with_conventions():
    for name in FAMS:
        f = registry.get(name)
        assert set(f.skills) == set(f.steps), name                      # one skill per step
        assert set(f.skills.values()) <= SKILLS, (name, f.skills)
        assert f.seed_blocks["train"] != f.seed_blocks["test"]
    blocks = [b for n in FAMS for b in registry.get(n).seed_blocks.values()]
    others = [b for n, f in registry.families().items() if n not in FAMS for b in f.seed_blocks.values()]
    assert not set(blocks) & set(others), "seed block reused by another family"


def test_blind_prompts():
    for name in FAMS:
        f = registry.get(name)
        s = f.draw(np.random.default_rng(0), f.knobs(np.random.default_rng(0)))
        blind, full = f.task_text(s.data, blind=True), f.task_text(s.data)
        assert f.task_text(s.data) == f.task_text(s.data, blind=False)
        leaked = [w for w in METHOD_WORDS[name] if w in blind]
        assert not leaked, (name, leaked)                               # the method is not named
        for k in f.keys:                                                # same question, same answer keys
            assert f'"{k}"' in blind and f'"{k}"' in full
        if name == "enzyme_mm":
            assert blind == full                                        # its prompt never named the method
        else:
            assert "Approach:" in full and len(full) > len(blind), name


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
        if name != "enzyme_mm":                                         # enzyme_mm keeps its pre-upgrade provenance
            assert cfg["_provenance"]["skills"] == {str(s): v for s, v in f.skills.items()}
        codes = [cell_code(c) for c in f.demo(cfg)]
        for code in codes:
            compile(code, name, "exec")
        lines = sum(len(c.splitlines()) for c in codes)
        assert lines <= MAX_DEMO_LINES, (name, lines)
        final = codes[-1][codes[-1].index('"FINAL '):]                   # the printed answer line, keys in order
        assert [final.index(f"{k} = ") for k in f.keys] == sorted(final.index(f"{k} = ") for k in f.keys), name


def test_demo_functions_are_the_reference():
    """The functions a demonstration embeds (Cell.uses) must be the module's own analysis functions,
    so the trace runs the code that defined the gold."""
    from sage.domains import arrhenius, enzyme, oscillator
    for name, mod in (("osc_damped", oscillator), ("arr_kinetics", arrhenius), ("enzyme_mm", enzyme)):
        f = registry.get(name)
        inst = make(f, f.seed_blocks["train"])
        for cell in f.demo(CANONICAL.config(inst, [])):
            for fn in cell.uses:
                assert getattr(mod, fn.__name__) is fn, (name, fn.__name__)


def test_enzyme_default_output_unchanged():
    """enzyme_mm's emitted files are the pre-upgrade toy's, byte for byte (on the pinned library stack)."""
    import scipy
    if (np.__version__, scipy.__version__) != PINNED_STACK:
        print(f"  (numpy {np.__version__} / scipy {scipy.__version__} differ from the pinned stack; skipped)")
        return
    f = registry.get("enzyme_mm")
    with tempfile.TemporaryDirectory() as tmp:
        for (split, i), (h_cfg, h_csv) in ENZYME_PINNED.items():
            inst = make(f, f.seed_blocks[split] + i)
            inst.id = f"enzyme_mm_{i:05d}"
            d = Path(tmp) / split / inst.id
            write_instance(inst, d)
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            assert sha(d / "eval_config.json") == h_cfg, (split, i, "eval_config.json")
            assert sha(d / "data_files" / "progress_curves.csv") == h_csv, (split, i, "progress_curves.csv")


def main() -> int:
    for fn in (test_registered_with_conventions, test_blind_prompts, test_invariants_and_demos,
               test_demo_functions_are_the_reference, test_enzyme_default_output_unchanged):
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
