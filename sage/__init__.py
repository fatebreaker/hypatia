"""SAGE: one engine for verified, trap-annotated analysis exercises; a science is a plug-in.

Every generated exercise in this project -- genomics, causal-effect estimation, Kepler
photometry -- makes the same four promises, whichever generator minted it:

  * the ground truth is the reference analysis applied to the SAMPLED data, so a correct
    analysis reproduces it (and it lands within tolerance of the latent truth);
  * every documented wrong analysis (a trap) lands outside tolerance, tagged with the
    pipeline step it skips, so a wrong answer says WHICH step was skipped;
  * every accepted variant of the reference lands inside tolerance, so judgment is not
    graded on a coin flip;
  * under rl/reward.py the gold scores 1.0 and each trap exactly its step's credit.

Three generator styles grew up keeping those promises separately (generators/emit.py +
families.py, generators/causal_effects.py, generators/astro_kepler.py), each with its own
loop, seeding, provenance layout and real-data builder. This package keeps them once:

  sage/core.py      the contract a science supplies (Family, Sample, Trap, Cell, RealSource)
  sage/engine.py    the validity loop, deterministic seeding, parallel emission, manifests, mix
  sage/layouts.py   eval_config.json layout (the canonical one; legacy ones live with adapters)
  sage/real.py      the real-set builder (same reference and trap table, published gold)
  sage/traces.py    executed demonstration traces of the family's own reference code
  sage/audit.py     `check`: the four invariants plus a shortcut audit, on draws or corpora
  sage/variants.py  task-text variants (verified rewordings) and blind prompts
  sage/skills.py    the shared skill taxonomy and its coverage report
  sage/corpus.py    SFT corpus assembly balanced by science, with a per-row manifest
  sage/pools.py     balanced mixed and leave-one-domain-out RL pools
  sage/author.py    `author`: a plug-in scaffolded from a spec, and its PASS/FAIL check
  sage/domains/     plug-ins: causal (5), astro (3), genomics (4 + hic), enzyme (toy), epidemiology (2),
                    pharmacokinetics, ecology (2), survival (2), oscillator, arrhenius

  venv/bin/python -m sage list
  venv/bin/python -m sage check --family enzyme_mm --n 20

Importing the package pins BLAS to one thread per process (set before numpy loads), for
the reason generators/emit.py gives: emission is parallel across instances, and a thread
pool per worker oversubscribes the node by the worker count.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:            # generators/, rl/, paths.py live at the repo root
    sys.path.insert(0, str(ROOT))

__all__ = ["ROOT"]
