"""The plug-in contract: what a science supplies, and nothing it does not have to.

A family is the data-generating process, the correct analysis and the documented wrong
analyses of ONE kind of exercise. Everything else -- the retry loop that makes an instance
valid, seeding, the ablation table, tolerance arithmetic, file layout, manifests, pooled
RL directories, the real-data builder, demonstration traces, the audit -- is the engine's,
so it is written once and cannot drift between sciences.

What a family supplies (methods on a `Family` subclass; * = has a default):

  name, domain, keys, steps   identity: answer keys, and step names indexed from 1 (the
                              numbers traps' `fails_at_step` refer to)
  task_text(data)             the prompt; it may depend on the draw (a real trait's name)
  knobs(rng)*                 the initial difficulty knobs / randomised config
  draw(rng, knobs)            -> Sample(data, latent): simulated data plus hidden truth
  write(data, dir)            data files -> their names, in listing order
  reference(data)             the correct analysis -> gold (answer keys, plus optional
                              "_"-prefixed diagnostics such as a standard error)
  traps(data)                 {name: Trap(answer, step, core)}: documented wrong analyses;
                              step None = an accepted variant of the reference
  tolerance(gold)             {key: absolute tolerance}; may read the gold's diagnostics
  strengthen(knobs, dead)*    harder knobs after `dead` core traps landed inside tolerance
  sanity(gold, tol, latent)*  extra reasons the gold is unreachable from the data (extends I1)
  shortcuts(data)*            data-light heuristics the audit should price
  demo(cfg)*                  cells a demonstration trace executes (the reference's code)
  max_tries*, rng_policy*, enforce*, seed_blocks*, tol_rule*, note*

Generality hooks, all optional (sage/variants.py, sage/skills.py); a plug-in without them works
unchanged and is reported as "blind unsupported" / "untagged":

  task_text(data, blind=False)  blind=True: the question and the data, not the method
  skills*                     {step: skill}, from the shared taxonomy sage.skills.SKILLS
  task_phrasings(data, blind=False)*   more templates of the task text (validated as variants)
  blind_from_text(task)*      the blind text derived from the default text (default: drop
                              the "Approach:" paragraph); used where only text is available

`reference` and `traps` receive `sample.data` and never `sample.latent`: the gold is
computed from what the analyst is given, so a correct analysis reproduces it exactly and
the latent truth cannot leak into it. The latent is recorded and used only for invariant
I1 (the gold must land within tolerance of it).

Real data plug in through `RealSource`: `keys()` enumerates targets (picklable, for the
worker pool) and `load(key)` returns a RealTarget(id, data, published answer, metadata,
files). The builder applies the SAME `reference` and `traps` to it (sage/real.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

# I1 reference within tolerance of the latent truth; I2 every core trap outside tolerance;
# I3 every accepted variant inside it; I4 the reward pays gold 1.0 and a trap its step.
INVARIANTS = ("I1", "I2", "I3", "I4")


@dataclass
class Sample:
    """One draw: `data` is everything the analyst gets (what `write` serialises and
    `reference`/`traps` analyse); `latent` is the hidden truth, recorded in provenance.
    A latent entry named like an answer key is the value I1 compares the gold with."""
    data: Any
    latent: dict = field(default_factory=dict)


class Trap(NamedTuple):
    """A documented wrong analysis and the pipeline step it skips.

    core=True: it must be live on every instance (I2); a draw where it is not is
    strengthened and redrawn. core=False (conditional): live only on some draws -- a short
    transit, an eccentric orbit -- and where it coincides with the reference it is recorded
    with step None, an accepted variant on that instance, rather than rejecting the draw.
    step=None: an accepted variant of the reference; it must land inside tolerance (I3)."""
    answer: dict
    step: int | None
    core: bool = True


@dataclass
class Cell:
    """One executed demonstration cell. `uses` are functions whose source is embedded
    verbatim above `code` (inspect.getsource), so the demonstration runs the same code
    that defined the ground truth. `prose` may be a callable of the earlier cells' stdout
    (e.g. to quote the noise level the inspection cell printed)."""
    prose: str | Callable[[list[str]], str]
    code: str
    uses: tuple = ()
    preamble: str = ""


class Family:
    """Base class for a plug-in. Subclasses override the methods that differ; the class
    attributes below are the knobs of the engine's loop, with the defaults a new science
    should keep."""

    name: str = ""
    domain: str = ""
    keys: tuple[str, ...] = ()
    exact_keys: tuple[str, ...] = ()          # categorical answer keys, graded by equality
    steps: dict[int, str] = {}
    skills: dict[int, str] = {}               # step -> sage.skills.SKILLS name; {} = untagged
    max_tries: int = 12
    # "stream": one generator seeded once per instance, continued across retries (a retry
    # is a fresh draw). "reseed": every retry restarts from the same seed with the
    # strengthened knobs, so a retry is the SAME instance made harder (emit.py's contract).
    rng_policy: str = "stream"
    enforce: frozenset = frozenset(INVARIANTS)  # invariants that reject a draw; the rest are audited
    seed_blocks: dict[str, int] = {}          # split -> first seed; blocks must not overlap
    tol_rule: str = ""
    note: str = ""
    demo_timeout: int = 300
    demonstration_only: bool = False          # a worked example, not part of any experiment
    layout: Any = None                        # None -> sage.layouts.CANONICAL

    # ------------------------------------------------------------------ required
    def task_text(self, data) -> str:
        raise NotImplementedError

    def draw(self, rng: np.random.Generator, knobs) -> Sample:
        raise NotImplementedError

    def write(self, data, data_dir: Path) -> list[str]:
        raise NotImplementedError

    def reference(self, data) -> dict:
        raise NotImplementedError

    def traps(self, data) -> dict[str, Trap]:
        raise NotImplementedError

    def tolerance(self, gold: dict) -> dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------ optional
    def knobs(self, rng: np.random.Generator):
        return 1.0

    def strengthen(self, knobs, dead: list[str]):
        return knobs

    def sanity(self, gold: dict, tol: dict, latent: dict) -> list[str]:
        return []

    def shortcuts(self, data) -> dict[str, dict]:
        return {}

    def demo(self, cfg: dict) -> list[Cell] | None:
        return None

    def extra_provenance(self, inst) -> dict:
        return {}

    def __repr__(self) -> str:
        return f"<Family {self.name} ({self.domain}) keys={list(self.keys)}>"


class RealTarget(NamedTuple):
    """One real question: the data as the family's `reference` reads it, the published
    answer that becomes the gold, provenance, and the files to ship (copied verbatim)."""
    id: str
    data: Any
    published: dict
    meta: dict
    files: dict                      # data-file name -> source path


class RealSource:
    """Real targets for one family. Subclasses set `family`, `label` (the provenance family
    name, e.g. "kepler_transit") and `note`, and implement `keys` and `load`."""
    family: Family
    label: str = ""
    note: str = ""
    keep_controls: bool = False      # ship reproduced targets with no live trap (see sage/real.py)
    # grade a target on the keys it has published values for (e.g. a hazard ratio with no
    # published median); False: a target missing any key is dropped (no_published_value)
    partial_keys: bool = False

    def keys(self) -> list:
        raise NotImplementedError

    def load(self, key) -> RealTarget:
        raise NotImplementedError

    def targets(self) -> Iterable[RealTarget]:
        return (self.load(k) for k in self.keys())

    def tolerance(self, published: dict) -> dict:
        return self.family.tolerance(published)

    def task_text(self, target: RealTarget) -> str:
        """The real question's prompt; by default the family's. A partial-key source may
        override it to ask only for the keys it grades."""
        return self.family.task_text(target.data)

    def provenance(self, target: RealTarget) -> dict:
        return dict(target.meta)


# ----------------------------------------------------------------------------- helpers
def seed_of(rng: np.random.Generator) -> int:
    """The integer a fresh engine generator was seeded with. Under rng_policy "reseed" the
    engine hands `draw` a new `default_rng(seed)` per try; a legacy simulator that seeds
    itself (`generate(cfg, seed)`) recovers the seed here and consumes an identical stream."""
    return int(rng.bit_generator.seed_seq.entropy)


def as_trap(t) -> Trap:
    """Accept a Trap or a plain (answer, step[, core]) tuple."""
    return t if isinstance(t, Trap) else Trap(*t)


def dev_units(answer: dict, gold: dict, tol: dict, keys, exact=()) -> float:
    """Deviation in units of tolerance, maxed over keys: the reward's own arithmetic
    (rl/reward.py `_linf`), so "live" here means exactly "fails the grader" there. A
    categorical key contributes 0 if equal and inf otherwise, as does a numeric key graded
    with zero tolerance (an integer code). nan if a key is missing or non-numeric."""
    try:
        num = [k for k in keys if k not in exact]
        d = max((abs(answer[k] - gold[k]) / tol[k]) if tol[k] else
                (0.0 if answer[k] == gold[k] else math.inf) for k in num) if num else 0.0
        for k in exact:
            if str(answer[k]) != str(gold[k]):
                d = math.inf
        return float(d)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def finite(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
