"""The validity loop, deterministic seeding, parallel emission, manifests and mix pools.

One loop for every science. An instance is drawn, analysed by the family's reference and
traps, and accepted only if the invariants the family enforces hold:

  I1  the reference lands within tolerance of the latent truth (and the family's own
      sanity checks pass) -- otherwise the gold is not what the task asks for; redraw
  I2  every CORE trap lands outside tolerance -- otherwise skipping that step still passes,
      so the instance teaches the wrong lesson; strengthen the knobs and redraw. A
      CONDITIONAL trap that coincides with the reference is recorded with step None
  I3  every accepted variant lands inside tolerance -- otherwise a defensible analysis is
      graded wrong, i.e. judgment is graded on a coin flip; redraw
  I4  under rl/reward.py the gold scores 1.0 and every credit-bearing trap earns exactly
      (s-1)/S -- otherwise the step credit a policy is trained on is not the step it
      skipped (two traps with one answer, a trap outside the grader's range)

After `max_tries` the last draw is kept and flagged (manifest `weak_ids`), as every legacy
generator did; filtering is the consumer's decision. Invariants a family does not enforce
are still measured and reported (manifest `unenforced`, and `python -m sage check`).

Seeding: instance i of a split is drawn from seed = seed0 + i, so output never depends on
the worker count, and a family's split seed blocks may not overlap (`check_seed_range`):
an overlap is train/test leakage that no downstream filter can see.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sage.core import Family, Sample, as_trap, dev_units

DEFAULT_SPAN = 50_000          # seeds reserved per split block when no later block bounds it


# ----------------------------------------------------------------------------- records
@dataclass
class Row:
    """One ablation-table line: a trap or accepted variant, as assessed on this draw."""
    name: str
    answer: dict
    nominal: int | None        # the step the family declared (None: accepted variant)
    step: int | None           # as recorded: None where a conditional trap coincides here
    core: bool
    dev: float                 # units of tolerance, maxed over keys

    @property
    def live(self) -> bool:
        return self.dev > 1.0


@dataclass
class Assessment:
    gold: dict                 # the reference's output (answer keys + "_" diagnostics)
    tol: dict
    rows: list[Row]
    weak: list[str]            # I2: core traps inside tolerance
    loose: list[str]           # I3: accepted variants outside tolerance
    insane: list[str]          # I1: keys where gold misses the latent, + sanity reasons
    unpaid: list[str] = field(default_factory=list)   # I4: rows the reward does not pay right

    def by_invariant(self) -> dict[str, list[str]]:
        return {"I1": self.insane, "I2": self.weak, "I3": self.loose, "I4": self.unpaid}


@dataclass
class Instance:
    family: Family
    seed: int
    tries: int
    knobs: Any
    sample: Sample
    a: Assessment
    secs: float                # reference + trap analysis time of the accepted draw
    id: str = ""

    def failures(self, enforced: bool = True) -> dict[str, list[str]]:
        fam = self.family
        return {k: v for k, v in self.a.by_invariant().items()
                if v and ((k in fam.enforce) == enforced)}

    @property
    def flags(self) -> list[str]:
        """Enforced failures, in the legacy generators' format (weak + loose + sanity:k)."""
        f = self.failures(True)
        return (list(f.get("I2", [])) + list(f.get("I3", [])) + list(f.get("I4", []))
                + [f"sanity:{k}" for k in f.get("I1", [])])

    @property
    def ground_truth(self) -> dict:
        return {k: self.a.gold[k] for k in self.family.keys}


# ----------------------------------------------------------------------------- the loop
def assess(fam: Family, sample: Sample) -> Assessment:
    """Reference, tolerance and trap table of one draw, with I1-I3 evaluated. The trap rule
    is astro_kepler.ablation_table's, which generalises causal_effects.make (every trap
    core) and emit.py (every trap core, variants unchecked)."""
    data = sample.data
    gold = fam.reference(data)
    tol = fam.tolerance(gold)
    keys, exact = fam.keys, fam.exact_keys
    insane = []
    for k in keys:
        if k not in sample.latent:
            continue
        if k in exact:
            ok = str(gold[k]) == str(sample.latent[k])
        else:
            ok = abs(gold[k] - sample.latent[k]) <= tol[k]
        if not ok:
            insane.append(k)
    insane += list(fam.sanity(gold, tol, sample.latent))
    rows, weak, loose = [], [], []
    for name, t in fam.traps(data).items():
        t = as_trap(t)
        dev = dev_units(t.answer, gold, tol, keys, exact)
        step = t.step
        if t.step is not None and dev <= 1.0:
            if t.core:
                weak.append(name)
            else:
                step = None                       # coincides with the reference on this draw
        elif t.step is None and dev > 1.0:
            loose.append(name)
        rows.append(Row(name, dict(t.answer), t.step, step, t.core, dev))
    return Assessment(gold, tol, rows, weak, loose, insane)


def layout_of(fam: Family):
    from sage.layouts import CANONICAL
    return fam.layout or CANONICAL


def unpaid(inst: Instance) -> list[str]:
    """I4 on the instance's own rendered eval_config: rows the reward does not pay exactly
    their step's credit, and "oracle" if the gold itself does not score 1.0."""
    from sage.audit import reward_check
    cfg = layout_of(inst.family).config(inst, [])
    rc = reward_check(cfg)
    return (["oracle"] if not rc["oracle_ok"] else []) + [r["name"] for r in rc["rows"] if not r["ok"]]


def make(fam: Family, seed: int, max_tries: int | None = None) -> Instance:
    """Draw until the enforced invariants hold, strengthening after dead core traps.

    rng_policy "stream" (causal, astro): one generator per instance, continued across
    retries. "reseed" (emit.py families): knobs from default_rng(seed), and every retry
    redraws from default_rng(seed) with the strengthened knobs. Knobs are strengthened
    only when another try follows, so the knobs recorded are the accepted draw's."""
    max_tries = max_tries or fam.max_tries
    stream = fam.rng_policy == "stream"
    rng = np.random.default_rng(seed)
    knobs = fam.knobs(rng if stream else np.random.default_rng(seed))
    inst = None
    for tries in range(1, max_tries + 1):
        sample = fam.draw(rng if stream else np.random.default_rng(seed), knobs)
        t0 = time.time()
        a = assess(fam, sample)
        inst = Instance(fam, seed, tries, knobs, sample, a, time.time() - t0)
        a.unpaid = unpaid(inst)
        if not inst.failures(True):
            break
        if a.weak and tries < max_tries:
            knobs = fam.strengthen(knobs, list(a.weak))
    return inst


# ----------------------------------------------------------------------------- emission
def check_seed_range(fam: Family, split: str, seed0: int, n: int) -> None:
    """Refuse a seed range that reaches into another split's block of the same family."""
    blocks = sorted(fam.seed_blocks.items(), key=lambda kv: kv[1])
    for j, (s, b) in enumerate(blocks):
        if s == split:
            continue
        end = blocks[j + 1][1] if j + 1 < len(blocks) else b + DEFAULT_SPAN
        if seed0 < end and b < seed0 + n:
            raise ValueError(f"{fam.name}/{split}: seeds [{seed0}, {seed0 + n}) overlap split "
                             f"'{s}' block [{b}, {end}) -- that is train/test leakage")


_CACHES: dict = {}                     # variant caches, loaded once per process


@dataclass(frozen=True)
class EmitOptions:
    """Opt-in generality features (all off by default: the default output is byte-identical
    to the legacy generators). None of them touches the draw, the gold or the trap table, so
    I1-I4 hold exactly as without them.

      variants  K > 1: K task texts per instance (the plug-in's, then verified rewordings:
                LLM cache, plug-in phrasings, generic); `task` is one chosen by the seed, the
                others go to _provenance.task_variants (sources in task_variant_sources)
      blind     task = task_text(data, blind=True) where the plug-in supports it
                (_provenance.task_mode records "blind" or "blind-unsupported")
      skills    _provenance.skills and trap_skills from the family's `skills` tags"""
    variants: int = 0
    blind: bool = False
    skills: bool = False
    variant_cache: tuple = ()

    @property
    def active(self) -> bool:
        return self.variants > 1 or self.blind or self.skills


def decorate(inst: Instance, cfg: dict, files_dir: Path | None, opts: EmitOptions, cache: dict | None = None) -> dict:
    """Apply the opt-in features to a rendered eval_config (in place)."""
    from sage import skills as SK
    from sage import variants as V
    fam, data = inst.family, inst.sample.data
    prov = cfg.setdefault("_provenance", {})
    label = "canonical"
    if opts.blind:
        text, kind = V.task_text_of(fam, data, blind=True)
        prov["task_mode"] = kind
        if kind == "blind":
            cfg["task"], label = text, "blind"
    if opts.variants > 1:
        if cache is None and opts.variant_cache:
            key = tuple(map(str, opts.variant_cache))
            if key not in _CACHES:
                _CACHES[key] = V.load_cache(opts.variant_cache)
            cache = _CACHES[key]
        cands = V.candidates(cfg["task"], opts.variants, inst.seed, label=label, fam=fam, data=data,
                             blind=opts.blind and label == "blind", cache=cache, answer_keys=fam.keys,
                             files_dir=files_dir)
        j = V.choose(len(cands), inst.seed)
        cfg["task"] = cands[j][1]
        prov["task_variant_chosen"] = cands[j][0]
        prov["task_variants"] = [t for i, (_, t) in enumerate(cands) if i != j]
        prov["task_variant_sources"] = [s for i, (s, _) in enumerate(cands) if i != j]
    if opts.skills:
        SK.record(fam, cfg)
    return cfg


def write_instance(inst: Instance, d: Path, opts: EmitOptions | None = None) -> dict:
    lay = layout_of(inst.family)
    (d / "data_files").mkdir(parents=True, exist_ok=True)
    files = inst.family.write(inst.sample.data, d / "data_files")
    cfg = lay.config(inst, files)
    if opts is not None and opts.active:
        cfg = decorate(inst, cfg, d / "data_files", opts)
    (d / "eval_config.json").write_text(lay.dumps(cfg))
    return cfg


def _emit_one(job) -> tuple:
    name, i, seed, out, prefix = job[:5]
    opts = job[5] if len(job) > 5 else None
    from sage.registry import get
    fam = get(name)
    inst = make(fam, seed)
    inst.id = f"{prefix or fam.name}_{i:05d}"
    cfg = write_instance(inst, Path(out) / inst.id, opts)
    extra = (cfg["_provenance"].get("task_mode"),) if opts is not None and opts.active else ()
    return (inst.id, inst.flags, inst.failures(False), inst.tries, inst.secs) + extra


def emit(fam: Family, split: str, n: int, out_root: Path, seed0: int | None = None,
         workers: int = 1, prefix: str | None = None, allow_seed_overlap: bool = False,
         opts: EmitOptions | None = None) -> dict:
    """Mint n instances into out_root/<family>/<split>/<prefix>_<i:05d>/ plus a manifest.
    Instances are seeded by index, so the output does not depend on `workers`. `opts`
    switches on the generality features (EmitOptions); None or all-off is the legacy output."""
    if seed0 is None:
        if split not in fam.seed_blocks:
            raise ValueError(f"{fam.name} has no seed block for split '{split}'; pass seed0")
        seed0 = fam.seed_blocks[split]
    if not allow_seed_overlap:
        check_seed_range(fam, split, seed0, n)
    out = Path(out_root) / fam.name / split
    out.mkdir(parents=True, exist_ok=True)
    on = opts is not None and opts.active
    jobs = [(fam.name, i, seed0 + i, str(out), prefix) + ((opts,) if on else ()) for i in range(n)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_emit_one, jobs, chunksize=1)
    else:
        res = [_emit_one(j) for j in jobs]
    weak_ids = [r[0] for r in res if r[1]]
    unenf: dict[str, list[str]] = {}
    for r in res:
        for inv in r[2]:
            unenf.setdefault(inv, []).append(r[0])
    man = {"family": fam.name, "split": split, "n": n, "seed0": seed0,
           "n_all_traps_valid": n - len(weak_ids), "n_weak": len(weak_ids), "weak_ids": weak_ids,
           "mean_tries": sum(r[3] for r in res) / max(n, 1),
           "mean_reference_seconds": sum(r[4] for r in res) / max(n, 1),
           "engine": "sage", "enforced": sorted(fam.enforce),
           "unenforced": {k: {"n": len(v), "ids": v} for k, v in sorted(unenf.items())}}
    if on:
        from collections import Counter
        man["generality"] = {"variants": opts.variants, "blind": opts.blind, "skills": opts.skills,
                             "variant_cache": list(map(str, opts.variant_cache)),
                             "task_mode": dict(Counter(r[5] for r in res if r[5]))}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def mix(names: list[str], split: str, out_root: Path, pool: str) -> int:
    """out_root/<pool>/<split>/: symlinks to every listed family's instances, one RL
    training pool (causal_effects.mix / astro_kepler.mix, for any set of families)."""
    out = Path(out_root) / pool / split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(names):
        for d in sorted((Path(out_root) / f / split).glob("*")):
            if not (d / "eval_config.json").exists():
                continue
            link = out / d.name
            if not link.exists():
                link.symlink_to(d.resolve(), target_is_directory=True)
            n += 1
    return n


def refuse_data_root(path: Path) -> None:
    """Scratch runs (tests, audits) must never write into the shared data root."""
    try:
        from paths import data_root
        root = Path(data_root()).resolve()
    except Exception:                                     # noqa: BLE001 -- no root, nothing to protect
        return
    p = Path(path).resolve()
    if p == root or root in p.parents:
        raise SystemExit(f"refusing to write under the shared data root {root}: {p}")


def cpu_workers(requested: int) -> int:
    return max(1, min(requested, os.cpu_count() or 1))
