"""The real-set builder: the family's own reference and trap table, applied to real data.

Generated instances are graded against simulated truth; a family's REAL test set asks
the same question of real data, with the published value as gold. Validation is the same
code path the generator uses, never a re-implementation (build_kepler_real.py's rule):

  reproduced   the reference, run on the real data, lands within tolerance of the
               published value on every key -- a careful analysis of THESE data recovers
               the published answer; otherwise the question is not answerable from them
  live traps   documented wrong analyses that miss the published value. On real data no
               trap can be forced live, so all are treated as conditional: a trap inside
               tolerance is recorded with fails_at_step None
  variants     accepted variants outside tolerance are recorded (the question may hinge
               on an unstated choice there)

and an EXPLICIT strictness policy decides what ships: reproduced, and at least one live
trap -- or, where the source keeps CONTROLS, reproduced with no live trap. A control is a
question on which the corrective move a trap teaches would be wrong: a single-longitude
rotator whose periodogram is already right, so "double the periodogram's period" is not a
shortcut the set rewards. Kepler keeps controls for rot and not for transit/eb (the policy
of build_kepler_real.py since 2026-09-14); `Policy.keep_controls` overrides the source.

Gold is the published value, and the ablation table is scored against it exactly as the
generator's is against its gold: the "reference" row IS the gold (deviation 0), and what
the reference analysis returned goes to _provenance.reference_answer with its deviation.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sage.core import RealSource, RealTarget, as_trap, dev_units, finite
from sage.layouts import grader


@dataclass(frozen=True)
class Policy:
    require_reproduced: bool = True
    keep_controls: bool | None = None       # reproduced, no live trap; None = the source's default
    require_variants_inside: bool = False
    allow_partial: bool | None = None       # grade only the published keys; None = source.partial_keys

    def as_json(self) -> dict:
        """The recorded policy; `allow_partial` only when set, so default files are unchanged."""
        d = asdict(self)
        if d["allow_partial"] is None:
            d.pop("allow_partial")
        return d


def assess_real(source: RealSource, t: RealTarget, allow_partial: bool | None = None) -> dict:
    """Validate one target. Pure: no writes.

    Partial targets (source.partial_keys, or allow_partial): a target that publishes only
    some answer keys is graded on those keys alone -- reproduction, trap liveness and the
    ablation table use the graded keys, and the tolerance is computed with the reference's
    own values standing in for the unpublished keys (they are not graded). A target with
    every key published takes exactly the path it always did."""
    fam = source.family
    keys = fam.keys
    gold = {k: finite(t.published.get(k)) for k in keys}
    row = {"id": t.id, "family": fam.name, **{f"gold_{k}": gold[k] for k in keys}}
    partial = source.partial_keys if allow_partial is None else allow_partial
    graded = tuple(k for k in keys if gold[k] is not None)
    if not graded or (len(graded) < len(keys) and not partial):
        return {**row, "status": "no_published_value"}
    if len(graded) < len(keys):
        return _assess_partial(source, t, row, gold, graded)
    tol = {k: float(v) for k, v in source.tolerance(gold).items()}
    try:
        ref = fam.reference(t.data)
        ref = {k: finite(ref.get(k)) for k in keys}
    except Exception as exc:                                              # noqa: BLE001
        return {**row, "status": "reference_error", "error": f"{type(exc).__name__}: {exc}"[:400],
                "tb": traceback.format_exc()[-1500:]}
    ref_dev = dev_units(ref, gold, tol, keys) if all(v is not None for v in ref.values()) else float("inf")
    try:
        traps = {k: as_trap(v) for k, v in fam.traps(t.data).items()}
    except Exception as exc:                                              # noqa: BLE001
        return {**row, "status": "trap_error", "error": f"{type(exc).__name__}: {exc}"[:400],
                "tb": traceback.format_exc()[-1500:]}
    table = {"reference": {**gold, "fails_at_step": None, "deviation": 0.0}}
    live, dead, variants_out, no_answer = [], [], [], []
    for name, tr in traps.items():
        vals = {k: finite(tr.answer.get(k)) for k in keys}
        if any(v is None for v in vals.values()):
            no_answer.append(name)                 # no number: cannot be attributed, left out
            continue
        dv = dev_units(vals, gold, tol, keys)
        step = tr.step
        if tr.step is None:
            if dv > 1.0:
                variants_out.append(name)
        elif dv > 1.0:
            live.append(name)
        else:
            dead.append(name)
            step = None                            # coincides with the published value here
        table[name] = {**vals, "fails_at_step": step, "deviation": dv}
    reproduced = bool(ref_dev <= 1.0)
    return {**row, "status": "assessed", "reproduced": reproduced, "ref_dev_units": ref_dev,
            **{f"ref_{k}": ref[k] for k in keys}, "live_traps": live, "dead_traps": dead,
            "variants_outside_tol": variants_out, "traps_without_answer": no_answer,
            "_gold": gold, "_tol": tol, "_ref": ref, "_table": table}


def _assess_partial(source: RealSource, t: RealTarget, row: dict, gold_all: dict, graded: tuple) -> dict:
    fam = source.family
    keys = fam.keys
    try:
        ref_full = fam.reference(t.data)
        ref = {k: finite(ref_full.get(k)) for k in keys}
    except Exception as exc:                                              # noqa: BLE001
        return {**row, "status": "reference_error", "error": f"{type(exc).__name__}: {exc}"[:400],
                "tb": traceback.format_exc()[-1500:]}
    stand_in = {**{k: v for k, v in ref_full.items() if str(k).startswith("_")},
                **{k: (gold_all[k] if gold_all[k] is not None else ref[k]) for k in keys}}
    try:
        tol_all = source.tolerance(stand_in)
    except Exception as exc:                                              # noqa: BLE001
        return {**row, "status": "tolerance_error", "error": f"{type(exc).__name__}: {exc}"[:400]}
    tol = {k: float(tol_all[k]) for k in graded}
    gold = {k: gold_all[k] for k in graded}
    ref_g = {k: ref[k] for k in graded}
    ref_dev = dev_units(ref_g, gold, tol, graded) if all(v is not None for v in ref_g.values()) else float("inf")
    try:
        traps = {k: as_trap(v) for k, v in fam.traps(t.data).items()}
    except Exception as exc:                                              # noqa: BLE001
        return {**row, "status": "trap_error", "error": f"{type(exc).__name__}: {exc}"[:400],
                "tb": traceback.format_exc()[-1500:]}
    table = {"reference": {**gold, "fails_at_step": None, "deviation": 0.0}}
    live, dead, variants_out, no_answer = [], [], [], []
    for name, tr in traps.items():
        vals = {k: finite(tr.answer.get(k)) for k in graded}
        if any(v is None for v in vals.values()):
            no_answer.append(name)
            continue
        dv = dev_units(vals, gold, tol, graded)
        step = tr.step
        if tr.step is None:
            if dv > 1.0:
                variants_out.append(name)
        elif dv > 1.0:
            live.append(name)
        else:
            dead.append(name)
            step = None
        table[name] = {**vals, "fails_at_step": step, "deviation": dv}
    return {**row, "status": "assessed", "reproduced": bool(ref_dev <= 1.0), "ref_dev_units": ref_dev,
            **{f"ref_{k}": ref[k] for k in keys}, "live_traps": live, "dead_traps": dead,
            "variants_outside_tol": variants_out, "traps_without_answer": no_answer,
            "graded_keys": list(graded), "ungraded_keys": [k for k in keys if k not in graded],
            "_gold": gold, "_tol": tol, "_ref": ref, "_table": table}


def decide(r: dict, policy: Policy, source_controls: bool = False) -> str:
    """kept | kept_control | not_reproduced | no_live_trap | variant_outside_tol | <error>."""
    if r["status"] != "assessed":
        return r["status"]
    if policy.require_reproduced and not r["reproduced"]:
        return "not_reproduced"
    status = "kept"
    if not r["live_traps"]:
        if not (source_controls if policy.keep_controls is None else policy.keep_controls):
            return "no_live_trap"
        status = "kept_control"
    if policy.require_variants_inside and r["variants_outside_tol"]:
        return "variant_outside_tol"
    return status


def write_real(source: RealSource, t: RealTarget, r: dict, policy: Policy, d: Path) -> None:
    fam = source.family
    if d.exists():
        shutil.rmtree(d)
    (d / "data_files").mkdir(parents=True)
    for fn, src in t.files.items():
        shutil.copy2(src, d / "data_files" / fn)
    graded = r.get("graded_keys") or list(fam.keys)
    part = {"graded_keys": graded, "ungraded_keys": r["ungraded_keys"]} if r.get("ungraded_keys") else {}
    cfg = {"id": t.id, "task": source.task_text(t), "data_files": [f"data_files/{fn}" for fn in t.files],
           "ground_truth": r["_gold"], "grader": grader(graded, r["_tol"]),
           "_provenance": {"family": source.label, **source.provenance(t), "tol_rule": fam.tol_rule,
                           "note": source.note, "steps": fam.steps, "reference_answer": r["_ref"],
                           "reference_deviation_units": r["ref_dev_units"], "reproduced": r["reproduced"],
                           "control": r["decision"] == "kept_control", "live_traps": r["live_traps"],
                           "variants_outside_tol": r["variants_outside_tol"], "policy": policy.as_json(),
                           **part, "engine": "sage", "ablations": r["_table"]}}
    (d / "eval_config.json").write_text(json.dumps(cfg, indent=1, default=float))


_SRC = None
_PARTIAL = None


def _init(factory, args, partial=None):
    global _SRC, _PARTIAL
    _SRC = factory(*args)
    _PARTIAL = partial


def _assess_key(key):
    return assess_real(_SRC, _SRC.load(key), _PARTIAL)


def build(factory, args: tuple, out: Path, policy: Policy = Policy(), workers: int = 1,
          dry: bool = False) -> dict:
    """Assess every target of `factory(*args)` and write the kept ones to out/<label>/test/."""
    src = factory(*args)
    keys = src.keys()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_init,
                                         initargs=(factory, args, policy.allow_partial)) as pool:
            rows = pool.map(_assess_key, keys, chunksize=1)
    else:
        _init(factory, args, policy.allow_partial)
        rows = [_assess_key(k) for k in keys]
    for r in rows:
        r["decision"] = decide(r, policy, src.keep_controls)
    root = Path(out) / src.label / "test"
    kept = [(k, r) for k, r in zip(keys, rows) if r["decision"] in ("kept", "kept_control")]
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
        for k, r in kept:
            write_real(src, src.load(k), r, policy, root / r["id"])
    rep = {"built_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "source": src.label,
           "policy": policy.as_json(), "targets": len(rows), "kept": len(kept),
           "decisions": dict(Counter(r["decision"] for r in rows)),
           "live_trap_counts": dict(Counter(x for r in rows if r.get("reproduced") for x in r["live_traps"])),
           "dead_trap_counts": dict(Counter(x for r in rows if r.get("reproduced") for x in r["dead_traps"])),
           "variant_outside_counts": dict(Counter(x for r in rows if r.get("reproduced")
                                                  for x in r["variants_outside_tol"])),
           "rows": [{k: v for k, v in r.items() if not k.startswith("_") and k != "tb"} for r in rows]}
    if not dry:
        (root.parent / "_validation_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
