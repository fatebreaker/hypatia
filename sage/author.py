"""`sage author`: a new science as a documented, checked procedure.

  1. write a short spec (YAML or JSON): the science, answer keys, pipeline steps with skill
     tags, the traps and the step each skips, the tolerance rule, the task text and the
     method paragraph blind mode drops -- see SPEC_EXAMPLE below;
  2. python -m sage author --name decay --spec decay.yaml --out sage/domains/decay.py
     scaffolds the plug-in: everything the spec states is code (identity, steps, skills,
     tolerance, the task text with blind support, the trap table), and draw / write /
     reference / the trap analyses are TODO bodies with the contract in their docstrings;
  3. fill in the TODOs, then
     python -m sage author --check decay [--n 40]
     runs the invariants and the shortcut audit on fresh draws and prints a PASS/FAIL report
     a scientist can act on: which invariant fails, on which seeds, by how much, and what
     to change; exit status 0 only on PASS;
  4. add the module to sage/registry.py MODULES (the scaffold prints the line).

The check also verifies the contract around the invariants: every trap's step exists and
is skill-tagged with the shared names, the blind text drops the method paragraph and keeps
the answer format, the task's JSON template asks for exactly the answer keys, tolerances
are positive and finite, and the family's seed blocks do not overlap another family's.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import re
import statistics as st
import sys
import textwrap
import traceback
from pathlib import Path

SPEC_EXAMPLE = """\
name: decay_halflife                 # family name (also the default file name)
science: nuclear-physics             # the plug-in's domain (a "science" in pools and corpora)
class: RadioactiveDecay              # optional class name
keys: [half_life, background]        # answer keys, in the answer's order
units: {half_life: s, background: counts/s}
steps:                               # step number -> what the correct analysis does, and its skill
  1: {does: "subtract the constant background", skill: normalise}
  2: {does: "fit counts by Poisson likelihood, not least squares on log counts", skill: model-choice}
traps:                               # documented wrong analyses: the step each skips (null: accepted variant)
  - {name: no_background, step: 1, does: "fit a pure exponential"}
  - {name: log_linear_ls, step: 2, does: "least squares on log counts"}
  - {name: coarse_bins, step: null, does: "the Poisson fit on 2x coarser bins"}
tolerance: {half_life: "5%", background: "abs 0.5"}   # "p%" of |gold|, or "abs x"
files: [counts.csv]
seed_block: 30000000                 # optional: first train seed (test = +50000); default: a free block
task: |
  You are a nuclear physicist. A source was counted in 10-second bins for {duration} s.

  Data files:
  - data_files/counts.csv: one row per bin. Columns: t_s (bin start, s), counts.

  Question: what are the source's half-life (s) and the constant background rate (counts/s)?
approach: |
  Fit N(t) = A exp(-ln2 t / T) + B by Poisson maximum likelihood.
"""


# ----------------------------------------------------------------------------- spec
def load_spec(path: Path) -> dict:
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        return json.loads(text)
    import collections
    import collections.abc
    if not hasattr(collections, "Hashable"):       # the venv's PyYAML 3.10 predates Python 3.10
        collections.Hashable = collections.abc.Hashable
    import yaml
    return yaml.safe_load(text)


def spec_problems(spec: dict) -> list[str]:
    from sage.skills import SKILLS
    out = []
    for k in ("name", "science", "keys", "steps", "traps", "tolerance", "task"):
        if not spec.get(k):
            out.append(f"spec: '{k}' is required")
    if out:
        return out
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec["name"]):
        out.append(f"name {spec['name']!r}: use lower_snake_case")
    steps = {int(s): v for s, v in spec["steps"].items()}
    for s, v in steps.items():
        sk = (v or {}).get("skill")
        if sk not in SKILLS:
            out.append(f"step {s}: skill {sk!r} is not one of {', '.join(SKILLS)}")
    for t in spec["traps"]:
        if t.get("step") is not None and int(t["step"]) not in steps:
            out.append(f"trap {t.get('name')}: step {t['step']} is not a declared step")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(t.get("name", ""))):
            out.append(f"trap name {t.get('name')!r}: use lower_snake_case")
    for k in spec["keys"]:
        if k not in spec["tolerance"]:
            out.append(f"tolerance: no rule for key {k!r}")
        elif _tol_expr(str(spec["tolerance"][k]), k) is None:
            out.append(f"tolerance {k}: {spec['tolerance'][k]!r} is not 'p%' or 'abs x'")
    if not any(t.get("step") is not None for t in spec["traps"]):
        out.append("traps: at least one trap with a step is needed (otherwise nothing earns step credit)")
    return out


def _tol_expr(rule: str, key: str) -> str | None:
    rule = rule.strip()
    m = re.fullmatch(r"([0-9.]+)\s*%", rule)
    if m:
        return f"{float(m.group(1)) / 100!r} * abs(gold[{key!r}])"
    m = re.fullmatch(r"(?:abs\s+)?([0-9.eE+-]+)", rule)
    if m:
        return f"{float(m.group(1))!r}"
    return None


def _free_seed_block() -> int:
    from sage import registry
    used = [b for f in registry.families().values() for b in (f.seed_blocks or {}).values()]
    return (max(used) // 1_000_000 + 1) * 1_000_000 if used else 20_000_000


# ----------------------------------------------------------------------------- scaffold
def scaffold(spec: dict) -> str:
    """Python source of the plug-in the spec describes (TODO bodies where science is needed)."""
    probs = spec_problems(spec)
    if probs:
        raise SystemExit("spec problems:\n  " + "\n  ".join(probs))
    name, sci = spec["name"], spec["science"]
    cls = spec.get("class") or "".join(w.capitalize() for w in name.split("_"))
    keys = list(spec["keys"])
    steps = {int(s): v for s, v in spec["steps"].items()}
    units = spec.get("units") or {}
    files = spec.get("files") or ["data.csv"]
    block = int(spec.get("seed_block") or _free_seed_block())
    traps = [(t["name"], (None if t.get("step") is None else int(t["step"])), t.get("does", ""),
              bool(t.get("core", True))) for t in spec["traps"]]
    task = spec["task"].rstrip("\n")
    approach = (spec.get("approach") or "").strip()
    answer = "Return your answer as JSON: " + json.dumps(
        {"answer": {k: f"<{units.get(k, 'value')}>" for k in keys}}).replace('"<', "<").replace('>"', ">")
    step_lines = "\n".join(f"    {s}: {v['does']!r}," for s, v in sorted(steps.items()))
    skill_lines = ", ".join(f"{s}: {v['skill']!r}" for s, v in sorted(steps.items()))
    trap_lines = "\n".join(f"    {n!r}: ({s!r}, {c!r}),{'  # ' + d if d else ''}" for n, s, d, c in traps)
    tol_lines = ", ".join(f"{k!r}: {_tol_expr(str(spec['tolerance'][k]), k)}" for k in keys)
    tol_rule = ", ".join(f"{k} {spec['tolerance'][k]}" for k in keys)
    trap_doc = "\n".join(f"  {n:<24} step {s if s is not None else '-'}: {d}" for n, s, d, _ in traps)
    q = '"""'
    return f'''{q}{spec.get("title") or sci}: the `{name}` family (scaffolded by `python -m sage author`).

Science: {sci}. Answer keys: {", ".join(keys)}.

Pipeline steps (fails_at_step numbers) and their skills (sage/skills.py):
{chr(10).join(f"  {s}  [{v['skill']}] {v['does']}" for s, v in sorted(steps.items()))}

Traps (documented wrong analyses; step None = an accepted variant that must agree with the reference):
{trap_doc}

Tolerance: {tol_rule}.

TODO, in the porting checklist's order (sage/README.md section 4):
  1. draw() and write(): the data-generating process and the files. Put everything the analyst
     gets in `data`, the hidden truth (named like the answer keys) in `latent`.
  2. analyse(data, **switches): the correct analysis; each trap switches ONE stage off at its
     source (never mutate `data`). reference() and traps() call it.
  3. python -m sage author --check {name} --n 40 ; calibrate knobs and tolerance until I1-I3 hold,
     and add strengthen() for each trap that dies.
  4. add "sage.domains.{name}" to MODULES in sage/registry.py.
{q}
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sage.core import Family, Sample, Trap

KEYS = {tuple(keys)!r}
FILES = {files!r}
STEPS = {{
{step_lines}
}}
SKILLS = {{{skill_lines}}}
# trap name -> (step it skips, core). core=False: live only on draws carrying its defect.
TRAPS = {{
{trap_lines}
}}
TASK = {task!r}
APPROACH = {approach!r}
ANSWER = {answer!r}


def analyse(data, **switches) -> dict:
    """The reference analysis. A trap is the same analysis with one stage switched off, e.g.
    analyse(data, {traps[0][0]}=True). Must use only `data` (never the latent truth).
    Returns the answer keys, plus optional "_"-prefixed diagnostics (e.g. "_se")."""
    raise NotImplementedError("TODO analyse: the correct analysis, with one switch per trap")


class {cls}(Family):
    name = {name!r}
    domain = {sci!r}
    keys = KEYS
    steps = STEPS
    skills = SKILLS
    tol_rule = {tol_rule!r}
    note = "ground truth is the reference analysis on the sampled data, not the latent parameters"
    seed_blocks = {{"train": {block}, "test": {block + 50_000}}}

    def task_text(self, data, blind: bool = False) -> str:
        """The question and the data; blind mode drops the method paragraph."""
        head = TASK.format(**data.get("meta", {{}})) if isinstance(data, dict) else TASK
        method = "" if blind or not APPROACH else "Approach: " + APPROACH + "\\n\\n"
        return head + "\\n\\n" + method + ANSWER

    def knobs(self, rng):
        return 1.0                     # TODO: the difficulty knob strengthen() turns up

    def draw(self, rng, knob) -> Sample:
        """TODO: simulate one exercise. Return Sample(data, latent): data = everything the
        analyst gets (plus data["meta"] for the task text's placeholders), latent = the hidden
        truth, with entries named like the answer keys for I1."""
        raise NotImplementedError("TODO draw: the data-generating process")

    def write(self, data, d: Path) -> list[str]:
        """TODO: write the data files; return their names in listing order."""
        data["df"].to_csv(d / FILES[0], index=False)
        return [FILES[0]]

    def reference(self, data) -> dict:
        return analyse(data)

    def traps(self, data) -> dict:
        out = {{}}
        for name, (step, core) in TRAPS.items():
            out[name] = Trap({{k: v for k, v in analyse(data, **{{name: True}}).items() if not k.startswith("_")}},
                             step, core)
        return out

    def tolerance(self, gold) -> dict:
        return {{{tol_lines}}}

    def strengthen(self, knob, dead):
        return knob * 1.5              # TODO: make the defects behind the `dead` traps stronger

    def shortcuts(self, data) -> dict:
        return {{}}                     # TODO: data-light heuristics a policy might learn


FAMILIES = [{cls}()]
'''


# ----------------------------------------------------------------------------- check
def load_family(name: str):
    """A registered family, or the families of sage.domains.<name> / a .py path (registered
    in-process so the worker pool can find them)."""
    from sage import registry
    if not name.endswith(".py"):
        try:
            return [registry.get(name)]
        except KeyError:
            pass
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(name).stem, name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(name if "." in name else f"sage.domains.{name}")
    fams = list(getattr(mod, "FAMILIES", []))
    if not fams:
        raise SystemExit(f"{name}: no FAMILIES in the module")
    for f in fams:
        registry.register(f)
    return fams


def _details(fam, a, lat) -> dict:
    """Per invariant: what failed on one assessed draw, and by how much (units of tolerance)."""
    det = {"I1": [], "I2": [], "I3": [], "I4": []}
    for k in fam.keys:
        if k in lat and k not in fam.exact_keys and a.tol.get(k):
            gap = abs(a.gold[k] - lat[k]) / a.tol[k]
            if gap > 1:
                det["I1"].append({"what": k, "by": round(gap, 3)})
    det["I1"] += [{"what": r, "by": None} for r in a.insane if r not in fam.keys]
    for r in a.rows:
        if r.name in a.weak:
            det["I2"].append({"what": r.name, "by": round(r.dev, 3), "inv": "I2"})
        if r.name in a.loose:
            det["I3"].append({"what": r.name, "by": round(r.dev, 3) if math.isfinite(r.dev) else None})
    det["I4"] = [{"what": u, "by": None} for u in a.unpaid]
    return det


def _first_draw(fam, seed):
    """The loop's FIRST draw of a seed, assessed (engine.make's first try, replayed)."""
    import numpy as np
    from sage.engine import Instance, assess, unpaid
    stream = fam.rng_policy == "stream"
    rng = np.random.default_rng(seed)
    knobs = fam.knobs(rng if stream else np.random.default_rng(seed))
    sample = fam.draw(rng if stream else np.random.default_rng(seed), knobs)
    a = assess(fam, sample)
    a.unpaid = unpaid(Instance(fam, seed, 1, knobs, sample, a, 0.0))
    return _details(fam, a, sample.latent)


def _one(job) -> dict:
    name, seed = job
    from sage.engine import layout_of, make
    from sage.registry import get
    fam = get(name)
    try:
        first = _first_draw(fam, seed)
        inst = make(fam, seed)
    except NotImplementedError as e:
        return {"seed": seed, "error": f"NotImplementedError: {e}"}
    except Exception as e:                                           # noqa: BLE001
        return {"seed": seed, "error": f"{type(e).__name__}: {e}",
                "where": traceback.format_exc(limit=-3).strip().splitlines()[-3:]}
    a = inst.a
    lat = inst.sample.latent
    det = _details(fam, a, lat)
    cfg = json.loads(json.dumps(layout_of(fam).config(inst, []), default=float))
    try:
        sc = fam.shortcuts(inst.sample.data)
    except Exception:                                                # noqa: BLE001
        sc = {}
    from sage import variants as V
    text, kind = V.task_text_of(fam, inst.sample.data, blind=True)
    return {"seed": seed, "tries": inst.tries, "shipped_flagged": bool(inst.flags), "det": det, "first": first,
            "cfg": cfg, "blind": text if kind == "blind" else None,
            "latent": {k: lat[k] for k in fam.keys if k in lat} or None, "shortcuts": sc,
            "nominal": {r.name: (r.nominal, r.core) for r in a.rows},
            "tol": {k: a.tol[k] for k in fam.keys}}


def contract(fam) -> list[tuple[str, str]]:
    """[(severity, message)] for the contract around the invariants."""
    from sage import registry
    from sage import skills as SK
    from sage import variants as V
    out = [("FAIL", f"skills: {p}") for p in SK.problems(fam)]
    if not SK.skills_of(fam):
        out.append(("WARN", "no `skills` tags: steps cannot be pooled or rewarded across sciences"))
    if not V.blind_supported(fam):
        out.append(("WARN", "task_text has no `blind` parameter: `--blind` will ship the default text"))
    mine = sorted((fam.seed_blocks or {}).values())
    for other in registry.families().values():
        if other is fam or other.name == fam.name:
            continue
        for b in (other.seed_blocks or {}).values():
            if any(abs(b - m) < 50_000 for m in mine):
                out.append(("WARN", f"seed block {mine} within 50000 of {other.name}'s {b}"))
    return out


def text_checks(fam, rec: dict) -> list[tuple[str, str]]:
    from sage import variants as V
    out = []
    cfg = rec["cfg"]
    task = cfg["task"]
    ks = V.answer_keys_of(task)
    if ks and sorted(ks) != sorted(fam.keys):
        out.append(("FAIL", f"the task's JSON template asks for {ks}, the family's keys are {list(fam.keys)}"))
    elif not ks:
        out.append(("WARN", "the task has no JSON answer template naming the keys"))
    blind = rec.get("blind")
    if blind is not None:
        if "Approach:" in blind:
            out.append(("FAIL", "the blind text still carries an 'Approach:' paragraph"))
        if V.answer_template(blind) != V.answer_template(task):
            out.append(("FAIL", "the blind text changes the answer format"))
        if blind == task and "Approach:" in task:
            out.append(("FAIL", "task_text(blind=True) returns the default text"))
    return out


def check(name: str, n: int = 30, seed0: int | None = None, workers: int = 1) -> tuple[bool, str, dict]:
    """(passed, report text, details) for one family on n fresh draws."""
    from sage.audit import audit
    fams = load_family(name)
    reports, ok_all, details = [], True, {}
    for fam in fams:
        s0 = seed0 if seed0 is not None else next(iter(fam.seed_blocks.values()), 0)
        jobs = [(fam.name, s0 + i) for i in range(n)]
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                recs = pool.map(_one, jobs, chunksize=1)
        else:
            recs = [_one(j) for j in jobs]
        ok, text, det = _report(fam, recs, s0, n, audit)
        reports.append(text)
        details[fam.name] = det
        ok_all &= ok
    return ok_all, "\n\n".join(reports), details


HINTS = {
    "I1": "the reference misses the latent truth: the estimator is biased on these draws, or the tolerance is too "
          "tight for the data size. Look at the listed seeds' gold vs latent; loosen the tolerance or give the draw "
          "more data. If the estimand is DEFINED as the finite-sample estimate, drop I1 from `enforce`.",
    "I2": "a core trap lands inside tolerance, so skipping its step still passes. Make its defect stronger in "
          "strengthen() (the engine redraws with stronger knobs), check that the trap switches its stage off at the "
          "source, or mark it core=False if the defect is only present on some draws.",
    "I3": "an accepted variant lands outside tolerance: a defensible analysis would be graded wrong. Widen the "
          "tolerance, or demote the variant to a trap with the step it actually changes.",
    "I4": "the reward does not pay a trap exactly its step's credit: two traps with the same answer (ambiguous "
          "attribution), a non-finite or missing answer, or a trap answer outside the grader's range.",
}


def _worst(rows: list[dict], k: int = 5) -> str:
    """The k worst seeds. For I2 (a trap INSIDE tolerance) small is worst; otherwise large."""
    trap = rows and rows[0].get("inv") == "I2"
    key = (lambda r: r["by"] if r.get("by") is not None else -1) if trap else \
          (lambda r: -(r["by"] if r.get("by") is not None else math.inf))
    return ", ".join(f"seed {r['seed']}" + (f" at {r['by']:.2f} tol" if r.get("by") is not None else "")
                     for r in sorted(rows, key=key)[:k])


def _by_what(rows: list[dict], where: str) -> list[str]:
    by = {}
    for r in rows:
        by.setdefault(r["what"], []).append(r)
    return [f"         {w} ({where}): {len(v)} draws; worst {_worst(v)}" for w, v in by.items()]


def _report(fam, recs: list[dict], s0: int, n: int, audit) -> tuple[bool, str, dict]:
    errs = [r for r in recs if "error" in r]
    good = [r for r in recs if "error" not in r]
    L = [f"== sage author --check {fam.name}   ({fam.domain}; {n} fresh draws, seeds {s0}..{s0 + n - 1}; "
         f"enforced {sorted(fam.enforce)})"]
    verdict = []
    for sev, msg in contract(fam):
        L.append(f"  {sev:<4} contract: {msg}")
        verdict.append(sev)
    if errs:
        todo = [e for e in errs if e["error"].startswith("NotImplementedError")]
        L.append(f"  FAIL draws: {len(errs)}/{n} raised" + (f" -- {len(todo)} at a TODO: {todo[0]['error']}" if todo
                                                            else f": {errs[0]['error']}"))
        if errs[0].get("where"):
            L += [f"         {w}" for w in errs[0]["where"]]
        verdict.append("FAIL")
    det = {"errors": len(errs)}
    if good:
        seen = {}
        for r in good[:3]:
            for sev, msg in text_checks(fam, r):
                seen[msg] = sev
        for msg, sev in seen.items():
            L.append(f"  {sev:<4} task text: {msg}")
            verdict.append(sev)
        for inv in ("I1", "I2", "I3", "I4"):
            bad = [{**d, "seed": r["seed"]} for r in good for d in r["det"][inv]]
            first = [{**d, "seed": r["seed"]} for r in good for d in r["first"][inv]]
            nseeds = len({b["seed"] for b in bad})
            enforced = inv in fam.enforce
            n1 = len({b["seed"] for b in first})
            redraw = (f"  (first draw failed on {n1}/{len(good)} seeds; the loop's redraws fixed "
                      f"{n1 - nseeds})" if n1 else "")
            if not bad:
                L.append(f"  PASS {inv} on {len(good)}/{len(good)} shipped draws" + redraw)
                if n1:
                    L += _by_what(first, "first draw")
                continue
            sev = "FAIL" if enforced else "WARN"
            L.append(f"  {sev} {inv} fails on {nseeds}/{len(good)} shipped draws"
                     + ("" if enforced else " (not enforced: audited only)") + redraw)
            L += _by_what(bad, "shipped")
            L.append("         -> " + textwrap.fill(HINTS[inv], 100, subsequent_indent="            "))
            verdict.append(sev)
            det[inv] = {b["what"]: [(x["seed"], x.get("by")) for x in bad if x["what"] == b["what"]] for b in bad}
            det[f"{inv}_first"] = n1
        tries = [r["tries"] for r in good]
        flagged = sum(r["shipped_flagged"] for r in good)
        L.append(f"  info mean tries {st.mean(tries):.2f} (max {max(tries)} of {fam.max_tries}); "
                 f"shipped after max_tries with a failure: {flagged}/{len(good)}")
        if flagged:
            verdict.append("FAIL")
        if len(good) >= 4:
            from sage.audit import record
            rep = audit([{**record(r["cfg"], r["latent"], r["nominal"], r["shortcuts"], r["tries"])} for r in good],
                        fam.name, fam.enforce)
            b = rep["constants"]["best"]
            sev = "PASS" if b["pass_rate"] <= 0.10 else "WARN"
            L.append(f"  {sev} shortcut: best constant ({b['which']}) passes {b['pass_rate']:.0%}, reward "
                     f"{b['mean_reward']:.3f}" + ("" if sev == "PASS" else "  -> widen the latent ranges, or make "
                                                   "the answer depend on the data more"))
            verdict.append(sev)
            for k, v in rep["shortcuts"].items():
                sev = "PASS" if v["pass_rate"] <= 0.10 else "WARN"
                L.append(f"  {sev} shortcut {k}: passes {v['pass_rate']:.0%}, reward {v['mean_reward']:.3f}")
                verdict.append(sev)
            for k, v in rep["I2"]["per_trap"].items():
                L.append(f"  info trap {k:<26} step {v['step']!s:<4} live {v.get('live', 0)}/{v['n']}   earns "
                         f"{rep['trap_earns'].get(k, 'n/a')}")
            det["audit"] = {"constants": rep["constants"]["best"], "shortcuts": rep["shortcuts"]}
    ok = "FAIL" not in verdict and bool(good)
    L.append(f"  VERDICT: {'PASS' if ok else 'FAIL'}" + (f" ({verdict.count('WARN')} warnings)" if verdict.count("WARN")
                                                         else ""))
    return ok, "\n".join(L), det
