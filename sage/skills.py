"""The shared skill taxonomy: eight skills every science's pipeline steps map onto.

Each family tags its pipeline steps with one skill each, as a class attribute
`skills = {step: name}`. A trap's `fails_at_step` then also names the skill the wrong
analysis skipped, so a skill can be rewarded, measured and compared across sciences
("does training on genomics QC transfer to astronomy QC?"). A family without `skills` is
untagged; nothing breaks, and the audit reports it.

  inspect       look at the data's structure before analysing it: which record, signal or
                column is which, units, roles (e.g. which eclipse is the primary)
  qc            remove or repair what would corrupt the analysis: outliers, glitches, failed
                runs, contamination, duplicate or re-tagged records, censored / BLQ values,
                related samples
  normalise     put the data on a common footing: detrending, calibration offsets, unit or
                scale conversion of inputs, allele alignment, batch / weekday / baseline terms
  model-choice  choose the estimator or design that identifies the quantity: adjustment set,
                instrument, compartment count, change-point, conditional likelihood
  estimate      compute the quantity correctly with the chosen model: the fit, its weighting,
                the right conversion of its parameters
  uncertainty   quantify uncertainty: standard errors, intervals, error models
  validate      check the answer against alternatives or diagnostics: period aliases,
                harmonics, sensitivity to an unstated choice
  report        report the quantity asked for in the form asked for: the estimand (ATT, not
                ATE), the natural not the damped frequency, units and scale

The names are fixed (other plug-ins and the reward analyses key on them): add a science by
mapping its steps onto these, never by adding a name.

Recording is opt-in, because a new provenance block would change emitted bytes:
`sage emit --skills` writes `_provenance.skills` ({"<step>": skill}, the format the newer
plug-ins already copy themselves; left untouched when present) and
`_provenance.trap_skills` ({trap: skill or None}, from each ablation row's recorded step).
"""
from __future__ import annotations

from collections import Counter, defaultdict

SKILLS = ("inspect", "qc", "normalise", "model-choice", "estimate", "uncertainty", "validate", "report")


def skills_of(fam) -> dict[int, str]:
    """{step: skill} for a family, {} if untagged (no attribute, or an empty one)."""
    sk = getattr(fam, "skills", None) or {}
    return {int(k): v for k, v in dict(sk).items()}


def problems(fam) -> list[str]:
    """Why a family's tagging is invalid: unknown names, steps that do not exist, steps left
    untagged. [] for a valid tagging and for an untagged family (untagged is not invalid)."""
    sk = skills_of(fam)
    if not sk:
        return []
    out = [f"step {s}: unknown skill {v!r} (allowed: {', '.join(SKILLS)})" for s, v in sk.items() if v not in SKILLS]
    steps = {int(s) for s in (getattr(fam, "steps", None) or {})}
    out += [f"step {s} is tagged but the family has no such step" for s in sorted(set(sk) - steps)]
    out += [f"step {s} ({fam.steps[s]!r}) has no skill" for s in sorted(steps - set(sk))]
    return out


def step_skill(fam, step) -> str | None:
    if step is None:
        return None
    return skills_of(fam).get(int(step))


def provenance_block(fam) -> dict:
    """`_provenance.skills` as the plug-ins already write it: {"<step>": skill}."""
    return {str(s): v for s, v in sorted(skills_of(fam).items())}


def trap_skills(fam, cfg: dict) -> dict:
    """{trap: skill of the step it skips} from an eval_config's own ablation table (the step
    as recorded there; None for accepted variants and coincident conditional traps)."""
    abl = (cfg.get("_provenance") or {}).get("ablations") or {}
    return {k: step_skill(fam, a.get("fails_at_step")) for k, a in abl.items() if k != "reference"}


def record(fam, cfg: dict) -> dict:
    """Write the skill blocks into cfg["_provenance"] (in place; idempotent). A `skills`
    block the plug-in already wrote is kept where it is when it agrees with the class
    attribute; a disagreeing copy is replaced (the attribute is the source of truth)."""
    if not skills_of(fam):
        return cfg
    prov = cfg.setdefault("_provenance", {})
    block = provenance_block(fam)
    have = prov.get("skills")
    if have is None or {str(k): v for k, v in have.items()} != block:
        prov["skills"] = block
    prov["trap_skills"] = trap_skills(fam, cfg)
    return cfg


# ----------------------------------------------------------------------------- coverage
def coverage(fams) -> dict:
    """Which skills each family and each domain exercises, from the step tags and from the
    trap tables' nominal steps (a skill with a step but no trap cannot be rewarded)."""
    per_family, per_domain = {}, defaultdict(Counter)
    for fam in fams:
        sk = skills_of(fam)
        entry = {"domain": fam.domain, "tagged": bool(sk), "problems": problems(fam),
                 "steps": {s: sk.get(s) for s in sorted(int(x) for x in (fam.steps or {}))}}
        entry["skills"] = sorted(set(sk.values()), key=SKILLS.index) if sk and not entry["problems"] else sorted(set(sk.values()))
        per_family[fam.name] = entry
        for v in sk.values():
            per_domain[fam.domain][v] += 1
    domains = {}
    for d in sorted({f["domain"] for f in per_family.values()}):
        fs = [n for n, f in per_family.items() if f["domain"] == d]
        domains[d] = {"families": fs, "untagged": [n for n in fs if not per_family[n]["tagged"]],
                      "skills": dict(per_domain[d]),
                      "missing": [s for s in SKILLS if s not in per_domain[d]]}
    total = Counter()
    for c in per_domain.values():
        total.update(c)
    return {"families": per_family, "domains": domains, "overall": {s: total.get(s, 0) for s in SKILLS},
            "never_tagged": [s for s in SKILLS if not total.get(s)]}


def trap_rows(fam, cfgs: list[dict]) -> dict:
    """{skill: number of credit-bearing trap rows (a recorded step) that target it} over the
    given eval_configs -- the skills a policy can actually be rewarded for on them."""
    rows = Counter()
    for cfg in cfgs:
        for name, a in ((cfg.get("_provenance") or {}).get("ablations") or {}).items():
            s = step_skill(fam, a.get("fails_at_step")) if name != "reference" else None
            if s:
                rows[s] += 1
    return {s: rows[s] for s in SKILLS if rows[s]}


def render(cov: dict) -> str:
    L = ["== skill coverage (step tags, sage/skills.py)"]
    head = f"  {'family':<18}{'domain':<18}" + "".join(f"{s[:6]:>7}" for s in SKILLS)
    L.append(head)
    for n, f in sorted(cov["families"].items(), key=lambda kv: (kv[1]["domain"], kv[0])):
        if not f["tagged"]:
            L.append(f"  {n:<18}{f['domain']:<18}  untagged")
            continue
        c = Counter(v for v in f["steps"].values() if v)
        L.append(f"  {n:<18}{f['domain']:<18}" + "".join(f"{(c.get(s) or '.'):>7}" for s in SKILLS)
                 + (f"   PROBLEMS: {f['problems']}" if f["problems"] else ""))
    L.append("  by domain:")
    for d, v in cov["domains"].items():
        L.append(f"    {d:<18} " + ", ".join(f"{k} {n}" for k, n in sorted(v["skills"].items(), key=lambda kv: SKILLS.index(kv[0])
                                                                  if kv[0] in SKILLS else 99))
                 + (f"   untagged: {v['untagged']}" if v["untagged"] else "")
                 + (f"   never: {v['missing']}" if v["missing"] and v["skills"] else ""))
    L.append(f"  overall: {cov['overall']}" + (f"   no family tags: {cov['never_tagged']}" if cov["never_tagged"] else ""))
    return "\n".join(L)
