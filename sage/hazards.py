"""Hazard toggles: learn the hazard, not the program (docs/DIRECTIONS.md A1).

WHY. Every family plants its hazards deterministically -- on surv_cox the hospital effect is always there, on
causal_confounded the post-treatment column is always a mediator -- so a policy never has to DETECT one: the repair
is learned as part of the surface's program, and one transcribed program solves the family. Here a hazard is drawn
per instance (present with probability 1/2, at a random magnitude), the prompt is blind (no "Approach" paragraph),
and `_provenance.hazards` records what was planted and which of its traps are live, so an evaluation can score
(hazard present?, repaired?) per hazard and per surface (scripts/hazard_endpoint.py).

HOW A TOGGLE WORKS. A wrapper family (HazardFamily) around an existing plug-in:
  * knobs(rng)  the base family's knobs, then per TOGGLE hazard a draw from an INDEPENDENT generator seeded by
                (instance seed, SALT): present w.p. P_PRESENT -> knob = default x U(0.5, 1.5); absent -> knob = the
                hazard's `absent` value. The base draw's own random stream is untouched. arm "fixed" draws nothing:
                every hazard present at the base default -- the base family's draw EXACTLY (same seed, same data).
  * draw        the base draw with those knobs; the planted set rides along in data["_planted"]
  * traps       the base trap table; where a hazard is ABSENT its traps are made conditional (core=False), so the
                engine records them with fails_at_step None where they coincide with the reference (the fix is then
                harmless and neither paid nor penalised -- the engine's existing rule), and as live traps where the
                data still carry the defect at a small magnitude
  * strengthen  the base rule, never raising an absent hazard's knob
  * task_text   ALWAYS blind (the base's task_text(data, blind=True)), plus a per-family wording edit where the
                base text would otherwise state the toggled mechanism (hc_dx)
  * provenance  _provenance.hazards: one record per hazard (toggled, fixed or natural): mode (scripts/trap_modes.py),
                planted, knob magnitude, governed traps with their recorded step and liveness, and `present` = any
                governed trap live on THIS instance (the operational definition the endpoint uses); `false_fix`
                traps are the over-application of the repair where the hazard is absent (two-sided hazards only)

THE FOUR SURFACES (toggled hazard in capitals; "fixed" = present in every instance of both arms; "natural" = already
variable in the base family, left as drawn):
  surv_cox           M2 SITE: hospital baseline hazards and hospital-dependent prescribing (knob `site`; absent =
                     1e-6, leaving the base DGP's residual N(0, 0.1) hospital effects) -> ignore_hospital.
                     fixed M1 censoring (censored_as_deaths, deaths_only): the surface cannot be made censoring-free
                     without breaking its other traps, so M1 is NOT toggled anywhere (deviation from DIRECTIONS.md).
  hc_dx              M5 REFERRAL: referral depends on the result and the covariate (new knob `dep` in
                     sage/domains/healthcare.py, default absent from knobs() so the base is byte-identical; 0 =
                     referral at random) -> referred_only, referral_by_test_only. fixed M5 unreferred_as_negative
                     (wrong whatever the referral mechanism). natural M4 units (units_not_harmonised, on ~45% of draws).
                     Wording: the base prompt says referral WAS based on the result and the covariate; both arms here
                     say it COULD depend on them and on nothing else (the identifying assumption, true in both states).
  epi_r0             M4 TESTING: the ascertainment step when testing was widened (knob `testing`; absent = 0: the
                     stated widening changed nothing in the counts) -> ignore_testing_change. natural M1 truncation and
                     M2 weekday (conditional in the base); fixed M3 single_exponential.
  causal_confounded  M5 POST-TREATMENT (TWO-SIDED): present = the base draw (the column is a mediator, described as
                     measured after treatment; adjusting for it is the trap post_treatment_control). Absent = the same
                     column is a PRE-treatment confounder, described as measured before treatment, that drives
                     treatment and outcome: the gold adjusts for it, and dropping it (the "fix" applied where it is
                     wrong) is the trap pretreatment_dropped. So "always drop the suspicious column" is penalised.

    venv/bin/python -m sage.hazards emit --family surv_cox --arm toggle --split train --n 400 --out $D/generated --workers 8
    venv/bin/python -m sage.hazards mix --arm toggle --split train --out $D/generated --pool a1_toggle4
    venv/bin/python -m sage.hazards blind --src $D/generated/pk_oral/test --out $D/generated/pk_oral_bl/test
    venv/bin/python -m sage.hazards stats --dir $D/generated/surv_cox_tg/train     # planted vs live, per hazard
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import sage  # noqa: F401  (BLAS pinning, repo root on sys.path)

import numpy as np
import pandas as pd

from sage.core import Family, Sample, Trap, as_trap, seed_of

SALT = 20260925
P_PRESENT = 0.5
MAG = (0.5, 1.5)
SURFACES = ("surv_cox", "hc_dx", "epi_r0", "causal_confounded")
ARM_SUFFIX = {"toggle": "tg", "fixed": "fx"}


@dataclass(frozen=True)
class Hazard:
    id: str
    mode: str                      # scripts/trap_modes.py MODES key
    traps: tuple                   # the traps this hazard makes live
    kind: str                      # "toggle" | "fixed" | "natural"
    knob: str | None = None        # toggle: the knob that carries its magnitude
    absent: float = 0.0            # toggle: the knob value when absent
    binary: bool = False           # toggle: present = knob 1.0 (no magnitude draw)
    false_fix: tuple = ()          # traps that apply the repair where the hazard is absent (two-sided)
    natural_flag: object = None    # natural: data -> bool (planted), or None (liveness only)


def _units_mixed(data) -> bool:
    return bool(data["meta"]["mixed"])


HAZARDS = {
    "surv_cox": (
        Hazard("site", "M2_heterogeneity", ("ignore_hospital",), "toggle", knob="site", absent=1e-6),
        Hazard("censoring", "M1_censoring", ("censored_as_deaths", "deaths_only"), "fixed"),
    ),
    "hc_dx": (
        Hazard("referral", "M5_selection", ("referred_only", "referral_by_test_only"), "toggle", knob="dep"),
        Hazard("unverified", "M5_selection", ("unreferred_as_negative",), "fixed"),
        Hazard("units", "M4_measurement", ("units_not_harmonised",), "natural", natural_flag=_units_mixed),
    ),
    "epi_r0": (
        Hazard("testing", "M4_measurement", ("ignore_testing_change",), "toggle", knob="testing"),
        Hazard("truncation", "M1_censoring", ("no_truncation_correction",), "natural"),
        Hazard("weekday", "M2_heterogeneity", ("no_weekday_effects",), "natural"),
        Hazard("changepoint", "M3_functional", ("single_exponential",), "fixed"),
    ),
    "causal_confounded": (
        Hazard("post_treatment", "M5_selection", ("post_treatment_control",), "toggle", knob="post", binary=True,
               false_fix=("pretreatment_dropped",)),
    ),
}


# ----------------------------------------------------------------------------- hc_dx wording (both arms)
_HC_RE = re.compile(r"the clinician decided, based on the (.+?) result and on (.+?), and nothing else about the patient "
                    r"influenced the decision\.")


def _hc_text(text: str, data) -> str:
    new, n = _HC_RE.subn(r"the decision was the clinician's, and it could depend on the \1 result and on \2 but on "
                         r"nothing else about the patient.", text)
    if n != 1:
        raise ValueError("hc_dx task text changed: the referral sentence was not found exactly once")
    return new


TEXT_EDITS = {"hc_dx": _hc_text}


# ----------------------------------------------------------------------------- causal_confounded, two-sided
# The mediator column of each context, described as a PRE-treatment measurement (the absent state).
PRE_DESC = {"hours_worked": "weekly hours worked, measured before the programme",
            "length_of_stay": "days in hospital during the patient's previous admission, before the protocol decision",
            "new_machines": "machines purchased in the year before the subsidy decision",
            "homework_hours": "weekly homework hours reported before enrolment"}


def gen_confounded_pre(rng, k: float):
    """generators/causal_effects.gen_confounded with the mediator turned into a PRE-treatment confounder: the column
    M is drawn before treatment (0.5 x0 + noise, as the mediator's baseline part), treatment depends on it, and the
    outcome loads on it. Same contexts, columns, rounding and description format; only M's role and description
    change. The gold (OLS on treatment + every pre-treatment column, M included) is recomputed by analyse_pre."""
    from generators import causal_effects as CE
    c = CE.CONFOUNDED[rng.integers(len(CE.CONFOUNDED))]
    n = int(rng.integers(1500, 4000))
    p = len(c["X"])
    X = rng.normal(size=(n, p))
    X[:, -1] = (X[:, -1] > rng.normal(0, 0.5)).astype(float)
    M = 0.5 * X[:, 0] + rng.normal(size=n)
    a = rng.normal(0, 0.6 * k, p) + np.sign(rng.normal(size=p)) * 0.4 * k
    aM = rng.uniform(0.5, 1.0) * k * rng.choice([-1, 1])
    T = (rng.random(n) < 1 / (1 + np.exp(-(X @ a + aM * M - 0.2)))).astype(float)
    tau_d = c["scale"] * rng.uniform(0.4, 1.0) * rng.choice([-1, 1])
    dlt = c["scale"] * rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
    beta = c["scale"] * rng.normal(0, 0.9, p) * k
    Y = tau_d * T + dlt * M + X @ beta + rng.normal(0, abs(c["scale"]) * 1.2, n)
    cols = {name: np.round(10 + 3 * X[:, j], 2) if j < p - 1 else X[:, j].astype(int) for j, (name, _) in enumerate(c["X"])}
    mname = c["M"][0]
    df = pd.DataFrame({c["T"]: T.astype(int), **cols, mname: np.round(M + 5, 3), c["Y"]: np.round(Y + 50, 3)})
    desc = (f"Observational data on {n} {c['unit']}. Variables in the Dataset:\n"
            + CE._vars([(c["T"], c["Tdesc"])] + c["X"] + [(mname, PRE_DESC[mname]), (c["Y"], c["Ydesc"])])
            + "\nAll covariates except the ones described as measured after treatment were recorded before treatment.")
    query = f"What is the effect of {c['T'].replace('_', ' ')} on {c['Y'].replace('_', ' ')}?"
    roles = {"treatment": c["T"], "outcome": c["Y"], "confounders": [nm for nm, _ in c["X"]] + [mname],
             "post_treatment": None, "lookalike": mname}
    return df, desc, query, roles


def analyse_pre(df, roles):
    """(gold, SE, {trap: (value, step, core)}) for the absent state, from the table and the roles alone."""
    from generators import causal_effects as CE
    ols, c1 = CE.ols, CE.c1
    Xa = np.column_stack([df[nm].to_numpy(float) for nm in roles["confounders"]])       # the lookalike is LAST
    y, t = df[roles["outcome"]].to_numpy(float), df[roles["treatment"]].to_numpy(float)
    b, se = ols(y, c1(t, Xa))
    gold, gse = float(b[1]), float(se[1])
    Xo = Xa[:, :-1]
    j = int(np.argmax(np.abs(b[2:2 + Xo.shape[1]]) * Xo.std(0)))    # the original confounder the outcome loads on most
    return gold, gse, {"no_adjustment": (float(y[t == 1].mean() - y[t == 0].mean()), 1, True),
                       "post_treatment_control": (gold, 2, False),          # adjusting for it is right here
                       "confounder_omitted": (float(ols(y, c1(t, np.delete(Xa, j, 1)))[0][1]), 2, True),
                       "pretreatment_dropped": (float(ols(y, c1(t, Xo))[0][1]), 2, True)}


def _confounded_family():
    from sage.domains import causal as C
    from generators import causal_effects as CE

    class ConfoundedPT(C.CausalDesign):
        """causal_confounded with knob `post`: 1 = the base draw (CE.gen_confounded, unchanged), 0 = the column is a
        pre-treatment confounder (gen_confounded_pre). Knobs are a dict {k, post}; `k` is the base's scalar knob."""

        def __init__(self):
            super().__init__("confounded")

        def knobs(self, rng):
            return {"k": 1.0, "post": 1.0}

        def strengthen(self, k, dead):
            k = dict(k)
            k["k"] = k["k"] * 1.4
            return k

        def draw(self, rng, k) -> Sample:
            if k["post"] > 0:
                df, desc, query, gold, gse, abl, roles = CE.gen_confounded(rng, k["k"])
            else:
                df, desc, query, roles = gen_confounded_pre(rng, k["k"])
            return Sample({"df": df, "desc": desc, "query": query, "roles": roles, "boot_se": None}, latent={})

        def _analysis(self, data):
            key = "_analysis"
            if key not in data:
                if data["roles"].get("post_treatment"):
                    g, s, tr = C.analyse("confounded", data["df"], data["roles"], None)
                    data[key] = (g, s, {nm: (v, st, True) for nm, (v, st) in tr.items()})
                else:
                    data[key] = analyse_pre(data["df"], data["roles"])
            return data[key]

        def traps(self, data) -> dict:
            return {nm: Trap({"effect": v}, st, core) for nm, (v, st, core) in self._analysis(data)[2].items()}

    return ConfoundedPT()


class CausalV1Hz:
    """CausalV1 (the causal corpora's layout, field for field) plus the hazard provenance, and each row's RECORDED step
    (a conditional trap that coincides with the reference is None) instead of the nominal one. Where every trap is
    core -- the whole FIXED arm -- recorded == nominal, so the ablation table is the base's."""
    indent = 1

    def config(self, inst, files) -> dict:
        from sage.domains.causal import CAUSAL_V1
        cfg = CAUSAL_V1.config(inst, files)
        for r in inst.a.rows:
            cfg["_provenance"]["ablations"][r.name]["fails_at_step"] = r.step
        cfg["_provenance"].update(inst.family.extra_provenance(inst))
        return cfg

    def dumps(self, cfg) -> str:
        return json.dumps(cfg, indent=self.indent)


# ----------------------------------------------------------------------------- the wrapper
def _base(name: str):
    if name == "causal_confounded":
        return _confounded_family()
    from sage.registry import get
    return get(name)


class HazardFamily(Family):
    DELEGATE = ("domain", "keys", "exact_keys", "steps", "skills", "max_tries", "rng_policy", "enforce", "seed_blocks",
                "tol_rule", "note", "demo_timeout", "demonstration_only")

    def __init__(self, base_name: str, arm: str):
        if arm not in ARM_SUFFIX:
            raise ValueError(f"arm must be one of {sorted(ARM_SUFFIX)}")
        self.base = _base(base_name)
        self.base_name, self.arm = base_name, arm
        self.hazards = HAZARDS[base_name]
        self.name = f"{base_name}_{ARM_SUFFIX[arm]}"
        for a in self.DELEGATE:
            setattr(self, a, getattr(self.base, a))
        self.layout = CausalV1Hz() if base_name == "causal_confounded" else self.base.layout
        self.note = (self.base.note + f" | docs/DIRECTIONS.md A1 hazard arm '{arm}' (sage/hazards.py): blind prompt; "
                     "_provenance.hazards records the planted hazards")

    # ---------------------------------------------------------------- knobs and draw
    def knobs(self, rng):
        k = self.base.knobs(rng)
        k = dict(k)
        planted = {}
        if self.arm == "toggle":
            trng = np.random.default_rng([seed_of(rng), SALT])
            for h in self.hazards:
                if h.kind != "toggle":
                    continue
                on, mag = bool(trng.random() < P_PRESENT), float(trng.uniform(*MAG))   # both drawn: fixed consumption
                planted[h.id] = on
                k[h.knob] = (1.0 if h.binary else k.get(h.knob, 1.0) * mag) if on else h.absent
        else:
            planted = {h.id: True for h in self.hazards if h.kind == "toggle"}
        k["_planted"] = planted
        return k

    def strengthen(self, k, dead):
        planted = dict(k.get("_planted", {}))
        k2 = dict(self.base.strengthen(k, dead))
        for h in self.hazards:
            if h.kind == "toggle" and planted.get(h.id) is False:
                k2[h.knob] = h.absent
        k2["_planted"] = planted
        return k2

    def draw(self, rng, k) -> Sample:
        s = self.base.draw(rng, k)
        s.data["_planted"] = dict(k.get("_planted", {}))
        return s

    # ---------------------------------------------------------------- analysis (the base's)
    def reference(self, data) -> dict:
        return self.base.reference(data)

    def tolerance(self, gold) -> dict:
        return self.base.tolerance(gold)

    def sanity(self, gold, tol, latent) -> list:
        return self.base.sanity(gold, tol, latent)

    def shortcuts(self, data) -> dict:
        return self.base.shortcuts(data)

    def write(self, data, d: Path) -> list:
        return self.base.write(data, d)

    def traps(self, data) -> dict:
        t = {n: as_trap(v) for n, v in self.base.traps(data).items()}
        planted = data.get("_planted", {})
        for h in self.hazards:
            if h.kind == "toggle" and planted.get(h.id) is False:
                for n in h.traps:
                    if n in t:
                        t[n] = Trap(t[n].answer, t[n].step, False)
        return t

    # ---------------------------------------------------------------- prompt
    def task_text(self, data, blind: bool = True) -> str:
        text = self.base.task_text(data, blind=True)
        edit = TEXT_EDITS.get(self.base_name)
        return edit(text, data) if edit else text

    # ---------------------------------------------------------------- provenance
    def extra_provenance(self, inst) -> dict:
        base = {} if self.base_name == "causal_confounded" else dict(self.base.extra_provenance(inst))
        rows = {r.name: r for r in inst.a.rows}
        planted = inst.knobs.get("_planted", {}) if isinstance(inst.knobs, dict) else {}
        recs = []
        for h in self.hazards:
            if h.kind == "toggle":
                pl = planted.get(h.id)
            elif h.kind == "fixed":
                pl = True
            else:
                pl = h.natural_flag(inst.sample.data) if h.natural_flag else None
            state = {n: {"step": rows[n].step, "live": bool(rows[n].dev > 1.0), "dev": float(rows[n].dev)}
                     for n in (*h.traps, *h.false_fix) if n in rows}
            recs.append({"id": h.id, "mode": h.mode, "kind": h.kind, "planted": pl, "knob": h.knob,
                         "magnitude": (float(inst.knobs[h.knob]) if h.knob and isinstance(inst.knobs, dict)
                                       and h.knob in inst.knobs else None),
                         "traps": list(h.traps), "false_fix": list(h.false_fix), "trap_state": state,
                         "present": any(state[n]["step"] is not None for n in h.traps if n in state)})
        out = {**base, "hazards": recs, "hazard_arm": self.arm, "hazard_family": self.base_name, "task_mode": "blind"}
        if self.base_name == "causal_confounded":
            out["knobs"] = {k: v for k, v in inst.knobs.items()}
        return out


def family(base_name: str, arm: str) -> HazardFamily:
    return HazardFamily(base_name, arm)


# ----------------------------------------------------------------------------- emission, pools, blind copies
def emit(base_name: str, arm: str, split: str, n: int, out_root: Path, workers: int = 1) -> dict:
    """Mint through the engine (sage.engine.emit: validity loop, seeds = the base family's split block + i, manifest).
    The family is registered in-process so the engine's forked workers find it by name."""
    from sage import registry
    from sage.engine import emit as engine_emit
    fam = family(base_name, arm)
    registry.register(fam)
    return engine_emit(fam, split, n, Path(out_root), workers=workers)


def mix(arm: str, split: str, out_root: Path, pool: str, surfaces=SURFACES) -> int:
    from sage.engine import mix as engine_mix
    return engine_mix([f"{s}_{ARM_SUFFIX[arm]}" for s in surfaces], split, Path(out_root), pool)


def blind_copy(src: Path, out: Path) -> dict:
    """A blind copy of an existing split (for held-out evaluation in the blind setting): each instance's eval_config
    with its task replaced by the family's blind text (sage.variants.blind_from_text: the plug-in's own hook, else the
    "Approach:" paragraph dropped), data_files a symlink to the original. Nothing else changes."""
    from sage.registry import get
    from sage.variants import blind_from_text
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n, same, fams = 0, 0, {}
    for d in sorted(Path(src).iterdir()):
        if not (d / "eval_config.json").is_file():
            continue
        cfg = json.loads((d / "eval_config.json").read_text())
        fname = cfg["_provenance"]["family"]
        if fname not in fams:
            fams[fname] = get(fname)
        bt = blind_from_text(fams[fname], cfg["task"])
        if bt is None:
            raise SystemExit(f"{fname}: no blind support")
        same += bt == cfg["task"]
        cfg["task"] = bt
        cfg["_provenance"]["task_mode"] = "blind"
        cfg["_provenance"]["blind_of"] = str(d.resolve())
        o = out / d.name
        o.mkdir(exist_ok=True)
        if not (o / "data_files").exists():
            (o / "data_files").symlink_to((d / "data_files").resolve(), target_is_directory=True)
        (o / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        n += 1
    man = {"src": str(src), "n": n, "unchanged_text": same, "builder": "sage/hazards.py blind_copy (docs/DIRECTIONS.md A1)"}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def stats(d: Path) -> dict:
    """Per hazard: instances planted / present (a governed trap live) / planted-but-not-present / present-unplanted."""
    out = {}
    for inst in sorted(Path(d).iterdir()):
        f = inst / "eval_config.json"
        if not f.is_file():
            continue
        for h in json.loads(f.read_text())["_provenance"].get("hazards", []):
            s = out.setdefault(h["id"], {"mode": h["mode"], "kind": h["kind"], "n": 0, "planted": 0, "present": 0,
                                         "planted_not_present": 0, "present_not_planted": 0, "false_fix_live": 0})
            s["n"] += 1
            s["planted"] += h["planted"] is True
            s["present"] += h["present"]
            s["planted_not_present"] += (h["planted"] is True) and not h["present"]
            s["present_not_planted"] += (h["planted"] is False) and h["present"]
            s["false_fix_live"] += any(h["trap_state"].get(n, {}).get("step") is not None for n in h["false_fix"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--family", required=True, choices=SURFACES)
    e.add_argument("--arm", required=True, choices=sorted(ARM_SUFFIX))
    e.add_argument("--split", required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--out", required=True, help="ROOT: writes ROOT/<family>_<tg|fx>/<split>")
    e.add_argument("--workers", type=int, default=1)
    m = sub.add_parser("mix")
    m.add_argument("--arm", required=True, choices=sorted(ARM_SUFFIX))
    m.add_argument("--split", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--pool", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--dir", required=True)
    a = ap.parse_args()
    if a.cmd == "emit":
        man = emit(a.family, a.arm, a.split, a.n, Path(a.out), a.workers)
        print(json.dumps({k: v for k, v in man.items() if k not in ("weak_ids",)}))
    elif a.cmd == "mix":
        print(mix(a.arm, a.split, Path(a.out), a.pool))
    elif a.cmd == "blind":
        print(json.dumps(blind_copy(Path(a.src), Path(a.out))))
    else:
        print(json.dumps(stats(Path(a.dir)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
