"""Time-domain astronomy (third domain): the Kepler families of generators/astro_kepler.py.

astro_kepler.py was written against the same four promises the engine keeps, so its
plug-in is the thinnest: every hook is one call into the module's public interface
(FAMILIES, reference, trap_answers, tolerance, task_text, TOL_RULE, STEPS, CORE_TRAPS), and
the canonical layout IS its layout. Nothing is re-implemented, so the adapter follows the
module when it changes (it is being edited as this is written): only make()'s loop rule and
_emit_one's field order are mirrored, and sage/tests/test_equivalence.py re-checks both.

Per-draw core traps: gen_<family> returns the traps that must be live on THIS draw
(`required`: no_detrending and no_limb_darkening for every transit; half_period for every
EB plus the phase traps on eccentric ones; ls_peak on two-longitude rotators). They become
Trap(core=True); the rest are conditional, recorded with step None where they coincide.

The transit family's extra identifiability check (the reference's linearised sigma(rp_rs)
must be below RPRS_ERR_MAX tolerances, else rp_rs hinges on where in the b-k valley the fit
lands) is the `sanity` hook, fed by a "_rp_rs_err" diagnostic on the reference's output.

The real half (Kepler catalogues) is a RealSource over the fetched targets of
experiments/third_domain/fetch_kepler.py; sage/real.py applies the same reference and trap
table to it, exactly as experiments/third_domain/build_kepler_real.py does.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from generators import astro_kepler as AK
from sage.core import Cell, Family, RealSource, RealTarget, Sample, Trap, finite

NOTE = ("ground truth is the reference analysis (generators/astro_analysis.py) on the sampled light curve, not "
        "the latent parameter; ablation deviation is in units of tolerance, maxed over the answer keys")

# Shared skill taxonomy (sage/skills.py), per step of AK.STEPS.
SKILLS = {"transit": {1: "normalise", 2: "estimate", 3: "model-choice", 4: "model-choice"},
          "eb": {1: "normalise", 2: "validate", 3: "inspect", 4: "estimate"},
          "rot": {1: "normalise", 2: "model-choice", 3: "validate"}}

# Blind prompts: astro_kepler's text names no method, but it WARNS about the defects the
# traps exploit. Blind mode keeps every fact about the data and the measurement (columns,
# units, gaps, the 29.4-min integration, the limb-darkening law the star file uses, the
# circular-orbit assumption) and drops the warnings: the contamination clause (step 1
# detrending), the eclipsing binary's "two eclipses / may be eccentric" paragraph (steps 2
# and 4; the primary is still defined as the deeper eclipse), and the rotator's "spots at
# more than one longitude" paragraph (steps 2-3).
_CONTAM = (", and besides the astrophysical signal it contains stellar variability, correlated noise and "
           "instrumental residuals.")
BLIND_EDITS = {
    "transit": [(_CONTAM, ".")],
    "eb": [(_CONTAM, "."),
           ("The binary shows two eclipses per orbit: the primary eclipse (the deeper one) and the secondary eclipse. "
            "The orbit may be eccentric, in which case the secondary eclipse does not fall half an orbit after the "
            "primary.\n\n", ""),
           ("measured from the primary eclipse, secondary_phase", "measured from the primary eclipse (the deeper "
            "one), secondary_phase")],
    "rot": [(_CONTAM, "."),
            ("Starspots grow and decay over a few rotations, and spots can be present at more than one longitude at "
             "the same time.\n\n", "")],
}


class AstroFamily(Family):
    domain = "astro"
    max_tries = 12
    rng_policy = "stream"
    enforce = frozenset({"I1", "I2", "I3"})     # astro_kepler.make's rule; I4 is audited
    note = NOTE

    def __init__(self, fam: str):
        self.fam = fam
        self.name = f"astro_{fam}"
        self.keys = tuple(AK.ANSWER_KEYS[fam])
        self.steps = AK.STEPS[fam]
        self.skills = SKILLS[fam]
        self.tol_rule = AK.TOL_RULE[fam]
        j = sorted(AK.FAMILIES).index(fam)       # seed block fixed per family (astro_kepler.main)
        self.seed_blocks = {"train": 9_000_000 + 100_000 * j, "test": 9_050_000 + 100_000 * j}

    def task_text(self, data, blind: bool = False) -> str:
        text = AK.task_text(self.fam, data["star"])
        return self.blind_from_text(text) if blind else text

    def blind_from_text(self, task: str) -> str:
        for old, new in BLIND_EDITS[self.fam]:
            if old not in task:
                raise ValueError(f"{self.name} blind: {old[:48]!r} is not in the task text "
                                 "(astro_kepler.task_text changed?)")
            task = task.replace(old, new, 1)
        return task

    def draw(self, rng, knob) -> Sample:
        lc, star, latent, required = AK.FAMILIES[self.fam](rng, knob)
        return Sample({"lc": lc, "star": star, "required": set(required)}, latent)

    def reference(self, data) -> dict:
        gold = AK.reference(self.fam, data["lc"], data["star"])
        if self.fam == "transit":
            gold["_rp_rs_err"] = AK._analyse(self.fam, data["lc"], data["star"])["diag"]["fit"]["rp_rs_err"]
        return gold

    def traps(self, data) -> dict:
        req = data.get("required", set(AK.CORE_TRAPS[self.fam]))
        return {k: Trap(ans, step, k in req) for k, (ans, step) in AK.trap_answers(self.fam, data["lc"], data["star"]).items()}

    def tolerance(self, gold) -> dict:
        return AK.tolerance(self.fam, gold)

    def sanity(self, gold, tol, latent) -> list[str]:
        if self.fam == "transit" and gold["_rp_rs_err"] > AK.RPRS_ERR_MAX * tol["rp_rs"]:
            return ["rp_rs_err"]
        return []

    def strengthen(self, knob, dead):
        return knob * 1.5 if "no_detrending" in dead else knob   # only the variability knob is tunable

    def shortcuts(self, data) -> dict:
        """Heuristics a policy could learn without the analysis: for a rotator, double the
        periodogram's peak (right on two-longitude stars, 2x wrong on the controls)."""
        if self.fam != "rot":
            return {}
        t = AK.trap_answers(self.fam, data["lc"], data["star"])
        return {"double_ls_peak": {"rotation_period": 2 * t["ls_peak"][0]["rotation_period"]}}

    def write(self, data, d: Path) -> list[str]:
        data["lc"].to_csv(d / AK.LC_FILE, index=False)
        files = [AK.LC_FILE]
        if data["star"] is not None:
            pd.DataFrame([data["star"]]).to_csv(d / AK.STAR_FILE, index=False)
            files.append(AK.STAR_FILE)
        return files

    def demo(self, cfg) -> list[Cell]:
        from rl import traces_astro as TA
        cells = []
        for prose, code in TA.moves(self.fam):
            if callable(prose):
                prose = (lambda p: (lambda outs: p(TA._obs(outs[0]))))(prose)
            cells.append(Cell(prose, code))
        return cells


FAMILIES = [AstroFamily(f) for f in sorted(AK.FAMILIES)]


# ----------------------------------------------------------------------------- real Kepler targets
def kepler_id(fam: str, kic: int) -> str:
    """build_kepler_real.inst_id: stable, free of star names."""
    return f"kepler_{fam}_{hashlib.sha1(f'{fam}:{int(kic)}'.encode()).hexdigest()[:8]}"


class KeplerSource(RealSource):
    """Fetched Kepler targets ($DEEP_DATA_ROOT/raw/kepler/<family>/KIC*/: lightcurve.csv,
    star.csv for transits, truth.json with the published catalogue answer)."""

    def __init__(self, fam: str, raw: Path | None = None, limit: int = 0):
        from sage.registry import get
        self.family = get(f"astro_{fam}")
        self.fam = fam
        self.label = f"kepler_{fam}"
        if raw is None:
            from paths import data_root
            raw = data_root() / "raw" / "kepler"
        self.raw = Path(raw)
        self.limit = limit
        self.keep_controls = fam == "rot"       # build_kepler_real.py's policy since 2026-09-14
        self.note = ("ground truth is the published catalogue value, not an analysis output; the reference "
                     "analysis (generators/astro_analysis.py) of these two quarters reproduces it within tolerance "
                     "on every key. Ablations are scored against the catalogue value: the reference row is the "
                     "gold, deviation is in units of tolerance maxed over the answer keys, and a trap inside "
                     "tolerance on this light curve has fails_at_step None")

    def keys(self) -> list:
        ds = sorted(str(p) for p in (self.raw / self.fam).glob("KIC*") if (p / "truth.json").exists())
        return ds[: self.limit] if self.limit else ds

    def load(self, key) -> RealTarget:
        d = Path(key)
        truth = json.loads((d / "truth.json").read_text())
        kic = int(truth["kic"])
        star = None
        if self.fam == "transit" and (d / AK.STAR_FILE).exists():
            star = pd.read_csv(d / AK.STAR_FILE).iloc[0].to_dict()
        files = {AK.LC_FILE: d / AK.LC_FILE}
        if self.fam == "transit":
            files[AK.STAR_FILE] = d / AK.STAR_FILE
        meta = {"benchmark": "Kepler real light curves (third domain)", "kic": kic, "name": truth.get("name"),
                "quarters": truth["quarters"], "n_points": truth.get("n_points"),
                "catalogue_reference": truth.get("reference"), "catalogue_source": truth.get("source"),
                "catalogue_uncertainty": truth.get("uncertainty"),
                "stratum": truth.get("stratum") or (truth.get("screen") or {}).get("class")}
        return RealTarget(kepler_id(self.fam, kic), {"lc": pd.read_csv(d / AK.LC_FILE), "star": star},
                          {k: finite(truth["answer"][k]) for k in self.family.keys}, meta, files)


REAL_SOURCES = {"kepler": KeplerSource}
