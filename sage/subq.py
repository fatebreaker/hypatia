"""Sub-estimand questions: a foothold for families the base model never solves (docs/DIRECTIONS.md A2).

WHY. On a 0%-pass family every GRPO group is dead: no rollout lands on the answer, none lands on a named trap, so
there is no gradient (enzyme_mm: 0.0% for the untrained Qwen3-8B, no general program ever sampled at 8B or 14B). But
the family's analysis computes intermediate quantities on the way to its estimand, each individually verifiable. This
module exposes them as SUB-QUESTIONS on the same instance -- same data, one intermediate asked for, its own gold,
tolerance and trap rows -- so a curriculum can mix them with the full question and give the group a live signal.

CONTRACT (per sub-question kind), kept to the engine's promises:
  * the gold is computed from the data files the analyst gets, by the family's own reference code (never the latent);
  * I1  the gold lands within tolerance of the latent truth (or, for derived data, of the parent's gold);
  * I2  the family's trap for the step the sub-question isolates lands OUTSIDE tolerance (else the sub-question is not
        emitted for that instance -- or, for a conditional row, it is recorded with fails_at_step None);
  * I3  the family's accepted variant for that step lands INSIDE tolerance;
  * I4  under rl/reward.py the gold scores 1.0 and each trap row earns exactly its step's credit (checked by `verify`).
  * the prompt states the QUANTITY, never how to get it (DIRECTIONS.md A2 risk: sub-questions must not leak the method).

The output is an ordinary eval_config.json instance (canonical fields: task, data_files, ground_truth,
multi_numeric_tolerance grader, _provenance.ablations), so rl/reward.py (StepReward), rl/dispersion.py and
rl/grpo_onpolicy.py grade and train on it with NO change. A sub-question that asks about the parent's own data links
the parent's data_files/ (a symlink, never a copy); one that supplies derived data writes its own data_files/.

ENZYME_MM (sage/domains/enzyme.py; reference = a global least-squares fit of the integrated Michaelis-Menten equation)
  sq1  initial_rate   "what was the initial rate of the reaction in assay A4 (s0 = X uM)?"      key v0 (uM/min)
       gold  the reference's own initial slope for that assay, Vmax_ref * s0 / (Km_ref + s0) (the derivative at t = 0
             of the fitted integrated equation; recomputed from the CSV, it equals the parent's gold to the bit)
       tol   10% of the gold
       rows  linear_whole_curve (step 1): the slope of a straight line through the whole progress curve -- the family's
             step-1 trap, applied to the one curve; initial_rate_quadratic (accepted): the quadratic through the origin
             over readings below 25% conversion (enzyme.initial_rates, the family's accepted-variant route)
       curve the assay is drawn per instance (seeded) among the 3rd-5th lowest substrate concentrations (s0/Km ~0.6-1.7),
             restricted to those where the trap is live and the accepted variant inside tolerance. Probe on the 460
             existing enzyme_mm instances: at 10% the quadratic is inside for 88/98/100% of those curves and the
             whole-curve slope for 0%; the lowest curves are excluded because a per-curve estimate is too noisy there
             (the quadratic's median error is 14% on the lowest curve).
  sq2  mm_from_rates  "initial rates were estimated at seven substrate concentrations; what are Vmax and Km?"
       data  initial_rates.csv: the parent's per-assay initial rates as enzyme.initial_rates estimates them (6 sig.
             figures); the progress curves are NOT supplied, so this isolates step 2
       gold  enzyme.hyperbola_fit on the supplied rates (nonlinear least squares of v = Vmax S / (Km + S))
       tol   the family's rule (5% of Vmax, 15% of Km)
       rows  lineweaver_burk (step 2) on the supplied rates; live on ~95% of instances, else recorded with step None
       I1    the gold is within the PARENT's tolerance of the parent's gold (the family's I3 guarantees it)

    venv/bin/python -m sage.subq emit --family enzyme_mm --kind sq1 --src $D/generated/enzyme_mm/train \\
                                      --out $D/generated/enzyme_mm_sq1/train [--workers 8]
    venv/bin/python -m sage.subq pool --full $D/generated/enzyme_mm/train \\
                                      --sub $D/generated/enzyme_mm_sq1/train $D/generated/enzyme_mm_sq2/train \\
                                      --out $D/generated/a2_enzyme_curr/train          # 50% full / 50% sub
    venv/bin/python -m sage.subq verify --dir $D/generated/enzyme_mm_sq1/train      # recompute gold, reward I4

pk_oral would plug in the same way (per-subject terminal slope / AUC from the reference's per-subject fits,
OralPK._study); not built -- enzyme_mm first, as the plan says.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import sage  # noqa: F401  (BLAS pinning, repo root on sys.path)

import numpy as np
import pandas as pd

SALT = 20260924                    # per-instance choices: default_rng([parent seed, SALT])


def _dev(a: float, g: float, tol: float) -> float:
    return abs(float(a) - float(g)) / tol if tol > 0 else math.inf


def _row(answer: dict, step, dev: float) -> dict:
    return {**{k: float(v) for k, v in answer.items()}, "fails_at_step": step, "deviation": float(dev)}


def _grader(tol: dict) -> dict:
    return {"type": "multi_numeric_tolerance",
            "config": {"keys": {k: {"absolute_tolerance": float(v)} for k, v in tol.items()}}}


# ----------------------------------------------------------------------------- enzyme_mm
class EnzymeSubQ:
    family = "enzyme_mm"
    kinds = {"sq1": "initial_rate", "sq2": "mm_from_rates"}
    RANKS = (2, 3, 4)              # 0-based rank by s0 of the candidate assays for sq1
    RATE_TOL = 0.10                # sq1 tolerance, fraction of the gold
    CSV = "progress_curves.csv"
    RATES = "initial_rates.csv"

    def __init__(self):
        from sage.domains import enzyme as E
        self.E = E

    # ---------------------------------------------------------------- shared
    def _parent(self, parent: Path):
        cfg = json.loads((parent / "eval_config.json").read_text())
        df = pd.read_csv(parent / "data_files" / self.CSV)
        a = self.E._arrays(df)
        vmax, km = self.E.fit_progress_curves(*a)
        gt = cfg["ground_truth"]
        # the reference re-run on the CSV the analyst gets must BE the parent's gold -- otherwise the sub-question's
        # gold would come from a different analysis than the full question's
        if not (vmax == gt["vmax"] and km == gt["km"]):
            raise ValueError(f"{parent.name}: reference on the CSV ({vmax}, {km}) != parent gold ({gt['vmax']}, {gt['km']})")
        return cfg, df, a, vmax, km

    def _header(self, task: str) -> str:
        i = task.find("\n\nQuestion:")
        if i < 0 or "Return your answer as JSON" not in task:
            raise ValueError("enzyme_mm task text changed: no 'Question:' paragraph")
        return task[:i]

    # ---------------------------------------------------------------- sq1: initial rate of a named assay
    def sq1(self, parent: Path):
        cfg, df, a, vmax, km = self._parent(parent)
        curve, s0, t, p = a
        S, vq = self.E.initial_rates(*a)
        _, vw = self.E.whole_curve_rates(*a)
        ids = np.unique(curve)
        order = np.argsort(S, kind="stable")
        lat = cfg["_provenance"]["latent"]
        cands = []
        for r in self.RANKS:
            i = int(order[r])
            g = vmax * S[i] / (km + S[i])
            tol = self.RATE_TOL * abs(g)
            v_lat = lat["vmax"] * S[i] / (lat["km"] + S[i])
            d_trap, d_var, d_lat = _dev(vw[i], g, tol), _dev(vq[i], g, tol), _dev(v_lat, g, tol)
            if d_trap > 1.0 and d_var <= 1.0 and d_lat <= 1.0:          # I2, I3, I1
                cands.append((r, i, g, tol, d_trap, d_var, v_lat))
        if not cands:
            return None, "no assay among ranks 2-4 with a live trap and the accepted variant inside tolerance"
        seed = int(cfg["_provenance"]["seed"])
        r, i, g, tol, d_trap, d_var, v_lat = cands[int(np.random.default_rng([seed, SALT]).integers(len(cands)))]
        cid, s = str(ids[i]), float(S[i])
        task = (self._header(cfg["task"]) + "\n\n"
                f"Question: what was the initial rate of the reaction in assay {cid} (initial substrate concentration "
                f"{s:g} uM), that is, the rate of product formation at the moment of mixing, in uM/min?\n\n"
                'Return your answer as JSON: {"answer": {"v0": <uM/min>}}')
        out = {"id": f"{self.family}_sq1_{parent.name.rsplit('_', 1)[1]}", "task": task,
               "data_files": [f"data_files/{self.CSV}"],
               "ground_truth": {"v0": float(g)},
               "grader": _grader({"v0": tol}),
               "_provenance": {
                   "family": f"{self.family}_sq1", "parent": cfg["id"], "parent_family": self.family,
                   "seed": seed, "subq": {"kind": "initial_rate", "assay": cid, "s0_uM": s, "rank_by_s0": r,
                                          "candidates": [c[0] for c in cands]},
                   "tol_rule": "10% of the gold",
                   "note": ("sub-question of the parent instance (docs/DIRECTIONS.md A2, sage/subq.py). Gold: the "
                            "reference's own initial slope for this assay, Vmax_ref*s0/(Km_ref+s0), from the global "
                            "integrated Michaelis-Menten fit that is the parent's gold, recomputed from the CSV."),
                   "latent": {"v0": float(v_lat)},
                   "steps": {"1": "initial rates account for substrate depletion"},
                   "ablations": {"reference": _row({"v0": g}, None, 0.0),
                                 "linear_whole_curve": _row({"v0": vw[i]}, 1, d_trap),
                                 "initial_rate_quadratic": _row({"v0": vq[i]}, None, d_var)}}}
        return (out, None), None

    # ---------------------------------------------------------------- sq2: Vmax, Km from supplied initial rates
    def sq2(self, parent: Path):
        cfg, df, a, vmax, km = self._parent(parent)
        S, vq = self.E.initial_rates(*a)
        ids = [str(c) for c in np.unique(a[0])]
        rates = pd.DataFrame({"assay": ids, "s0_uM": S.astype(float), "v0_uM_per_min": [float(f"{v:.6g}") for v in vq]})
        # gold from the table AS WRITTEN (to_csv writes shortest-repr floats, so reading back is exact; checked in verify)
        Sx, vx = rates["s0_uM"].to_numpy(float), rates["v0_uM_per_min"].to_numpy(float)
        try:
            gv, gk = self.E.hyperbola_fit(Sx, vx)
            lv, lk = self.E.lineweaver_burk(Sx, vx)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            return None, f"fit failed: {exc}"
        if not (np.isfinite([gv, gk, lv, lk]).all() and gv > 0 and gk > 0):
            return None, "non-finite or non-positive gold"
        tol = {"vmax": 0.05 * abs(gv), "km": 0.15 * abs(gk)}
        d_lb = max(_dev(lv, gv, tol["vmax"]), _dev(lk, gk, tol["km"]))
        ptol = {k: v["absolute_tolerance"] for k, v in cfg["grader"]["config"]["keys"].items()}
        d_parent = max(_dev(gv, vmax, ptol["vmax"]), _dev(gk, km, ptol["km"]))
        if d_parent > 1.0:                                              # I1 (derived data): the parent's I3
            return None, f"hyperbola on the supplied rates misses the parent's gold ({d_parent:.2f} tol)"
        seed = int(cfg["_provenance"]["seed"])
        head = ("You are an expert in enzymology. You will analyse initial-rate data from a continuous enzyme assay.\n\n"
                "Data files:\n"
                f"- data_files/{self.RATES}: one row per assay. Columns: assay (assay id), s0_uM (initial substrate "
                "concentration in uM), v0_uM_per_min (the initial rate of product formation in uM/min, estimated from "
                "the assay's progress curve, with estimation error).\n\n"
                "Every assay uses the same enzyme concentration; the enzyme is stable over the assay and the product "
                "does not inhibit it.\n\n")
        task = (head + "Question: what are the enzyme's maximal rate Vmax (uM/min) and Michaelis constant Km (uM)?\n\n"
                'Return your answer as JSON: {"answer": {"vmax": <uM/min>, "km": <uM>}}')
        out = {"id": f"{self.family}_sq2_{parent.name.rsplit('_', 1)[1]}", "task": task,
               "data_files": [f"data_files/{self.RATES}"],
               "ground_truth": {"vmax": float(gv), "km": float(gk)},
               "grader": _grader(tol),
               "_provenance": {
                   "family": f"{self.family}_sq2", "parent": cfg["id"], "parent_family": self.family, "seed": seed,
                   "subq": {"kind": "mm_from_rates", "parent_gold": {"vmax": float(vmax), "km": float(km)},
                            "dev_from_parent_gold_in_parent_tol": float(d_parent)},
                   "tol_rule": "vmax 5% of gold, km 15% of gold (the family's rule)",
                   "note": ("sub-question of the parent instance (docs/DIRECTIONS.md A2, sage/subq.py). Data: the parent's "
                            "per-assay initial rates as enzyme.initial_rates estimates them. Gold: enzyme.hyperbola_fit "
                            "on them (the family's accepted-variant route, step 2)."),
                   "latent": {"vmax": float(cfg["_provenance"]["latent"]["vmax"]),
                              "km": float(cfg["_provenance"]["latent"]["km"])},
                   "steps": {"1": "initial rates account for substrate depletion",
                             "2": "fit the hyperbola, not a linearisation"},
                   "ablations": {"reference": _row({"vmax": gv, "km": gk}, None, 0.0),
                                 "lineweaver_burk": _row({"vmax": lv, "km": lk}, 2 if d_lb > 1.0 else None, d_lb)}}}
        return (out, rates), None

    def gold_from_files(self, inst: Path, cfg: dict) -> dict:
        """Recompute a sub-question's gold from the files an analyst gets (verify / tests)."""
        kind = cfg["_provenance"]["subq"]["kind"]
        if kind == "initial_rate":
            a = self.E._arrays(pd.read_csv(inst / "data_files" / self.CSV))
            v, k = self.E.fit_progress_curves(*a)
            s = cfg["_provenance"]["subq"]["s0_uM"]
            return {"v0": v * s / (k + s)}
        r = pd.read_csv(inst / "data_files" / self.RATES)
        gv, gk = self.E.hyperbola_fit(r["s0_uM"].to_numpy(float), r["v0_uM_per_min"].to_numpy(float))
        return {"vmax": gv, "km": gk}


SUBQ = {"enzyme_mm": EnzymeSubQ}


# ----------------------------------------------------------------------------- emission
def _emit_one(job):
    family, kind, parent, out = job
    sq = SUBQ[family]()
    parent, out = Path(parent), Path(out)
    res, why = getattr(sq, kind)(parent)
    if res is None:
        return parent.name, None, why
    cfg, table = res
    d = out / cfg["id"]
    tmp = out / (".tmp_" + cfg["id"])
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if table is None:                           # asks about the parent's own data: link it, never copy
        (tmp / "data_files").symlink_to((parent / "data_files").resolve(), target_is_directory=True)
    else:
        (tmp / "data_files").mkdir()
        table.to_csv(tmp / "data_files" / Path(cfg["data_files"][0]).name, index=False)
    (tmp / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    return parent.name, cfg["id"], None


def emit(family: str, kind: str, src: Path, out: Path, workers: int = 1) -> dict:
    parents = sorted(p for p in Path(src).iterdir() if (p / "eval_config.json").is_file())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(family, kind, str(p), str(out)) for p in parents]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_emit_one, jobs, chunksize=4)
    else:
        res = [_emit_one(j) for j in jobs]
    skipped = {p: why for p, i, why in res if i is None}
    man = {"family": f"{family}_{kind}", "parent_family": family, "kind": SUBQ[family].kinds[kind], "src": str(src),
           "n_parents": len(parents), "n": len(parents) - len(skipped), "skipped": skipped, "salt": SALT,
           "builder": "sage/subq.py (docs/DIRECTIONS.md A2)"}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def pool(full: Path, subs: list[Path], out: Path) -> dict:
    """A curriculum pool: every full-question instance of `full`, plus ONE sub-question per parent, cycling through
    `subs` by parent index (parent i takes subs[i % len(subs)], falling back to the next kind if that one was not
    emitted for it). Symlinks only. With one sub-question per full question the pool is 50% / 50%, and the trainer's
    uniform draw (rng.sample) keeps that ratio in expectation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parents = sorted(p for p in Path(full).iterdir() if (p / "eval_config.json").is_file())
    by_kind = []
    for s in subs:
        m = {}
        for d in Path(s).iterdir():
            if (d / "eval_config.json").is_file():
                m[json.loads((d / "eval_config.json").read_text())["_provenance"]["parent"]] = d
        by_kind.append(m)
    n_full, n_sub, missing, comp = 0, 0, [], {}

    def link(src: Path):
        l = out / src.name
        if not l.exists():
            l.symlink_to(src.resolve(), target_is_directory=True)

    for i, p in enumerate(parents):
        link(p)
        n_full += 1
        pid = json.loads((p / "eval_config.json").read_text())["id"]
        for j in range(len(subs)):
            k = (i + j) % len(subs)
            if pid in by_kind[k]:
                link(by_kind[k][pid])
                n_sub += 1
                comp[str(subs[k])] = comp.get(str(subs[k]), 0) + 1
                break
        else:
            missing.append(pid)
    man = {"full": str(full), "subs": [str(s) for s in subs], "n_full": n_full, "n_sub": n_sub,
           "sub_by_source": comp, "parents_without_sub": missing,
           "builder": "sage/subq.py pool (docs/DIRECTIONS.md A2 CURRICULUM)"}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def _reward_steps(cfg: dict, default: int = 5) -> int:
    """rl/grpo_onpolicy.reward_steps (the trainer's per-instance reward scale S), without importing torch."""
    return max([v.get("fails_at_step") for v in (cfg.get("_provenance", {}).get("ablations") or {}).values()
                if v.get("fails_at_step")], default=default)


def _verify(d: Path, partial_ceiling: float = 0.5, limit: int = 0) -> dict:
    """Every sub-question under d: gold recomputed from its files equals the recorded gold; under rl/reward.py
    (StepReward, the trainer's own S = deepest trap step, c = partial_ceiling) the gold scores 1.0, every trap row is
    attributed to itself and earns exactly c*(step-1)/S, an accepted row passes, and an answer 3 tolerances off fails."""
    from rl.reward import StepReward
    insts = sorted(p for p in Path(d).iterdir() if (p / "eval_config.json").is_file())
    if limit:
        insts = insts[:limit]
    bad, n = [], 0
    for inst in insts:
        cfg = json.loads((inst / "eval_config.json").read_text())
        fam = cfg["_provenance"].get("parent_family")
        if fam not in SUBQ:
            continue                                          # a full-question instance in a mixed pool
        n += 1
        gold = SUBQ[fam]().gold_from_files(inst, cfg)
        if any(gold[k] != cfg["ground_truth"][k] for k in gold):
            bad.append((inst.name, "gold != recomputed"))
        tol = {k: v["absolute_tolerance"] for k, v in cfg["grader"]["config"]["keys"].items()}
        S = _reward_steps(cfg)
        r = StepReward(tolerance=min(tol.values()), n_steps=S, partial_ceiling=partial_ceiling)
        if r.score(cfg, {"answer": dict(cfg["ground_truth"])}).reward != 1.0:
            bad.append((inst.name, "gold does not score 1.0"))
        far = {k: v + 3 * tol[k] for k, v in cfg["ground_truth"].items()}
        if r.score(cfg, {"answer": far}).passed:
            bad.append((inst.name, "3 tolerances off passes"))
        for name, row in cfg["_provenance"]["ablations"].items():
            if name == "reference":
                continue
            ans = {k: row[k] for k in cfg["ground_truth"]}
            res = r.score(cfg, {"answer": ans})
            if row["fails_at_step"] is None:
                if row["deviation"] <= 1.0 and not res.passed:
                    bad.append((inst.name, f"accepted row {name} does not pass"))
            else:
                exp = round(partial_ceiling * (row["fails_at_step"] - 1) / S, 6)
                if res.passed or res.attribution != name or abs(res.reward - exp) > 1e-9:
                    bad.append((inst.name, f"trap {name}: reward {res.reward} ({res.component}, {res.attribution}), "
                                           f"expected {exp}"))
    return {"dir": str(d), "n_checked": n, "n_bad": len(bad), "bad": bad[:20]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--family", required=True, choices=sorted(SUBQ))
    e.add_argument("--kind", required=True, help="sq1 | sq2 (enzyme_mm)")
    e.add_argument("--src", required=True, help="the parent split directory (generated/<family>/<split>)")
    e.add_argument("--out", required=True, help="generated/<family>_<kind>/<split>")
    e.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("pool")
    p.add_argument("--full", required=True)
    p.add_argument("--sub", nargs="+", required=True)
    p.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--c", type=float, default=0.5)
    v.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "emit":
        if a.kind not in SUBQ[a.family].kinds:
            ap.error(f"{a.family} kinds: {sorted(SUBQ[a.family].kinds)}")
        man = emit(a.family, a.kind, Path(a.src), Path(a.out), a.workers)
        print(json.dumps({k: v for k, v in man.items() if k != "skipped"} | {"n_skipped": len(man["skipped"])}))
    elif a.cmd == "pool":
        print(json.dumps(pool(Path(a.full), [Path(s) for s in a.sub], Path(a.out))))
    else:
        rep = _verify(Path(a.dir), a.c, a.limit)
        print(json.dumps(rep, indent=1))
        return 1 if rep["n_bad"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
