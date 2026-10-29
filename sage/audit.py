"""`check`: the four invariants and a shortcut audit, on fresh draws or an emitted corpus.

Why each number is here. Step-attributed partial credit (rl/reward.py) is only as good as
the ablation table behind it, and a table can be wrong in ways no training curve shows:

  I1  gold vs latent truth. A reference that misses the truth on a draw grades the truth
      as wrong. Families whose estimand is DEFINED as the finite-sample estimate (the
      genomics families) do not enforce it; the audit still reports the gap.
  I2  per-trap liveness. A dead trap passes the grader, so the skipped step earns 1.0 and
      the step credit for it is unreachable -- the instance rewards the shortcut.
  I3  accepted variants inside tolerance, or a defensible analysis is graded wrong.
  I4  the reward itself, run on the instance's own eval_config: the gold must score 1.0
      and each trap exactly partial_ceiling*(s-1)/S (S = max(5, deepest step), reward.py's
      rule). Duplicate trap answers (ambiguous attribution), answers outside the grader's
      range, and dead traps all show up here as unpaid rows.

and the shortcut audit, because a reward that a data-ignoring policy can farm is what the
paper shows RL converging to: the best CONSTANT answer (fitted in-sample: the mean,
quantiles, and every instance's own gold tried as a constant), what each trap earns, and
any data-light heuristics the family declares (`Family.shortcuts`).

  venv/bin/python -m sage check --family causal_iv --n 50
  venv/bin/python -m sage check --corpus $DEEP_DATA_ROOT/generated/causal_iv   # read-only
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sage.core import dev_units


def _reward_mod():
    from rl import reward as R          # imported lazily: reward.py resolves paths on import
    return R


# ----------------------------------------------------------------------------- I4
def reward_check(cfg: dict, r=None) -> dict:
    """Score the gold and every credit-bearing ablation row with rl/reward.py."""
    R = _reward_mod()
    r = r or R.StepReward()
    keys = R.StepReward._fields(cfg)
    abl = (cfg.get("_provenance") or {}).get("ablations") or {}
    steps = []
    for name, a in abl.items():
        if name in R.PASSING_ABLATIONS or a.get("fails_at_step") is None:
            continue
        try:
            [float(a[k]) for k in keys]
        except (KeyError, TypeError, ValueError):
            continue
        steps.append(a["fails_at_step"])
    n_steps = max(max(steps, default=r.n_steps), r.n_steps)
    o = r.score(cfg, dict(cfg["ground_truth"]))
    rows = []
    for name, a in abl.items():
        if name in R.PASSING_ABLATIONS or a.get("fails_at_step") is None:
            continue
        s = int(a["fails_at_step"])
        res = r.score(cfg, {k: a.get(k) for k in cfg["ground_truth"]})
        exp = round(r.partial_ceiling * (s - 1) / n_steps, 6)
        rows.append({"name": name, "step": s, "expected": exp, "got": res.reward,
                     "component": res.component, "ok": abs(res.reward - exp) < 1e-9,
                     "attributed": res.attribution == name and res.component == "step_attribution"})
    return {"oracle": o.reward, "oracle_ok": abs(o.reward - 1.0) < 1e-12, "n_steps": n_steps, "rows": rows}


# ----------------------------------------------------------------------------- records
def _tol(cfg: dict) -> dict:
    cc = (cfg.get("grader") or {}).get("config") or {}
    kc = dict(cc.get("keys") or {})
    kc.update(cc.get("numeric_keys") or {})
    return {k: float((v or {}).get("absolute_tolerance") or 0.0) for k, v in kc.items()}


def _exact(cfg: dict) -> list[str]:
    return _reward_mod().StepReward._exact_keys(cfg)


def record(cfg: dict, latent=None, nominal=None, shortcuts=None, tries=None) -> dict:
    return {"cfg": cfg, "latent": latent, "nominal": nominal, "shortcuts": shortcuts, "tries": tries}


def audit(recs: list[dict], label: str, enforced=None) -> dict:
    """The report for a list of records (see `record`)."""
    R = _reward_mod()
    r = R.StepReward()
    n = len(recs)
    cfg0 = recs[0]["cfg"]
    gkeys = list(cfg0["ground_truth"])
    exact = _exact(cfg0)
    num = [k for k in gkeys if k not in exact]
    rep = {"label": label, "n": n, "keys": gkeys, "enforced": sorted(enforced) if enforced else None}

    i1 = {"checked": 0, "pass": 0, "fail_ids": []}
    per_trap = defaultdict(lambda: Counter())
    trap_step, trap_core = {}, defaultdict(set)
    variants = defaultdict(lambda: Counter())
    all_live = all_inside = 0
    i4 = {"oracle_ok": 0, "rows": 0, "rows_ok": 0, "unpaid": Counter(), "why": Counter()}
    earn = defaultdict(list)
    tries = [x["tries"] for x in recs if x.get("tries")]
    no_table = 0
    # From files, a conditional trap that coincided with the reference (recorded None) looks
    # like an accepted variant; a name that carries a step anywhere in the corpus is a trap.
    seen_step = defaultdict(Counter)
    for x in recs:
        for name, a in ((x["cfg"].get("_provenance") or {}).get("ablations") or {}).items():
            if a.get("fails_at_step") is not None:
                seen_step[name][a["fails_at_step"]] += 1
    known = {k: c.most_common(1)[0][0] for k, c in seen_step.items()}
    for x in recs:
        cfg = x["cfg"]
        gold, tol = cfg["ground_truth"], _tol(cfg)
        lat = x.get("latent") or (cfg.get("_provenance") or {}).get("latent") or {}
        if lat and all(k in lat for k in gkeys):
            i1["checked"] += 1
            ok = all((str(lat[k]) == str(gold[k])) if k in exact else
                     (abs(float(lat[k]) - float(gold[k])) <= tol[k]) for k in gkeys)
            i1["pass"] += ok
            if not ok and len(i1["fail_ids"]) < 8:
                i1["fail_ids"].append(cfg["id"])
        abl = (cfg.get("_provenance") or {}).get("ablations")
        if not abl:
            no_table += 1
        live_here = inside_here = True
        nominal = x.get("nominal") or {}
        for name, a in (abl or {}).items():
            if name == "reference":
                continue
            dev = dev_units(a, gold, tol, gkeys, exact)
            step = a.get("fails_at_step")
            nom, core = nominal.get(name, (step if step is not None else known.get(name), None))
            if nom is None and step is None:                      # accepted variant
                c = variants[name]
                c["n"] += 1
                c["inside" if dev <= 1.0 else ("no_answer" if math.isnan(dev) else "outside")] += 1
                inside_here &= not dev > 1.0
                continue
            c = per_trap[name]
            trap_step[name] = nom
            if core is not None:
                trap_core[name].add(core)
            c["n"] += 1
            if math.isnan(dev):
                c["no_answer"] += 1
            elif dev > 1.0:
                c["live"] += 1
            elif step is None:
                c["coincident"] += 1                              # conditional, recorded None
            else:
                c["dead"] += 1
                live_here = False
        all_live += live_here and bool(abl)
        all_inside += inside_here and bool(abl)
        rc = reward_check(cfg, r) if abl else {"oracle_ok": abs(r.score(cfg, dict(gold)).reward - 1) < 1e-12,
                                               "rows": []}
        i4["oracle_ok"] += rc["oracle_ok"]
        for row in rc["rows"]:
            i4["rows"] += 1
            i4["rows_ok"] += row["ok"]
            earn[row["name"]].append(row["got"])
            if not row["ok"]:
                i4["unpaid"][row["name"]] += 1
                i4["why"][row["component"]] += 1
    rep["I1"] = i1
    rep["I2"] = {"instances_all_live": all_live, "no_ablation_table": no_table,
                 "per_trap": {k: {"step": trap_step[k],
                                  "core": (None if not trap_core[k] else all(trap_core[k])), **dict(v)}
                              for k, v in per_trap.items()}}
    rep["I3"] = {"instances_all_inside": all_inside, "variants": {k: dict(v) for k, v in variants.items()}}
    rep["I4"] = {"oracle_ok": i4["oracle_ok"], "rows": i4["rows"], "rows_ok": i4["rows_ok"],
                 "unpaid": dict(i4["unpaid"]), "unpaid_by_branch": dict(i4["why"])}
    rep["trap_earns"] = {k: round(st.mean(v), 4) for k, v in earn.items()}
    if tries:
        rep["mean_tries"] = round(st.mean(tries), 3)
    rep["constants"] = constants(recs, gkeys, num, exact, r)
    sc = defaultdict(list)
    for x in recs:
        for name, ans in (x.get("shortcuts") or {}).items():
            res = r.score(x["cfg"], ans)
            sc[name].append((res.reward, res.passed))
    rep["shortcuts"] = {k: {"mean_reward": round(st.mean(a for a, _ in v), 4),
                            "pass_rate": round(st.mean(b for _, b in v), 4)} for k, v in sc.items()}
    return rep


def constants(recs, gkeys, num, exact, r, max_candidates: int = 200) -> dict:
    """What a data-ignoring constant answer earns: the mean, joint quantiles, and up to
    `max_candidates` instances' own gold tried as the constant. A gold candidate is priced
    on the OTHER instances only (leave-one-out), or it would pass its own instance for free."""
    golds = [x["cfg"]["ground_truth"] for x in recs]
    vals = {k: np.array([float(g[k]) for g in golds]) for k in num}
    mode = {k: Counter(str(g[k]) for g in golds).most_common(1)[0][0] for k in exact}
    cands = {"mean": ({**{k: float(vals[k].mean()) for k in num}, **mode}, None)}
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        cands[f"q{int(q * 100)}"] = ({**{k: float(np.quantile(vals[k], q)) for k in num}, **mode}, None)
    step = max(1, len(golds) // max_candidates)
    if len(golds) > 1:
        for j in range(0, len(golds), step):
            cands[f"gold#{j} (leave-one-out)"] = ({k: golds[j][k] for k in gkeys}, j)

    def price(c, skip):
        res = [r.score(x["cfg"], dict(c)) for i, x in enumerate(recs) if i != skip]
        return (st.mean(z.reward for z in res), st.mean(z.passed for z in res),
                Counter(z.component for z in res))
    out = {}
    for name in ("mean", "q50"):
        m, p, b = price(*cands[name])
        out[name] = {"answer": cands[name][0], "mean_reward": round(m, 4), "pass_rate": round(p, 4),
                     "branches": dict(b)}
    best = None
    for name, (c, skip) in cands.items():
        m, p, b = price(c, skip)
        if best is None or (m, p) > (best[1], best[2]):
            best = (name, m, p, b, c)
    out["best"] = {"which": best[0], "answer": {k: (round(v, 6) if isinstance(v, float) else v)
                                               for k, v in best[4].items()},
                   "mean_reward": round(best[1], 4), "pass_rate": round(best[2], 4),
                   "branches": dict(best[3])}
    return out


# ----------------------------------------------------------------------------- fresh draws
def _check_one(job) -> dict:
    name, seed, i = job
    from sage.engine import layout_of, make
    from sage.registry import get
    fam = get(name)
    inst = make(fam, seed)
    inst.id = f"{fam.name}_check_{i:05d}"
    cfg = layout_of(fam).config(inst, [])
    nominal = {r.name: (r.nominal, r.core) for r in inst.a.rows}
    lat = {k: inst.sample.latent[k] for k in fam.keys if k in inst.sample.latent}
    try:
        sc = fam.shortcuts(inst.sample.data)
    except Exception as exc:                                      # noqa: BLE001
        sc = {}
        print(f"  shortcut failed on {inst.id}: {type(exc).__name__}: {exc}", flush=True)
    return {**record(json.loads(json.dumps(cfg, default=float)), lat or None, nominal, sc, inst.tries),
            "flags": inst.flags, "unenforced": inst.failures(False)}


def check_family(fam, n: int, seed0: int | None = None, workers: int = 1) -> dict:
    """Run the engine's loop on n seeds, in memory (nothing is written), and audit them.
    Default seeds are the family's first split block, so the audit describes exactly the
    instances `emit` would write there."""
    if seed0 is None:
        seed0 = next(iter(fam.seed_blocks.values()), 0)
    jobs = [(fam.name, seed0 + i, i) for i in range(n)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            recs = pool.map(_check_one, jobs, chunksize=1)
    else:
        recs = [_check_one(j) for j in jobs]
    rep = audit(recs, f"{fam.name} (fresh draws, seeds {seed0}..{seed0 + n - 1})", fam.enforce)
    rep["shipped_flagged"] = sum(bool(x["flags"]) for x in recs)
    unenf = Counter()
    for x in recs:
        for k in x["unenforced"]:
            unenf[k] += 1
    rep["unenforced_failures"] = dict(unenf)
    return rep


# ----------------------------------------------------------------------------- corpora
def load_corpus(path: Path, limit: int = 0) -> dict[str, list[dict]]:
    """{split: [eval_config, ...]} for a family directory with split subdirectories, or
    {"": [...]} for a directory of instances."""
    path = Path(path)
    flat = sorted(glob.glob(str(path / "*" / "eval_config.json")))
    if flat:
        fs = flat[:limit] if limit else flat
        return {"": [json.loads(Path(f).read_text()) for f in fs]}
    out = {}
    for sp in sorted(p for p in path.iterdir() if p.is_dir()):
        fs = sorted(glob.glob(str(sp / "*" / "eval_config.json")))
        if fs:
            out[sp.name] = [json.loads(Path(f).read_text()) for f in (fs[:limit] if limit else fs)]
    return out


def leaks(splits: dict[str, list[dict]]) -> dict:
    """Instances shared between splits: same seed, or same ground truth, or same id."""
    seen_seed, seen_gt, seen_id = defaultdict(set), defaultdict(set), defaultdict(set)
    for sp, cfgs in splits.items():
        for c in cfgs:
            p = c.get("_provenance") or {}
            if "seed" in p:
                seen_seed[(p.get("family"), p["seed"])].add(sp)
            seen_gt[json.dumps(c["ground_truth"], sort_keys=True)].add(sp)
            seen_id[c["id"]].add(sp)
    pair = lambda d: Counter(" & ".join(sorted(v)) for v in d.values() if len(v) > 1)
    return {"same_seed": dict(pair(seen_seed)), "same_ground_truth": dict(pair(seen_gt)),
            "same_id": dict(pair(seen_id))}


def check_corpus(path: Path, limit: int = 0) -> list[dict]:
    """Audit each split, one family at a time (a pooled directory such as kepler_strict
    mixes families whose answer keys differ)."""
    splits = load_corpus(path, limit)
    reps = []
    for sp, cfgs in splits.items():
        groups = defaultdict(list)
        for c in cfgs:
            groups[((c.get("_provenance") or {}).get("family") or "?", tuple(c["ground_truth"]))].append(c)
        for (fam, _), cs in sorted(groups.items()):
            label = f"{Path(path).name}/{sp}" if sp else Path(path).name
            reps.append(audit([record(c) for c in cs], label + (f" [{fam}]" if len(groups) > 1 else "")))
    if len(splits) > 1:
        reps.append({"label": f"{Path(path).name} (cross-split)", "leaks": leaks(splits)})
    return reps


# ----------------------------------------------------------------------------- skills
def skill_report(names: list[str], corpora=None, limit: int = 200) -> str:
    """Skill coverage (sage/skills.py): which of the shared skills each family's steps and
    each domain exercise, and -- for emitted corpora (read-only) -- how many credit-bearing
    trap rows target each skill, i.e. what a policy can actually be rewarded for there."""
    from sage import registry
    from sage import skills as SK
    fams = [registry.get(n) for n in names]
    out = [SK.render(SK.coverage(fams))]
    for c in corpora or []:
        for sp, cfgs in load_corpus(Path(c), limit).items():
            by_fam = defaultdict(list)
            for cfg in cfgs:
                by_fam[(cfg.get("_provenance") or {}).get("family")].append(cfg)
            for fname, cs in sorted(by_fam.items(), key=lambda kv: str(kv[0])):
                try:
                    fam = registry.get(fname)
                except KeyError:
                    out.append(f"  {Path(c).name}/{sp} [{fname}]: not a registered family")
                    continue
                rows = SK.trap_rows(fam, cs)
                out.append(f"  {Path(c).name}/{sp} [{fname}] n={len(cs)}: credit-bearing trap rows by skill "
                           + (", ".join(f"{k} {v}" for k, v in rows.items()) if rows else "none (untagged or no traps)"))
    return "\n".join(out)


# ----------------------------------------------------------------------------- printing
def render(rep: dict) -> str:
    if "leaks" in rep:
        lk = rep["leaks"]
        lines = [f"== {rep['label']}"]
        for k, v in lk.items():
            lines.append(f"  {k:<18} " + (", ".join(f"{a}: {b}" for a, b in v.items()) if v else "none"))
        return "\n".join(lines)
    n = rep["n"]
    L = [f"== {rep['label']}   n={n}   keys={rep['keys']}"
         + (f"   enforced={rep['enforced']}" if rep.get("enforced") else "")]
    if "mean_tries" in rep:
        L.append(f"  mean tries {rep['mean_tries']}; shipped flagged {rep.get('shipped_flagged', 0)}/{n}"
                 + (f"; unenforced failures {rep['unenforced_failures']}" if rep.get("unenforced_failures") else ""))
    i1 = rep["I1"]
    L.append(f"  I1 gold within tol of latent: " + (f"{i1['pass']}/{i1['checked']}" if i1["checked"] else "n/a (no latent answer)")
             + (f"  e.g. {i1['fail_ids'][:3]}" if i1["fail_ids"] else ""))
    i2 = rep["I2"]
    if i2["no_ablation_table"] == n:
        L.append("  I2 no ablation table: step credit impossible, traps unmeasurable from the files")
    else:
        L.append(f"  I2 instances with every credit-bearing trap live: {i2['instances_all_live']}/{n}")
        for k, v in i2["per_trap"].items():
            core = {True: "core", False: "cond", None: ""}[v.get("core")]
            L.append(f"     {k:<30} step {v['step']!s:<4} {core:<4} live {v.get('live', 0):>4}/{v['n']:<4}"
                     + "".join(f" {t} {v[t]}" for t in ("dead", "coincident", "no_answer") if v.get(t))
                     + f"   earns {rep['trap_earns'].get(k, 'n/a')}")
    i3 = rep["I3"]
    if i3["variants"]:
        L.append(f"  I3 instances with every accepted variant inside: {i3['instances_all_inside']}/{n}")
        for k, v in i3["variants"].items():
            L.append(f"     {k:<30} inside {v.get('inside', 0):>4}/{v['n']}"
                     + (f"  outside {v['outside']}" if v.get("outside") else ""))
    i4 = rep["I4"]
    L.append(f"  I4 oracle scores 1.0: {i4['oracle_ok']}/{n};  trap rows paid exactly their step: "
             f"{i4['rows_ok']}/{i4['rows']}" + (f"  unpaid {i4['unpaid']} via {i4['unpaid_by_branch']}" if i4["unpaid"] else ""))
    c = rep["constants"]
    L.append(f"  constant = mean {c['mean']['answer']}: reward {c['mean']['mean_reward']}, pass {c['mean']['pass_rate']}"
             f"  {c['mean']['branches']}")
    b = c["best"]
    L.append(f"  best constant ({b['which']}) {b['answer']}: reward {b['mean_reward']}, pass {b['pass_rate']}  {b['branches']}")
    for k, v in rep.get("shortcuts", {}).items():
        L.append(f"  shortcut {k}: reward {v['mean_reward']}, pass {v['pass_rate']}")
    return "\n".join(L)
