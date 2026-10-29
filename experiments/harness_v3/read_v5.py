#!/usr/bin/env python
"""HARNESS v5 PHASE-1 reading -- "WRITE THE METHOD, TEST IT ON NEW DATA" vs v4 (the soft gate) and v3 (docs/RL_PLAN.md
2026-09-29 14:15 UTC, "HARNESS v5 ... REGISTERED before any build or run"). The untrained Qwen3-8B, one run per cell, paired by
instance (synthetic TEST, 17 sealed families 12x2) and by problem (externals):
  v3  <root>/V3      (HV3_17: the v3 evaluation harness)            externals <root>/V3/ext
  v4  <root>/V3EFS   (EFS_1: v3 + the soft execution gate)           externals <root>/V3EFSX/ext   (EFS_2)
  v5  <root>/V3V5    (V5_1: v3 + program-first, program_first.py)    externals <root>/V3V5X/ext    (V5_2)

    venv/bin/python experiments/harness_v3/read_v5.py [--root results/pods/harness_v3] [--base V3] [--v4 V3EFS] [--v4x V3EFSX]
        [--v5 V3V5] [--v5x V3V5X] [--bench-root benchmarks] [--perms 20000] [--seed 0] [--out DIR]
        [--oracle [--oracle-k 8] [--workers 8]]

A NEW FILE: read_ef2.py / read_ef.py / analyze.py are imported for their helpers (TEST-all twin-weighted cell_pass, the paired
sign-flip test, completeness, the exact McNemar, the externals' loaders, the QRData causal split) and never edited.

REGISTERED RULE (verbatim in substance), Phase 1:
  V5-ADOPT       iff TEST-all(v5) >= TEST-all(v4) + 2 points (one-sided sign-flip p <= 0.05) AND no external (QRData, DABench,
                 BixBench) is down by more than 3 points vs v4.
  V5-EQUIVALENT  iff not ADOPT but TEST-all(v5) >= TEST-all(v3) + 3 (one-sided sign-flip p <= 0.05) and no external down by more
                 than 3 vs v3 (kept as the ACQUISITION harness, not the default).
  V5-NULL        otherwise.
TEST-all = the twin-weighted mean over the 17 sealed TEST families (analyze.py's TEST weights, as read_efs.py / read_ef.py);
the sign-flip test is analyze.paired's (E.pair). "Down by more than 3" is the point difference on the problems both cells
have; the exact McNemar p of every external contrast is printed beside it (never deciding). STATUS = the label once every
cell is COMPLETE (every TEST family at the draw's n x k in v3 / v4 / v5; every problem of every benchmark with summary.json in
the three external cells); otherwise INCOMPLETE with the provisional label on the rows present.
BESIDE (never deciding): HO6 (v5 vs v4 and v3); the per-family table; the protocol's mechanics per cell -- rollouts with >= 1
VALID submission, final answer by solve / by the soft fallback / none (with pass rates), submissions and revisions used,
refused JSON answers, the sibling-flag rates per submission (exception on a sibling, missing keys, non-numeric, constant
across siblings, identical to the target), mean portability of the final submissions; the QRData causal / non-causal
split; provenance (every v5 cell records program_first_v5 -- never exec_forcing / exec_forcing_soft -- over the plain v3
harness, on the untrained 8B); the ORACLE GENERAL RATE of the submitted programs (--oracle, below).
ORACLE GENERAL RATE (--oracle; offline, CPU; read_gr's rule): every TEST rollout whose final answer is a solve() output has its
final submission re-run, unchanged, by program_first.run_once (the harness's own runner cell, the v3 kernel sandbox class,
180 s per run; solve(data_dir, task) with each sibling's OWN task text, as in the harness since the 15:55 amendment) on K=8
layout-matched TRAIN siblings of its target (program_first.siblings_for with the key "v5_oracle": a
draw apart from the 3 the agent saw), each graded by the SIBLING's own grader (rl/grpo_prog.PortableStage.grade:
StepReward at instance_tol -- the census / acquire path); a program is GENERAL iff it passes >= 7/8 of its applications
(experiments/acquire/read_gr.GENERAL; the fraction is over its applications when the layout has fewer siblings). Results are
cached in <out>/V5_ORACLE.jsonl (resumable). NOTE printed with it: in 7 of the 17 families the sibling's QUESTION differs from
the target's in a parameter that lives in the task text, not the data (hic_loop / hic_var: the bin pair; gwas_loci: the trait;
hc_quality: the hospital reviewed; eco_size: the legal size; hc_dx: the units per site; pk_oral: the assay error model and
LLOQ) -- marked [task-param]: a program is general there only if it reads that parameter from `task` (the census's GENERAL
counts, whose programs never saw the sibling's task, are bounded by construction in those families).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_ef2 as R2  # noqa: E402  (read_ef / read_v3 / analyze helpers; never edited)

E = R2.E
A = R2.A
HO6 = E.HO6
fmt = E.fmt
BENCHES = R2.BENCHES
ADOPT_BAR, ADOPT_P, EQ_BAR, EQ_P, DOWN_BAR = 2.0, 0.05, 3.0, 0.05, 3.0
EPS = 1e-9
HARNESS = "program_first_v5"
GATES = ("exec_forcing", "exec_forcing_soft")
PARAM_FAMILIES = ("hic_loop", "hic_var", "gwas_loci", "hc_quality", "eco_size", "hc_dx", "pk_oral")
ORACLE_KEY, ORACLE_K = "v5_oracle", 8


# ----------------------------------------------------------------------------------------------------------- the rule
def ext_ok(x: dict | None) -> bool | None:
    return None if x is None else x["d"] >= -DOWN_BAR - EPS


def verdict(t_v4: dict | None, t_v3: dict | None, x_v4: dict, x_v3: dict, complete: bool) -> dict:
    """t_v4 / t_v3: E.pair(v5, v4 / v3) on TEST-all; x_v4 / x_v3: {bench: R2.mcnemar(v5, v4 / v3)}."""
    legs = {}
    a_test = None if t_v4 is None else (t_v4["d"] >= ADOPT_BAR - EPS and t_v4["p_gt"] <= ADOPT_P)
    legs["ADOPT.TEST"] = {"ok": a_test, "why": "no paired rows" if t_v4 is None else
                          f"TEST-all v5 - v4 {t_v4['d']:+.1f} (sign-flip p>0 {t_v4['p_gt']:.4f}) "
                          + ("is" if a_test else "is not") + f" >= +{ADOPT_BAR:.0f} with p <= {ADOPT_P}"}
    for b in BENCHES:
        o = ext_ok(x_v4.get(b))
        legs[f"ADOPT.{b}"] = {"ok": o, "why": "no paired rows" if o is None else
                              f"{b} v5 - v4 {x_v4[b]['d']:+.1f} (+{x_v4[b]['b01']}/-{x_v4[b]['b10']}) "
                              + ("is not" if o else "IS") + f" down by more than {DOWN_BAR:.0f}"}
    e_test = None if t_v3 is None else (t_v3["d"] >= EQ_BAR - EPS and t_v3["p_gt"] <= EQ_P)
    legs["EQUIV.TEST"] = {"ok": e_test, "why": "no paired rows" if t_v3 is None else
                          f"TEST-all v5 - v3 {t_v3['d']:+.1f} (sign-flip p>0 {t_v3['p_gt']:.4f}) "
                          + ("is" if e_test else "is not") + f" >= +{EQ_BAR:.0f} with p <= {EQ_P}"}
    for b in BENCHES:
        o = ext_ok(x_v3.get(b))
        legs[f"EQUIV.{b}"] = {"ok": o, "why": "no paired rows" if o is None else
                              f"{b} v5 - v3 {x_v3[b]['d']:+.1f} (+{x_v3[b]['b01']}/-{x_v3[b]['b10']}) "
                              + ("is not" if o else "IS") + f" down by more than {DOWN_BAR:.0f}"}
    adopt = all(legs[k]["ok"] is True for k in legs if k.startswith("ADOPT."))
    equiv = all(legs[k]["ok"] is True for k in legs if k.startswith("EQUIV."))
    label = "V5-ADOPT" if adopt else ("V5-EQUIVALENT" if equiv else "V5-NULL")
    return {"status": label if complete else "INCOMPLETE", "label": label, "final": bool(complete), "legs": legs,
            "failed": [k for k, v in legs.items() if v["ok"] is False]}


# ----------------------------------------------------------------------------------------------------------- mechanics
def v5_of(r: dict) -> dict | None:
    x = r.get("v5") if isinstance(r.get("v5"), dict) else (r.get("v2") or {}).get("v5")
    return x if isinstance(x, dict) else None


def mechanics(rows: list, passed, errored) -> dict:
    """The protocol's bookkeeping over one cell's rows (synthetic rollouts or external problem records)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    logs = [(r, v5_of(r)) for r in rows]
    has = [(r, x) for r, x in logs if x is not None]
    subs = [s for _, x in has for s in (x.get("subs") or [])]
    sib_subs = [s for s in subs if s.get("siblings")]
    fin = {}
    for lab in ("solve", "soft_fallback", None):
        sel = [r for r, x in has if x.get("final") == lab]
        k = sum(1 for r in sel if passed(r))
        fin[str(lab)] = {"n": len(sel), "passed": k, "rate": (100.0 * k / len(sel)) if sel else None}
    dist = {str(i): sum(1 for _, x in has if int(x.get("submissions") or 0) == i) for i in range(0, 4)}

    def frac(pred, pool):
        return (100.0 * sum(1 for s in pool if pred(s)) / len(pool)) if pool else None
    port = [float(x.get("portability") or 0.0) for _, x in has if x.get("final") == "solve"]
    return {"n": n, "with_log": len(has), "errored": sum(1 for r in rows if errored(r)),
            "valid_rollouts": sum(1 for _, x in has if int(x.get("valid") or 0) > 0),
            "valid_rate": 100.0 * sum(1 for _, x in has if int(x.get("valid") or 0) > 0) / n,
            "final": fin, "submissions_total": len(subs), "submissions_mean": len(subs) / n, "submissions_dist": dist,
            "revisions_total": sum(int(x.get("revisions") or 0) for _, x in has),
            "refused_answers": sum(int(x.get("refused") or 0) for _, x in has),
            "refused_submissions": sum(int(x.get("refused_submissions") or 0) for _, x in has),
            "offered_no_answer": sum(1 for _, x in has if x.get("final") is None and int(x.get("offered") or 0) > 0),
            "valid_submission_rate": frac(lambda s: s.get("valid"), subs),
            "flags_per_submission": {"target_exception": frac(lambda s: "target" in (s.get("flags") or {}).get("exception", []),
                                                               subs),
                                     "sibling_exception": frac(lambda s: any(x != "target" for x in (s.get("flags") or {})
                                                                             .get("exception", [])), sib_subs),
                                     "missing": frac(lambda s: bool((s.get("flags") or {}).get("missing")), subs),
                                     "nonnumeric": frac(lambda s: bool((s.get("flags") or {}).get("nonnumeric")), subs),
                                     "constant": frac(lambda s: bool((s.get("flags") or {}).get("constant")), sib_subs),
                                     "identical": frac(lambda s: bool((s.get("flags") or {}).get("identical")), sib_subs)},
            "portability_mean_final": float(np.mean(port)) if port else None,
            "k_siblings": sorted({int(x.get("k") or 0) for _, x in has})}


def syn_rows(res, members=None) -> list:
    if res is None:
        return []
    return [r for (s, f), rows in res["rollouts"].items() if s == "test" and (members is None or f in members) for r in rows]


def syn_mech(res, members=None) -> dict | None:
    if res is None:
        return None
    return mechanics(syn_rows(res, members), lambda r: bool(r.get("passed")),
                     lambda r: str(r.get("attribution") or "").startswith("error:"))


def ext_mech(recs: dict) -> dict | None:
    if not recs:
        return None
    return mechanics(list(recs.values()), lambda r: bool((r.get("grade") or {}).get("passed")), lambda r: bool(r.get("error")))


def v5_provenance(root: Path, name: str, sets, plain_harness: dict | None, ext: bool) -> list:
    """Problems with a v5 cell's provenance (empty list = clean)."""
    c = R2.load_cfg(root / name)
    if c is None:
        return [f"{root / name}/config.json missing"]
    bad = []
    h = dict(c.get("harness") or {})
    for g in GATES:
        if g in h:
            bad.append(f"config.json records {g} (an execution gate) -- not a v5 directory")
            h.pop(g, None)
    rec = h.pop(HARNESS, None) or {}
    if rec.get("harness") != HARNESS or not rec.get("on"):
        bad.append("config.json does not record program_first_v5 -- NOT a v5 directory")
    elif rec.get("k_siblings") != 3 or rec.get("max_revisions") != 2:
        bad.append(f"program_first_v5 recorded with K={rec.get('k_siblings')} / revisions {rec.get('max_revisions')}, "
                   "not the registered 3 / 2")
    elif rec.get("signature") != "solve(data_dir, task)":
        bad.append(f"program_first_v5 recorded with signature {rec.get('signature')!r}, not the amended solve(data_dir, task) "
                   "(docs/RL_PLAN.md 2026-09-29 15:55 UTC)")
    if c.get("model", "Qwen3-8B") != "Qwen3-8B" or c.get("ckpt"):
        bad.append(f"config.json model={c.get('model')} ckpt={c.get('ckpt')!r} (Phase 1 is read on the untrained 8B)")
    want = dict(plain_harness) if plain_harness else dict(R2.V3_HARNESS, system="science")
    off = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    if off:
        bad.append(f"config.json harness differs from the plain v3 cell's: {off}")
    if not ext:
        for s in ("design", "test"):
            for f in sets[s]:
                p = root / name / s / f"{f}.json"
                if p.is_file():
                    hv = json.loads(p.read_text()).get("harness_v2") or {}
                    if (hv.get(HARNESS) or {}).get("harness") != HARNESS or any(g in hv for g in GATES):
                        bad.append(f"{s}/{f}.json does not record program_first_v5 (only)")
    else:
        for b in BENCHES:
            hj = root / name / "ext" / b / "harness_v2.json"
            if hj.is_file():
                hv = json.loads(hj.read_text())
                if (hv.get(HARNESS) or {}).get("harness") != HARNESS or any(g in hv for g in GATES):
                    bad.append(f"ext/{b}/harness_v2.json does not record program_first_v5 (only)")
    return bad


# ----------------------------------------------------------------------------------------------------------- oracle
_OLOCK = threading.Lock()


def oracle_jobs(res, sets, gen_root: Path) -> list:
    """(family, instance, rollout index, code, code_sha, target data dir) of every TEST rollout answered by solve()."""
    jobs = []
    fams = sorted({f for (s, f) in (res or {}).get("rollouts", {}) if s == "test"})
    for f in fams:
        p = res["_dir"] / "test" / f"{f}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        for inst, rs in (d.get("groups") or {}).items():
            for i, r in enumerate(rs):
                x = v5_of(r) or {}
                if x.get("final") != "solve" or not x.get("final_submission"):
                    continue
                subs = x.get("subs") or []
                n = int(x["final_submission"])
                if n < 1 or n > len(subs) or not subs[n - 1].get("code"):
                    continue
                code = subs[n - 1]["code"]
                jobs.append({"family": f, "instance": inst, "rollout": i, "code": code,
                             "code_sha": hashlib.sha256(code.encode()).hexdigest()[:16],
                             "target": str(gen_root / f / "test" / inst)})
    return jobs


def oracle(res, sets, out: Path, k: int, workers: int, gen_root: Path, sandbox_cls=None) -> dict:
    """Re-run every final solve() on K layout-matched train siblings, graded by each sibling's grader; cached, resumable."""
    import program_first as PF
    from rl import dispersion
    from rl import grpo_prog as GP
    from rl.reward import StepReward
    GEN = 7 / 8
    try:
        sys.path.insert(0, str(ROOT / "experiments" / "acquire"))
        import read_gr
        GEN = read_gr.GENERAL
    except Exception:                                                   # noqa: BLE001
        pass
    if sandbox_cls is None:
        sys.path.insert(0, str(ROOT / "harness"))
        import v2
        sandbox_cls = v2.sandbox_class(v2.HarnessConfig(kernel=True))
    cache = out / "V5_ORACLE.jsonl"
    done = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            try:
                x = json.loads(line)
                done[(x["instance"], x["code_sha"], x["sibling"])] = x
            except (ValueError, KeyError):
                pass
    stage = GP.PortableStage(dispersion)
    jobs = oracle_jobs(res, sets, gen_root)
    work = []
    for j in jobs:
        sibs = PF.siblings_for(Path(j["target"]) / "data_files", k, ORACLE_KEY)
        j["siblings"] = [s.name for s in sibs]
        for s in sibs:
            if (j["instance"], j["code_sha"], s.name) not in done:
                work.append((j, s))

    class _Host:                          # what run_once needs of an "episode sandbox": its class and timeout
        timeout = 180
        data_dir = None
        workdir = None

    def go(item):
        j, s = item
        run = PF.run_once(PF.runner_cell(j["code"], PF.sibling_task(s)), _Host(), s, 180)   # the sibling's OWN task
        scfg = json.loads((s / "eval_config.json").read_text())
        c = PF.check(run, list(scfg.get("ground_truth") or {}), StepReward._fields(scfg))
        passed = False
        if c["ran"]:
            try:
                passed = stage.grade({"answer": run["value"]}, s, 5, 1.0)
            except Exception:                                           # noqa: BLE001
                passed = False
        rec = {"family": j["family"], "instance": j["instance"], "code_sha": j["code_sha"], "sibling": s.name,
               "status": run.get("status"), "clean": c["clean"], "passed": bool(passed)}
        with _OLOCK:
            with open(cache, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
        return rec

    saved = PF.FRESH
    PF.FRESH = lambda sb, data_dir, workdir, timeout: sandbox_cls(data_dir=data_dir, workdir=workdir, timeout=timeout)
    try:
        out.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for rec in ex.map(go, work):
                done[(rec["instance"], rec["code_sha"], rec["sibling"])] = rec
    finally:
        PF.FRESH = saved
    per_fam: dict = {}
    progs = []
    for j in jobs:
        apps = [done.get((j["instance"], j["code_sha"], s)) for s in j["siblings"]]
        apps = [a for a in apps if a is not None]
        orc = (sum(a["passed"] for a in apps) / len(apps)) if apps else None
        g = bool(orc is not None and orc >= GEN - 1e-9)
        progs.append({**{k2: j[k2] for k2 in ("family", "instance", "rollout", "code_sha")}, "n_apps": len(apps),
                      "oracle": orc, "general": g})
        pf = per_fam.setdefault(j["family"], {"programs": 0, "general": 0})
        pf["programs"] += 1
        pf["general"] += int(g)
    n = len(progs)
    return {"k": k, "general_rule": GEN, "programs": n, "general": sum(p["general"] for p in progs),
            "rate": (100.0 * sum(p["general"] for p in progs) / n) if n else None, "per_family": per_fam,
            "param_families": list(PARAM_FAMILIES), "rows": progs, "cache": str(cache)}


# ----------------------------------------------------------------------------------------------------------- report helpers
def ptxt(x):
    if not x:
        return "-"
    return f"{x['d']:+.1f} (p>0 {x['p_gt']:.4f}, p<0 {x['p_lt']:.4f})"


def mtxt(x):
    if not x:
        return "-"
    return f"{x['d']:+.1f} (+{x['b01']} / -{x['b10']}; p>0 {x['p_gt']:.4f}, p<0 {x['p_lt']:.4f}, two-sided {x['p2']:.4f})"


def rate_txt(p: dict) -> str:
    return f"{100.0 * sum(p.values()) / len(p):.1f} ({len(p)})" if p else "-"


def mech_line(tag: str, m: dict | None) -> str:
    if not m or not m.get("n"):
        return f"* {tag}: (no rows)"
    f = m["final"]

    def fr(lab):
        x = f[lab]
        return f"{x['n']}" + (f" ({x['passed']} pass, {x['rate']:.1f}%)" if x["n"] else "")
    fl = m["flags_per_submission"]
    ftxt = ", ".join(f"{k} {fmt(v)}%" for k, v in fl.items() if v is not None)
    return (f"* {tag}: rows {m['n']} (v5 log {m['with_log']}, errored {m['errored']}); >= 1 VALID submission "
            f"{m['valid_rollouts']} ({m['valid_rate']:.1f}%); final answer by solve {fr('solve')}, by the soft fallback "
            f"{fr('soft_fallback')}, none {fr('None')}; submissions {m['submissions_total']} ({m['submissions_mean']:.2f} / row; "
            f"rows with 0/1/2/3: {'/'.join(m['submissions_dist'][str(i)].__str__() for i in range(4))}), revisions "
            f"{m['revisions_total']}; valid submissions {fmt(m['valid_submission_rate'])}%; refused JSON answers "
            f"{m['refused_answers']}, refused submissions {m['refused_submissions']}; flags per submission: {ftxt or '-'}; "
            f"mean portability of final submissions {fmt(m['portability_mean_final'], 2)}; K {m['k_siblings']}")


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--bench-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--base", default="V3", help="v3, TEST (externals under <base>/ext)")
    ap.add_argument("--v4", default="V3EFS", help="v4 (the soft gate), TEST (EFS_1)")
    ap.add_argument("--v4x", default="V3EFSX", help="v4, externals (EFS_2)")
    ap.add_argument("--v5", default="V3V5", help="v5, TEST (V5_1)")
    ap.add_argument("--v5x", default="V3V5X", help="v5, externals (V5_2)")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where V5_READING.md/.json (and V5_ORACLE.jsonl) go (default: <root>/<v5>, "
                                                "else <root>/<v5x>, if either exists; else nothing is written)")
    ap.add_argument("--oracle", action="store_true", help="compute the oracle general rate (offline CPU re-runs; cached)")
    ap.add_argument("--oracle-k", type=int, default=ORACLE_K)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-root", default=None, help="generated/ (default: paths.data_root()/generated)")
    a = ap.parse_args(argv)
    root, broot = Path(a.root), Path(a.bench_root)
    sets = A.load_families()
    test_all = A.cell_members(sets, "test")

    def load(name):
        if not (root / name).is_dir():
            return None
        r = A.load_config(root, name, sets)
        r["_dir"] = root / name
        return r
    v3, v4, v5 = load(a.base), load(a.v4), load(a.v5)
    miss = {"v3": E.complete(v3, sets, test_all), "v4": E.complete(v4, sets, test_all), "v5": E.complete(v5, sets, test_all)}
    plain_cfg = R2.load_cfg(root / a.base)
    plain_h = (plain_cfg or {}).get("harness")

    # externals
    nprob = R2.problem_counts(broot)
    xdir = {"v3": root / a.base / "ext", "v4": root / a.v4x / "ext", "v5": root / a.v5x / "ext"}
    recs = {k: {b: R2.load_ext(d / b) for b in BENCHES} for k, d in xdir.items()}
    xcomp = {k: {b: (xdir[k] / b / "summary.json").is_file() and len(recs[k][b]) == nprob[b] for b in BENCHES} for k in xdir}
    passed = {k: {b: R2.ext_passed(recs[k][b]) for b in BENCHES} for k in xdir}
    x_v4 = {b: R2.mcnemar(passed["v5"][b], passed["v4"][b]) if recs["v5"][b] and recs["v4"][b] else None for b in BENCHES}
    x_v3 = {b: R2.mcnemar(passed["v5"][b], passed["v3"][b]) if recs["v5"][b] and recs["v3"][b] else None for b in BENCHES}
    x_43 = {b: R2.mcnemar(passed["v4"][b], passed["v3"][b]) if recs["v4"][b] and recs["v3"][b] else None for b in BENCHES}

    # the registered reading (its own rng: nothing beside can move it)
    rng = np.random.default_rng(a.seed)
    t_v4 = E.pair(v5, v4, sets, a.perms, rng, test_all)
    t_v3 = E.pair(v5, v3, sets, a.perms, rng, test_all)
    complete = (all(x is not None for x in (v3, v4, v5)) and not any(miss.values())
                and all(xcomp[k][b] for k in xdir for b in BENCHES))
    V = verdict(t_v4, t_v3, x_v4, x_v3, complete)

    # beside
    rng_b = np.random.default_rng(a.seed + 1)
    cp = {k: (A.cell_pass(v, sets, "test", test_all)[0] if v is not None else None) for k, v in (("v3", v3), ("v4", v4), ("v5", v5))}
    ho = {k: (A.cell_pass(v, sets, "test", HO6)[0] if v is not None else None) for k, v in (("v3", v3), ("v4", v4), ("v5", v5))}
    beside = {"test_all": {**cp, "v5_minus_v4": t_v4, "v5_minus_v3": t_v3,
                           "v4_minus_v3": E.pair(v4, v3, sets, a.perms, rng_b, test_all)},
              "ho6": {**ho, "v5_minus_v4": E.pair(v5, v4, sets, a.perms, rng_b, HO6),
                      "v5_minus_v3": E.pair(v5, v3, sets, a.perms, rng_b, HO6),
                      "v4_minus_v3": E.pair(v4, v3, sets, a.perms, rng_b, HO6)}}
    cau = R2.qrdata_causal(broot)
    split = {}
    for lab, want in (("causal", True), ("non-causal", False)):
        ids = {k for k, v in cau.items() if v == want}
        split[lab] = {"n_ids": len(ids),
                      "v5_minus_v4": R2.mcnemar(passed["v5"]["qrdata"], passed["v4"]["qrdata"], ids)
                      if recs["v5"]["qrdata"] and recs["v4"]["qrdata"] else None,
                      "v5_minus_v3": R2.mcnemar(passed["v5"]["qrdata"], passed["v3"]["qrdata"], ids)
                      if recs["v5"]["qrdata"] and recs["v3"]["qrdata"] else None}
    mech = {"synthetic": syn_mech(v5), "externals": {b: ext_mech(recs["v5"][b]) for b in BENCHES}}
    per = {}
    for f in test_all:
        m = syn_mech(v5, [f])
        per[f] = {"v3": E.fam(v3, sets, f), "v4": E.fam(v4, sets, f), "v5": E.fam(v5, sets, f),
                  "v5_minus_v4": E.pair(v5, v4, sets, a.perms, rng_b, [f]), "v5_minus_v3": E.pair(v5, v3, sets, a.perms, rng_b, [f]),
                  "valid_rate": (m or {}).get("valid_rate"), "final_solve": ((m or {}).get("final") or {}).get("solve"),
                  "param_family": f in PARAM_FAMILIES}

    # provenance / warnings
    warn = []
    if v5 is not None:
        warn += [f"PROVENANCE {a.v5}: {x}" for x in v5_provenance(root, a.v5, sets, plain_h, ext=False)]
    if (root / a.v5x).is_dir():
        warn += [f"PROVENANCE {a.v5x}: {x}" for x in v5_provenance(root, a.v5x, sets, plain_h, ext=True)]
    shas = {}
    for n in (a.v5, a.v5x):
        c = R2.load_cfg(root / n)
        if c:
            shas[n] = ((c.get("harness") or {}).get(HARNESS) or {}).get("sha256") or {}
    if len(shas) == 2 and shas[a.v5] != shas[a.v5x]:
        warn.append(f"the v5 cells were written by different v5 builds: {json.dumps(shas)}")
    for tag, m in [("TEST", mech["synthetic"])] + [(b, mech["externals"][b]) for b in BENCHES]:
        if m and m.get("n"):
            if m["with_log"] < m["n"] - m["errored"]:
                warn.append(f"{tag}: {m['n'] - m['with_log']} v5 rows carry no v5 log (and only {m['errored']} errored)")
            if m["offered_no_answer"]:
                warn.append(f"{tag}: {m['offered_no_answer']} v5 rows end with NO answer although the model offered one -- "
                            "the soft fallback did not fire")

    out = Path(a.out) if a.out else ((root / a.v5) if (root / a.v5).is_dir() else ((root / a.v5x) if (root / a.v5x).is_dir()
                                                                                  else None))
    orc = None
    if a.oracle:
        if v5 is None or out is None:
            warn.append("--oracle: no v5 TEST cell -- nothing to re-run")
        else:
            gen = Path(a.gen_root) if a.gen_root else None
            if gen is None:
                sys.path.insert(0, str(ROOT))
                from paths import data_root
                gen = data_root() / "generated"
            orc = oracle(v5, sets, out, a.oracle_k, a.workers, gen)

    # report
    L = ["# HARNESS v5 -- program-first ('write the method, test it on new data') vs v4 and v3 (docs/RL_PLAN.md 2026-09-29 "
         "14:15 UTC, Phase 1)\n",
         f"root `{root}`: v3 `{a.base}` (+ `{a.base}/ext`), v4 `{a.v4}` / `{a.v4x}/ext`, v5 `{a.v5}` / `{a.v5x}/ext`; untrained "
         f"Qwen3-8B; paired by instance (TEST, one-sided sign-flip, {a.perms} perms) and by problem (externals, exact McNemar "
         f"beside). Rule: read_v5.py docstring.\n"]
    for n, what in ((a.v5, "V5_1 (TEST)"), (a.v5x, "V5_2 (externals)")):
        if not (root / n).is_dir():
            L.append(f"**{root / n} does not exist yet ({what}) -- PENDING.**")
    for k, v in miss.items():
        if v and not (k == "v5" and v5 is None):
            L.append(f"* TEST {k}: {len(v)} families missing or short: {', '.join(v)}")
    for k in xdir:
        part = [f"{b} {len(recs[k][b])}/{nprob[b]}" for b in BENCHES if not xcomp[k][b]]
        if part:
            L.append(f"* externals {k}: INCOMPLETE -- " + ", ".join(part))
    L.append("\n## Registered reading (Phase 1)\n")
    L.append("| leg | holds | detail |")
    L.append("|---|---|---|")
    for k, lg in V["legs"].items():
        L.append(f"| {k} | {({True: 'yes', False: '**NO**', None: '-'})[lg['ok']]} | {lg['why']} |")
    L.append(f"\n**{V['status']}**" + ("" if V["final"] else f" (provisional on the rows present: {V['label']})"))
    if V["label"] == "V5-ADOPT" and V["final"]:
        L.append("* ADOPT: v5 becomes the default harness for untrained-8B reads.")
    elif V["label"] == "V5-EQUIVALENT" and V["final"]:
        L.append("* EQUIVALENT: v5 is kept as the ACQUISITION harness, not the default.")
    L.append("\n## Beside: TEST-all / HO6 (never deciding)\n")
    L.append("| set | v3 | v4 | v5 | v5 - v4 | v5 - v3 | v4 - v3 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, key in (("TEST-all (twin-weighted)", "test_all"), ("HO6", "ho6")):
        b = beside[key]
        L.append(f"| {lab} | {fmt(b['v3'])} | {fmt(b['v4'])} | {fmt(b['v5'])} | {ptxt(b['v5_minus_v4'])} | "
                 f"{ptxt(b['v5_minus_v3'])} | {ptxt(b['v4_minus_v3'])} |")
    L.append("\n## Externals (the down-by-3 legs use the point difference; McNemar beside)\n")
    L.append("| benchmark | v3 (n) | v4 (n) | v5 (n) | v5 - v4 | v5 - v3 | v4 - v3 |")
    L.append("|---|---|---|---|---|---|---|")
    for b in BENCHES:
        L.append(f"| {b} | " + " | ".join(rate_txt(passed[k][b]) for k in ("v3", "v4", "v5"))
                 + f" | {mtxt(x_v4[b])} | {mtxt(x_v3[b])} | {mtxt(x_43[b])} |")
    L.append("\n| QRData subset | v5 - v4 | v5 - v3 |")
    L.append("|---|---|---|")
    for lab, s in split.items():
        L.append(f"| {lab} ({s['n_ids']}) | {mtxt(s['v5_minus_v4'])} | {mtxt(s['v5_minus_v3'])} |")
    L.append("\n## The protocol's mechanics (v5 cells)\n")
    L.append(mech_line("TEST", mech["synthetic"]))
    for b in BENCHES:
        L.append(mech_line(b, mech["externals"][b]))
    L.append("\n## Per TEST family (pass %, 12x2)\n")
    L.append("| family | v3 | v4 | v5 | v5 - v4 (p>0) | v5 - v3 (p>0) | rollouts with a valid solve | answered by solve (pass) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in test_all:
        x = per[f]
        fs = x["final_solve"]
        L.append(f"| {f}{' (HO6)' if f in HO6 else ''}{' [task-param]' if x['param_family'] else ''} | {fmt(x['v3'])} | "
                 f"{fmt(x['v4'])} | {fmt(x['v5'])} | "
                 + (f"{x['v5_minus_v4']['d']:+.1f} ({x['v5_minus_v4']['p_gt']:.3f})" if x["v5_minus_v4"] else "-") + " | "
                 + (f"{x['v5_minus_v3']['d']:+.1f} ({x['v5_minus_v3']['p_gt']:.3f})" if x["v5_minus_v3"] else "-") + " | "
                 + (f"{x['valid_rate']:.1f}%" if x["valid_rate"] is not None else "-") + " | "
                 + (f"{fs['n']} ({fs['passed']})" if fs else "-") + " |")
    L.append("\n[task-param] = the sibling's question differs from the target's in a parameter of the TASK TEXT (read_v5.py "
             "docstring): solve(data_dir, task) receives each dataset's own task, so a method is general there only if it "
             "reads the parameter from task.")
    L.append("\n## Oracle general rate of the submitted solve() programs (beside; read_gr's rule)\n")
    if orc is None:
        L.append("* not computed (run with --oracle on a CPU allocation: every final solve() re-run on 8 layout-matched train "
                 "siblings, graded by each sibling's grader; cached in V5_ORACLE.jsonl).")
    else:
        L.append(f"* {orc['general']} of {orc['programs']} final programs are GENERAL (>= {orc['general_rule']:.3f} of K="
                 f"{orc['k']} applications pass): {fmt(orc['rate'])}%. Per family: "
                 + "; ".join(f"{f} {v['general']}/{v['programs']}{' [task-param]' if f in PARAM_FAMILIES else ''}"
                             for f, v in sorted(orc["per_family"].items())))
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    L.insert(2, f"**Status: {V['status']}**" + ("" if V["final"] else f" (provisional: {V['label']})") + "\n")
    obj = {"root": str(root), "cells": {"v3": a.base, "v4": a.v4, "v4x": a.v4x, "v5": a.v5, "v5x": a.v5x},
           "status": V["status"], "label": V["label"], "final": V["final"], "legs": V["legs"], "failed": V["failed"],
           "values": {"test_v5_minus_v4": t_v4, "test_v5_minus_v3": t_v3, "ext_v5_minus_v4": x_v4, "ext_v5_minus_v3": x_v3},
           "complete": complete, "missing": miss,
           "ext_completeness": {k: xcomp[k] for k in xdir}, "ext_n": {k: {b: len(recs[k][b]) for b in BENCHES} for k in xdir},
           "beside": beside, "ext_v4_minus_v3": x_43, "qrdata_split": split, "mechanics": mech, "per_family": per,
           "oracle": orc, "v5_sha256": shas, "warnings": warn}
    text = "\n".join(L) + "\n"
    print(text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "V5_READING.md").write_text(text)
        (out / "V5_READING.json").write_text(json.dumps(obj, indent=1, default=str))
    print(f"V5 SUMMARY: {V['status']}" + ("" if V["final"] else f" (provisional {V['label']})")
          + (f"; failing legs: {', '.join(V['failed'])}" if V["failed"] else "")
          + (f"; written {out / 'V5_READING.md'}" if out is not None else "; nothing written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
