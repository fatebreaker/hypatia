#!/usr/bin/env python
"""SAME-DRAW PREDICTION of HARNESS v4 (EFS) from the stored hard-gate rollouts (docs/RL_PLAN.md 2026-09-29 05:35 UTC:
logged before computing; BESIDE, NOT the registered reading -- EFS_1 / EFS_2 on fresh draws decide, and nothing about the
rule changes after this).

WHY IT IS EXACT ON THESE DRAWS. EFS (experiments/harness_v3/exec_forcing_soft.py) differs from the hard gate EF only in
SoftGate.fallback: at the end of an episode with no accepted answer, EF's fallback runs first (the last offered answer
grounded against every executed cell, logged "fallback"); only when it returns none AND the model offered an answer does
EFS submit the LAST offered answer. Every turn before that point is identical, so a stored EF rollout whose log says
accepted = None with offered >= 1 is exactly the episode EFS would have run, and EFS's submission is the last answer the
model offered in it. Everything else (accepted, grounded fallback, no offer at all, errored) is unchanged.

WHAT IT DOES (CPU only, reads only; writes only to --out if given):
  TEST       every rollout of <root>/<ef>/test/<family>.json (V3EF, EF_1) with v2.ef.accepted None and offered >= 1 gets
             the last answer offered in its stored messages (agents.extract_answer on every assistant turn, the parser
             the loop used), graded by rl/dispersion.py's own path: StepReward(tolerance=instance_tol(cfg, 0.025),
             n_steps=<the family's n_steps as dispersion.main reads it>, partial_ceiling=1.0).score(cfg, submission).passed
             (dispersion_v2.py is called without --tol, so 0.025 is the fallback tolerance it used).
  EXTERNALS  every problem of <root>/<efx>/ext/<bench> (V3EFX, EF_3) with ef.accepted None and offered >= 1 gets the last
             answer offered in its stored transcript, graded by harness/runner.py's own path:
             runner.load_grader(problem_dir).evaluate(cfg, submission)["passed"].
  CHECKS     the answers found in each stored episode number exactly log["offered"]; for every episode EF accepted
             ("gate" / "na" / "gate_error") the last offered answer IS the stored submission; and regrading every stored
             submission through the same path reproduces its stored pass flag (printed; a mismatch is a warning).
  REPORT     per benchmark and TEST-all (twin-weighted, as read_ef.py) / HO6: plain vs EF vs EFS-regraded, EFS - plain with
             read_efs.py's tests (one-sided sign-flip on TEST, exact McNemar on the externals), EFS - EF beside, the
             soft-fallback count and pass rate, and the SOFT-GATE-ADOPT rule applied to these numbers -- a PREDICTION.

    venv/bin/python experiments/harness_v3/regrade_efs.py [--root results/pods/harness_v3] [--base V3] [--ef V3EF]
        [--efx V3EFX] [--bench-root benchmarks] [--gen-root <data_root>/generated] [--perms 20000] [--seed 0] [--out DIR]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_efs as RS  # noqa: E402  (read_ef2 / read_ef / analyze helpers, the v4 rule)

R2, E, A = RS.R2, RS.E, RS.A
fmt = E.fmt
BENCHES = RS.BENCHES
LABEL = "SAME-DRAW PREDICTION (beside, not the registered reading)"
ACCEPTED = ("gate", "na", "gate_error")
TOL_FALLBACK = 0.025        # rl/dispersion.py --tol default; screen_any.sh / dispersion_v2.py never pass --tol


def _harness():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "harness"))
    import agents
    import runner
    from rl.dispersion import instance_tol
    from rl.reward import StepReward
    return agents, runner, instance_tol, StepReward


def offers(msgs: list, extract) -> list:
    """Every answer the model offered in an episode, in order (assistant turns the loop's parser reads as an answer)."""
    out = []
    for m in msgs or []:
        if m.get("role") == "assistant":
            a = extract(m.get("content") or "")
            if a is not None:
                out.append(a)
    return out


def family_n_steps(gen: Path, fam: str, split: str = "test") -> int:
    """rl/dispersion.main's n_steps: max fails_at_step over the FIRST sorted instance's provenance ablations, default 5."""
    d = gen / fam / split
    dirs = sorted(p for p in d.iterdir() if (p / "eval_config.json").exists())
    probe = json.loads((dirs[0] / "eval_config.json").read_text())
    steps = [v.get("fails_at_step") for v in ((probe.get("_provenance") or {}).get("ablations") or {}).values()]
    return max([s for s in steps if s], default=5)


class Grader:
    """The eval's own grading paths, loaded once."""

    def __init__(self, gen: Path, bench_root: Path):
        self.agents, self.runner, self.instance_tol, self.StepReward = _harness()
        self.gen, self.bench_root = gen, bench_root
        self._ns, self._gr, self._idx = {}, {}, {}

    def extract(self, text: str):
        return self.agents.extract_answer(text)

    def synthetic(self, fam: str, inst: str, submission) -> bool:
        if fam not in self._ns:
            self._ns[fam] = family_n_steps(self.gen, fam)
        cfg = json.loads((self.gen / fam / "test" / inst / "eval_config.json").read_text())
        res = self.StepReward(tolerance=self.instance_tol(cfg, TOL_FALLBACK), n_steps=self._ns[fam],
                              partial_ceiling=1.0).score(cfg, submission)
        return bool(res.passed)

    def external(self, bench: str, pid: str, submission) -> bool:
        if bench not in self._idx:
            self._idx[bench] = {json.loads((p / "eval_config.json").read_text())["id"]: p
                                for p in self.runner.problems(self.bench_root / bench)}
        pdir = self._idx[bench][pid]
        if bench not in self._gr:
            self._gr[bench] = self.runner.load_grader(pdir)
        cfg = json.loads((pdir / "eval_config.json").read_text())
        if submission is None:
            return False
        return bool(self._gr[bench].evaluate(cfg, submission)["passed"])


def _new_checks() -> dict:
    return {"episodes": 0, "offer_count_mismatch": 0, "accepted_last_offer_mismatch": 0, "regrade_checked": 0,
            "regrade_mismatch": 0, "errored_skipped": 0, "examples": []}


def regrade_test(root: Path, ef_name: str, sets: dict, G: Grader) -> tuple[dict, dict, dict]:
    """(the EF res with the soft submissions regraded, soft stats per family, checks) over TEST."""
    ef = A.load_config(root, ef_name, sets)
    efs = copy.deepcopy(ef)
    soft, chk = {}, _new_checks()
    for f in A.cell_members(sets, "test"):
        p = root / ef_name / "test" / f"{f}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        fams, rolls = {}, []
        st = {"n": 0, "passed": 0}
        for inst, rs in (d.get("groups") or {}).items():
            fams[inst] = []
            for r in rs:
                r = dict(r)
                chk["episodes"] += 1
                log = ((r.get("v2") or {}).get("ef")) or {}
                acc, off = log.get("accepted"), int(log.get("offered") or 0)
                if str(r.get("attribution") or "").startswith("error:"):
                    chk["errored_skipped"] += 1
                    fams[inst].append(bool(r.get("passed")))
                    rolls.append(r)
                    continue
                offs = offers(r.get("messages"), G.extract)
                if len(offs) != off:
                    chk["offer_count_mismatch"] += 1
                    if len(chk["examples"]) < 5:
                        chk["examples"].append(f"test/{f}/{inst}: {len(offs)} answers in the messages, log offered {off}")
                if r.get("answered") and offs:
                    chk["regrade_checked"] += 1
                    if acc in ACCEPTED:
                        if (offs[-1] or {}).get("answer") != r.get("answer"):
                            chk["accepted_last_offer_mismatch"] += 1
                        elif G.synthetic(f, inst, offs[-1]) != bool(r.get("passed")):
                            chk["regrade_mismatch"] += 1
                    elif acc == "fallback":
                        sub = next((o for o in reversed(offs) if (o or {}).get("answer") == r.get("answer")), None)
                        if sub is None or G.synthetic(f, inst, sub) != bool(r.get("passed")):
                            chk["regrade_mismatch"] += 1
                if acc is None and off >= 1 and offs:
                    ok = G.synthetic(f, inst, offs[-1])
                    r.update(passed=ok, answered=True, answer=(offs[-1] or {}).get("answer"),
                             efs_regraded=True)
                    st["n"] += 1
                    st["passed"] += int(ok)
                fams[inst].append(bool(r.get("passed")))
                rolls.append(r)
        efs["families"][("test", f)] = fams
        efs["rollouts"][("test", f)] = rolls
        soft[f] = st
    return efs, soft, chk


def regrade_ext(root: Path, efx_name: str, G: Grader) -> tuple[dict, dict, dict, dict]:
    """({bench: {pid: passed}} EF, the same with the soft submissions regraded, soft stats, checks) over the externals."""
    ef_p, efs_p, soft, chk = {}, {}, {}, {}
    for b in BENCHES:
        recs = R2.load_ext(root / efx_name / "ext" / b)
        ef_p[b] = R2.ext_passed(recs)
        efs_p[b] = dict(ef_p[b])
        soft[b] = {"n": 0, "passed": 0}
        c = chk[b] = _new_checks()
        for pid, rec in recs.items():
            c["episodes"] += 1
            if rec.get("error"):
                c["errored_skipped"] += 1
                continue
            log = rec.get("ef") or {}
            acc, off = log.get("accepted"), int(log.get("offered") or 0)
            offs = offers(rec.get("transcript"), G.extract)
            if len(offs) != off:
                c["offer_count_mismatch"] += 1
                if len(c["examples"]) < 5:
                    c["examples"].append(f"{b}/{pid}: {len(offs)} answers in the transcript, log offered {off}")
            if rec.get("submission") is not None:
                c["regrade_checked"] += 1
                if acc in ACCEPTED and offs and offs[-1] != rec["submission"]:
                    c["accepted_last_offer_mismatch"] += 1
                if G.external(b, pid, rec["submission"]) != bool((rec.get("grade") or {}).get("passed")):
                    c["regrade_mismatch"] += 1
            if acc is None and off >= 1 and offs:
                ok = G.external(b, pid, offs[-1])
                efs_p[b][pid] = ok
                soft[b]["n"] += 1
                soft[b]["passed"] += int(ok)
    return ef_p, efs_p, soft, chk


def rate(st: dict) -> str:
    return f"{st['n']} ({st['passed']} pass, {100.0 * st['passed'] / st['n']:.1f}%)" if st["n"] else "0"


def cell_rate(p: dict) -> str:
    return f"{100.0 * sum(p.values()) / len(p):.1f} ({len(p)})" if p else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--base", default="V3")
    ap.add_argument("--ef", default="V3EF")
    ap.add_argument("--efx", default="V3EFX")
    ap.add_argument("--bench-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--gen-root", default=None, help="the generated instances (default: paths.data_root()/generated)")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="also write EFS_SAMEDRAW.md / .json here (default: print only)")
    a = ap.parse_args(argv)
    root, broot = Path(a.root), Path(a.bench_root)
    if a.gen_root:
        gen = Path(a.gen_root)
    else:
        sys.path.insert(0, str(ROOT))
        from paths import data_root
        gen = data_root() / "generated"
    G = Grader(gen, broot)
    sets = A.load_families()
    test_all = A.cell_members(sets, "test")
    base = A.load_config(root, a.base, sets) if (root / a.base).is_dir() else None
    ef = A.load_config(root, a.ef, sets) if (root / a.ef).is_dir() else None
    efs, soft_t, chk_t = regrade_test(root, a.ef, sets, G) if ef is not None else (None, {}, _new_checks())
    ef_p, efs_p, soft_x, chk_x = regrade_ext(root, a.efx, G)
    plain_p = {b: R2.ext_passed(R2.load_ext(root / a.base / "ext" / b)) for b in BENCHES}
    nprob = R2.problem_counts(broot)

    rng = np.random.default_rng(a.seed)
    d_all = E.pair(efs, base, sets, a.perms, rng, test_all)
    vs_plain = {b: R2.mcnemar(efs_p[b], plain_p[b]) if efs_p[b] and plain_p[b] else None for b in BENCHES}
    vals = {"TEST": d_all, **{leg: vs_plain[b] for leg, b in RS.LEG_BENCH.items()}}
    miss = {"plain": E.complete(base, sets, test_all), "ef": E.complete(ef, sets, test_all)}
    complete = {"TEST": base is not None and ef is not None and not miss["plain"] and not miss["ef"],
                **{leg: len(efs_p[b]) == nprob[b] and len(plain_p[b]) == nprob[b] for leg, b in RS.LEG_BENCH.items()}}
    V = RS.verdict(vals, complete)
    rng_b = np.random.default_rng(a.seed + 1)
    cells = {"plain": base, "EF": ef, "EFS-regraded": efs}
    tab = {}
    for lab, members in (("TEST-all (twin-weighted)", test_all), ("HO6", E.HO6)):
        tab[lab] = {k: (A.cell_pass(v, sets, "test", members)[0] if v is not None else None) for k, v in cells.items()}
        tab[lab]["efs_minus_plain"] = d_all if members is test_all else E.pair(efs, base, sets, a.perms, rng_b, members)
        tab[lab]["efs_minus_ef"] = E.pair(efs, ef, sets, a.perms, rng_b, members)
        tab[lab]["ef_minus_plain"] = E.pair(ef, base, sets, a.perms, rng_b, members)
    soft_all = {"n": sum(s["n"] for s in soft_t.values()), "passed": sum(s["passed"] for s in soft_t.values())}

    L = [f"# {LABEL}",
         "",
         "HARNESS v4 (EFS) regraded on the SAME draws as the hard gate (docs/RL_PLAN.md 2026-09-29 05:35 UTC): every stored "
         f"`{a.ef}` / `{a.efx}/ext` episode that ended with NO answer after >= 1 offered answer gets EFS's submission (the "
         "last offered answer), graded by the eval's own path; everything else unchanged. NOT the registered reading "
         "(EFS_1 / EFS_2 on fresh draws decide).",
         "",
         "| set | plain | EF (hard gate) | EFS-regraded | EFS - plain (read_efs.py's test) | EFS - EF | soft fallbacks (pass) |",
         "|---|---|---|---|---|---|---|"]
    for lab, t in tab.items():
        sf = soft_all if lab.startswith("TEST-all") else {
            "n": sum(soft_t.get(f, {"n": 0})["n"] for f in E.HO6), "passed": sum(soft_t.get(f, {"passed": 0})["passed"]
                                                                                 for f in E.HO6)}
        L.append(f"| {lab} | {fmt(t['plain'])} | {fmt(t['EF'])} | {fmt(t['EFS-regraded'])} | "
                 f"{RS.ptxt(t['efs_minus_plain'])} | {RS.ptxt(t['efs_minus_ef'])} | {rate(sf)} |")
    for b in BENCHES:
        x, y = vs_plain[b], (R2.mcnemar(efs_p[b], ef_p[b]) if efs_p[b] and ef_p[b] else None)
        L.append(f"| {b} | {cell_rate(plain_p[b])} | {cell_rate(ef_p[b])} | {cell_rate(efs_p[b])} | {RS.mtxt(x)} | "
                 f"{RS.mtxt(y)} | {rate(soft_x[b])} |")
    L.append("")
    L.append(f"**The v4 rule applied to these numbers (a PREDICTION): {V['label']}**"
             + ("" if all(complete.values()) else " (some cells incomplete)"))
    for leg in RS.LEGS:
        lg = V["legs"][leg]
        L.append(f"* {leg}: {'holds' if lg['ok'] else ('FAILS' if lg['ok'] is False else '-')} -- {lg['why']}")
    L.append("")
    L.append("| TEST family | plain | EF | EFS-regraded | soft fallbacks (pass) |")
    L.append("|---|---|---|---|---|")
    per = {}
    for f in test_all:
        per[f] = {k: E.fam(v, sets, f) for k, v in cells.items()}
        per[f]["soft"] = soft_t.get(f, {"n": 0, "passed": 0})
        L.append(f"| {f}{' (HO6)' if f in E.HO6 else ''} | {fmt(per[f]['plain'])} | {fmt(per[f]['EF'])} | "
                 f"{fmt(per[f]['EFS-regraded'])} | {rate(per[f]['soft'])} |")
    L.append("")
    L.append("CHECKS (the regrade reproduces the eval):")
    warn = []
    for lab, c in [("TEST", chk_t)] + [(b, chk_x[b]) for b in BENCHES]:
        L.append(f"* {lab}: {c['episodes']} episodes; answers found == log 'offered' in all but {c['offer_count_mismatch']}; "
                 f"accepted episodes whose last offer is not the stored submission {c['accepted_last_offer_mismatch']}; "
                 f"stored submissions regraded {c['regrade_checked']}, pass flag NOT reproduced {c['regrade_mismatch']}; "
                 f"errored (left unchanged) {c['errored_skipped']}")
        if c["offer_count_mismatch"] or c["accepted_last_offer_mismatch"] or c["regrade_mismatch"]:
            warn.append(f"{lab}: the regrade does not reproduce every stored episode ({c['examples'][:2]})")
    if warn:
        L.append("\n**WARNINGS**: " + "; ".join(warn))
    text = "\n".join(L) + "\n"
    print(text)
    obj = {"label": LABEL, "rule_prediction": V["label"], "legs": V["legs"], "complete": complete, "table": tab,
           "externals": {b: {"plain": plain_p[b] and 100.0 * sum(plain_p[b].values()) / len(plain_p[b]),
                             "ef": ef_p[b] and 100.0 * sum(ef_p[b].values()) / len(ef_p[b]),
                             "efs_regraded": efs_p[b] and 100.0 * sum(efs_p[b].values()) / len(efs_p[b]),
                             "efs_minus_plain": vs_plain[b], "soft": soft_x[b]} for b in BENCHES},
           "soft_test": soft_all, "soft_test_per_family": soft_t, "per_family": per,
           "checks": {"test": chk_t, **chk_x}, "warnings": warn}
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "EFS_SAMEDRAW.md").write_text(text)
        (out / "EFS_SAMEDRAW.json").write_text(json.dumps(obj, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
