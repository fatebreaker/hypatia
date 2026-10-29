"""H-V51RL -- RL INSIDE HARNESS v5.1 (docs/RL_PLAN.md 2026-09-29 21:10 UTC, "HARNESS v5.1 REGISTERED", PHASE 2': "RL inside
v5.1 on causal_rdd, mu 0.3, dynamic sampling (draw 12), LR 3e-6, seed 1, 100 steps (HV3_142). Comparator: H-LRGR (HV3_140)
... Bar: v5.1-RL >= HV3_140 + 3 on sealed TEST-all AND on HO6 at step 100 (paired one-sided p <= 0.05); ALIVE reported
beside. Also reported: v5.1-RL's labelled sibling pass rate over training ... and its inspect rate").

v5.1 = v5 (rl/grpo_v5.py + experiments/harness_v3/program_first.py, BOTH UNCHANGED IN BEHAVIOUR: rl/grpo_v5.py is not edited,
program_first.py's v5.1 lines are flag-gated) with two changes:
  (a) LOOK FIRST -- program_first.configure(look_first=True), called here before the loop is installed: a solve() submission
      is refused (one fixed harness message; the refusal uses a turn) until an EXECUTED notebook cell of the episode has
      READ a data file of the target (program_first.LookWatch: inotify IN_ACCESS on an original data file while the cell
      ran; static fallback). The same switch the evaluation uses (screen_v3v5.sh V51=1).
  (b) LABELLED PORTABILITY -- TRAINING ONLY, THIS FILE (the harness never reads a label):
        r = r_answer + MU x labelled,   MU = 0.3 (registered)
      r_answer  as in v5: grpo_onpolicy's reward (StepReward at --c) of the episode's final answer = solve(target) of the
                last valid submission (else the soft fallback), passed through grpo_v3's scoring at lambda 0 -- never
                recomputed; --require-code's zero kept (a rollout with no executed code earns 0 in total).
      labelled  the fraction of the K = 3 sibling datasets on which the FINAL valid submission's solve(sibling_dir,
                sibling_task) returned that sibling's CORRECT answer under the family's own grader: rl.dispersion.StepReward
                with tolerance instance_tol(cfg, instance_tol(cfg, 0.025)), n_steps reward_steps(cfg), partial_ceiling --c,
                on the SIBLING's eval_config -- the grader r_answer is computed with (rl/dispersion.one_rollout via
                rl/grpo_onpolicy.rollouts; rl/grpo_prog.PortableStage.grade); correct = its `passed`. The graded values are
                the ones the episode's own sibling runs returned (program_first logs each sibling's value; nothing is run
                again). 0 when the final answer is not a solve() output (soft fallback / none); the denominator is the
                siblings run (3 whenever the family has >= 3 layout-matched train instances: causal_rdd has 400).
      The v5 measure (portability: ran cleanly with a non-constant output) is still computed and LOGGED for comparison;
      it no longer enters the reward. A passed rollout earns >= 1.0 and an unpassed one <= c + MU = 0.8 (install refuses
      c + MU >= 1 - grpo_v3.MARGIN), so a correct answer always out-scores an incorrect one.
SEALED GUARD (hard): sibling labels are read ONLY for DESIGN-family TRAIN instances -- guard_labels(inst), called before any
sibling eval_config is opened: the instance path must resolve (symlinks followed; the path as given is checked too) to
generated/<family>/train/<instance> with <family> in experiments/harness_v2/families.json 'design' and in neither its 'test'
list nor SEALED_17 (the 17 sealed TEST families, verbatim as experiments/selector/common.SEALED; families.json 'test' must
equal it or every read is refused). Anything else -- a sealed family, an external, a real-data or unknown family, a test
split -- raises SealedLabelError (a RuntimeError, never caught here: it stops the trainer with the reason). install()
applies the same guard to --family before anything runs. Evaluation never imports this file, so it never grades siblings.

DYNAMIC SAMPLING: rl/grpo_ds.py, unchanged (--dyn-sample N required); the reward is composed inside grpo_v3's scoring, i.e.
BEFORE selection, so a group is alive (and kept) iff its v5.1 reward varies.

LOGS (in --out). process.jsonl: r_answer, reward = r_answer + 0.3 x labelled, bonus = reward - r_answer. v5.jsonl (grpo_v5's
file and fields, one line per DRAWN rollout) gains: labelled, sib_pass (per sibling), sib_graded, portability_old (= the v5
field portability), inspect (the harness/process_signals inspect signal: 1 / 0 / None), looked, look_turn, look_how,
look_refusals, look_detector, look_overflow. process_history.jsonl keeps grpo_v3's / grpo_ds's / grpo_v5's fields
(rate_inspect over the kept batch; v5_portability_mean = the OLD measure) and gains, over the KEPT batch and with _drawn
over all N x K drawn rollouts:
    v51_labelled_mean     mean labelled sibling pass fraction (0 for rollouts not answered by solve)   <- the (b) signal
    v51_sib_pass / v51_sib_graded   sibling passes / sibling runs graded (final solve submissions only)
    v51_inspect_rate      the inspect signal's rate (applicable rollouts; n in v51_n_inspect)
    v51_look_refusals     look-first refusals (submissions refused before a look)
    v51_looked            rollouts whose episode looked (any detector); v51_look_static / v51_look_vacuous: of those, by
                          the static fallback / vacuously; v51_never_looked_refused: rollouts refused >= 1 and never looked
harness_v51.json: the reward's record (version, reward, MU, K, revision cap, look_first, grader, label guard, program_first's
v5.1 record); a resume with another record, of a dir with steps trained without it, or of a v5 dir, is refused.

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_v51.py --dyn-sample 12 \\
        --process-lambda 0 --process-sham off <every rl/grpo_onpolicy.py argument>
Launch through experiments/harness_v3/run_v3v5.sh with V51=1 (run_v3.sh's configuration; run dir -hv3v51lp0.3-ds<N>).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_v5 as G5  # noqa: E402  (unchanged: load_pf, compose, v5_log, portability_of, row_of, V5Reward)

DS, V3 = G5.DS, G5.V3
FAMJ = ROOT / "experiments" / "harness_v2" / "families.json"
MU = 0.3
V51RL_VERSION = 1
# the 17 sealed TEST families, verbatim as experiments/selector/common.SEALED (tests/test_v51.py checks both are equal)
SEALED_17 = frozenset("""causal_confounded causal_att surv_cox surv_tvc hc_dx hc_quality enzyme_mm osc_damped pk_oral eco_mb
eco_size carrier_cnv carrier_var hic_loop hic_var crispri_casrx gwas_loci""".split())
V51_REWARD = ("r = r_answer(solve(target)) + 0.3 x labelled portability; labelled = fraction of the K=3 layout-matched train "
              "siblings on which the FINAL valid solve(sibling_dir, sibling_task) returned that sibling's CORRECT answer under "
              "the family's own grader (rl.dispersion.StepReward at instance_tol, passed); 0 without a valid final "
              "submission; design-family labels only (sealed guard); harness v5.1 (look first); no process bonus (lambda 0); "
              "--require-code's zero kept")
GRADER = ("rl.dispersion.StepReward(tolerance=instance_tol(cfg, instance_tol(cfg, 0.025)), n_steps=reward_steps(cfg), "
          "partial_ceiling=--c).score(sibling cfg, {'answer': value}).passed -- r_answer's grader (rl/dispersion.one_rollout)")


class SealedLabelError(RuntimeError):
    """A label read outside DESIGN-family TRAIN instances was attempted -- refused (never caught in this file)."""


# ----------------------------------------------------------------------------------------------------------- the guard
@lru_cache(maxsize=1)
def label_families() -> tuple:
    """(design, sealed) from experiments/harness_v2/families.json; sealed = its 'test' list, which must equal SEALED_17."""
    try:
        fj = json.loads(FAMJ.read_text())
        design = frozenset(k for k in fj["design"] if k != "note")
        test = frozenset(k for k in fj["test"] if k != "note")
    except Exception as exc:                                              # noqa: BLE001
        raise SealedLabelError(f"REFUSED: cannot read the family registry {FAMJ}: {exc}") from None
    if test != SEALED_17:
        raise SealedLabelError(f"REFUSED: {FAMJ} 'test' {sorted(test)} != the 17 sealed families {sorted(SEALED_17)} -- "
                               "no label is read until the registry is fixed")
    if design & SEALED_17:
        raise SealedLabelError(f"REFUSED: {FAMJ} lists sealed families as design: {sorted(design & SEALED_17)}")
    return design, SEALED_17


def guard_labels(inst) -> Path:
    """The resolved instance dir, iff its labels may be read: generated/<design family>/train/<instance>. Else raise."""
    p = Path(inst)
    design, sealed = label_families()
    try:
        real = p.resolve(strict=True)
    except OSError as exc:
        raise SealedLabelError(f"REFUSED: cannot resolve {p} ({exc}) -- no label read") from None
    named = {p.parent.parent.name, real.parent.parent.name}
    if named & sealed:
        raise SealedLabelError(f"REFUSED: {p} (-> {real}) is an instance of the SEALED TEST family "
                               f"{sorted(named & sealed)} -- its labels are never read (docs/RL_PLAN.md 2026-09-29 21:10 UTC)")
    fam, split, gen = real.parent.parent.name, real.parent.name, real.parent.parent.parent.name
    if gen != "generated" or split != "train" or fam not in design or p.parent.parent.name != fam:
        raise SealedLabelError(f"REFUSED: labels are read only for DESIGN-family TRAIN instances (generated/<design family>/"
                               f"train/<instance>); {p} -> {real} is family {fam!r}, split {split!r} under {gen!r}")
    return real


def _reward_steps(cfg: dict, default: int = 5) -> int:
    """rl/grpo_onpolicy.reward_steps, verbatim (that module imports torch; `passed` does not depend on it anyway)."""
    return max([v.get("fails_at_step") for v in (cfg.get("_provenance", {}).get("ablations") or {}).values()
                if v.get("fails_at_step")], default=default)


def grade_sibling(sib, value: dict, c: float) -> bool:
    """Is `value` the sibling's CORRECT answer under its own grader (r_answer's)? The guard runs first."""
    real = guard_labels(sib)
    cfg = json.loads((real / "eval_config.json").read_text())
    prov = (cfg.get("_provenance") or {}).get("family")
    if prov and prov in SEALED_17:
        raise SealedLabelError(f"REFUSED: {real}'s provenance names the sealed family {prov!r}")
    from rl.dispersion import StepReward, instance_tol
    res = StepReward(tolerance=instance_tol(cfg, instance_tol(cfg, 0.025)), n_steps=_reward_steps(cfg),
                     partial_ceiling=c).score(cfg, {"answer": value})
    return bool(res.passed)


def labelled_of(o: dict, inst: Path, c: float) -> dict:
    """The rollout's labelled portability from its v5 log: every sibling run of the FINAL valid submission graded."""
    out = {"frac": 0.0, "pass": [], "graded": 0}
    log = G5.v5_log(o)
    if log.get("final") != "solve":
        return out
    n, subs = int(log.get("final_submission") or 0), log.get("subs") or []
    if not 1 <= n <= len(subs):
        return out
    rows = subs[n - 1].get("siblings") or []
    if not rows:
        return out
    train = Path(inst).resolve().parent.parent / "train"     # program_first.siblings_for: the (resolved) target's train split
    res = []
    for row in rows:
        name = str(row.get("name") or "")
        if not name or "/" in name or name in (".", ".."):
            raise SealedLabelError(f"REFUSED: a sibling name {name!r} that is not an instance name")
        sib = train / name
        guard_labels(sib)                                     # every sibling, graded or not: refuse early and loudly
        val = row.get("value") if row.get("ran") else None
        res.append(bool(isinstance(val, dict) and grade_sibling(sib, val, c)))
    out.update(frac=sum(res) / len(res), graded=len(res))
    out["pass"] = res
    return out


def look_of(o: dict) -> dict:
    log = G5.v5_log(o)
    return {k: log.get(k) for k in ("looked", "look_turn", "look_how", "look_refusals", "look_detector", "look_overflow")}


# ----------------------------------------------------------------------------------------------------------- the reward
class V51Reward(G5.V5Reward):
    """grpo_v5's V5Reward (logs, v5.jsonl, the v5 counts) with the reward term replaced: labelled portability; the v5
    portability is logged beside it (the instance is re-classed by install)."""
    c = 0.5                                   # --c (install sets it): StepReward's partial ceiling, as for r_answer

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = V3.ProcessReward.score_group(self, inst, outs, step)      # grpo_v3's scoring at lambda 0 (r_answer)
        if self._v5_rows is None:
            self._v5_rows = {}
        rows = []
        for o, r in zip(outs, recs):
            lab = labelled_of(o, inst, self.c)
            port = G5.portability_of(o)                                   # the OLD v5 measure: logged only
            pre = float(r["reward"])
            total = 0.0 if (self.require_code and not r["cells"]) else G5.compose(pre, lab["frac"], MU)
            r.update(reward=total, bonus=round(total - r["r_answer"], 6), labelled=lab["frac"], portability=port)
            o["reward"] = total
            o["v51_reward"] = {"pre": pre, "labelled": lab["frac"], "sib_pass": lab["pass"], "portability_old": port,
                               "mu": MU}
            row = G5.row_of(o, r, port)
            row.update(labelled=lab["frac"], sib_pass=lab["pass"], sib_graded=lab["graded"], portability_old=port,
                       inspect=(r.get("fired") or {}).get("inspect"), **look_of(o))
            rows.append(row)
        self._v5_rows[inst.name] = rows
        return recs

    def _v5_counts(self, insts, suffix: str) -> dict:
        rec = super()._v5_counts(insts, suffix)
        rows = [x for n in insts for x in (self._v5_rows or {}).get(n, [])]
        ins = [x["inspect"] for x in rows if x.get("inspect") is not None]
        looked = [x for x in rows if x.get("looked")]
        rec.update({f"v51_labelled_mean{suffix}": round(statistics.fmean([x["labelled"] for x in rows]), 4) if rows else 0.0,
                    f"v51_sib_pass{suffix}": sum(sum(x["sib_pass"]) for x in rows),
                    f"v51_sib_graded{suffix}": sum(x["sib_graded"] for x in rows),
                    f"v51_inspect_rate{suffix}": round(sum(ins) / len(ins), 4) if ins else None,
                    f"v51_n_inspect{suffix}": len(ins),
                    f"v51_look_refusals{suffix}": sum(int(x.get("look_refusals") or 0) for x in rows),
                    f"v51_looked{suffix}": len(looked),
                    f"v51_look_static{suffix}": sum(1 for x in looked if x.get("look_how") == "static"),
                    f"v51_look_vacuous{suffix}": sum(1 for x in looked if x.get("look_how") == "vacuous"),
                    f"v51_never_looked_refused{suffix}": sum(1 for x in rows if not x.get("looked")
                                                             and int(x.get("look_refusals") or 0) > 0)})
        return rec


# ----------------------------------------------------------------------------------------------------------- install
def config_record(PF) -> dict:
    design, sealed = label_families()
    return {"version": V51RL_VERSION, "reward": V51_REWARD, "mu": MU, "k_siblings": PF.K_SIBLINGS,
            "max_revisions": PF.MAX_REVISIONS, "look_first": bool(PF.LOOK_FIRST), "grader": GRADER,
            "label_guard": {"design": sorted(design), "sealed": sorted(sealed),
                            "families_json_sha16": hashlib.sha256(FAMJ.read_bytes()).hexdigest()[:16]},
            "program_first": PF.record(),
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "grpo_v5_sha256": hashlib.sha256(Path(G5.__file__).read_bytes()).hexdigest()[:16]}


def _same_reward(old: dict, rec: dict) -> bool:
    return all(old.get(k) == rec[k] for k in ("version", "reward", "mu", "k_siblings", "max_revisions", "look_first"))


def install(grpo_mod, dispersion_mod, v3args, dsargs, rest: list, agents_mod=None, v2_mod=None) -> DS.DynSampler:
    """program_first's loop in LOOK-FIRST mode into the trainer's agents / v2, then grpo_ds.install (grpo_v3's harness +
    lambda-0 reward + dynamic sampling), then the labelled-portability term. agents_mod / v2_mod: tests only."""
    if float(v3args.process_lambda) != 0.0 or v3args.process_sham != "off":
        raise SystemExit(f"rl/grpo_v51.py: the v5.1 reward has NO process bonus -- --process-lambda must be 0 and "
                         f"--process-sham off (got {v3args.process_lambda} / {v3args.process_sham})")
    if not int(dsargs.dyn_sample or 0):
        raise SystemExit("rl/grpo_v51.py: Phase 2' is registered with dynamic sampling -- --dyn-sample N is required")
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    pa.add_argument("--family")
    pa.add_argument("--c", type=float, default=0.5)
    g, _ = pa.parse_known_args(rest)
    if g.c + MU >= 1.0 - V3.MARGIN:
        raise SystemExit(f"rl/grpo_v51.py: --c {g.c} + MU {MU} >= 1 - {V3.MARGIN}: an unpassed rollout could out-score a "
                         "passed one under the uncapped registered reward -- refused")
    design, sealed = label_families()
    if not g.family or g.family in sealed or g.family not in design:
        raise SystemExit(f"rl/grpo_v51.py: REFUSED -- --family {g.family!r} is not a DESIGN family: labelled portability "
                         "reads sibling labels, which is allowed for design families only (sealed guard)")
    PF = G5.load_pf()
    PF.configure(look_first=True)                                     # v5.1 (a), before any episode
    out = Path(g.out) if g.out else None
    rec = config_record(PF)
    cfgf = out / "harness_v51.json" if out is not None else None
    if cfgf is not None:
        if (out / "harness_v5.json").exists():
            raise SystemExit(f"rl/grpo_v51.py: {out} is a v5 run (harness_v5.json) -- refusing to resume it as v5.1")
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if not _same_reward(old, rec):
                raise SystemExit(f"rl/grpo_v51.py: {out} was trained with {old.get('reward')!r} (mu {old.get('mu')}, "
                                 f"look_first {old.get('look_first')}) -- refusing to resume it under {rec['reward']!r}")
            if (old.get("program_first") or {}).get("sha256") != rec["program_first"]["sha256"]:
                print(f"rl/grpo_v51.py: WARNING program_first.py / v5_site changed since this run started "
                      f"({(old.get('program_first') or {}).get('sha256')} -> {rec['program_first']['sha256']})", flush=True)
        elif (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_v51.py: {out} has steps trained WITHOUT the v5.1 harness / reward (no "
                             "harness_v51.json) -- refusing to resume it")
    if agents_mod is None:
        import agents as agents_mod                                   # the module rl/dispersion's make_agent lives in
    if v2_mod is None:
        import v2 as v2_mod                                           # the module rl/grpo_v3.install uses
    PF.install_agents(agents_mod)
    PF.install_v2(v2_mod)                                             # BEFORE grpo_ds.install: install_grpo -> v5 wrap_rollout
    ds = DS.install(grpo_mod, dispersion_mod, v3args, dsargs, rest)
    if type(ds.pr) is not DS.DSReward:
        raise SystemExit(f"rl/grpo_v51.py: grpo_ds.install left a {type(ds.pr).__name__}, not grpo_ds's DSReward")
    ds.pr.__class__ = V51Reward
    ds.pr._v5_rows = {}
    ds.pr.c = float(g.c)
    if cfgf is not None and not cfgf.exists():
        out.mkdir(parents=True, exist_ok=True)
        cfgf.write_text(json.dumps(rec, indent=1))
    print(f"H-V51RL (RL_PLAN 2026-09-29 21:10, Phase 2'): training rollouts in harness v5.1 (look first: a solve() runs only "
          f"after an executed cell has read the data; submit solve(data_dir, task); run on the target + {PF.K_SIBLINGS} "
          f"layout-matched train siblings; <= {PF.MAX_REVISIONS} revisions); reward = r_answer + {MU} x LABELLED portability "
          f"(the siblings graded by the {g.family} grader; design-family labels only); no process bonus; dynamic sampling on "
          "the v5.1 reward", flush=True)
    return ds


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    dsargs, rest = DS.parse_ds_args(rest)
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, dsargs, rest)
    raise SystemExit(grpo_onpolicy.main())
