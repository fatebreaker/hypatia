"""H-V5RL -- RL INSIDE HARNESS v5 (docs/RL_PLAN.md 2026-09-29 14:15 UTC, "HARNESS v5 ... REGISTERED before any build or run",
PHASE 2: "RL INSIDE v5 on causal_rdd, reward = r_answer(solve(target)) + 0.3 x portability (fraction of the K siblings where
solve ran without exception with a non-constant output), dynamic sampling (draw 12), 100 steps, 8B LoRA").

TRAINING ROLLOUTS RUN IN v5. The trainer's rollouts are rl/dispersion.one_rollout -> make_agent(llm:<policy>:rl) ->
LLMAgent.solve, in THIS process (grpo_onpolicy.rollouts' thread pool). install() puts the v5 loop in place exactly as the
evaluation hook (experiments/harness_v3/v5_site) does, without a hook: experiments/harness_v3/program_first.py is loaded by path
(module _gbp_program_first, the hook's own name) and its install_agents(agents) / install_v2(v2) run on the SAME `agents` and
`v2` modules rl/dispersion and rl/grpo_v3 use -- BEFORE grpo_ds.install, so v2.install_grpo builds its rollout wrapper with
v5's wrap_rollout (the stored episode carries the v5 note the policy read) and every episode's log reaches rollout["v2"]["v5"]
through V2Sandbox.stats. So a training episode is a v5 episode: the policy submits solve(data_dir, task), the harness runs it (each dataset with its OWN task text) on the
target and on K=3 layout-matched siblings of the causal_rdd TRAIN split (target excluded), reports, allows <= 2 revisions, and
the answer is solve(target) of the last valid submission (else the last offered JSON answer).

REWARD, per rollout (no process bonus: --process-lambda 0, --process-sham off):
    r = r_answer + MU * portability,   MU = 0.3 (registered)
  r_answer     what rl/grpo_v3.py passes through at lambda 0: grpo_onpolicy's reward (StepReward at --c) of the episode's
               final answer = solve(target) of the last valid submission (else the soft fallback) -- never recomputed; 0 under
               --require-code when the episode executed no code (a submission counts as executed code: program_first
               decision 6).
  portability  program_first's per-rollout log: the fraction of the K siblings on which the FINAL valid submission ran without
               exception, returned every answer key as a finite number, and an output different from the target's and from
               every other clean sibling's; 0 when the final answer is not a solve() output (soft fallback / none).
  A passed rollout earns >= 1.0; an unpassed one < c + MU (install refuses --c + MU >= 1 - grpo_v3.MARGIN, so with the
  registered c = 0.5 a correct answer always out-scores an incorrect one: the registered formula needs no cap).
DYNAMIC SAMPLING: rl/grpo_ds.py, unchanged (--dyn-sample N required): the reward is composed inside grpo_v3's scoring, i.e.
BEFORE selection, so a group is alive (and kept) iff its v5 reward varies.

HOW, WITH NO PINNED FILE EDITED (rl/grpo_onpolicy.py, rl/grpo_v3.py, rl/grpo_ds.py, harness/*.py): grpo_ds.install (which runs
grpo_v3.install at lambda 0) is called unchanged; its reward object (grpo_ds.DSReward) is re-classed to V5Reward, whose
score_group = grpo_v3's (lambda 0), then the portability term; summary / ds_log / log add the v5 counts (the rl/grpo_grds.py
pattern).

LOGS (in --out). process.jsonl: r_answer = the answer reward, reward = r_answer + 0.3 x portability, bonus = reward - r_answer.
v5.jsonl -- one line per DRAWN rollout: step, inst, ds_batch, r_answer, passed, reward, portability, submissions, valid,
revisions, refused (JSON answers refused before a valid submission), final ("solve" | "soft_fallback" | None), end, siblings,
and per submission (valid, why, flags, portability, code_sha). process_history.jsonl gains, per step, over the KEPT batch (and
with _drawn over all N x K drawn rollouts -- read these, as for arm B):
    v5_with_log        rollouts that carry a v5 log (all, unless an episode raised before its first turn)
    v5_submissions     submissions made (<= 3 per rollout)
    v5_sibling_runs    sibling executions of submissions (K per submission when K siblings exist) -- the smoke gate
    v5_valid           rollouts with >= 1 valid submission
    v5_final_solve     rollouts whose answer is solve(target);  v5_soft_fallback / v5_no_answer: the others
    v5_refused         JSON answers refused before a valid submission
    v5_portability_mean  mean portability (0 for rollouts not answered by solve)
harness_v5.json: the reward's record (version, reward, MU, K, revision cap, program_first's config record); a resume with
another record (or of a dir with steps trained without it) is refused.

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_v5.py --dyn-sample 12 \
        --process-lambda 0 --process-sham off <every rl/grpo_onpolicy.py argument>
Launch through experiments/harness_v3/run_v3v5.sh (run_v3.sh's configuration and naming; run dir -hv3v5port0.3-ds<N>).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_ds as DS  # noqa: E402

V3 = DS.V3
PF_PY = ROOT / "experiments" / "harness_v3" / "program_first.py"
PF_NAME = "_gbp_program_first"                 # the evaluation hook's module name: one v5 instance per process
MU = 0.3
V5RL_VERSION = 1
V5_REWARD = ("r = r_answer(solve(target)) + 0.3 x portability; portability = fraction of the K=3 layout-matched train "
             "siblings on which the final valid solve ran without exception, returned every answer key as a finite number "
             "and an output different from the target's and every other clean sibling's; 0 without a valid final "
             "submission; no process bonus (lambda 0); --require-code's zero kept")


def load_pf():
    """experiments/harness_v3/program_first.py, loaded by path once per process (as the hook loads it)."""
    mod = sys.modules.get(PF_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(PF_NAME, PF_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PF_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


def compose(r_answer: float, portability: float, mu: float = MU) -> float:
    """The registered reward: r_answer + mu x portability (portability clipped to [0, 1])."""
    return float(r_answer) + mu * max(0.0, min(1.0, float(portability or 0.0)))


def v5_log(o: dict) -> dict:
    x = (o.get("v2") or {}).get("v5")
    return x if isinstance(x, dict) else {}


def portability_of(o: dict) -> float:
    """The rollout's portability: its log's, only when its final answer is a solve() output."""
    log = v5_log(o)
    return float(log.get("portability") or 0.0) if log.get("final") == "solve" else 0.0


def row_of(o: dict, r: dict, port: float) -> dict:
    log = v5_log(o)
    subs = log.get("subs") or []
    return {"has_log": bool(log), "submissions": int(log.get("submissions") or 0),
            "valid": int(log.get("valid") or 0), "final": log.get("final"), "end": log.get("end"),
            "sibling_runs": sum(len(s.get("siblings") or []) for s in subs), "refused": int(log.get("refused") or 0),
            "portability": port, "passed": bool(r.get("passed")), "r_answer": r.get("r_answer"), "reward": r.get("reward"),
            "revisions": int(log.get("revisions") or 0), "siblings": log.get("siblings"),
            "subs": [{"n": s.get("n"), "phase": s.get("phase"), "valid": s.get("valid"), "why": s.get("why"),
                      "flags": s.get("flags"), "portability": s.get("portability"), "code_sha": s.get("code_sha"),
                      "sibling_status": [x.get("status") for x in s.get("siblings") or []]} for s in subs]}


class V5Reward(DS.DSReward):
    """grpo_ds's DSReward (the instance is re-classed by install): the portability term after grpo_v3's scoring (lambda 0),
    and the v5 counts."""
    _v5_drawn = None
    _v5_rows = None            # {instance name: [row]} for the step being scored

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = super().score_group(inst, outs, step)
        if self._v5_rows is None:
            self._v5_rows = {}
        rows = []
        for o, r in zip(outs, recs):
            port = portability_of(o)
            pre = float(r["reward"])
            total = 0.0 if (self.require_code and not r["cells"]) else compose(pre, port)
            r.update(reward=total, bonus=round(total - r["r_answer"], 6), portability=port)
            o["reward"] = total
            o["v5_reward"] = {"pre": pre, "portability": port, "mu": MU}
            rows.append(row_of(o, r, port))
        self._v5_rows[inst.name] = rows
        return recs

    def _v5_counts(self, insts, suffix: str) -> dict:
        rows = [x for n in insts for x in (self._v5_rows or {}).get(n, [])]
        return {f"v5_rollouts{suffix}": len(rows),
                f"v5_with_log{suffix}": sum(1 for x in rows if x["has_log"]),
                f"v5_submissions{suffix}": sum(x["submissions"] for x in rows),
                f"v5_sibling_runs{suffix}": sum(x["sibling_runs"] for x in rows),
                f"v5_valid{suffix}": sum(1 for x in rows if x["valid"] > 0),
                f"v5_final_solve{suffix}": sum(1 for x in rows if x["final"] == "solve"),
                f"v5_soft_fallback{suffix}": sum(1 for x in rows if x["final"] == "soft_fallback"),
                f"v5_no_answer{suffix}": sum(1 for x in rows if x["final"] is None),
                f"v5_refused{suffix}": sum(x["refused"] for x in rows),
                f"v5_portability_mean{suffix}": round(statistics.fmean([x["portability"] for x in rows]), 4) if rows else 0.0}

    def summary(self, step: int, lines: list, stats: dict, n_groups: int) -> dict:
        rec = super().summary(step, lines, stats, n_groups)
        rec["mu"] = MU
        rec.update(self._v5_counts(list(dict.fromkeys(x["inst"] for x in lines)), ""))
        if self._v5_drawn is not None:
            rec.update(self._v5_counts(self._v5_drawn, "_drawn"))
        return rec

    def _write_v5(self, step: int, batch) -> None:
        if self.out is None:
            return
        with self.lock:
            with open(self.out / "v5.jsonl", "a") as f:
                for name, rows in (self._v5_rows or {}).items():
                    for x in rows:
                        f.write(json.dumps({"step": step, "inst": name, "ds_batch": (name in batch) if batch is not None
                                            else None, **x}, default=str) + "\n")

    def ds_log(self, step: int, lines: list, drawn: list, alive: list, kept: list, handed: list, p: int) -> dict:
        self._v5_drawn = list(drawn)
        try:
            rec = super().ds_log(step, lines, drawn, alive, kept, handed, p)
            self._write_v5(step, set(handed))
            return rec
        finally:
            self._v5_drawn = None
            self._v5_rows = {}

    def log(self, step: int, lines: list, stats: dict, n_groups: int) -> None:
        held = self._held is not None
        super().log(step, lines, stats, n_groups)
        if not held:
            self._write_v5(step, None)
            self._v5_rows = {}


def config_record() -> dict:
    PF = load_pf()
    return {"version": V5RL_VERSION, "reward": V5_REWARD, "mu": MU, "k_siblings": PF.K_SIBLINGS,
            "max_revisions": PF.MAX_REVISIONS, "program_first": PF.record(),
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


def _same_reward(old: dict, rec: dict) -> bool:
    return all(old.get(k) == rec[k] for k in ("version", "reward", "mu", "k_siblings", "max_revisions"))


def install(grpo_mod, dispersion_mod, v3args, dsargs, rest: list, agents_mod=None, v2_mod=None) -> DS.DynSampler:
    """program_first's v5 loop into the trainer's agents / v2, then grpo_ds.install (grpo_v3's harness + lambda-0 reward +
    dynamic sampling), then the portability term. agents_mod / v2_mod: tests only (default: the modules rl/dispersion uses)."""
    import argparse
    if float(v3args.process_lambda) != 0.0 or v3args.process_sham != "off":
        raise SystemExit(f"rl/grpo_v5.py: the v5 reward has NO process bonus -- --process-lambda must be 0 and --process-sham "
                         f"off (got {v3args.process_lambda} / {v3args.process_sham})")
    if not int(dsargs.dyn_sample or 0):
        raise SystemExit("rl/grpo_v5.py: Phase 2 is registered with dynamic sampling -- --dyn-sample N is required")
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    pa.add_argument("--c", type=float, default=0.5)
    g, _ = pa.parse_known_args(rest)
    if g.c + MU >= 1.0 - V3.MARGIN:
        raise SystemExit(f"rl/grpo_v5.py: --c {g.c} + MU {MU} >= 1 - {V3.MARGIN}: an unpassed rollout could out-score a passed "
                         "one under the uncapped registered reward -- refused")
    out = Path(g.out) if g.out else None
    rec = config_record()
    cfgf = out / "harness_v5.json" if out is not None else None
    if cfgf is not None:
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if not _same_reward(old, rec):
                raise SystemExit(f"rl/grpo_v5.py: {out} was trained with {old.get('reward')!r} (mu {old.get('mu')}, K "
                                 f"{old.get('k_siblings')}, revisions {old.get('max_revisions')}) -- refusing to resume it "
                                 f"under {rec['reward']!r}")
            if (old.get("program_first") or {}).get("sha256") != rec["program_first"]["sha256"]:
                print(f"rl/grpo_v5.py: WARNING program_first.py / v5_site changed since this run started "
                      f"({(old.get('program_first') or {}).get('sha256')} -> {rec['program_first']['sha256']})", flush=True)
        elif (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_v5.py: {out} has steps trained WITHOUT the v5 harness / reward (no harness_v5.json) -- "
                             "refusing to resume it")
    PF = load_pf()
    if agents_mod is None:
        import agents as agents_mod                                   # the module rl/dispersion's make_agent lives in
    if v2_mod is None:
        import v2 as v2_mod                                           # the module rl/grpo_v3.install uses
    PF.install_agents(agents_mod)
    PF.install_v2(v2_mod)                                             # BEFORE grpo_ds.install: install_grpo -> v5 wrap_rollout
    ds = DS.install(grpo_mod, dispersion_mod, v3args, dsargs, rest)
    if type(ds.pr) is not DS.DSReward:
        raise SystemExit(f"rl/grpo_v5.py: grpo_ds.install left a {type(ds.pr).__name__}, not grpo_ds's DSReward")
    ds.pr.__class__ = V5Reward
    ds.pr._v5_rows = {}
    if cfgf is not None and not cfgf.exists():
        out.mkdir(parents=True, exist_ok=True)
        cfgf.write_text(json.dumps(rec, indent=1))
    print(f"H-V5RL (RL_PLAN 2026-09-29 14:15, Phase 2): training rollouts in harness v5 (submit solve(data_dir, task); run on the "
          f"target + {PF.K_SIBLINGS} layout-matched train siblings; <= {PF.MAX_REVISIONS} revisions); reward = r_answer + "
          f"{MU} x portability; no process bonus; dynamic sampling on the v5 reward", flush=True)
    return ds


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    dsargs, rest = DS.parse_ds_args(rest)
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, dsargs, rest)
    raise SystemExit(grpo_onpolicy.main())
