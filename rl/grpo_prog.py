"""HARNESS V3 + PORTABLE-PROGRAM REWARD (docs/RL_PLAN.md s.18, ARM E): rl/grpo_v3.py's process reward plus one signal,
`portable` -- does the analysis the rollout RAN also solve a sibling dataset of the same design?

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_prog.py \
        --port-mu 0.3 [--port-sham perm] [--port-extract printed] [--port-sibling layout] --process-lambda 0.3 \
        [--process-sham off] <every rl/grpo_onpolicy.py argument>

Launch through experiments/harness_v3/run_prog.sh (run dir ...-hv3pr<lambda>port<mu>-... / ...portsham<mu>-...; the tag
gains "pos" for --port-extract printed and "lay" for --port-sibling layout: ...portposlay<mu>-... / ...portposlaysham<mu>-...).
rl/grpo_onpolicy.py and rl/grpo_v3.py are untouched: this file installs grpo_v3 (v2 harness + process reward) and adds
two things around it.

WHY. The goal's missing half is PROCEDURES for unseen sciences: a program that solves a design, not an answer that solves
an instance. The answer reward cannot tell them apart (a number typed from a lucky cell passes), the s.15 process reward
pays for checking but not for generality, and R2's generality reward re-ran the POLICY on other instances (a new sample,
not the same program). Here the rollout's OWN program is re-run, unchanged, on new data of the same design.

PORTABLE, per rollout (computed for every rollout that has a transcript and executed >= 1 cell):
  1. the program = the rollout's EXECUTED code cells in order, as the harness decides what ran
     (harness/process_signals.executed_cells: the first ```python block of each non-answer, non-forced turn);
  2. ONE sibling (--port-sibling, default family = the registered spec) -- the same sibling for every rollout of a GRPO
     group, redrawn each step, seeded as rl/siblings.pick is (target, key = the policy spec):
       family  rl/siblings.pick(target, 1, key): another instance of the same family in the target's own split.
               MEASURED: causal_rdd and causal_iv each hold FOUR scenario layouts (column names + roles such as the cutoff,
               which lives in the task text, not the data) of ~25% each, so a family-drawn sibling shares the target's
               layout ~1 time in 4 and a correct program that names its columns and cutoff -- every R0 program does --
               cannot run on the other 3: R0 s3 @150 passes 93/96 and is portable on 28/96 (printed mode).
       layout  opt-in: a sibling drawn only among the split's instances with the SAME layout (data-file names, their header
               lines and the eval_config roles) -- same seeding; none -> portable 0, counted as an error. Portable then
               asks only whether the program computes its answer from the data (a general program) or not (a lucky or
               typed one), within the design.
  3. a fresh sandbox of the rollout's own class (rl.dispersion.Sandbox as installed = the v3 KERNEL sandbox) on a data
     directory holding the SIBLING's data files under the target's file names (symlinks, the siblings.composite layout
     without the target); the cells run in order in that one session -- the notebook the rollout ran, on new data
     (a cell that raises does not stop the later ones, exactly as during the rollout); same per-cell --timeout, and no new
     cell starts after the rollout's wall budget (GBP_WALL_BUDGET) has elapsed;
  4. answer (--port-extract, default json = the registered spec):
       json     harness/agents.extract_answer(the concatenated STDOUT of those cells); none -> portable 0.
                MEASURED BEFORE ANY RUN (2026-09-26, kernel sandbox, one sibling each): 0 of 122 eligible stored causal_rdd
                rollouts print an extractable answer -- the untrained 8B in the v3 harness (26 eligible, HV3_17) AND R0
                seed 3 @150 (96 eligible, 93 passed) print "FINAL effect = 2.86" and type the JSON into their reply. So
                under json, portable is 0 for every rollout until the policy starts printing a JSON answer on its own:
                the arm is the PR arm until then (no group variance, no gradient from the portable term).
       printed  opt-in: the number the re-run prints at the place the ORIGINAL run printed its submitted answer --
                for each numeric answer key, the LAST (cell, number) of the rollout's executed-cell output that shows the
                submitted value (process_signals' grounded match: rounding-aware, sign included; a number equal to a
                numeric literal of that cell's code does not count, so a typed-in answer is never mapped), addressed as
                (cell, index from the end of that cell's printed numbers after the typed text is removed); the re-run's
                number at that address is the sibling answer. Unmappable (the answer was never printed by a cell) or
                out of range -> portable 0. Same measurement: see experiments/harness_v3/tests/test_arms_ce.py / s.18.
  5. graded by the SIBLING's own grader through the trainer's reward path (rl.dispersion.StepReward at instance_tol, the
     call rl.dispersion.one_rollout makes for the target): portable = 1.0 if it passes, else 0.0.
  Any exception anywhere in 1-5 -> portable 0 and counted (process_history.jsonl: portable_errors). A program that hard-codes
  the target's number fails the sibling; a program that computes the answer and PRINTS it (json) passes. The re-execution
  runs inside grpo_onpolicy.rollouts(), so it is part of the step's t_rollout / t_step wall time (also logged: t_portable).

REWARD (the s.15 composition with one more bounded term; the group statistics, clipping and update are grpo_onpolicy's):
    r = r_answer + lambda * p + mu * portable                             (passed)
    r = r_answer + min(lambda * p + mu * portable, 1 - MARGIN - r_answer)  (not passed; MARGIN = grpo_v3.MARGIN = 0.01)
  so a passed rollout earns >= 1.0 and an unpassed one < 1.0: a correct answer always out-scores an incorrect one.
  mu = 0 is grpo_v3.compose exactly; lambda = mu = 0 returns r_answer itself (bit-identical to grpo_onpolicy's reward).
PORTABLE-SHAM (--port-sham perm, the control): portable permuted within each GRPO group over its eligible rollouts
  (grpo_v3.sham_perm, keyed (seed, step, instance + "#portable")): the group's multiset of portable bonuses is the real
  arm's, which rollout earns one is independent of its program. The process term stays real (it is the PR arm's).

LOGS (in --out): grpo_v3's process.jsonl / process_history.jsonl (reward and bonus there are the TOTAL; each step line
also carries mu, port_sham, portable_mean (all rollouts, ineligible = 0), portable_mean_eligible, portable_used_mean,
portable_eligible, portable_errors, portable_no_answer, portable_mapped (printed mode), portable_pass_both, t_portable);
portable.jsonl -- one line per rollout: step, inst, sibling, value, used, passed, eligible, error, no_answer, mapped, cells,
cells_run, truncated, seconds, answer; harness_prog.json (mu, sham, extract, sibling; a resume under other settings is
refused).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_v3 as V3  # noqa: E402

PORT_SHAM_MODES = ("off", "perm")
EXTRACT_MODES = ("json", "printed")
SIBLING_MODES = ("family", "layout")
_LAYOUT: dict = {}                 # instance dir -> its layout key (read once per process)
_LAYOUT_LOCK = threading.Lock()


def layout_key(inst: Path) -> str:
    """Data-file names + each file's header line + the eval_config roles: what a program may legitimately hard-code."""
    inst = Path(inst).resolve()
    with _LAYOUT_LOCK:
        if inst in _LAYOUT:
            return _LAYOUT[inst]
    import gzip
    parts = []
    for f in sorted((inst / "data_files").iterdir()):
        head = ""
        if f.is_file():
            try:
                with (gzip.open(f, "rt", errors="replace") if f.name.endswith(".gz") else open(f, errors="replace")) as fh:
                    head = fh.readline().rstrip("\n")
            except (OSError, EOFError):
                head = "?"
        parts.append((f.name, head))
    try:
        roles = (json.loads((inst / "eval_config.json").read_text()).get("_provenance") or {}).get("roles") or {}
    except (OSError, ValueError):
        roles = {}
    key = json.dumps([parts, roles], sort_keys=True, default=str)
    with _LAYOUT_LOCK:
        _LAYOUT[inst] = key
    return key


def pick_layout(target: Path, key: str) -> Path:
    """One sibling of `target` with the same layout, seeded exactly as rl/siblings.pick (target, key)."""
    import random
    from rl import siblings
    real = Path(target).resolve()
    mine = layout_key(real)
    pool = [p for p in siblings._instances(real.parent) if p.name != real.name and layout_key(p) == mine]
    if not pool:
        raise RuntimeError(f"{real.parent}: no sibling with {real.name}'s layout")
    seed = int(hashlib.sha256(f"{real.parent}|{real.name}|{key}".encode()).hexdigest()[:16], 16)
    return random.Random(seed).sample(pool, 1)[0]


PORT_SALT = "#portable"
PROG_VERSION = 1


def compose(r_answer: float, passed: bool, p: float, port: float, lam: float, mu: float,
            margin: float = V3.MARGIN) -> float:
    """r_answer + bounded (process + portable) term (module docstring). mu == 0 is grpo_v3.compose; lam == mu == 0 returns
    r_answer unchanged."""
    if not lam and not mu:
        return r_answer
    bonus = lam * max(0.0, min(1.0, float(p))) + mu * max(0.0, min(1.0, float(port)))
    if not passed:
        bonus = min(bonus, max(0.0, 1.0 - margin - r_answer))
    return r_answer + bonus


def program_cells(messages: list, n_prefix: int = 0) -> list:
    """The rollout's executed cells' code, in order (the harness's own rules: process_signals.executed_cells)."""
    from process_signals import executed_cells
    cells, _ = executed_cells(messages, n_prefix)
    return [c.code for c in cells]


def sibling_data_dir(sib: Path, root: str | None = None) -> Path:
    """A temp dir holding the sibling's data files under their own (= the design's) file names, as symlinks."""
    d = Path(tempfile.mkdtemp(prefix="port_", dir=root))
    for p in sorted((Path(sib) / "data_files").iterdir()):
        (d / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    return d


def reexecute(codes: list, data_dir: Path, sandbox_factory, timeout: int, wall_budget: float = 0.0):
    """Run the cells in order in ONE fresh sandbox; returns (each cell's stdout, truncated by the wall budget)."""
    sb = sandbox_factory(data_dir=data_dir, timeout=timeout)
    outs, trunc, t0 = [], False, time.monotonic()
    try:
        for code in codes:
            if wall_budget and time.monotonic() - t0 >= wall_budget:
                trunc = True
                break
            r = sb.run(code)
            outs.append(getattr(r, "stdout", "") or "")
    finally:
        sb.cleanup()
    return outs, trunc


def _numeric_literals(code: str) -> set:
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    return {float(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}


def _printed_numbers(code: str, stdout: str) -> list:
    """(value, half-unit, has_decimals) of every number the cell PRINTED, typed text removed (process_signals' evidence)."""
    import ast
    from process_signals import Cell, num_tokens
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        tree = None
    return num_tokens(Cell(0, code, stdout or "", "", tree).evidence)


def answer_values(answer, keys: list) -> list | None:
    obj = answer if isinstance(answer, dict) else {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    out = []
    for k in keys:
        try:
            v = float(obj.get(k))
        except (TypeError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        out.append((k, v))
    return out or None


def answer_positions(codes: list, stdouts: list, answer, keys: list) -> dict | None:
    """key -> (cell index, index from the end of that cell's printed numbers) of the LAST printed occurrence of the
    submitted value (the grounded detector's match); None if any key was never printed by a cell."""
    vals = answer_values(answer, keys)
    if not vals:
        return None
    toks = [_printed_numbers(c, o) for c, o in zip(codes, stdouts)]
    lits = [_numeric_literals(c) for c in codes]
    pos = {}
    for k, v in vals:
        s = repr(v)
        mant, _, ex = s.lower().partition("e")
        dv = len(mant.split(".")[1]) if "." in mant else 0
        half_v = 0.5 * 10.0 ** ((int(ex) if ex else 0) - dv)
        found = None
        for ci in range(len(toks) - 1, -1, -1):
            for j in range(len(toks[ci]) - 1, -1, -1):
                t, half_t, has_dec = toks[ci][j]
                if not has_dec and not v.is_integer():
                    continue
                if any(abs(t - x) <= 1e-9 * max(1.0, abs(x)) for x in lits[ci]):
                    continue                               # a number the cell's own code typed in: not the analysis's
                if abs(t - v) <= max(half_t, half_v) + 1e-12 * max(1.0, abs(v)):
                    found = (ci, len(toks[ci]) - 1 - j)
                    break
            if found:
                break
        if found is None:
            return None
        pos[k] = found
    return pos


def read_positions(codes: list, stdouts: list, pos: dict) -> dict | None:
    out = {}
    for k, (ci, back) in pos.items():
        if ci >= len(stdouts):
            return None
        toks = _printed_numbers(codes[ci], stdouts[ci])
        if back >= len(toks):
            return None
        out[k] = toks[len(toks) - 1 - back][0]
    return out


class PortableStage:
    """Installed INSIDE grpo_v3's process reward (it wraps grpo_onpolicy.rollouts first): after a step's episodes, each
    eligible rollout's program is re-run on one sibling and o["portable"] is set; the process reward (ProgReward) then
    composes the total."""

    def __init__(self, dispersion_mod, grpo_mod=None, out: Path | None = None, sandbox_factory=None,
                 tmp_root: str | None = None, extract: str = "json", sibling: str = "family"):
        if extract not in EXTRACT_MODES:
            raise ValueError(f"--port-extract must be one of {EXTRACT_MODES}")
        if sibling not in SIBLING_MODES:
            raise ValueError(f"--port-sibling must be one of {SIBLING_MODES}")
        self.dispersion, self.grpo, self.out, self.extract, self.sibling = dispersion_mod, grpo_mod, out, extract, sibling
        self.factory = sandbox_factory or (lambda **kw: self.dispersion.Sandbox(**kw))
        self.tmp_root = tmp_root
        self.seconds: dict = {}
        self.lock = threading.Lock()

    @staticmethod
    def wall_budget() -> float:
        try:
            return float(os.environ.get("GBP_WALL_BUDGET") or 0)
        except ValueError:
            return 0.0

    def grade(self, sub: dict, sib: Path, n_steps, c: float) -> bool:
        cfg = json.loads((sib / "eval_config.json").read_text())
        it = self.dispersion.instance_tol
        if n_steps is None:
            rs = getattr(self.grpo, "reward_steps", None)
            n_steps = rs(cfg) if rs else 5
        res = self.dispersion.StepReward(tolerance=it(cfg, it(cfg, 0.025)), n_steps=n_steps,
                                         partial_ceiling=c).score(cfg, sub)
        return bool(res.passed)

    def one(self, o: dict, sib: Path, timeout: int, n_steps, c: float, target: Path | None = None) -> dict:
        t = time.time()
        rec = {"value": 0.0, "eligible": True, "error": None, "sibling": sib.name, "no_answer": False,
               "cells": 0, "cells_run": 0, "truncated": False, "answer": None}
        try:
            from process_signals import executed_cells
            cells, _ = executed_cells(o["messages"], int(o.get("n_prefix") or 0))
            codes = [x.code for x in cells]
            rec["cells"] = len(codes)
            pos = None
            if codes and self.extract == "printed":
                from rl.reward import StepReward
                keys = StepReward._fields(json.loads((Path(target) / "eval_config.json").read_text()))
                pos = answer_positions(codes, [x.stdout for x in cells], o.get("answer"), keys)
                rec["mapped"] = pos is not None
            if codes and (self.extract == "json" or pos is not None):
                d = sibling_data_dir(sib, self.tmp_root)
                try:
                    outs, rec["truncated"] = reexecute(codes, d, self.factory, timeout, self.wall_budget())
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                rec["cells_run"] = len(outs)
                if self.extract == "json":
                    from agents import extract_answer
                    sub = extract_answer("\n".join(outs))
                else:
                    vals = read_positions(codes, outs, pos)
                    sub = {"answer": vals} if vals is not None else None
                if sub is None:
                    rec["no_answer"] = True
                else:
                    rec["answer"] = sub.get("answer", sub) if isinstance(sub, dict) else sub
                    rec["value"] = 1.0 if self.grade(sub, sib, n_steps, c) else 0.0
            else:
                rec["no_answer"] = True
        except Exception as exc:                                               # noqa: BLE001
            rec.update(value=0.0, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        rec["seconds"] = round(time.time() - t, 1)
        return rec

    def run(self, insts, groups: dict, spec: str, timeout: int, n_steps, workers: int, c: float) -> None:
        import agents  # noqa: F401  -- imported here, once, before the worker threads use them
        import process_signals  # noqa: F401
        from rl import siblings
        t0 = time.time()
        step = V3.step_of(spec)
        by_name = {i.name: i for i in insts}
        jobs = []
        for name, outs in groups.items():
            sib, err = None, None
            try:
                sib = (pick_layout(by_name[name], spec) if self.sibling == "layout"
                       else siblings.pick(by_name[name], 1, spec)[0])
            except Exception as exc:                                           # noqa: BLE001
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
            for o in outs:
                if not (o.get("messages") and int(o.get("n_cells") or 0) > 0):
                    o["portable"] = {"value": 0.0, "eligible": False, "error": None, "sibling": None}
                elif sib is None:
                    o["portable"] = {"value": 0.0, "eligible": True, "error": err, "sibling": None}
                else:
                    jobs.append((o, sib, by_name[name]))

        def go(job):
            o, sib, inst = job
            try:
                return self.one(o, sib, timeout, n_steps, c, target=inst)
            except Exception as exc:                                           # noqa: BLE001  (belt and braces)
                return {"value": 0.0, "eligible": True, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                        "sibling": getattr(sib, "name", None)}

        if jobs:
            with cf.ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as ex:
                for (o, _, _), res in zip(jobs, ex.map(go, jobs)):
                    o["portable"] = res
        with self.lock:
            self.seconds[step] = round(time.time() - t0, 1)

    def wrap(self, orig_rollouts):
        def prog_rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
            groups = orig_rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c,
                                   require_code=require_code, demo_for=demo_for)
            self.run(insts, groups, spec, timeout, n_steps, workers, c)
            return groups

        prog_rollouts.portable_stage = self
        return prog_rollouts


class ProgReward(V3.ProcessReward):
    """grpo_v3's ProcessReward (its instance is re-classed after grpo_v3.install): the process score as v3 computes it,
    then the total recomposed with the portable term, the per-rollout portable log, and portable fields per step."""

    def setup_prog(self, mu: float, port_sham: str, stage: PortableStage) -> "ProgReward":
        if port_sham not in PORT_SHAM_MODES:
            raise ValueError(f"--port-sham must be one of {PORT_SHAM_MODES}")
        self.mu, self.port_sham, self.stage = float(mu), port_sham, stage
        self._port: dict = {}
        return self

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = super().score_group(inst, outs, step)
        port = [float((o.get("portable") or {}).get("value") or 0.0) for o in outs]
        used = list(port)
        if self.port_sham == "perm":
            elig = [i for i, r in enumerate(recs) if r["eligible"]]
            perm = V3.sham_perm(self.seed, step, inst.name + PORT_SALT, len(elig))
            for dst, k in zip(elig, perm):
                used[dst] = port[elig[k]]
        lines = []
        for o, r, pv, pu in zip(outs, recs, port, used):
            total = compose(r["r_answer"], r["passed"], r["p_used"], pu, self.lam, self.mu)
            if self.require_code and not r["cells"]:
                total = 0.0
            r.update(portable=pv, portable_used=pu, bonus=round(total - r["r_answer"], 6), reward=total)
            o["reward"] = total
            if isinstance(o.get("process"), dict):
                o["process"].update(portable=pv, portable_used=pu)
            info = o.get("portable") or {}
            lines.append({"step": step, "inst": inst.name, "sibling": info.get("sibling"), "value": pv, "used": pu,
                          "passed": r["passed"], "eligible": bool(info.get("eligible")), "error": info.get("error"),
                          "no_answer": info.get("no_answer"), "mapped": info.get("mapped"), "cells": info.get("cells"),
                          "cells_run": info.get("cells_run"), "truncated": info.get("truncated"),
                          "seconds": info.get("seconds"), "answer": info.get("answer")})
        with self.lock:
            acc = self._port.setdefault(step, [])
            acc += lines
            if self.out is not None:
                with open(self.out / "portable.jsonl", "a") as f:
                    for x in lines:
                        f.write(json.dumps(x, default=str) + "\n")
        return recs

    def summary(self, step: int, lines: list, stats: dict, n_groups: int) -> dict:
        rec = super().summary(step, lines, stats, n_groups)
        with self.lock:
            acc = self._port.pop(step, [])
        el = [x for x in acc if x["eligible"]]
        rec.update({
            "mu": self.mu, "port_sham": self.port_sham, "port_extract": self.stage.extract if self.stage else None,
            "port_sibling": self.stage.sibling if self.stage else None,
            "portable_mean": round(statistics.fmean([x["value"] for x in acc]), 4) if acc else 0.0,
            "portable_mean_eligible": round(statistics.fmean([x["value"] for x in el]), 4) if el else 0.0,
            "portable_used_mean": round(statistics.fmean([x["used"] for x in acc]), 4) if acc else 0.0,
            "portable_eligible": len(el),
            "portable_errors": sum(1 for x in acc if x["error"]),
            "portable_no_answer": sum(1 for x in el if x["no_answer"]),
            "portable_mapped": sum(1 for x in el if x.get("mapped")),
            "portable_pass_both": sum(1 for x in acc if x["passed"] and x["value"] >= 1.0),
            "t_portable": self.stage.seconds.get(step) if self.stage is not None else None})
        return rec


def config_record(mu: float, sham: str, extract: str = "json", sibling: str = "family") -> dict:
    return {"version": PROG_VERSION, "port_mu": mu, "port_sham": sham, "port_extract": extract, "port_sibling": sibling,
            "port_salt": PORT_SALT,
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


def parse_prog_args(argv: list):
    """(grpo_v3's args, this file's args, the rest for grpo_onpolicy)."""
    v3args, rest = V3.parse_v3_args(argv)
    # allow_abbrev=False: the trainer's own --port would otherwise be read as an ambiguous abbreviation of --port-*
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--port-mu", type=float, default=0.3)
    ap.add_argument("--port-sham", choices=PORT_SHAM_MODES, default="off")
    ap.add_argument("--port-extract", choices=EXTRACT_MODES, default="json")
    ap.add_argument("--port-sibling", choices=SIBLING_MODES, default="family")
    pargs, rest = ap.parse_known_args(rest)
    return v3args, pargs, rest


def install(grpo_mod, dispersion_mod, v3args, pargs, rest: list, sandbox_factory=None, tmp_root: str | None = None):
    """Portable stage around grpo_onpolicy.rollouts, then grpo_v3.install (v2 harness + process reward) around that, and
    grpo_v3's reward object re-classed to ProgReward. Returns the reward object."""
    mu, sham = float(pargs.port_mu), pargs.port_sham
    extract, sibling = getattr(pargs, "port_extract", "json"), getattr(pargs, "port_sibling", "family")
    if extract not in EXTRACT_MODES:
        raise SystemExit(f"rl/grpo_prog.py: --port-extract must be one of {EXTRACT_MODES}")
    if sibling not in SIBLING_MODES:
        raise SystemExit(f"rl/grpo_prog.py: --port-sibling must be one of {SIBLING_MODES}")
    if not 0.0 <= mu < 1.0:
        raise SystemExit("rl/grpo_prog.py: --port-mu must be in [0, 1)")
    if sham not in PORT_SHAM_MODES:
        raise SystemExit(f"rl/grpo_prog.py: --port-sham must be one of {PORT_SHAM_MODES}")
    if sham != "off" and mu == 0:
        raise SystemExit("rl/grpo_prog.py: a portable sham needs --port-mu > 0")
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--out")
    g, _ = pa.parse_known_args(rest)
    out = Path(g.out) if g.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cfgf = out / "harness_prog.json"
        rec = config_record(mu, sham, extract, sibling)
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if ((old.get("port_mu"), old.get("port_sham"), old.get("port_extract", "json"), old.get("port_sibling", "family"))
                    != (rec["port_mu"], rec["port_sham"], rec["port_extract"], rec["port_sibling"])):
                raise SystemExit(f"rl/grpo_prog.py: {out} was trained with {old} -- refusing to resume it as {rec}")
        else:
            cfgf.write_text(json.dumps(rec, indent=1))
    stage = PortableStage(dispersion_mod, grpo_mod, out, sandbox_factory, tmp_root, extract, sibling)
    grpo_mod.rollouts = stage.wrap(grpo_mod.rollouts)
    pr = V3.install(grpo_mod, dispersion_mod, v3args, rest)
    pr.__class__ = ProgReward
    pr.setup_prog(mu, sham, stage)
    print(f"portable-program reward: mu={mu} sham={sham} extract={extract} sibling={sibling} (one sibling per group and step, the rollout's "
          f"executed cells re-run in a fresh sandbox, the sibling's grader)", flush=True)
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3a, pa_, rest_ = parse_prog_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest_
    install(grpo_onpolicy, dispersion, v3a, pa_, rest_)
    raise SystemExit(grpo_onpolicy.main())
