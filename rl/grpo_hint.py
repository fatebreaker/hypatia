"""HINT-GUIDED GROUNDED RL -- H-HINT9 (docs/RL_PLAN.md, registered 2026-10-08): rl/grpo_rg.py (rl/grpo_v3.py + the grounded
reward, installed exactly as it installs itself) with ONE change, in the rollouts:

  For every chosen training instance whose k policy samples ALL fail (none passed with a positive reward), --hint-k
  extra rollouts are drawn from the same policy with the calculator's documented formula appended to the task. The hint
  is the text that calculator's TRAINING-split reference explanations share (scripts/build_formula_hints.py: the
  formula or scoring criteria, no patient values). The hinted rollouts that pass with a positive reward join the
  instance's group with the hint REMOVED from their prompt, so the policy is trained to produce, from the plain task,
  the analysis it could produce with the formula in view (rationalization: the trainer's ratio is 1 by construction,
  as for every sample). Groups with any success are untouched; nothing changes at evaluation (no hints there).

    venv_vllm/bin/python rl/grpo_hint.py --hints <json> --hint-k 4 <every rl/grpo_rg.py argument>
                                                        (experiments/harness_v3/run_v3hint.sh launches it)

LOGS (in --out): harness_hint.json -- this file's sha256, the hints file's path, sha256 and size, hint_k; resuming a run
dir with other hints, or one trained without this file, is refused. hint_log.jsonl -- per call: failing groups, hinted
rollouts, rescued episodes and the instances rescued.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HINT_HEAD = ("\n\nReference for this calculator (documentation for your own use; do not mention or quote this note in "
             "your answer):\n")


def load_hints(path: Path) -> dict[str, str]:
    d = json.loads(path.read_text())
    hints = d.get("hints", d)
    if not isinstance(hints, dict) or not hints or not all(isinstance(v, str) and v.strip() for v in hints.values()):
        raise SystemExit(f"rl/grpo_hint.py: {path} holds no {{calculator: hint}} map")
    return hints


def calculator(cfg: dict) -> str | None:
    prov = cfg.get("_provenance")
    if isinstance(prov, str):
        try:
            prov = ast.literal_eval(prov)
        except (ValueError, SyntaxError):
            return None
    return (prov or {}).get("calculator") if isinstance(prov, dict) else None


def suffix_for(hint: str) -> str:
    return HINT_HEAD + hint.strip()


def hinted_copy(inst: Path, root: Path, suffix: str) -> Path:
    """A copy of the instance (same name, its own parent) whose task ends with the hint; data files copied unchanged."""
    d = root / inst.name
    d.mkdir(parents=True)
    cfg = json.loads((inst / "eval_config.json").read_text())
    cfg["task"] = cfg["task"] + suffix
    (d / "eval_config.json").write_text(json.dumps(cfg))
    if (inst / "data_files").is_dir():
        shutil.copytree(inst / "data_files", d / "data_files")
    return d


def strip(o: dict, suffix: str) -> dict | None:
    """The episode with the hint removed from its prompt; None unless the hint sits exactly once in exactly one
    non-assistant message (anything else cannot be stripped cleanly and is dropped)."""
    msgs = o.get("messages") or []
    hits = [i for i, m in enumerate(msgs)
            if m.get("role") != "assistant" and isinstance(m.get("content"), str) and suffix in m["content"]]
    if len(hits) != 1 or msgs[hits[0]]["content"].count(suffix) != 1:
        return None
    new = [dict(m) for m in msgs]
    new[hits[0]]["content"] = new[hits[0]]["content"].replace(suffix, "")
    assert not any(m.get("role") != "assistant" and isinstance(m.get("content"), str) and HINT_HEAD in m["content"]
                   for m in new), "hint left in the prompt"
    out = dict(o)
    out["messages"] = new
    out["guided"] = "hint"
    return out


def solved(o: dict) -> bool:
    return bool(o.get("passed")) and float(o.get("reward") or 0) > 0


def guided_rollouts(base, hints: dict[str, str], hint_k: int, tmp_root: Path, log_path: Path | None, force: bool = False):
    """A drop-in for grpo_onpolicy.rollouts: base rollouts, then hinted rollouts for the groups that all failed."""
    def rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
        groups = base(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code, demo_for)
        failing = [i for i in insts if not any(solved(o) for o in groups.get(i.name, []))]
        forced = False
        if force and not failing and insts:             # smoke: exercise the hint path at least once
            failing, forced = [insts[0]], True
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="hint_", dir=tmp_root))
        todo, sfx, orig = [], {}, {}
        try:
            for inst in failing:
                cfg = json.loads((inst / "eval_config.json").read_text())
                h = hints.get(calculator(cfg) or "")
                if not h:
                    continue
                s = suffix_for(h)
                d = hinted_copy(inst, tmp, s)
                todo.append(d); sfx[d.name] = s; orig[d.name] = inst
            n_hinted = n_ok = n_resc = 0
            rescued = []
            if todo:
                hg = base(todo, spec, hint_k, temperature, timeout, n_steps, workers, c, require_code, demo_for)
                for d in todo:
                    outs = hg.get(d.name, [])
                    n_hinted += len(outs)
                    ok = [o for o in outs if solved(o)]
                    n_ok += len(ok)
                    added = [e for e in (strip(o, sfx[d.name]) for o in ok) if e is not None]
                    if added:
                        groups[orig[d.name].name].extend(added)
                        n_resc += len(added); rescued.append(d.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "groups": len(insts),
               "failing": len(failing) - int(forced), "forced": forced, "with_hint": len(todo), "hinted_rollouts": n_hinted,
               "hinted_solved": n_ok, "rescued_episodes": n_resc, "rescued": rescued}
        if log_path is not None:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"grpo_hint: {rec['failing']}/{len(insts)} groups failed{' (smoke: forced 1)' if forced else ''}; "
              f"{len(todo)} hinted x {hint_k} -> {n_ok} solved, {n_resc} added to {len(rescued)} groups", flush=True)
        return groups
    return rollouts


def adopt(grpo_mod, hints_path: Path, hints: dict[str, str], hint_k: int, out: Path | None, force: bool = False) -> None:
    if out is not None:
        f = out / "harness_hint.json"
        rec = {"objective": "grounded GRPO + hint-guided rollouts for all-failed groups (hint stripped before training)",
               "trainer_hint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
               "hints": str(hints_path), "hints_sha256": hashlib.sha256(hints_path.read_bytes()).hexdigest()[:16],
               "calculators": len(hints), "hint_k": hint_k}
        if f.exists():
            old = json.loads(f.read_text())
            if (old.get("hints_sha256"), old.get("hint_k")) != (rec["hints_sha256"], rec["hint_k"]):
                raise SystemExit(f"rl/grpo_hint.py: {out} was trained with hints {old.get('hints_sha256')} / k "
                                 f"{old.get('hint_k')} -- refusing to resume it with {rec['hints_sha256']} / k {hint_k}")
        else:
            if (out / "history.jsonl").exists():
                raise SystemExit(f"rl/grpo_hint.py: {out} has training history but no harness_hint.json -- "
                                 "refusing to resume another objective with hints")
            out.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rec, indent=1))
    root = (out if out is not None else Path(tempfile.gettempdir())) / "hint_instances"
    grpo_mod.rollouts = guided_rollouts(grpo_mod.rollouts, hints, hint_k, root,
                                        (out / "hint_log.jsonl") if out is not None else None, force=force)


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    from rl import grpo_rg as RG
    from rl import grpo_v3 as V3
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--hints", required=True)
    ap.add_argument("--hint-k", type=int, default=4)
    ha, rest = ap.parse_known_args(sys.argv[1:])
    if ha.hint_k < 1:
        raise SystemExit("rl/grpo_hint.py: --hint-k must be >= 1")
    v3args, rest = V3.parse_v3_args(rest)
    sys.argv = [sys.argv[0]] + rest
    RG.adopt(V3.install(grpo_onpolicy, dispersion, v3args, rest))        # exactly rl/grpo_rg.py's __main__
    po = argparse.ArgumentParser(add_help=False)
    po.add_argument("--out")
    po.add_argument("--smoke", action="store_true")
    g, _ = po.parse_known_args(rest)
    hp = Path(ha.hints)
    hints = load_hints(hp)
    adopt(grpo_onpolicy, hp, hints, ha.hint_k, Path(g.out) if g.out else None, force=g.smoke)
    print(f"grpo_rg: GROUNDED reward on; grpo_hint: {len(hints)} calculator hints from {hp}, {ha.hint_k} hinted rollouts "
          "per all-failed group (hint stripped before training)", flush=True)
    raise SystemExit(grpo_onpolicy.main())
