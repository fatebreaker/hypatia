#!/usr/bin/env python
"""The per-hazard endpoint (docs/DIRECTIONS.md "Measurement changes" 1-3; A1's registered reading): trap rate by
hazard MODE per surface, for any checkpoint's evaluation.

WHAT IT READS. Result directories of rl/dispersion.py JSONs (<family>.json, e.g. results/pods/rl_plan/base_echo/ or
results/pods/directions/a1/<run>/step150/). Each JSON names its family and split, so every rollout's instance config is
found at $DEEP_DATA_ROOT/generated/<family>/<split>/<instance>/eval_config.json. Or, with --train-run, a training run's
rollouts.jsonl (instances under generated/<pool>/train).

HAZARDS OF AN INSTANCE.
  * toggled corpora (sage/hazards.py) carry _provenance.hazards: one record per hazard with its mode, governed traps,
    false-fix traps, `planted` and `present` (a governed trap is live on this instance). Used as is.
  * every other corpus: hazards are derived from the instance's own ablation table and scripts/trap_modes.py -- one
    hazard per MODE, its traps the ablation rows mapped to that mode; present = at least one of them has a recorded
    step (live on this instance; an accepted variant or a conditional trap coinciding with the reference has step None
    and never counts). Traps with no mode are pooled as "unmapped" so their rates stay visible.

WHAT COUNTS AS A TRAP HIT. STRICT (default): the reward attributed the answer to that trap (fails_at_step recorded:
within 0.5 tolerance of the trap's output, unambiguous -- rl/reward.py). NEAREST (secondary, --nearest column): the
trap was merely the closest ablation, however far -- the older mode re-score (scratchpad mode_rescore.py) counted this,
which inflates rates on 0%-pass families (every wrong enzyme_mm answer is "nearest" to an M3 trap).

OUTPUT per (surface, hazard): rollouts on instances where it is present -> trap%, nearest%, pass%, mean reward (step
credit); where absent (toggled corpora) -> false-fix% (two-sided hazards) and pass%. Then the hazard x surface matrix
by MODE (strict trap% on present instances), and with --base, the change vs a base directory cell by cell. Families
are tagged with their foothold class if --classes is given (judgement vs procedure; never average across them).

    venv/bin/python scripts/hazard_endpoint.py results/pods/directions/a1/<run>/step150 --base results/pods/directions/a1/base_sysscience
    venv/bin/python scripts/hazard_endpoint.py results/pods/rl_plan/base_echo results/pods/rl_plan/<run>/step150 --json out.json
    venv/bin/python scripts/hazard_endpoint.py --train-run $D/ckpt/<run> --pool a1_toggle4 --steps 1-25
    venv/bin/python scripts/hazard_endpoint.py --check-mapping      # trap_modes.py vs Trap.step in the registry (slow)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from trap_modes import TRAP_MODE  # noqa: E402

NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?")


def data_root() -> Path:
    from paths import data_root as dr
    return Path(dr())


_CFG: dict = {}


def cfg_of(family: str, split: str, inst: str) -> dict | None:
    key = (family, split, inst)
    if key not in _CFG:
        p = data_root() / "generated" / family / split / inst / "eval_config.json"
        _CFG[key] = json.loads(p.read_text()) if p.is_file() else None
    return _CFG[key]


def hazards_of(cfg: dict) -> list[dict]:
    """[{id, mode, kind, traps, false_fix, present, planted}] for one instance (see module docstring)."""
    prov = cfg.get("_provenance") or {}
    if prov.get("hazards"):
        return [{"id": h["id"], "mode": h["mode"], "kind": h["kind"], "traps": h["traps"],
                 "false_fix": h.get("false_fix", []), "present": bool(h["present"]), "planted": h.get("planted")}
                for h in prov["hazards"]]
    abl = prov.get("ablations") or {}
    by_mode: dict = collections.defaultdict(lambda: {"traps": [], "live": []})
    for name, row in abl.items():
        if name == "reference":
            continue
        mode = TRAP_MODE.get(name, "unmapped")
        by_mode[mode]["traps"].append(name)
        if row.get("fails_at_step") is not None:
            by_mode[mode]["live"].append(name)
    return [{"id": m, "mode": m, "kind": "derived", "traps": v["traps"], "false_fix": [], "present": bool(v["live"]),
             "planted": None} for m, v in by_mode.items()
            if v["live"] or m != "unmapped"]         # an "unmapped" group of accepted variants only is not a hazard


def distinct_printed_numbers(r: dict) -> int | None:
    """HABIT metric (DIRECTIONS.md A2, secondary): distinct numbers the episode's executed cells printed before the
    answer (tool outputs only; the task text excluded). None if the transcript was not stored."""
    msgs = r.get("messages")
    if not msgs:
        return None
    seen = set()
    for m in msgs[2:]:
        if m.get("role") == "user" and str(m.get("content", "")).startswith(("STDOUT", "STDERR")):
            seen.update(NUM.findall(m["content"].split("STDERR:")[0]))
    return len(seen)


class Tally:
    def __init__(self):
        self.c = collections.defaultdict(lambda: collections.Counter())

    def add(self, key, **kv):
        for k, v in kv.items():
            self.c[key][k] += v


def rollouts_from_dir(d: Path):
    """(family, split, instance, rollout) for every rollout of every dispersion JSON in d."""
    for f in sorted(Path(d).glob("*.json")):
        try:
            js = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        if not isinstance(js, dict) or "groups" not in js:
            continue
        fam, split = js.get("family"), js.get("split", "test")
        for inst, rs in js["groups"].items():
            for r in rs:
                yield fam, split, inst, r


def rollouts_from_train(run: Path, pool: str, steps: tuple[int, int] | None):
    for line in (Path(run) / "rollouts.jsonl").open():
        r = json.loads(line)
        if steps and not (steps[0] <= r["step"] <= steps[1]):
            continue
        yield pool, "train", r["inst"], {"reward": r["reward"], "passed": r["passed"], "attribution": r["attr"],
                                         "fails_at_step": r["fails_at_step"], "answered": r["answered"]}


def score(stream, habit: bool = False) -> dict:
    """Per (family, hazard id): counts on present / absent instances."""
    T = Tally()
    fam_of_inst = {}
    for fam, split, inst, r in stream:
        cfg = cfg_of(fam, split, inst)
        if cfg is None:
            T.add((fam, "_missing_config", "-", "-"), n=1)
            continue
        src_fam = cfg.get("_provenance", {}).get("family", fam)
        fam_of_inst[(fam, inst)] = src_fam
        attr = str(r.get("attribution"))
        strict = r.get("fails_at_step") is not None
        passed = bool(r.get("passed"))
        rew = float(r.get("reward") or 0.0)
        T.add((src_fam, "_all", "-", "-"), n=1, passed=passed, reward=rew,
              **({"habit_n": 1, "habit_sum": distinct_printed_numbers(r)} if habit and r.get("messages") else {}))
        for h in hazards_of(cfg):
            key = (src_fam, h["id"], h["mode"], h["kind"])
            if h["present"]:
                T.add(key, n_present=1, hit=int(strict and attr in h["traps"]), nearest=int(attr in h["traps"]),
                      pass_present=int(passed), reward_present=rew)
            else:
                T.add(key, n_absent=1, false_fix=int(strict and attr in h["false_fix"]),
                      hit_absent=int(strict and attr in h["traps"]), pass_absent=int(passed))
    return T.c


def pct(a, b):
    return None if not b else round(100.0 * a / b, 1)


def table(c: dict) -> list[dict]:
    rows = []
    for (fam, hid, mode, kind), v in sorted(c.items()):
        if hid.startswith("_"):
            continue
        rows.append({"family": fam, "hazard": hid, "mode": mode, "kind": kind,
                     "n_present": v["n_present"], "trap_pct": pct(v["hit"], v["n_present"]),
                     "nearest_pct": pct(v["nearest"], v["n_present"]),
                     "pass_pct_present": pct(v["pass_present"], v["n_present"]),
                     "reward_present": None if not v["n_present"] else round(v["reward_present"] / v["n_present"], 4),
                     "n_absent": v["n_absent"], "false_fix_pct": pct(v["false_fix"], v["n_absent"]),
                     "trap_pct_absent": pct(v["hit_absent"], v["n_absent"]),
                     "pass_pct_absent": pct(v["pass_absent"], v["n_absent"])})
    return rows


def family_summary(c: dict) -> dict:
    out = {}
    for (fam, hid, _, _), v in c.items():
        if hid == "_all":
            out[fam] = {"n": v["n"], "pass_pct": pct(v["passed"], v["n"]), "mean_reward": round(v["reward"] / v["n"], 4),
                        **({"habit_distinct_numbers": round(v["habit_sum"] / v["habit_n"], 1)} if v.get("habit_n") else {})}
    return out


def mode_matrix(rows: list[dict]) -> dict:
    """mode -> family -> strict trap% on present instances (the hazard x surface matrix), pooling hazards of one mode."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        if r["n_present"]:
            a = acc[r["mode"]][r["family"]]
            a[0] += round((r["trap_pct"] or 0) * r["n_present"] / 100)
            a[1] += r["n_present"]
    return {m: {f: {"trap_pct": pct(h, n), "n": n} for f, (h, n) in fams.items()} for m, fams in acc.items()}


def print_report(label: str, rows: list[dict], fams: dict, base_rows: list[dict] | None = None, classes=None):
    print(f"\n=== {label}")
    bmap = {(r["family"], r["hazard"]): r for r in (base_rows or [])}
    print(f"{'surface':22s} {'class':10s} {'hazard':22s} {'mode':18s} {'kind':8s} {'n_pres':>6s} {'trap%':>6s} "
          f"{'d_base':>7s} {'near%':>6s} {'pass%':>6s} {'rew':>6s} | {'n_abs':>5s} {'ffix%':>6s} {'pass%':>6s}")
    for r in rows:
        b = bmap.get((r["family"], r["hazard"]))
        d = (None if not b or b["trap_pct"] is None or r["trap_pct"] is None else round(r["trap_pct"] - b["trap_pct"], 1))
        f = lambda x: "-" if x is None else f"{x}"
        print(f"{r['family']:22s} {(classes or {}).get(r['family'], ''):10s} {r['hazard']:22s} {r['mode']:18s} "
              f"{r['kind']:8s} {r['n_present']:6d} {f(r['trap_pct']):>6s} {f(d):>7s} {f(r['nearest_pct']):>6s} "
              f"{f(r['pass_pct_present']):>6s} {f(r['reward_present']):>6s} | {r['n_absent']:5d} "
              f"{f(r['false_fix_pct']):>6s} {f(r['pass_pct_absent']):>6s}")
    print("per surface: " + "; ".join(f"{k} pass {v['pass_pct']}% (n={v['n']}, reward {v['mean_reward']}"
                                      + (f", distinct printed numbers {v['habit_distinct_numbers']}" if
                                         'habit_distinct_numbers' in v else "") + ")" for k, v in sorted(fams.items())))


def check_mapping() -> dict:
    """scripts/trap_modes.py against Trap.step in the registry (DIRECTIONS.md: the mapping drifted twice):
    mapped traps that are accepted variants (step None), mapped traps no family has, and registry traps with a step
    that are neither mapped nor named in the file's deliberately-unmapped notes."""
    import warnings
    warnings.filterwarnings("ignore")
    import sage  # noqa: F401  (FIRST: it pins BLAS to one thread before numpy loads)
    import numpy as np
    from sage.registry import families
    src = (ROOT / "scripts" / "trap_modes.py").read_text()
    notes = src[src.find("# THREE ACCEPTED"):src.find("def mode_of")]
    seen = collections.defaultdict(dict)
    for name, f in sorted(families().items()):
        for s in range(3):
            try:
                rng = np.random.default_rng(s)
                traps = f.traps(f.draw(rng, f.knobs(rng)).data)
            except Exception:                                   # noqa: BLE001
                break
            for t, spec in traps.items():
                step = spec.step if hasattr(spec, "step") else spec[1]
                seen[name].setdefault(t, set()).add(step)
    variants = sorted(f"{t} ({fam})" for fam, ts in seen.items() for t, st in ts.items()
                      if t in TRAP_MODE and st == {None})
    absent = sorted(t for t in TRAP_MODE if not any(t in ts for ts in seen.values()))
    unlisted = sorted(f"{t} ({fam})" for fam, ts in seen.items() for t, st in ts.items()
                      if st != {None} and t not in TRAP_MODE and t not in notes)
    return {"mapped_but_accepted_variant": variants, "mapped_but_not_in_registry": absent,
            "unmapped_and_not_explained": unlisted,
            "drifted": bool(variants or absent)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="result directories of rl/dispersion.py JSONs (one report each)")
    ap.add_argument("--base", default=None, help="a base result directory: every report shows the change vs it")
    ap.add_argument("--train-run", default=None, help="an RL run directory (reads its rollouts.jsonl)")
    ap.add_argument("--pool", default=None, help="with --train-run: the training pool under generated/")
    ap.add_argument("--steps", default=None, help="with --train-run: A-B, the step range")
    ap.add_argument("--habit", action="store_true", help="also the distinct-printed-numbers transcript metric")
    ap.add_argument("--classes", default=None, help="JSON {family: judgement|procedure} to tag rows")
    ap.add_argument("--json", default=None, help="write every report here")
    ap.add_argument("--check-mapping", action="store_true")
    a = ap.parse_args()
    out = {}
    if a.check_mapping:
        rep = check_mapping()
        print(json.dumps(rep, indent=1))
        out["mapping_check"] = rep
    classes = json.loads(Path(a.classes).read_text()) if a.classes else None
    base_rows = None
    if a.base:
        bc = score(rollouts_from_dir(Path(a.base)), a.habit)
        base_rows = table(bc)
        print_report(f"BASE {a.base}", base_rows, family_summary(bc), None, classes)
        out["base"] = {"dir": a.base, "rows": base_rows, "families": family_summary(bc), "matrix": mode_matrix(base_rows)}
    for d in a.dirs:
        c = score(rollouts_from_dir(Path(d)), a.habit)
        rows = table(c)
        print_report(d, rows, family_summary(c), base_rows, classes)
        out[d] = {"rows": rows, "families": family_summary(c), "matrix": mode_matrix(rows)}
    if a.train_run:
        if not a.pool:
            ap.error("--train-run needs --pool")
        st = tuple(int(x) for x in a.steps.split("-")) if a.steps else None
        c = score(rollouts_from_train(Path(a.train_run), a.pool, st))
        rows = table(c)
        print_report(f"TRAIN {a.train_run} steps {a.steps or 'all'}", rows, family_summary(c), None, classes)
        out["train"] = {"run": a.train_run, "steps": a.steps, "rows": rows, "families": family_summary(c)}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
