"""H-MC-DISTILL stage 1 -> SFT corpus (docs/RL_PLAN.md): the teacher's CORRECT transcripts on the MedCalc TRAIN sample,
rebuilt into the exact conversation the harness sent (system = SYSTEMS[GBP_SYSTEM], user = build_prompt(task, listing),
then assistant turns and tool outputs as user turns -- the same reconstruction as rl/dispersion.conversation), written as
{"messages": [...]} lines for rl/sft_train.py (--corpus medcalc_<tag>).

    venv/bin/python scripts/build_mc_distill_sft.py --results results/pods/frontier_api/o4-mini-teacher/medcalc_train_1000 \
        --problems benchmarks/medcalc_train_1000 --tag o4mini [--system science] [--val-frac 0.05]
Only passed, error-free episodes with at least one executed code cell are kept (a correct answer without code would teach
guessing). Test notes never enter: the TRAIN pool already excludes every note in either MedCalc test file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "harness"))
from paths import data_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--problems", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--system", default="science")
    ap.add_argument("--val-frac", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["GBP_SYSTEM"] = a.system
    import agents
    from sandbox import Sandbox
    rows, skipped = [], {"failed": 0, "error": 0, "no_code": 0, "no_cfg": 0}
    for f in sorted(Path(a.results).glob("*.json")):
        if f.name in ("summary.json", "harness_v2.json"):
            continue
        r = json.loads(f.read_text())
        if "grade" not in r:
            continue
        g = r["grade"] if isinstance(r["grade"], dict) else eval(r["grade"])  # noqa: S307 -- our own result files
        if r.get("error"):
            skipped["error"] += 1; continue
        if not g.get("passed"):
            skipped["failed"] += 1; continue
        if not int(r.get("n_cells") or 0):
            skipped["no_code"] += 1; continue
        pdir = Path(a.problems) / "problems" / r["id"]
        cfgp = pdir / "eval_config.json"
        if not cfgp.exists():
            skipped["no_cfg"] += 1; continue
        cfg = json.loads(cfgp.read_text())
        sb = Sandbox(data_dir=pdir / "data_files", timeout=10)
        try:
            listing = sb.listing()
        finally:
            sb.cleanup()
        msgs = [{"role": "system", "content": agents.SYSTEMS[a.system]},
                {"role": "user", "content": agents.build_prompt(cfg["task"], listing)}]
        for m in r["transcript"]:
            if m.get("role") == "assistant":
                msgs.append({"role": "assistant", "content": m.get("content") or ""})
            elif m.get("role") in ("tool", "system"):
                msgs.append({"role": "user", "content": m.get("sent") or m.get("content") or ""})
        rows.append({"messages": msgs, "id": r["id"], "calculator": cfg["_provenance"].get("calculator")})
    random.Random(a.seed).shuffle(rows)
    nval = max(1, int(len(rows) * a.val_frac)) if rows else 0
    out = data_root() / "sft"; out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", rows[nval:]), ("val", rows[:nval])):
        with open(out / f"medcalc_{a.tag}_{name}.jsonl", "w") as fh:
            for x in part:
                fh.write(json.dumps(x) + "\n")
    calcs = len({x["calculator"] for x in rows})
    print(f"[mc_distill] kept {len(rows)} correct transcripts ({calcs} calculators): train {len(rows) - nval}, val {nval}; "
          f"skipped {skipped} -> {out}/medcalc_{a.tag}_{{train,val}}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
