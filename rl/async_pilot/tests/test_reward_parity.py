"""Reward parity on STORED R1 rollouts: the adapter scores every logged R1 answer exactly as R1 did.

For each row of an R1 run's rollouts.jsonl (step, inst, reward, passed, answered, answer, cells, attr):
  * DIRECT: StepReward(tolerance=instance_tol, n_steps=reward_steps(cfg), partial_ceiling=0.5) on {"answer": answer},
    zeroed when cells == 0 (--require-code) -- the arithmetic alone;
  * ADAPTER: the full pilot path with a scripted policy that reproduces the logged behaviour -- a code cell first when
    R1 executed cells, then the logged answer as the final JSON (or prose until the stall abort when R1 never
    answered) -- through TokenLLMAgent -> harness LLMAgent.solve -> real Sandbox -> rl/dispersion.one_rollout ->
    rl/grpo_onpolicy.rollouts (--require-code, per-instance S). Must equal the logged reward and pass flag.
Rows that cannot be replayed from the log are counted, not guessed: errored rollouts (context overflow; reward 0 by
construction -- checked) and answers submitted as a bare object without an "answer" key (the log keeps only
submission["answer"]).

    venv_async/bin/python rl/async_pilot/tests/test_reward_parity.py [--runs DIR ...] [--replay-limit N] [--workers 12]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("GBP_ECHO_LAST", "1")
os.environ.setdefault("GBP_SANDBOX_PYTHON", str(REPO / "venv" / "bin" / "python"))

CK = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/ckpt")
RUNS = [CK / f"rl-czero-rlmix6-base-echo-rc-lr1e-5-s{s}" for s in range(1, 6)]
MODEL = "/scratch4/workspace/hieutran_umass_edu-deeprs-bench/models/Qwen3-8B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(r) for r in RUNS])
    ap.add_argument("--replay-run", default=str(RUNS[0]), help="run whose rows are replayed through the adapter")
    ap.add_argument("--replay-limit", type=int, default=0, help="0 = every replayable row")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from paths import data_root
    from rl.dispersion import instance_tol
    from rl.grpo_onpolicy import reward_steps
    from rl.reward import StepReward
    from rl.async_pilot.episode import run_episode

    pool = data_root() / "generated" / "rlmix6" / "train"
    cfgs: dict[str, dict] = {}

    def cfg_of(name: str) -> dict:
        if name not in cfgs:
            cfgs[name] = json.loads((pool / name / "eval_config.json").read_text())
        return cfgs[name]

    # ---------------------------------------------------------------- DIRECT re-score, every row of every run
    print("[direct] StepReward(instance tol, per-instance S, c=0.5) + require-code on logged answers")
    tot = Counter()
    worst = []
    for run in map(Path, args.runs):
        c = Counter()
        for line in open(run / "rollouts.jsonl"):
            r = json.loads(line)
            c["rows"] += 1
            if str(r.get("attr") or "").startswith("error"):
                c["errored"] += 1
                c["errored_reward0"] += r["reward"] == 0.0
                continue
            if r.get("answered") and r.get("answer") is None:
                c["bare_object_unreplayable"] += 1
                continue
            cfg = cfg_of(r["inst"])
            res = StepReward(tolerance=instance_tol(cfg, 0.025), n_steps=reward_steps(cfg),
                             partial_ceiling=0.5).score(cfg, {"answer": r["answer"]} if r.get("answered") else None)
            rew = res.reward if (r.get("cells") or 0) else 0.0
            ok = abs(rew - r["reward"]) < 1e-9 and bool(res.passed) == bool(r["passed"])
            c["checked"] += 1
            c["match"] += ok
            if not ok:
                worst.append((run.name, r["step"], r["inst"], r["reward"], rew, r["passed"], res.passed))
        print(f"  {run.name}: {dict(c)}")
        tot.update(c)
    print(f"  TOTAL: {tot['match']}/{tot['checked']} rescored rows match exactly; "
          f"{tot['errored']} errored rows (reward 0: {tot['errored_reward0']}); "
          f"{tot['bare_object_unreplayable']} bare-object answers not replayable")
    for w in worst[:10]:
        print("   MISMATCH", w)

    # ---------------------------------------------------------------- ADAPTER replay through the full pilot path
    print(f"\n[adapter] replay through TokenLLMAgent + harness + sandbox + one_rollout + rollouts(require_code): "
          f"{Path(args.replay_run).name}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = [json.loads(line) for line in open(Path(args.replay_run) / "rollouts.jsonl")]
    todo = [r for r in rows if not str(r.get("attr") or "").startswith("error")
            and not (r.get("answered") and r.get("answer") is None)]
    if args.replay_limit and len(todo) > args.replay_limit:
        todo = random.Random(0).sample(todo, args.replay_limit)

    def replay(r):
        cells = int(r.get("cells") or 0)
        answer = f"```json\n{json.dumps({'answer': r['answer']})}\n```"
        script = []
        if r.get("answered"):
            if cells:
                script.append("Checking the data.\n```python\nprint('replay')\n```")
            script.append(answer)
        else:
            script = ["I am still thinking about this problem."] * 30      # never acts: stall abort, no answer
        state = {"t": 0}

        def gen(ids, n, temp, top_p):
            text = script[min(state["t"], len(script) - 1)]
            state["t"] += 1
            out = tok.encode(text, add_special_tokens=False) + [im_end]
            return out, [-0.1] * len(out), [0] * len(out), "stop"
        out, _ = run_episode(pool / r["inst"], gen, tok, temperature=1.0, timeout=180, c=0.5, require_code=True,
                             wall_budget=600.0)
        return r, out

    t0 = time.time()
    ok = n = 0
    bad = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, out in ex.map(replay, todo):
            n += 1
            good = abs(float(out["reward"]) - float(r["reward"])) < 1e-9 and bool(out.get("passed")) == bool(r["passed"])
            ok += good
            if not good:
                bad.append((r["step"], r["inst"], r["reward"], out["reward"], r["passed"], out.get("passed"),
                            r.get("cells"), out.get("n_cells"), str(out.get("attribution"))[:40]))
    print(f"  {ok}/{n} replayed rollouts reproduce R1's logged reward and pass flag exactly "
          f"({time.time() - t0:.0f}s, {args.workers} workers)")
    for b in bad[:10]:
        print("   MISMATCH (step, inst, r1_reward, adapter_reward, r1_pass, adapter_pass, r1_cells, adapter_cells, attr)", b)
    fail = (tot["match"] != tot["checked"]) or (ok != n) or (tot["errored"] != tot["errored_reward0"])
    print(f"\n{'ALL PASS' if not fail else 'FAILED'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
