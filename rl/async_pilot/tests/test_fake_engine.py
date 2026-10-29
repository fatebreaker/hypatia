"""The adapter end to end with a FAKE model (CPU; real tokenizer, real harness loop, real sandbox, real reward).

A scripted engine stands in for vLLM behind AReaL's InferenceEngine interface (agenerate + get_version). One R1 group
of six episodes on a real rlmix6 train instance, each episode a different script:

  S0 code-then-right   a cell ending in a bare expression (notebook display must show it), then the right answer
  S1 code-then-wrong   a cell, then a wrong answer                          -> proximity credit only
  S2 no-code-right     the right answer with no cell                        -> --require-code: reward 0, still trained
  S3 stall-then-right  prose (harness nudges), a cell, then the right answer
  S4 overflow          long cells until the prompt cannot fit a reply       -> R1's 400 path: error, reward 0, stub
  S5 cut-off           a reply that hits max_tokens with no EOS, then code, then the right answer

and the engine bumps its policy version while the group runs (as an asynchronous trainer would). Checked:
  * every trained row decodes to EXACTLY the conversation R1 would train on (RL_TEMPLATE over one_rollout's messages)
    up to the policy's last token; its loss mask covers exactly the sampled tokens; logprobs/versions are the engine's;
  * rewards and pass flags are what StepReward/--require-code give; advantages are rl/grpo_train.advantages;
  * the overflow episode is a stub with no loss tokens (R1: errored rollouts never enter the gradient);
  * rollouts.jsonl rows are written; a trimmed history (LLMAgent._trim) is detected as desync and kept out of training.

    venv_async/bin/python rl/async_pilot/tests/test_fake_engine.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("GBP_ECHO_LAST", "1")
os.environ.setdefault("GBP_SANDBOX_PYTHON", str(REPO / "venv" / "bin" / "python"))

INST = "causal_did_00150"
MODEL = "/scratch4/workspace/hieutran_umass_edu-deeprs-bench/models/Qwen3-8B"
MAX_LEN, MAX_NEW = 4096, 1024             # small window so the overflow script overflows in a few turns
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


@dataclass
class _Resp:
    output_tokens: list
    output_logprobs: list
    output_versions: list
    stop_reason: str


@dataclass
class FakeEngine:
    tok: object
    scripts: list
    bump_every: int = 4
    version: int = 0
    calls: int = 0
    convs: list = field(default_factory=list)         # per script: token history the engine has seen+produced
    turn: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_version(self) -> int:
        return self.version

    def _which(self, ids: list) -> int:
        for i, h in enumerate(self.convs):
            if h is not None and len(ids) >= len(h) and ids[:len(h)] == h:
                return i
        i = len([h for h in self.convs if h is not None])        # a new episode: next unassigned script
        self.convs.append(None)
        self.turn.append(0)
        return i

    async def agenerate(self, req):
        await asyncio.sleep(0)
        with self.lock:
            i = self._which(list(req.input_ids))
            t = self.turn[i]
            text, eos = self.scripts[i](t)
            out = self.tok.encode(text, add_special_tokens=False)
            if eos:
                out = out + [self.tok.convert_tokens_to_ids("<|im_end|>")]
            stop = "stop"
            n = req.gconfig.max_new_tokens
            if len(out) > n:
                out, stop = out[:n], "length"
            self.turn[i] += 1
            self.convs[i] = list(req.input_ids) + out
            self.calls += 1
            if self.calls % self.bump_every == 0:
                self.version += 1
            v = self.version
        return _Resp(out, [-0.01 * (k % 7) - 0.001 for k in range(len(out))], [v] * len(out), stop)


def scripts_for(gt: dict):
    right = json.dumps({"answer": gt})
    wrong = json.dumps({"answer": {k: (v * 1.7 + 0.3 if isinstance(v, (int, float)) else v) for k, v in gt.items()}})
    code = "Let me inspect the data first.\n```python\nimport os\nsorted(os.listdir('data_files'))\n```"
    long_code = "```python\n# " + " ".join(["filler"] * 900) + "\nprint(1)\n```"

    def s0(t): return (code, True) if t == 0 else (f"```json\n{right}\n```", True)
    def s1(t): return (code, True) if t == 0 else (f"```json\n{wrong}\n```", True)
    def s2(t): return (f"The effect is clear.\n```json\n{right}\n```", True)
    def s3(t): return [("I will think about the design carefully before acting.", True), (code, True),
                       (f"```json\n{right}\n```", True)][min(t, 2)]
    def s4(t): return (long_code, True)
    def s5(t): return [(" ".join(["ramble"] * 2000), True), (code, True), (f"```json\n{right}\n```", True)][min(t, 2)]
    return [s0, s1, s2, s3, s4, s5]


def main() -> int:
    from transformers import AutoTokenizer
    import rl.async_pilot.workflow as W
    from paths import data_root
    from rl.grpo_onpolicy import RL_TEMPLATE
    from rl.grpo_train import advantages
    from rl.reward import StepReward
    from rl.dispersion import instance_tol
    from rl.grpo_onpolicy import reward_steps

    tok = AutoTokenizer.from_pretrained(MODEL)
    inst = data_root() / "generated" / "rlmix6" / "train" / INST
    cfg = json.loads((inst / "eval_config.json").read_text())
    log_dir = Path(tempfile.mkdtemp(prefix="apilot_fake_", dir=os.environ.get("TMPDIR")))
    wf = W.DeeprsGroupWorkflow(tokenizer_path=MODEL, lora_name="lora-test", k=6, max_model_len=MAX_LEN,
                               max_new_tokens=MAX_NEW, log_dir=str(log_dir), threads=8)
    eng = FakeEngine(tok=tok, scripts=scripts_for(cfg["ground_truth"]))

    # the workflow runs all six at once; to give each script a fixed episode, serialise the first turn by
    # handing scripts out in arrival order (FakeEngine._which) and mapping back through the recorded conversations
    print(f"[fake engine] one R1 group on {INST} (max_model_len {MAX_LEN}, max_new {MAX_NEW})")
    batch = asyncio.run(wf.arun_episode(eng, {"inst": str(inst)}))
    captured = wf.last_results            # row order
    outs = [o for o, _ in captured]
    trajs = [t for _, t in captured]
    # map each episode to its script by its first generated reply
    first = {}
    for j, (o, tr) in enumerate(captured):
        msgs = o.get("messages") or []
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if a is None:     # errored: no messages -> identify from the trajectory text
            a = tok.decode([x for x, m in zip(tr.ids, tr.loss) if m][:20])
        first[j] = a
    kind = {}
    for j, a in first.items():
        kind[j] = ("S2" if a.startswith("The effect is clear") else "S3" if a.startswith("I will think") else
                   "S4" if a.startswith("```python\n# filler") or "filler" in a else
                   "S5" if a.startswith("ramble") else "S0/S1")
    rewards = [float(o["reward"]) for o in outs]
    advs = advantages(rewards)
    print("  episodes:", [(kind[j], round(rewards[j], 4), bool(outs[j].get("passed")), outs[j].get("n_cells"),
                           str(outs[j].get("attribution"))[:22]) for j in range(6)])
    check("batch has 6 rows", batch["input_ids"].shape[0] == 6, str(tuple(batch["input_ids"].shape)))
    check("versions advanced during the group (async)", eng.version > 0, f"engine version {eng.version}")
    tol = instance_tol(cfg, 0.025)
    for j in range(6):
        o, tr = outs[j], trajs[j]
        row_mask = batch["loss_mask"][j]
        k = kind[j]
        if k == "S4":
            check("S4 overflow -> error, reward 0, no messages, stub row (no loss tokens)",
                  str(o.get("attribution", "")).startswith("error") and o["reward"] == 0.0 and not o.get("messages")
                  and int(row_mask.sum()) == 0, str(o.get("error_msg", ""))[:120])
            continue
        # R1's training text for this episode
        r1_text = tok.apply_chat_template(o["messages"], chat_template=RL_TEMPLATE, tokenize=False)
        n = int(batch["attention_mask"][j].sum())
        ids = batch["input_ids"][j][:n].tolist()
        ours = tok.decode(ids, skip_special_tokens=False)
        rest = r1_text[len(ours):] if r1_text.startswith(ours) else None
        check(f"{k}: trajectory text == R1's training conversation up to the policy's last token",
              rest is not None and rest.startswith("\n") and "<|im_start|>assistant" not in rest,
              f"{n} tokens" if rest is not None else f"mismatch at char {next((i for i, (x, y) in enumerate(zip(ours, r1_text)) if x != y), -1)}")
        gen = [x for x, m in zip(ids, row_mask.tolist()) if m]
        check(f"{k}: loss mask = exactly the sampled tokens ({len(gen)})", len(gen) == tr.gen_tokens)
        lp = batch["logprobs"][j][row_mask.bool()]
        vv = batch["versions"][j][row_mask.bool()]
        check(f"{k}: behaviour logprobs and versions carried", float(lp.max()) < 0 and int(vv.min()) >= 0,
              f"versions {sorted(set(vv.tolist()))}")
        ans = next((m["content"] for m in reversed(o["messages"]) if m["role"] == "assistant"), "")
        from agents import extract_answer
        sub = extract_answer(ans)
        direct = StepReward(tolerance=tol, n_steps=reward_steps(cfg), partial_ceiling=0.5).score(cfg, sub)
        exp = 0.0 if not o.get("n_cells") else direct.reward
        check(f"{k}: reward {o['reward']:.4f} = StepReward(c=0.5, S={reward_steps(cfg)}) with require-code ({exp:.4f})",
              abs(o["reward"] - exp) < 1e-12 and bool(o.get("passed")) == direct.passed)
    s0 = [j for j in range(6) if kind[j] == "S0/S1"]
    fb = [m["content"] for m in outs[s0[0]]["messages"] if m["role"] == "user"][1] if s0 else ""
    check("sandbox round trip: bare last expression displayed (GBP_ECHO_LAST=1)", "did_data.csv" in fb, fb[:80])
    s2 = [j for j in range(6) if kind[j] == "S2"]
    check("S2 (answer without code): passed but reward 0, and trained on with its advantage",
          bool(s2) and outs[s2[0]]["passed"] and outs[s2[0]]["reward"] == 0.0 and int(batch["loss_mask"][s2[0]].sum()) > 0)
    s5 = [j for j in range(6) if kind[j] == "S5"]
    if s5:
        tr5 = trajs[s5[0]]
        check("S5 cut-off: no EOS -> '<|im_end|>' appended as context (loss 0)",
              any(x == tok.convert_tokens_to_ids("<|im_end|>") and m == 0 for x, m in zip(tr5.ids, tr5.loss)))
    rw = batch["rewards"].tolist()
    check("row rewards = rl/grpo_train.advantages over the group (stubs 0)",
          all(abs(rw[j] - (advs[j] if int(batch["loss_mask"][j].sum()) else 0.0)) < 1e-6 for j in range(6)),
          f"{[round(x, 4) for x in rw]}")
    rows = [json.loads(line) for line in open(log_dir / "rollouts.jsonl")]
    check("rollouts.jsonl: one row per episode with versions and staleness fields",
          len(rows) == 6 and all("head_version" in r and "version_at_finish" in r for r in rows))

    print("[desync] LLMAgent._trim rewriting history is detected")
    from rl.async_pilot.episode import TokenLLMAgent
    from sandbox import Sandbox
    long_out = "```python\nprint('x' * 3000)\n```"
    right = f"```json\n{json.dumps({'answer': cfg['ground_truth']})}\n```"
    n_calls = {"n": 0}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def gen2(ids, n, temp, top_p):                 # 6 cells of 3000 chars each, then the answer
        n_calls["n"] += 1
        out = tok.encode(long_out if n_calls["n"] <= 6 else right, add_special_tokens=False) + [im_end]
        return out, [-0.1] * len(out), [0] * len(out), "stop"
    agent = TokenLLMAgent(model="x", base_url="http://x.invalid/v1", render="rl", gen=gen2,
                          tokenizer=tok, max_model_len=32768, history_char_budget=6000, wall_budget=0.0)
    agent.temperature = 1.0
    sb = Sandbox(data_dir=inst / "data_files", timeout=60)
    try:
        sub, transcript = agent.solve(cfg, sb)
    finally:
        sb.cleanup()
    check("trimmed history -> desync flagged, episode completed, not trainable",
          agent.traj.desync is not None and sub is not None and agent.traj.trainable_slice() is None,
          str(agent.traj.desync))
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
