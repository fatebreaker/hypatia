"""One R1 episode, token-faithful, with the model call routed to an asynchronous RL engine.

WHAT IS REUSED UNCHANGED (imported, never copied)
-------------------------------------------------
* harness/agents.py  LLMAgent.solve(): the whole multi-turn loop -- 25 turns, the answer-forcing tail, the stall abort,
  the repeated-error hint, the wall budget (RL_PLAN 2026-09-24 straggler cap), extract_code / extract_answer, and
  render_prompt(mode "rl"), the prompt format R1 samples in.
* harness/sandbox.py Sandbox: the notebook that replays every earlier cell on each exec; GBP_ECHO_LAST=1 display.
* rl/dispersion.py   one_rollout(): sandbox construction, StepReward(tolerance=instance_tol, n_steps, c) scoring,
  the stored conversation, and the error path (an exception -> reward 0, no messages, never trained on).
* rl/grpo_onpolicy.py rollouts(): R1's wrapper around one_rollout -- `--require-code` (reward 0 for an answer given
  without executing a cell) and `--per-instance-steps` (n_steps=None -> reward_steps(cfg) per instance).

WHAT IS NEW: TokenLLMAgent._chat
--------------------------------
R1's agent renders the conversation to text and POSTs it to vLLM /completions; vLLM re-tokenises the whole history
every turn and the trainer later re-tokenises the finished conversation (RL_TEMPLATE) to score it. An asynchronous
trainer needs more: the behaviour policy's log-prob of every sampled token and the policy VERSION that sampled it
(importance correction and staleness). So the model call goes to the engine token-in/token-out, and the agent keeps
the episode as one token sequence:

    context tokens (loss 0)  |  sampled tokens (loss 1, logp, version)  |  context tokens (loss 0) | ...

Each turn the harness renders the conversation exactly as in R1; the text it adds since the last call (tool output,
nudges, the forcing message, the next generation prompt) is tokenised and appended, and the sampled tokens are kept
VERBATIM (never decoded and re-encoded), so the policy is trained on exactly the tokens it produced in exactly the
context it produced them in. If the harness ever rewrites history (LLMAgent._trim elides old turns past 60k chars --
at a 16k-token server this essentially never happens) the episode continues exactly as in R1 on the re-tokenised
prompt, but it is marked `desync` and kept out of the gradient (its reward still enters its group, as R1's errored
rollouts do).

R1's context-overflow behaviour is reproduced exactly: LLMAgent._chat asks for max_tokens=8192 and halves it on each
HTTP 400 down to 512 (five attempts), then raises. vLLM returns 400 when prompt_tokens + max_tokens > max_model_len,
so the budget here is the first of 8192/4096/2048/1024/512 that fits, and when none fits the same RuntimeError is
raised -> one_rollout records error:RuntimeError, reward 0, no gradient.
"""
from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agents import IM_END, LLMAgent, render_prompt  # noqa: E402  (harness/agents.py, unchanged)
import rl.dispersion as _dispersion  # noqa: E402       (one_rollout, instance_tol, conversation; unchanged)
from rl import grpo_onpolicy as _grpo  # noqa: E402     (rollouts, reward_steps, family_of; unchanged)

# gen(prompt_ids, max_new_tokens, temperature, top_p) -> (output_ids, output_logprobs, output_versions, stop_reason)
GenFn = Callable[[list, int, float, float], tuple]

OVERFLOW_TAG = "maximum context length"          # the substring R1's overflow errors carry (vLLM's 400 text)


@dataclass
class TokenTrajectory:
    ids: list = field(default_factory=list)       # every token of the episode so far
    loss: list = field(default_factory=list)      # 1 = sampled by the policy, 0 = context the harness wrote
    logp: list = field(default_factory=list)      # behaviour log-prob of each sampled token (0.0 on context)
    ver: list = field(default_factory=list)       # policy version that sampled each token (-1 on context)
    text: str = ""                                # the text these ids stand for, used to find each turn's new context
    turns: int = 0
    gen_tokens: int = 0
    desync: str | None = None                     # why the episode can no longer be trained on as one sequence

    def trainable_slice(self) -> dict | None:
        """The sequence to train on: everything up to the policy's last sampled token (context after it carries no
        loss and no information), or None when nothing was sampled or the trajectory desynchronised."""
        if self.desync is not None or not any(self.loss):
            return None
        last = max(i for i, m in enumerate(self.loss) if m)
        return {"ids": self.ids[:last + 1], "loss": self.loss[:last + 1],
                "logp": self.logp[:last + 1], "ver": self.ver[:last + 1]}


@dataclass
class TokenLLMAgent(LLMAgent):
    """harness LLMAgent with one method replaced: the model call. Everything solve() does is inherited."""

    gen: GenFn | None = None
    tokenizer: Any = None
    max_model_len: int = 16384
    stop_ids: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        self.traj = TokenTrajectory()
        tok = self.tokenizer
        self._im_end_id = tok.convert_tokens_to_ids(IM_END)
        if not self.stop_ids:
            # R1 stops at "<|im_end|>" (payload stop) and at the generation config's EOS ids; for Qwen3 that is
            # {<|im_end|>, <|endoftext|>}.
            self.stop_ids = tuple(sorted({self._im_end_id, tok.convert_tokens_to_ids("<|endoftext|>")}))

    def budgets(self) -> list[int]:
        """max_tokens LLMAgent._chat tries, in order: it halves on every HTTP 400 (floor 512), five attempts."""
        b, out = self.max_tokens, [self.max_tokens]
        for _ in range(4):
            b = max(512, b // 2)
            out.append(b)
        return out

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _chat(self, messages: list[dict]) -> tuple[str, str]:
        text = render_prompt(messages, self.render)
        tr = self.traj
        if tr.desync is None and (not tr.ids or text.startswith(tr.text)):
            new = self._encode(text[len(tr.text):] if tr.ids else text)
            prompt, faithful = tr.ids + new, True
        else:
            if tr.desync is None:
                tr.desync = f"turn {tr.turns}: the harness rewrote history (prompt no longer extends the episode)"
            prompt, faithful = self._encode(text), False          # the model sees what R1's model would see
        budgets = self.budgets()
        budget = next((b for b in budgets if len(prompt) + b <= self.max_model_len), None)
        if budget is None:
            raise RuntimeError(
                f"HTTP 400 after {len(budgets)} attempts (max_tokens={budgets[-1]}): This model's {OVERFLOW_TAG} is "
                f"{self.max_model_len} tokens. However, you requested {len(prompt) + budgets[-1]} tokens "
                f"({len(prompt)} in the messages, {budgets[-1]} in the completion). [emulated: rl/async_pilot]")
        out, lp, ver, stop = self.gen(prompt, budget, self.temperature, self.top_p)
        out, lp, ver = list(out), list(lp), list(ver)
        if not (len(out) == len(lp) == len(ver)):
            raise RuntimeError(f"engine returned {len(out)} tokens, {len(lp)} logprobs, {len(ver)} versions")
        eos = bool(out) and out[-1] in self.stop_ids
        body = out[:-1] if eos else out
        # vLLM /completions detokenises with skip_special_tokens=True and no clean-up; so does this.
        reply = self.tokenizer.decode(body, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if faithful:
            tr.ids = prompt + out
            tr.loss += [0] * len(new) + [1] * len(out)
            tr.logp += [0.0] * len(new) + [float(x) for x in lp]
            tr.ver += [-1] * len(new) + [int(v) for v in ver]
            # render_prompt writes an assistant turn as "<generation prompt>{content}<|im_end|>\n": the text of the
            # ids is the prompt text plus the reply, plus "<|im_end|>" when the policy itself emitted it.
            tr.text = text + reply + (IM_END if eos and out[-1] == self._im_end_id else "")
        tr.turns += 1
        tr.gen_tokens += len(out)
        return reply, ("length" if stop == "length" else "stop")


# ---------------------------------------------------------------------------------------------------------------------
# one_rollout builds its agent with make_agent(spec). A spec "areal-token:<id>" hands it the TokenLLMAgent built for
# that episode; every other spec goes to harness make_agent as before. The attribute is replaced in THIS process's
# rl.dispersion module object only -- the file is not edited and no other process is affected.
_REG: dict[str, TokenLLMAgent] = {}
_REG_LOCK = threading.Lock()
_ORIG_MAKE_AGENT = _dispersion.make_agent
SPEC_PREFIX = "areal-token:"


def _make_agent(spec: str):
    if spec.startswith(SPEC_PREFIX):
        with _REG_LOCK:
            agent = _REG.pop(spec, None)
        if agent is None:
            raise RuntimeError(f"no registered agent for {spec}")
        return agent
    return _ORIG_MAKE_AGENT(spec)


if getattr(_dispersion.make_agent, "__name__", "") != "_make_agent":
    _dispersion.make_agent = _make_agent


def run_episode(inst: Path, gen: GenFn, tokenizer, *, temperature: float = 1.0, timeout: int = 180, c: float = 0.5,
                require_code: bool = True, wall_budget: float = 600.0, max_model_len: int = 16384,
                max_tokens: int = 8192) -> tuple[dict, TokenTrajectory]:
    """Exactly R1's per-rollout path: grpo_onpolicy.rollouts([inst], spec, k=1, temperature, timeout,
    n_steps=None (per-instance S), workers=1, c, require_code) -> one_rollout -> LLMAgent.solve -> StepReward.
    Returns (R1's rollout record, the token trajectory)."""
    agent = TokenLLMAgent(model="areal-policy", base_url="http://areal-engine.invalid/v1", api_key="unused",
                          render="rl", wall_budget=float(wall_budget), max_tokens=int(max_tokens),
                          gen=gen, tokenizer=tokenizer, max_model_len=int(max_model_len))
    spec = f"{SPEC_PREFIX}{uuid.uuid4().hex}"
    with _REG_LOCK:
        _REG[spec] = agent
    try:
        groups = _grpo.rollouts([Path(inst)], spec, 1, temperature, timeout, None, 1, c, require_code)
    finally:
        with _REG_LOCK:
            _REG.pop(spec, None)
    out = groups[Path(inst).name][0]
    return out, agent.traj


def is_infrastructure_error(out: dict) -> bool:
    """An errored rollout that is NOT the policy overflowing its context (a dead engine, a broken sandbox)."""
    return str(out.get("attribution") or "").startswith("error") and OVERFLOW_TAG not in str(out.get("error_msg", ""))
