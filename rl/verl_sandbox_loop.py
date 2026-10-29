"""verl agent_loop adapter: the GeneBench sandbox as an RL environment.

WHY THIS EXISTS ALONGSIDE rl/grpo_train.py
------------------------------------------
`rl/grpo_train.py` is a ~300-line hand-rolled GRPO. It passes its own smoke
assertions, but "I wrote a GRPO loop and it looked right" is weak provenance
for a number in a paper — and this project's entire method is distrusting
self-checked numbers. verl owns the optimizer, the clipped objective, batching
and the distributed path; this file supplies only the environment, which is the
part that genuinely has to be ours.

The plan is two-stage: the hand-rolled pilot answers *is there signal at all*
cheaply (if held-out scRNA stays at 0.022, nothing here is worth running), and
this produces the number that someone who distrusts my 300 lines can reproduce.

WHAT verl WANTS BACK
--------------------
    AgentLoopOutput.response_ids    every token after the prompt, INCLUDING
                                    tool output
    AgentLoopOutput.response_mask   1 = model generated it, 0 = the sandbox did
    AgentLoopOutput.reward_score    terminal, from rl/reward.py

That `response_mask` is exactly the assistant-only masking `sft_train.py`
documents and `grpo_train.py` recomputes by hand. Here it is maintained
incrementally, which is both cheaper and harder to get silently wrong: tokens
are appended with their provenance at the moment they are produced, rather than
re-derived afterwards from a chat template that must agree with how the
conversation was actually built.

WHY NOT ToolAgentLoop
---------------------
verl ships `tool_agent_loop`, but it assumes the model emits structured tool
CALLS. Our protocol — and both SFT corpora, and every trace in `rl/traces_*` —
is a ```python fenced block executed in a notebook-like sandbox. Switching to
tool-calling would invalidate both adapters and every trace. So this subclasses
`AgentLoopBase` directly and keeps the protocol the policy was trained on.

THE SANDBOX IS STATEFUL AND MUST BE PER-EPISODE
-----------------------------------------------
`Sandbox` keeps a working directory and replays prior cells, so it is episode
state, not a shared service. It is created inside `run()` and always cleaned up
— a leaked sandbox is a leaked temp dir per rollout, and RL does tens of
thousands of rollouts.
"""
from __future__ import annotations

import asyncio
import json
import random as _random
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

from verl.experimental.agent_loop.agent_loop import (  # noqa: E402
    AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register,
)
from verl.utils.profiler import simple_timer  # noqa: E402
from verl.utils.rollout_trace import rollout_trace_op  # noqa: E402

_rng = _random.Random()
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("genebench_sandbox_agent")
class GeneBenchSandboxAgentLoop(AgentLoopBase):
    """Multi-turn code execution against a GeneBench-Pro style instance.

    One turn = the model emits a ```python block, the sandbox runs it, stdout
    comes back as a user message. The episode ends when the model emits a final
    JSON answer, runs out of turns, or stalls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # MUST MATCH THE EVAL HARNESS (harness/agents.py default 25). This was
        # 12 while eval allowed 25, and scRNA episodes average 21 turns -- most
        # training rollouts were cut off mid-analysis and scored as failures.
        # Step-1 reward read 0.0095 against the 0.298 the same adapter scores
        # under the eval harness. cis-MVMR hid it by finishing in 4-6 turns.
        cfg = getattr(self.config.actor_rollout_ref.rollout, "genebench", None)
        self.max_turns = int(getattr(cfg, "max_turns", 25) if cfg else 25)
        self.cell_timeout = int(getattr(cfg, "cell_timeout", 180) if cfg else 180)

        # WHY THIS EXISTS. Two training runs hung with the identical signature:
        # `pending: 0, running: 1, finished: 7` repeating for over an hour with
        # both GPUs at 0 %. Exactly one rollout of the batch never returned, and
        # because verl waits for the whole batch, one stuck episode stalls the
        # entire run -- the first cost 69 minutes and 12 unsaved steps.
        #
        # Neither `await` in the turn loop was bounded: `server_manager.generate`
        # can block forever if a generation request is lost, and `to_thread(
        # sb.run, ...)` inherits no deadline from the sandbox's own per-cell
        # limit if the worker thread wedges. A rollout that breaks out early is
        # scored with submission=None -- a zero-reward sample, which is a cheap
        # loss. A rollout that never returns is not.
        self.generate_timeout = float(
            os.environ.get("GBP_GENERATE_TIMEOUT",
                           getattr(cfg, "generate_timeout", 600) if cfg else 600))
        self.exec_timeout = float(
            os.environ.get("GBP_EXEC_TIMEOUT", self.cell_timeout + 60))
        self._last_prompt_ids: list[int] | None = None
        self.episode_budget = float(
            os.environ.get("GBP_EPISODE_BUDGET",
                           getattr(cfg, "episode_budget", 2400) if cfg else 2400))

        # Thinking is left at verl's default (ON), which is the mode the policy
        # is good in: 28.5% solved on held-out scRNA versus 5.6% with it off.
        #
        # It was forced OFF while this loop appended tokens by hand, because
        # <think> blocks then persisted in the context where the chat template
        # would have stripped them. Re-rendering history through the template
        # removes that constraint -- verl's continuous-token builder handles
        # the boundaries, so prior turns are rendered exactly as the eval path
        # renders them.
        # Reserve the tail of the budget for actually answering. Without this
        # the loop runs out mid-analysis and the grader sees nothing, which is
        # strictly less informative than a wrong answer (FINDINGS C5).
        self.force_answer_turns = int(
            getattr(cfg, "force_answer_turns", 3) if cfg else 3)
        self.partial_ceiling = float(
            getattr(cfg, "partial_ceiling", 0.5) if cfg else 0.5)

        # THINKING MODE STAYS ON -- verl's default, and the mode the policy is
        # actually good in.
        #
        # Qwen3's template defaults to thinking enabled. Every evaluation used
        # "llm:MODEL:nothink" (enable_thinking=False), and that mismatch is what
        # made RL look broken: the SAME checkpoint scores 0.992 with thinking on
        # and 0.505 with it off on held-out cis-MVMR, and solves the real
        # published problem to five decimals in one mode and not at all in the
        # other.
        #
        # I first "fixed" this by forcing TRAINING to non-thinking to match the
        # eval. That was backwards -- it matched them by degrading training to
        # the worse mode, and step-1 reward fell to 0.010 against the 0.298 the
        # same adapter scores with thinking on. The evaluation was what needed
        # changing.

    # ------------------------------------------------------------------ utils
    async def _encode(self, messages: list[dict]) -> list[int]:
        return await self.apply_chat_template(messages)

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # -------------------------------------------------------------------- run
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Hard deadline around the WHOLE episode.

        Per-await guards were not enough. Three runs have hung with the same
        signature -- `pending: 0, running: 1, finished: 7` repeating while both
        GPUs sit at 0 % -- and the last one held a card for 4.5 hours without
        either guard firing, because the two continuous-token merges and the
        prompt encode were still unbounded. verl waits for the whole batch, so
        ONE unbounded await stalls the entire run.
        
        Bounding the episode from the outside is the only version of this that
        cannot be defeated by adding another await inside: whatever hangs, the
        rollout is abandoned and scored zero. A lost sample is cheap; a stalled
        run is not.
        """
        try:
            return await asyncio.wait_for(
                self._run_episode(sampling_params, **kwargs),
                timeout=self.episode_budget + 300)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("rollout %s: ABANDONED after %.0fs -- the episode "
                           "exceeded its hard deadline",
                           kwargs.get("instance_dir", "?"),
                           self.episode_budget + 300)
            return self._abandoned_output(kwargs)

    def _abandoned_output(self, kwargs: dict) -> AgentLoopOutput:
        """A syntactically valid, zero-reward rollout for an abandoned episode."""
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", 0) or 0
        prompt_ids = self._last_prompt_ids or [pad]
        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=[pad],
            response_mask=[0],
            response_logprobs=None,
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(),
            extra_fields={"instance": Path(kwargs.get("instance_dir", "?")).name,
                          "timed_out": "abandoned", "passed": False,
                          "fails_at_step": None, "attribution": "abandoned"},
        )

    async def _run_episode(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from agents import SYSTEM, build_prompt, extract_answer, strip_reasoning
        from rl.reward import StepReward
        from sandbox import Sandbox

        metrics: dict = {}
        # verl passes the dataset row through; the instance directory is what
        # identifies which generated problem this rollout is solving.
        inst = Path(kwargs["instance_dir"])
        cfg = json.loads((inst / "eval_config.json").read_text())
        n_steps = kwargs.get("n_steps") or self._n_steps(cfg)
        tol = kwargs.get("tolerance") or self._tolerance(cfg)

        # ROLLOUT INSTRUMENTATION.
        #
        # Training scored 0 solves in 336 rollouts while the eval harness
        # scored 5.6% on the same weights, and the cause -- every cell dying on
        # "import statsmodels" because the Ray interpreter lacked it -- took a
        # day to find because rewards were the only thing being read. Cell
        # stderr would have said ModuleNotFoundError immediately.
        #
        # Sampled, not universal: writing every episode of every step would be
        # gigabytes. GBP_TRACE_FRAC controls the rate.
        trace_sink = None
        trace_frac = float(os.environ.get("GBP_TRACE_FRAC", "0.02"))
        trace_dir = os.environ.get("GBP_TRACE_DIR")
        if trace_dir and trace_frac > 0 and _rng.random() < trace_frac:
            trace_sink = []

        # to_thread: BixBench instances carry up to 1.1 GB of data files, and
        # any synchronous filesystem work here blocks the event loop for every
        # other rollout -- the cause of three hangs that an outer asyncio
        # deadline could not interrupt. link_data hard-links instead of
        # copying, which is what makes a 12 GB benchmark affordable per step.
        sb = await asyncio.to_thread(
            Sandbox, data_dir=inst / "data_files",
            timeout=self.cell_timeout, link_data=True)
        try:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user",
                         "content": build_prompt(cfg["task"], sb.listing())}]
            prompt_ids = await self._encode(messages)
            self._last_prompt_ids = list(prompt_ids)
            # `runtime` is the full token stream the policy conditions on. With
            # continuous-token enabled it is maintained by verl's builder, which
            # RE-RENDERS non-assistant turns through the chat template -- the
            # whole point of this rewrite. The previous version appended
            # hand-built turn separators, and six separate train/eval
            # divergences followed from that one shortcut (RESULTS: 0 solves in
            # 336 training rollouts against 5.6% under the eval harness).
            runtime = list(prompt_ids)
            response_mask: list[int] = []
            response_logprobs: list[float] = []
            versions: dict[str, int] = {}
            submission = None
            assistant_turns = 0

            started = time.monotonic()
            timed_out = None
            for turn in range(self.max_turns):
                if len(response_mask) >= self.response_length:
                    break
                if time.monotonic() - started > self.episode_budget:
                    timed_out = "episode_budget"
                    break
                try:
                    with simple_timer("generate_sequences", metrics):
                        out = await asyncio.wait_for(
                            self.server_manager.generate(
                                request_id=f"{inst.name}-{turn}",
                                prompt_ids=runtime,
                                sampling_params=sampling_params,
                            ),
                            timeout=self.generate_timeout,
                        )
                except (asyncio.TimeoutError, TimeoutError):
                    # Nothing useful can follow a lost generation, so end the
                    # episode here rather than blocking the batch.
                    timed_out = "generate"
                    logger.warning("rollout %s: generation timed out after %.0fs "
                                   "on turn %d", inst.name, self.generate_timeout,
                                   turn)
                    break
                gen = list(out.token_ids)
                self._track_versions(versions, getattr(out, "extra_fields", None))
                assistant_turns += 1

                if self.enable_continuous_token:
                    merge, response_mask, lp = await self.ct_merge_assistant_token(
                        runtime, gen, response_mask,
                        response_logprobs if (response_logprobs or out.log_probs) else None,
                        assistant_logprobs=out.log_probs if out.log_probs else None,
                    )
                    runtime = merge.token_ids
                    if lp is not None:
                        response_logprobs = lp
                else:
                    runtime += gen
                    response_mask += [1] * len(gen)
                    if out.log_probs:
                        response_logprobs += list(out.log_probs)

                reply = self.tokenizer.decode(gen, skip_special_tokens=True)
                clean = strip_reasoning(reply)
                messages.append({"role": "assistant", "content": reply})

                ans = extract_answer(clean)
                if isinstance(ans, dict) and ans:
                    submission = ans
                    break

                code = self._code_block(clean)
                if code is None:
                    obs = ("No ```python block and no final JSON answer. Emit "
                           "one code block, or your final answer as JSON.")
                else:
                    try:
                        res = await asyncio.wait_for(
                            asyncio.to_thread(sb.run, code),
                            timeout=self.exec_timeout)
                    except (asyncio.TimeoutError, TimeoutError):
                        # The sandbox has its own per-cell limit; reaching this
                        # one means the worker thread itself is wedged, so stop
                        # waiting on it. The thread is abandoned, not joined --
                        # a leaked thread is survivable, a hung batch is not.
                        timed_out = "exec"
                        logger.warning("rollout %s: sandbox exec exceeded %.0fs "
                                       "on turn %d", inst.name, self.exec_timeout,
                                       turn)
                        break
                    obs = res.as_feedback()
                    if trace_sink is not None:
                        trace_sink.append({
                            "turn": turn, "code": code[:2000],
                            "stdout": (res.stdout or "")[:2000],
                            "stderr": (res.stderr or "")[:2000],
                        })

                if turn >= self.max_turns - self.force_answer_turns:
                    obs += ("\n\nTurns are nearly exhausted. Emit your final "
                            "JSON answer now.")

                previous = list(messages)
                messages.append({"role": "user", "content": obs})

                if self.enable_continuous_token:
                    merge, response_mask, lp = await self.ct_merge_non_assistant_msg(
                        previous, messages, runtime, response_mask,
                        response_logprobs if response_logprobs else None,
                    )
                    runtime = merge.token_ids
                    if response_logprobs:
                        response_logprobs = lp or []
                else:
                    # Fallback only. The opener must match what the template
                    # emits for this mode or turn 1 and turn 2+ disagree.
                    tail = self._tok(
                        f"<|im_start|>user\n{obs}<|im_end|>\n"
                        f"<|im_start|>assistant\n")
                    runtime += tail
                    response_mask += [0] * len(tail)
                    if response_logprobs:
                        response_logprobs += [0.0] * len(tail)

            if trace_sink is not None:
                try:
                    d = Path(trace_dir); d.mkdir(parents=True, exist_ok=True)
                    (d / f"{inst.name}-{os.getpid()}-{len(trace_sink)}.json").write_text(
                        json.dumps({"instance": inst.name,
                                    "submission": submission,
                                    "assistant_turns": assistant_turns,
                                    "n_cells": sb.attempts,
                                    "n_cell_failures": sb.failures,
                                    "cells": trace_sink}, indent=1))
                except Exception:                                   # noqa: BLE001
                    pass

            score = StepReward(tolerance=tol, n_steps=n_steps,
                               partial_ceiling=self.partial_ceiling
                               ).score(cfg, submission)

            response_ids = runtime[len(prompt_ids):]
            assert len(response_ids) == len(response_mask), (
                f"mask/id length mismatch {len(response_ids)} vs "
                f"{len(response_mask)} -- credit would be assigned to the wrong "
                f"tokens")
            assert not response_logprobs or len(response_logprobs) == len(response_mask), (
                f"logprob/mask length mismatch {len(response_logprobs)} vs "
                f"{len(response_mask)} -- the rollout-vs-actor divergence check "
                f"would compare unrelated positions and look like drift")
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=(response_logprobs[: self.response_length]
                                   if response_logprobs else None),
                reward_score=float(score.reward),
                num_turns=assistant_turns * 2 + 1,
                metrics=AgentLoopMetrics(**metrics) if metrics else AgentLoopMetrics(),
                extra_fields={
                    **versions,
                    "instance": inst.name,
                    "timed_out": timed_out or "",
                    "passed": bool(score.passed),
                    "fails_at_step": score.fails_at_step,
                    "attribution": score.attribution,
                    "n_cells": sb.attempts,
                    "n_cell_failures": sb.failures,
                    # Zero cells means the model answered without computing --
                    # a guess. runner.py flags these and they must not be read
                    # as capability (FINDINGS: 'answered-without-code').
                    "answered_without_code": sb.attempts == 0,
                },
            )
        finally:
            sb.cleanup()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _track_versions(versions: dict, extra: dict | None) -> None:
        """Fold one generation's policy-version stamps into the episode's.

        min = the oldest weights any turn was served by, max = the newest.
        """
        if not extra:
            return
        lo = extra.get("min_global_steps")
        if lo is not None:
            versions["min_global_steps"] = min(
                versions.get("min_global_steps", lo), int(lo))
        hi = extra.get("max_global_steps")
        if hi is not None:
            versions["max_global_steps"] = max(
                versions.get("max_global_steps", hi), int(hi))

    @staticmethod
    def _code_block(text: str) -> str | None:
        import re
        m = re.search(r"```python[ \t]*\r?\n(.*?)```", text, re.DOTALL)
        return m.group(1) if m else None

    @staticmethod
    def _n_steps(cfg: dict) -> int:
        """Pipeline depth from the instance: cis-MVMR has 5 stages, scRNA 3.

        A fixed value mis-scales every attributed reward on the other family --
        the bug that made the first transfer run score the wrong thing.
        """
        abl = (cfg.get("_provenance") or {}).get("ablations") or {}
        steps = [v.get("fails_at_step") for v in abl.values() if v.get("fails_at_step")]
        return max(steps, default=5)

    @staticmethod
    def _tolerance(cfg: dict) -> float:
        """Grading tolerance from the instance; it differs by family."""
        kc = ((cfg.get("grader") or {}).get("config") or {})
        keys = kc.get("keys") or {}
        tols = [v.get("absolute_tolerance") for v in keys.values()
                if isinstance(v, dict) and v.get("absolute_tolerance")]
        if not tols and kc.get("absolute_tolerance"):
            tols = [kc["absolute_tolerance"]]
        return float(min(tols)) if tols else 0.025
