"""anthropic_agent.py (2026-10-25; the user supplied an Anthropic API key and $180 to score a non-OpenAI frontier model on every
benchmark) -- the harness's LLMAgent with ONE change: _chat goes to the Anthropic Messages API through the official SDK. The loop,
system prompt, task prompt, history trimming, forcing tail, sandbox and grader are LLMAgent's, exactly as every other system.

Spec 'claude:MODEL' (harness/agents.py make_agent). Settings, all from the environment the launcher (scripts/eval_anthropic.sh) sets:
  GBP_ANTHROPIC_EFFORT      output_config.effort (default 'high': every API model in the paper is compared at high effort)
  GBP_ANTHROPIC_MAX_TOKENS  per-call output ceiling, thinking included (default 32000; streamed, so no HTTP timeout)
  thinking                  not sent: adaptive thinking is the default on current models (Claude Opus 5.5 cannot disable it)
  caching                   top-level cache_control {'type': 'ephemeral'}: each turn re-reads the cached conversation prefix
  fallbacks                 none -- a fallback model would answer some problems; the comparison is of ONE model, so a refusal is
                            kept and scored like any other unanswered problem
  ledger                    every call's usage priced into GBP_USAGE_LEDGER (GBP_PRICE_IN / _CACHE_WRITE / _CACHED / _OUT, $ per 1M
                            tokens); agents._budget_check stops the run once the ledger reaches GBP_BUDGET_USD
The SDK is installed apart from the project venv (GBP_ANTHROPIC_LIB, default /scratch4/.../pylib_anthropic) and put on this
process's path only, so the sandbox kernels that run the model's code never see it."""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from dataclasses import dataclass, field

_LIB = os.environ.get("GBP_ANTHROPIC_LIB", "/scratch4/workspace/hieutran_umass_edu-deeprs-bench/pylib_anthropic")
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import anthropic  # noqa: E402

from agents import LLMAgent, _budget_check  # noqa: E402


def _record(model: str, rec: dict) -> None:
    led = os.environ.get("GBP_USAGE_LEDGER")
    if not led:
        return
    rec = {"t": round(time.time(), 1), "model": model, **rec}
    with open(led, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(rec) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)


def _price(usage) -> tuple[dict, float]:
    p = lambda k: float(os.environ.get(k, "0"))  # noqa: E731
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    tr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    usd = (tin * p("GBP_PRICE_IN") + tw * p("GBP_PRICE_CACHE_WRITE") + tr * p("GBP_PRICE_CACHED") + tout * p("GBP_PRICE_OUT")) / 1e6
    return {"in": tin, "cache_write": tw, "cached": tr, "out": tout}, usd


@dataclass
class AnthropicAgent(LLMAgent):
    base_url: str = "https://api.anthropic.com"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    name: str = "claude"

    def __post_init__(self):
        if not self.api_key:
            raise RuntimeError("claude: ANTHROPIC_API_KEY is not set (scripts/eval_anthropic.sh reads it from the key file)")
        self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=4, timeout=900.0)

    def _chat(self, messages: list[dict]) -> tuple[str, str]:
        """Return (content, finish_reason) as LLMAgent._chat does: 'length' when the output ceiling cut the reply, 'refusal'
        when the model declined, else 'stop'. Thinking is not returned or replayed (the harness keeps visible text only, as
        for every other system); an empty assistant turn is sent back as '(no output)', which the API requires."""
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
        msgs = [{"role": m["role"], "content": m["content"] if (m.get("content") or "").strip() else "(no output)"}
                for m in messages if m["role"] != "system"]
        max_out = int(os.environ.get("GBP_ANTHROPIC_MAX_TOKENS", "32000"))
        kw = dict(model=self.model, max_tokens=max_out, messages=msgs, cache_control={"type": "ephemeral"},
                  output_config={"effort": os.environ.get("GBP_ANTHROPIC_EFFORT", "high")})
        if system:
            kw["system"] = system
        _budget_check()
        t0 = time.time()
        try:
            with self._client.messages.stream(**kw) as stream:
                msg = stream.get_final_message()
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # no usage comes back: record a WORST-CASE estimate (the whole output ceiling) so the budget stop counts it
            usd = max_out * float(os.environ.get("GBP_PRICE_OUT", "0")) / 1e6
            _record(f"{self.model}~failed-est", {"in": 0, "cache_write": 0, "cached": 0, "out": max_out, "usd": round(usd, 6),
                                                 "estimated": True, "kind": type(e).__name__, "elapsed_s": round(time.time() - t0, 1)})
            print(f"[api-failure] {self.model} {type(e).__name__} after {time.time() - t0:.0f}s: {str(e)[:200]!r}",
                  file=sys.stderr, flush=True)
            raise RuntimeError(f"Anthropic API call failed: {type(e).__name__}: {str(e)[:300]}") from e
        tok, usd = _price(msg.usage)
        _record(self.model, {**tok, "usd": round(usd, 6), "stop": msg.stop_reason, "s": round(time.time() - t0, 1),
                             "id": getattr(msg, "_request_id", None)})
        text = "".join(b.text for b in msg.content if b.type == "text")
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            print(f"[refusal] {self.model} category={cat}", file=sys.stderr, flush=True)
            return text, "refusal"
        return text, ("length" if msg.stop_reason == "max_tokens" else "stop")
