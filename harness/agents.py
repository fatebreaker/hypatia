"""Agents for the GeneBench-Pro harness.

Protocol is deliberately plain text, not native tool-calling: the model writes
```python fenced blocks, we execute them and feed stdout back. Native tool
calling would be tidier with frontier APIs but excludes local/base models
served through vLLM, which is exactly where the RL work has to run.

Agents:
  oracle  -- submits ground truth. Harness self-test; must score 10/10.
  null    -- submits zeros. Must score ~0/10.
  llm     -- OpenAI-compatible chat endpoint (works for vLLM and hosted APIs).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from sandbox import Sandbox

CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning-model scratchpad before parsing.

    Reasoning models (Qwen3, R1-distills) emit <think>...</think>. Parsing that
    region is actively harmful: a number the model *hypothesised* mid-reasoning
    gets extracted and submitted as the final answer, which is how a run can
    score non-zero while executing zero code.

    An unclosed <think> means the trace was truncated by max_tokens -- there is
    no actionable output, so return "" and let the caller nudge rather than
    scavenge a guess out of half-finished reasoning.
    """
    if not text:
        return ""
    cleaned = THINK_RE.sub("", text)
    if THINK_OPEN.search(cleaned):          # opened but never closed
        return cleaned[: THINK_OPEN.search(cleaned).start()]
    return cleaned

SYSTEM = """You are an expert computational biologist and statistician.

You solve a data-analysis problem by writing Python. Rules:

1. To run code, emit exactly one ```python fenced block. You will receive its
   stdout and stderr. State persists across blocks like a notebook.
2. Inspect the data before modelling it. These datasets contain deliberate
   irregularities -- batch effects, ascertainment, winner's curse, ambient
   contamination, pseudogene mismapping, LD structure. Diagnose before you
   estimate, and let what you find change your estimator.
3. Available: numpy, scipy, pandas, statsmodels, scikit-learn. Files are under
   data_files/ in the working directory.
4. When confident, emit your final answer as ONE JSON object in a ```json
   block, with no code block in that same message.

Do not guess numbers. Compute them."""

# A domain-neutral variant, opt-in with GBP_SYSTEM=neutral (second referee round: the default names genomics artefacts even
# on CauSciBench's social-science questions). Same rules; only the role and the list of irregularities change.
SYSTEM_NEUTRAL = """You are an expert statistician and data analyst.

You solve a data-analysis problem by writing Python. Rules:

1. To run code, emit exactly one ```python fenced block. You will receive its
   stdout and stderr. State persists across blocks like a notebook.
2. Inspect the data before modelling it: the study design, how each variable is
   coded, missing or irregular values. Diagnose before you estimate, and let what
   you find change your estimator.
3. Available: numpy, scipy, pandas, statsmodels, scikit-learn. Files are under
   data_files/ in the working directory.
4. When confident, emit your final answer as ONE JSON object in a ```json
   block, with no code block in that same message.

Do not guess numbers. Compute them."""
# SCIENCE-NEUTRAL, opt-in with GBP_SYSTEM=science (docs/DIRECTIONS.md "Measurement changes" 7, 2026-09-24): for NEW RL arms
# only (A1/A2), never for running conditions. The default names genomics artefacts on every family, a content prior toward
# one science in every unseen-domain test; "neutral" above names "the study design" (a causal/survey prior). This one names
# no science, no artefact and no hazard class -- only the generic habit of inspecting the data -- so it does not tell the
# policy about the hazard modes A1 trains against. Same rules 1, 3, 4 as the default.
SYSTEM_SCIENCE = """You are an expert scientist and data analyst.

You solve a data-analysis problem by writing Python. Rules:

1. To run code, emit exactly one ```python fenced block. You will receive its
   stdout and stderr. State persists across blocks like a notebook.
2. Inspect the data before modelling it: what each file and column records, how
   values are coded, and anything missing, irregular or surprising. Diagnose
   before you estimate, and let what you find change your estimator.
3. Available: numpy, scipy, pandas, statsmodels, scikit-learn. Files are under
   data_files/ in the working directory.
4. When confident, emit your final answer as ONE JSON object in a ```json
   block, with no code block in that same message.

Do not guess numbers. Compute them."""
SYSTEMS = {"default": SYSTEM, "neutral": SYSTEM_NEUTRAL, "science": SYSTEM_SCIENCE}


def extract_code(text: str) -> str | None:
    m = CODE_RE.findall(strip_reasoning(text))
    return m[0] if m else None


def extract_answer(text: str) -> dict[str, Any] | None:
    """Pull the submission JSON out of a model message.

    Models wrap JSON in fences despite being told not to, emit prose around it,
    or produce several objects. Try fenced blocks first, then any balanced
    object, preferring ones that look like a submission.
    """
    text = strip_reasoning(text)
    if not text:
        return None
    # Walk fenced blocks once, keeping the language tag. Two things depend on
    # getting this right, and a `(?:json)?` pattern gets both wrong when a
    # ```python block precedes the answer: it pairs the CLOSING fence of the
    # code block with the OPENING fence of the json block and captures the
    # prose between them.
    cands: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"```([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", text, re.DOTALL):
        spans.append(m.span())
        if m.group(1).lower() in ("", "json"):
            cands.append(m.group(2))

    # The bare-object scan below must run over PROSE ONLY, never over code.
    # A model writing ordinary Python emits dict literals -- `results = {}` --
    # and json.loads happily turns `{}` into a valid, empty submission. The
    # episode then ends on turn 1 having executed nothing: observed as
    # n_cells=0, submission={}, score 0, on a model that had in fact written a
    # correct-looking analysis. Same family as C1: never harvest an answer from
    # something the model was merely writing rather than submitting.
    prose_parts, last = [], 0
    for a, b in spans:
        prose_parts.append(text[last:a])
        last = b
    prose_parts.append(text[last:])
    prose = " ".join(prose_parts)
    depth, start = 0, None
    for i, ch in enumerate(prose):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(prose[start : i + 1])
                start = None
    best = None
    for c in reversed(cands):
        try:
            obj = json.loads(c.strip())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "answer" in obj and isinstance(obj["answer"], dict):
            return obj
        best = best or obj
    return best


COT_SUFFIX = """
Work as a careful analyst:
1. Inspect the data before choosing a method -- check shapes, dtypes, missingness
   and any obvious quality problems.
2. State which analysis is appropriate and why, before running it.
3. Run it, then VERIFY the result (sanity bounds, a second estimator, or a
   residual check). If the check fails, revise and re-run.
4. Only then report the final answer in the required JSON form.
"""

FEWSHOT_SUFFIX = COT_SUFFIX + """
Worked example of the expected shape of a solution (a DIFFERENT problem):

    # 1. inspect
    df = pd.read_csv("data_files/counts.csv"); print(df.shape, df.isna().sum())
    # 2. choose: paired design with a batch covariate -> mixed model, not a t-test
    # 3. run
    m = smf.mixedlm("y ~ treat + batch", df, groups=df["subject"]).fit()
    # 4. verify: effect should survive dropping the largest batch
    chk = smf.mixedlm("y ~ treat + batch", df[df.batch != "B3"],
                      groups=df["subject"]).fit()
    print(m.params["treat"], chk.params["treat"])   # agree to ~2 decimals

    {"answer": {"<key>": 0.42}}
"""


ICL_BRIDGE = "That task is complete. Here is a new task of the same kind, with new data in data_files/.\n\n"


def build_prompt(task: str, listing: str, style: str | None = None) -> str:
    extra = ""
    if style == "cot":
        extra = COT_SUFFIX
    elif style == "fewshot":
        extra = FEWSHOT_SUFFIX
    return f"{task}\n\nFiles in your working directory:\n{listing}\n{extra}"


# --------------------------------------------------------------------------- #
# reference agents -- validate the harness without spending tokens
# --------------------------------------------------------------------------- #
class OracleAgent:
    """Submits ground truth. If this is not 10/10 the harness is broken."""

    name = "oracle"

    def solve(self, cfg: dict, sandbox: Sandbox) -> tuple[dict, list]:
        return {"answer": dict(cfg["ground_truth"]), "reasoning": "oracle"}, []


class NullAgent:
    """Submits zeros / empty strings. Establishes the floor."""

    name = "null"

    def solve(self, cfg: dict, sandbox: Sandbox) -> tuple[dict, list]:
        ans = {}
        for k, v in cfg["ground_truth"].items():
            ans[k] = "" if isinstance(v, str) else 0
        return {"answer": ans, "reasoning": "null baseline"}, []


# --------------------------------------------------------------------------- #
# Prompt rendering that does not depend on the inference server
# --------------------------------------------------------------------------- #
# WHY. Our adapters were trained on conversations rendered by sft_train's
# TRAIN_TEMPLATE, whose assistant turns start directly after
# "<|im_start|>assistant\n". Qwen3's own template, which the server applies to
# every request, puts an empty "<think>\n\n</think>\n\n" before the reply when
# thinking is off. Older vLLM served each LoRA with the chat template saved in
# its adapter directory (TRAIN_TEMPLATE); vLLM 0.28 uses the base model's for
# all of them, and the ten-family SFT+RL model -- 0.994 on the public ten under
# the old serving -- scored 0.013 behind those four extra tokens, skipping its
# trained first turn and improvising code. Rendering here and calling
# /completions pins the prompt to the training format, whatever the server does.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
NOTHINK = "<think>\n\n</think>\n\n"


def render_prompt(messages: list[dict], mode: str) -> str:
    """Render a conversation plus the generation prompt, byte-for-byte.

    "train": sft_train.TRAIN_TEMPLATE -- how every SFT adapter was trained.
    "rl"   : the same, but EVERY assistant turn (and the generation prompt)
             carries Qwen3's empty no-think block. This is the base model's
             native no-think format at the generation point, and because past
             turns carry it too, the whole episode renders as one sequence in
             which each turn was sampled in exactly the context it is later
             scored in (rl/grpo_onpolicy.py's RL_TEMPLATE).
    """
    if mode in ("mtrain", "ltrain"):          # Mistral / Llama-3 families: rl/chat_formats.py
        import sys as _s
        from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from rl.chat_formats import render_llama3, render_mistral
        return render_mistral(messages) if mode == "mtrain" else render_llama3(messages)
    pre = NOTHINK if mode == "rl" else ""
    if mode not in ("train", "rl"):
        raise ValueError(f"unknown render mode {mode!r}")
    out = []
    for m in messages:
        if m["role"] == "assistant":
            out.append(f"{IM_START}assistant\n{pre}{m['content']}{IM_END}\n")
        else:
            out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(out) + f"{IM_START}assistant\n{pre}"


# --------------------------------------------------------------------------- #
# LLM agent
# --------------------------------------------------------------------------- #
_BASE_URL_RR = itertools.count()


def _budget_check() -> None:
    """GBP_USAGE_LEDGER + GBP_BUDGET_USD (2026-09-30): refuse the call once the ledger's dollar total reaches the budget.
    Unset = no accounting (every local-vLLM run is unchanged)."""
    led, cap = os.environ.get("GBP_USAGE_LEDGER"), os.environ.get("GBP_BUDGET_USD")
    if not led or not cap:
        return
    spent = 0.0
    try:
        with open(led) as f:
            for line in f:
                try: spent += float(json.loads(line).get("usd") or 0)
                except Exception: pass                      # noqa: BLE001
    except FileNotFoundError:
        pass
    if spent >= float(cap):
        raise RuntimeError(f"API budget exhausted: ${spent:.4f} spent of ${float(cap):.4f} (GBP_BUDGET_USD)")


def _budget_record(model: str, usage: dict | None, tier: str | None = None) -> None:
    """Append one call's usage and cost to GBP_USAGE_LEDGER (flock). Prices in $ per 1M tokens: GBP_PRICE_IN (uncached
    prompt), GBP_PRICE_CACHED (cached prompt), GBP_PRICE_OUT (completion incl. reasoning)."""
    led = os.environ.get("GBP_USAGE_LEDGER")
    if not led or not usage:
        return
    import fcntl
    pin = float(os.environ.get("GBP_PRICE_IN", "0")); pc = float(os.environ.get("GBP_PRICE_CACHED", "0"))
    pout = float(os.environ.get("GBP_PRICE_OUT", "0"))
    tin = int(usage.get("prompt_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    tout = int(usage.get("completion_tokens") or 0)
    reas = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    usd = ((tin - cached) * pin + cached * pc + tout * pout) / 1e6
    if tier == "flex":   # 2026-10-05: OpenAI flex processing is billed at half the standard rate (batch pricing)
        usd *= 0.5
    rec = {"t": round(time.time(), 1), "model": model, "in": tin, "cached": cached, "out": tout, "reasoning": reas, "usd": round(usd, 6)}
    if tier:
        rec["tier"] = tier
    with open(led, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(rec) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)


def _budget_record_failure(model: str, kind: str, max_out: int, elapsed: float, detail: str) -> None:
    """2026-10-04 (gpt-6.1-sol pilot): a failed API call (HTTP 400 'output limit reached', or a timeout) returns no usage,
    so _budget_record never saw it and the GBP_BUDGET_USD hard stop could not count it. Log every such failure to stderr
    and, when a ledger is configured (API runs only), record a WORST-CASE estimate -- the call's whole output budget at
    the output price -- under the model name '<model>~failed-est', so the hard stop counts it conservatively."""
    led = os.environ.get("GBP_USAGE_LEDGER")
    if not led:          # local vLLM runs (RL rollouts, our evals): nothing to account, keep their logs quiet
        return
    import sys as _sys
    print(f"[api-failure] {model} {kind} after {elapsed:.0f}s (max_out={max_out}): {detail[:160]!r}", file=_sys.stderr, flush=True)
    import fcntl
    pout = float(os.environ.get("GBP_PRICE_OUT", "0")) * (0.5 if os.environ.get("GBP_SERVICE_TIER") == "flex" else 1.0)
    rec = {"t": round(time.time(), 1), "model": f"{model}~failed-est", "in": 0, "cached": 0, "out": int(max_out), "reasoning": 0,
           "usd": round(int(max_out) * pout / 1e6, 6), "estimated": True, "kind": kind, "elapsed_s": round(elapsed, 1)}
    with open(led, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(rec) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)


def _default_base_url() -> str:
    """GBP_BASE_URL. A comma-separated list (docs/RL_PLAN.md 2026-09-29 'RP_1 on both GPUs': two vLLM servers serving
    the same policy) is spread round-robin over AGENTS -- one agent per episode -- so every turn of an episode stays on
    one server and keeps its prefix cache. A single URL, the only form used before, behaves exactly as it always did."""
    raw = os.environ.get("GBP_BASE_URL", "https://api.openai.com/v1")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if len(urls) <= 1:
        return urls[0] if urls else raw
    return urls[next(_BASE_URL_RR) % len(urls)]


@dataclass
class LLMAgent:
    model: str
    base_url: str = field(default_factory=_default_base_url)
    api_key: str = field(
        default_factory=lambda: os.environ.get("GBP_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
    )

    def __post_init__(self):
        # A local-model run launched without GBP_BASE_URL used to fall through to the
        # OpenAI API with the placeholder key and fail every rollout with HTTP 401 --
        # after sending the task text off-cluster. Refuse that combination outright.
        if self.base_url.startswith("https://api.openai.com") and self.api_key == "EMPTY":
            raise RuntimeError(f"no GBP_BASE_URL for model {self.model!r}: set it to the pod's server "
                               "(http://127.0.0.1:PORT/v1); refusing to call api.openai.com without a key")
    # Configurable so evaluation can be matched to whatever budget a policy was
    # TRAINED under. rl/verl_sandbox_loop.py trains at 12; scoring that policy
    # at 25 measures a different objective than the one it optimised, and the
    # difference is not cosmetic -- a policy that learned to finish inside 12
    # turns has been selected for brevity that a 25-turn eval never rewards.
    # Default stays 25 so every previously reported harness number reproduces.
    max_turns: int = field(
        default_factory=lambda: int(os.environ.get("GBP_MAX_TURNS", "25")))
    max_stalls: int = 5   # consecutive no-progress turns before abandoning
    force_answer_turns: int = 3   # tail turns reserved for emitting the answer
    # WALL-CLOCK budget per episode, seconds (0 = none, the default: every reported number is unaffected).
    # docs/RL_PLAN.md 2026-09-24: an RL step waits for its slowest rollout, and ~5% of rollouts ran code to the
    # cell timeout for 30-60 min (2 of 191 passed) while the median took under a minute. Once the budget is
    # spent the episode enters the same answer-forcing tail as at the turn limit -- it is asked for its answer,
    # not discarded, so partial credit survives. Checked between turns: a cell already running finishes first.
    wall_budget: float = field(
        default_factory=lambda: float(os.environ.get("GBP_WALL_BUDGET", "0")))
    # GBP_TEMPERATURE (2026-09-30, H-TTS self-consistency samples): unset = 0.2, exactly as before.
    temperature: float = field(default_factory=lambda: float(os.environ.get("GBP_TEMPERATURE", "0.2")))
    # None | 'cot' | 'fewshot' -- prompting baselines; see build_prompt.
    prompt_style: str | None = None
    top_p: float = 0.95
    max_tokens: int = 8192
    enable_thinking: bool | None = None
    # None: the server's chat template (/chat/completions). "train" | "rl":
    # render_prompt() here and call /completions -- see render_prompt.
    render: str | None = None
    name: str = "llm"
    last_400: str = ""
    # Budgeted in CHARS but the server's limit is in TOKENS, and the ratio is
    # not the ~3.7 chars/token of prose. Scientific tool output -- gene ids,
    # numeric tables, accessions -- tokenizes near 2 chars/token
    # (`ENSG00000223972.5` is 17 chars and ~9 tokens), so 90k chars reached
    # ~45k tokens and blew a 40,960 window. Three BixBench problems died on
    # HTTP 400 that way. 60k chars keeps the worst case under ~30k tokens with
    # room for the system prompt, task and reply. See FINDINGS C2.
    history_char_budget: int = 60_000
    # IN-CONTEXT BASELINE. GBP_ICL_DEMO names a JSON list of messages: one executed demonstration
    # in this harness's own format, everything after the system prompt (experiments/icl/extract_demo.py
    # writes the trace a one-demonstration fine-tune repeats). The conversation then opens with that
    # solved episode and the task follows as a new turn, so the same trace is tested in context
    # instead of in the weights. The demonstration is never trimmed.
    icl_demo: list = field(default_factory=lambda: json.loads(open(os.environ["GBP_ICL_DEMO"]).read())
                           if os.environ.get("GBP_ICL_DEMO") else [])
    # E3: prepend STAGE_DIAGNOSE and ask for a MODES line. Graded offline from the transcript
    # (scripts/mode_recognition.py); the harness scores the analysis exactly as it does for any other arm.
    diagnose: bool = False

    def _chat(self, messages: list[dict]) -> tuple[str, str]:
        """Return (content, finish_reason)."""
        budget = self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": budget,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        # GBP_OPENAI_STYLE=1 (2026-09-30, frontier API comparison): OpenAI reasoning models take max_completion_tokens and
        # refuse temperature/top_p; GBP_REASONING_EFFORT sets reasoning_effort. Unset = the payload above, unchanged.
        if os.environ.get("GBP_OPENAI_STYLE") == "1" and not self.render:
            payload["max_completion_tokens"] = int(os.environ.get("GBP_MAX_COMPLETION_TOKENS") or payload.pop("max_tokens"))
            payload.pop("max_tokens", None)
            budget = payload["max_completion_tokens"]
            payload.pop("temperature", None); payload.pop("top_p", None)
            if os.environ.get("GBP_REASONING_EFFORT"):
                payload["reasoning_effort"] = os.environ["GBP_REASONING_EFFORT"]
            if os.environ.get("GBP_SERVICE_TIER"):   # e.g. 'flex' (2026-10-05): cheaper, slower, same API
                payload["service_tier"] = os.environ["GBP_SERVICE_TIER"]
        _budget_check()
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if self.render:
            payload.pop("messages"); payload.pop("chat_template_kwargs", None)
            payload["prompt"] = render_prompt(messages, self.render)
            payload["stop"] = (["</s>"] if self.render == "mtrain" else ["<|eot_id|>"] if self.render == "ltrain"
                               else [IM_END])
            url = f"{self.base_url.rstrip('/')}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(5):
            # Build a fresh Request each attempt. Mutating req.data on a
            # already-constructed Request does not reliably resend the new
            # body, which is why the previous 400-retry silently never fired
            # and two problems died on their first call.
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(), headers=headers
            )
            t_call = time.time()
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    body = json.loads(r.read())
                    _budget_record(self.model, body.get("usage"), body.get("service_tier"))
                    ch = body["choices"][0]
                    text = ch["text"] if self.render else ch["message"].get("content")
                    return text or "", ch.get("finish_reason", "")
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode()[:400]
                except Exception:                      # noqa: BLE001
                    detail = "<no body>"
                # 400 == prompt + max_tokens exceeds the context window. Halve
                # the budget and retry rather than losing the whole problem.
                if e.code == 400 and ("output limit" in detail or "max_tokens" in detail):
                    _budget_record_failure(self.model, "http400-output-limit", budget, time.time() - t_call, detail)
                if e.code == 400 and attempt < 4:
                    self.last_400 = detail
                    budget = max(512, budget // 2)
                    # OpenAI style (GBP_OPENAI_STYLE=1) sends max_completion_tokens; re-adding max_tokens would turn every
                    # retry into "Unsupported parameter: 'max_tokens'" (seen with gpt-6.1-sol, 2026-09-30).
                    payload["max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"] = budget
                    continue
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(5 * (attempt + 1))   # module-level time (a local import here would shadow it in t_call)
                    continue
                raise RuntimeError(
                    f"HTTP {e.code} after {attempt+1} attempts "
                    f"(max_tokens={budget}): {detail}") from e
            except (TimeoutError, urllib.error.URLError) as e:   # HTTPError (a URLError) is handled above
                _budget_record_failure(self.model, "timeout-or-network", budget, time.time() - t_call, repr(e))
                raise
        raise RuntimeError("chat failed after retries")

    def _trim(self, messages: list[dict]) -> list[dict]:
        """Keep the conversation inside the model's context window.

        Tool outputs are capped at 8k chars each, so ~25 turns reliably blows
        past a 32k window and the server returns 400 -- killing the episode
        outright. Raising --max-model-len only postpones that, so instead drop
        the OLDEST middle exchanges, always preserving:
          * the system prompt and the original task (indices 0,1)
          * the most recent exchanges, which carry the live analysis state
        A marker is inserted so the model knows history was elided rather than
        silently forgetting what it already did.
        """
        nh = 2 + len(self.icl_demo)                     # system, any demonstration, the task
        budget = self.history_char_budget + sum(len(m["content"]) for m in self.icl_demo)
        total = sum(len(m["content"]) for m in messages)
        if total <= budget:
            return messages

        head, tail = messages[:nh], messages[nh:]
        kept: list[dict] = []
        used = sum(len(m["content"]) for m in head)
        for m in reversed(tail):                       # newest first
            if used + len(m["content"]) > budget and kept:
                break
            kept.append(m)
            used += len(m["content"])
        kept.reverse()
        if len(kept) < len(tail):
            note = {"role": "user",
                    "content": f"[{len(tail) - len(kept)} earlier exchanges elided to "
                               f"fit the context window. Your notebook state is intact "
                               f"-- variables from earlier cells are still defined.]"}
            return head + [note] + kept
        return head + kept

    @staticmethod
    def _force_msg(left: int) -> str:
        return (
            f"STOP. You have {left} turn(s) left and must answer now.\n"
            "Do NOT run more code. Emit ONLY the final JSON object in the exact "
            "schema you were given, using your best current estimates from the "
            "analysis you have already done. If a quantity is still uncertain, "
            "give your best numeric estimate rather than omitting the key -- an "
            "omitted or missing key scores zero, whereas an imperfect number may "
            "be within tolerance. Every key is required."
        )

    def solve(self, cfg: dict, sandbox: Sandbox) -> tuple[dict | None, list]:
        task = build_prompt(cfg["task"], sandbox.listing(), self.prompt_style)
        if self.diagnose:
            task = STAGE_DIAGNOSE + task
        messages = [
            {"role": "system", "content": SYSTEMS[os.environ.get("GBP_SYSTEM", "default")]},
            *self.icl_demo,
            {"role": "user", "content": (ICL_BRIDGE + task) if self.icl_demo else task},
        ]
        transcript: list[dict] = []
        stalled = 0          # consecutive turns producing neither code nor an answer
        last_err, repeats = "", 0

        t0, wall_turn = time.monotonic(), None     # wall_turn: the turn at which wall_budget ran out
        for turn in range(self.max_turns):
            # Reserve the tail of the budget for actually answering. Without
            # this the loop simply runs out mid-analysis: 6 of 10 problems
            # burned every turn running code and returned submission=None, so
            # the grader saw nothing and a dense reward had nothing to read.
            # An absent answer is strictly less informative than a wrong one.
            if wall_turn is None and self.wall_budget and time.monotonic() - t0 > self.wall_budget:
                wall_turn = turn
            left = self.max_turns - turn
            if wall_turn is not None:              # out of wall time: the forcing tail starts at wall_turn
                left = min(left, wall_turn + self.force_answer_turns - turn)
                if left <= 0:
                    break
            forcing = left <= self.force_answer_turns
            if forcing:
                fm = self._force_msg(left)
                messages.append({"role": "user", "content": fm})
                # "content" keeps the marker the analyses look for (index_pods, judge_gap); "sent" is what the
                # model actually read, so an RL episode rebuilt from this transcript conditions on the real prompt.
                transcript.append({"turn": turn, "role": "tool",
                                   "content": "[forcing final answer]", "sent": fm,
                                   **({"wall_budget_spent": True} if wall_turn is not None else {})})

            messages = self._trim(messages)
            reply, finish = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            transcript.append({"turn": turn, "role": "assistant", "content": reply})

            # While forcing, an answer is the only acceptable output; running
            # more code would just consume the reserve we set aside.
            ans = extract_answer(reply)
            if ans is not None:
                return ans, transcript
            if forcing:
                continue

            code = extract_code(reply)
            if code:
                stalled = 0
                res = sandbox.run(code)
                fb = res.as_feedback()
                # Guessing the same wrong thing repeatedly (e.g. npz key names)
                # burns the whole turn budget. Point it at introspection.
                sig = (res.stderr or "").strip()[-160:]
                if sig and sig == last_err:
                    repeats += 1
                    if repeats >= 2:
                        fb += ("\n\nHINT: this is the same error as your previous "
                               "attempt. Stop guessing and inspect first -- e.g. "
                               "`np.load(path).files` for .npz keys, `df.columns` "
                               "for table columns, `df.head()` for values.")
                        repeats = 0
                else:
                    repeats = 0
                last_err = sig
                messages.append({"role": "user", "content": fb})
                transcript.append({"turn": turn, "role": "tool", "content": fb})
                continue

            if finish == "length":
                # The trace was cut off mid-thought, so there is nothing to
                # parse. Telling it to "emit JSON" is useless; tell it to stop
                # reasoning and act.
                nudge = (
                    "Your reply was cut off before producing any output. Stop "
                    "reasoning and immediately emit a short ```python block "
                    "that loads and inspects the data files. Keep it brief."
                )
            else:
                nudge = (
                    "No ```python block and no parseable JSON answer outside your "
                    "reasoning. Emit an actual ```python block to inspect the data, "
                    "or the final JSON object exactly as specified. Do not answer "
                    "from reasoning alone -- compute the numbers."
                )
            # Give up early on a model that will not act. Two problems
            # previously burned all 25 turns emitting prose and never ran a
            # single cell; nudging further just wastes the allocation.
            stalled += 1
            if stalled >= self.max_stalls:
                transcript.append({"turn": turn, "role": "system",
                                   "content": f"aborted: {stalled} turns with no code and no answer"})
                return None, transcript

            messages.append({"role": "user", "content": nudge})
            transcript.append({"turn": turn, "role": "tool", "content": nudge})

        # Last chance: the loop is exhausted, so ask once more for the answer
        # alone. Cheap, and it is the difference between a scoreable wrong
        # answer and no data point at all.
        try:
            messages = self._trim(messages + [
                {"role": "user", "content": self._force_msg(0)}])
            reply, _ = self._chat(messages)
            transcript.append({"turn": self.max_turns, "role": "assistant",
                               "content": reply})
            ans = extract_answer(reply)
            if ans is not None:
                return ans, transcript
        except Exception as exc:                                  # noqa: BLE001
            transcript.append({"turn": self.max_turns, "role": "system",
                               "content": f"final forced call failed: {exc}"})
        return None, transcript


# --------------------------------------------------------------------------- #
# the self-verifying agent: acquire the procedure at test time
# --------------------------------------------------------------------------- #
# Every transfer measurement in this project says the same thing: a model that has
# not seen a science does not derive its analysis, and training on other sciences
# does not help (leave-one-family-out, leave-one-domain-out, and mixtures of one to
# seven sciences at a fixed budget all land at the untrained rate). What the model
# CAN do is execute an analysis it has been given, and write code. This agent turns
# the second ability into the first: it makes its own ground truth.
#
#   READ       state the estimand and the intended analysis, after inspecting the files
#   REPLICATE  write a simulator of the observed data with the target quantity PLANTED,
#              its nuisance parameters fitted to what was observed
#   CALIBRATE  run the intended analysis on fresh synthetic draws and report the worst
#              relative error against the planted value, as a machine-readable line
#   REPAIR     while that error exceeds the tolerance, find the responsible step and fix it
#   ABLATE     show that disabling each step moves the estimate: a step that changes
#              nothing is not doing anything
#   APPLY      run the verified analysis on the real files and answer
#
# The check is parsed by the harness, not judged by the model: the agent advances only
# on a SELFCHECK line it can read, so a model that merely claims success does not pass.
SELFCHECK_RE = re.compile(r"SELFCHECK\s+worst_rel_err\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

STAGE_READ = (
    "Before analysing anything: inspect the files with code, then state in two lines --\n"
    "  ESTIMAND: <the quantity asked for, and its units>\n"
    "  PLAN: <the analysis you intend, as numbered steps>\n"
    "Do not answer the question yet.")
# v7: the other half of "account for the data". Adjusting for more is right when the reference analysis differs from
# the naive one by what it INCLUDES, and wrong when it differs by what it correctly LEAVES OUT -- controlling a
# variable measured after the exposure is as wrong as omitting a confounder, and the loop increases that error.
CAUTION_OVERADJUST = (
    "\nOne caution while you write the PLAN. Adjusting for more is not always safer. Adjust for what could confound "
    "the comparison -- anything that influences both the exposure and the outcome and was fixed before the exposure. "
    "Do NOT adjust for a variable that is measured after the exposure, or that the exposure could itself have "
    "changed: conditioning on a consequence of the treatment distorts the estimate exactly as omitting a confounder "
    "does, and it is the more tempting mistake because the variable is sitting there in the file. Say, for each "
    "variable you adjust for, whether it was fixed before the exposure."
)

# v8: the simulator's content without the simulator. Everything REPLICATE asks the model to decide, asked as prose --
# so that if the loop works because it makes the model articulate a generating process, this captures it, and if it
# works because the model gets to RUN one, this does not.
STAGE_DESCRIBE = (
    "You have not seen this kind of problem before. Before you analyse anything, say IN WORDS how the numbers in "
    "these files came to be. Do not write any code for this step -- no simulator, no draws, no files.\n"
    "Cover four things, briefly:\n"
    "  (1) THE UNITS: what is one row, and what population of units could have been measured.\n"
    "  (2) THE MECHANISM: what determines the quantity in ESTIMAND, and what else moves the recorded outcome.\n"
    "  (3) THE OBSERVATION: which units actually reach this file and what is recorded for each -- who is missed, who "
    "is followed to the end and who is censored, what is truncated, what is measured through a proxy or with error. "
    "If whether a unit is observed depends on what happened to it, say so here.\n"
    "  (4) WHAT THAT COSTS THE OBVIOUS ANALYSIS: name the analysis someone would reach for first, and say which of "
    "(1)-(3) it ignores and in which direction that pushes the estimate.\n"
    "Then state the analysis you will run instead.")

# v9: the falsification stage. Deliberately about the REAL file, not a synthetic draw -- the point is to give the
# description something that can contradict it at the lowest possible cost.
STAGE_FALSIFY = (
    "Now try to show that what you just said is WRONG.\n"
    "Pick the one claim in your description that most affects which analysis is correct -- that units are missing in "
    "a way that depends on the outcome, that follow-up is censored, that a variable is measured after the exposure, "
    "whatever it is for these data. State what that claim implies you should SEE in the file you were given: a rate "
    "that differs between two groups, a distribution truncated at some value, a column whose missingness tracks "
    "another, a total that does not add up.\n"
    "Then compute it, on the real data, and print the number beside what your description predicted.\n"
    "Finish with one line, exactly:  FALSIFY claim=<short name> predicted=<value> observed=<value> holds=<yes|no>\n"
    "If it does not hold, say what that rules out and revise the plan before you analyse."
)

# v11: choose among analyses instead of auditing one. The planted value is ground truth the agent made itself, and
# ground truth can rank candidates -- which is what every earlier stage failed to use it for.
STAGE_PROPOSE = (
    "Before you analyse the real data, write down THREE genuinely different analyses that a competent person might "
    "run on it, as a numbered list. They must differ in substance, not in wording:\n"
    "  - they should disagree about what to CONDITION ON (include a variable, or deliberately leave one out), or\n"
    "  - about the FUNCTIONAL FORM (linear/additive against something else the data may require), or\n"
    "  - about which UNITS OR ROWS enter the estimate (all of them, or a subset, or reweighted).\n"
    "At least one should be the analysis you would reach for first. Do not decide between them yet, and do not run "
    "them on the real data. Just state the three, each in one sentence, and say what each assumes.")
STAGE_SELECT = (
    "Now let your simulator decide. Run ALL THREE analyses on a draw from your simulator, where you know the planted "
    "value, and print for each one:\n"
    "  CANDIDATE <n> estimate=<value> planted=<value> rel_err=<|estimate-planted|/|planted|>\n"
    "Then state which candidate has the smallest rel_err and use THAT analysis on the real data. If two are close, "
    "draw again with a different seed and larger n to break the tie.\n"
    "If the one you would have reached for first is not the winner, say so explicitly before you proceed -- that is "
    "the case this step exists for.")

# v12: the generator, written in causal order. The point is not to describe the data better -- it is to make the
# synthetic draw capable of exhibiting the error the real data punishes.
# ---------------------------------------------------------------------------- M6: identification, stated at the
# MODE level rather than the family level. See docs/CROSS_DOMAIN_TASK_DESIGN.md.
#
# WHY. Every content instruction in this ladder is written for ONE FAMILY'S SURFACE -- v12 names "the exposure" and
# "variables fixed before it", which IS the confounding recipe, and the agent applies it everywhere including where
# it is wrong (28 predicted / 0 contrary, docs/GENERALIZATION_PLAN.md). This one names an ABSTRACT CONDITION and how
# to test for it, in words true of an economics instrument (a distance, a lottery, an administrative rule) and of a
# genetic variant in Mendelian randomisation. `causal_iv` and `cis_mvmr` are the SAME identification strategy in two
# maximally different surfaces, so an agent that can act on this in both has learned the MODE; one that can only act
# on it where it was written has learned a surface -- the programme's existing conclusion, but measured.
# DE-LEAKED 2026-09-21, same reason: this stage offered "a genetic variant" as an example, which is the held-out
# genomics surface. Examples that name an evaluation family defeat the test.
STAGE_REPLICATE_IDENT = (
    "You have not seen this kind of problem before. Before estimating anything, decide WHERE THE ANSWER IS "
    "IDENTIFIED FROM, and print these four lines first, one per line:\n"
    "  SOURCE <name|none> -- a measured variable that moves the exposure but that could not affect the outcome "
    "except through the exposure. It may be an assignment, a lottery, a distance, an administrative rule, or "
    "a scheduling quirk: anything that varies for reasons outside the process under study.\n"
    "  WHY-EXCLUDED <the reason it cannot reach the outcome by any other path, or n/a>\n"
    "  CONFOUNDED <yes|no> -- could something you cannot measure move both the exposure and the outcome?\n"
    "  IDENTIFY <from-source|from-adjustment> -- answer from-source when SOURCE is not none AND CONFOUNDED is yes; "
    "otherwise from-adjustment.\n"
    "Then estimate the way you just said. FROM-SOURCE: the effect is the outcome's response to the SOURCE divided "
    "by the exposure's response to the same SOURCE, with any variable entering both responses held fixed in both; "
    "do not read the effect off a regression of the outcome on the exposure. FROM-ADJUSTMENT: condition on what was "
    "fixed before the exposure and on nothing the exposure could have changed.\n"
    "Write a simulator of these data with the quantity in ESTIMAND set to a value you plant, generate it so your "
    "SOURCE line is true of it, and check your chosen estimator recovers the planted value on a draw before you run "
    "it on the real data. Put synthetic files under ./sim/, never in data_files/."
)

# ---------------------------------------------------------------------------- M5: selection and post-exposure
# conditioning, stated at the MODE level. See docs/CROSS_DOMAIN_TASK_DESIGN.md sections 7-8.
#
# WHY. Re-reading the tree by mode (scripts/mode_sweep.py) shows the contentless scaffold CAUSES the trap on
# exactly the two modes about WHICH DATA IDENTIFIES THE ANSWER (M5 selection, M6 identification) and REDUCES it on
# the four about HOW TO MODEL WHAT YOU HAVE (censoring, heterogeneity, functional form, measurement). M5 spans five
# families in three domains -- causal_confounded post_treatment_control, hc_dx referred_only, hc_quality
# in_hospital_deaths_only, surv_tvc ever_treated_from_diagnosis, rare_burden panel_maf_filter -- and there is one
# clean existing row of a content arm reducing it OUTSIDE its home family (v12 @32B on hc_dx, 40.5% -> 27.4%,
# p = 0.0035). This instruction names the condition without naming any family's variables, so the same sentence can
# be evaluated on all of them.
#
# v12 says "which variables were fixed BEFORE THE EXPOSURE" -- that presumes an exposure, which is the confounding
# surface. This asks the two questions that generate the whole mode: could this COLUMN have been changed by the
# thing whose effect I am estimating, and did this ROW get into the data because of it.
# DE-LEAKED 2026-09-21 after an adversarial design review: this stage listed "(who was referred, who was admitted,
# who survived long enough to be recorded, who passed a filter)" -- the selection mechanisms of hc_dx, hc_quality
# and surv_tvc BY NAME, which are the families it was written to be evaluated on held-out. A mode-level
# instruction that names the held-out surfaces is not testing transfer. Examples removed; the condition stays.
STAGE_REPLICATE_SELECT = (
    "You have not seen this kind of problem before. Before estimating anything, decide WHICH ROWS AND WHICH "
    "COLUMNS the question is actually about, and print your decision first.\n"
    "For every column, print one line:  COL <name> AFTER <yes|no|unclear> because <reason> -- where AFTER means "
    "its value could have been changed by the thing whose effect you are estimating, or was recorded later than "
    "it. Count anything downstream: a consequence, a response, a later measurement, a status that follows.\n"
    "Then print one line about the rows:  ROWS <all|selected> because <reason> -- SELECTED when whether a unit "
    "appears in these data at all, or appears with a usable value, depends on the thing under study or on its "
    "consequences.\n"
    "Analyse accordingly: do not condition on, stratify by, or adjust for any column you marked AFTER, and when "
    "ROWS is selected say what the selection does to the answer and handle it rather than ignoring it. If you "
    "marked a column unclear, treat it as AFTER and say you did.\n"
    "Write a simulator of these data with the quantity in ESTIMAND set to a value you plant, generate the AFTER "
    "columns and the row selection the way you just described, and confirm your analysis recovers the planted "
    "value there before running it on the real data. Put synthetic files under ./sim/, never in data_files/."
)

# ---------------------------------------------------------------------------- M2: unmodelled heterogeneity,
# stated at the MODE level. See docs/CROSS_DOMAIN_TASK_DESIGN.md sections 7 and 10.
#
# WHY M2 AND WHY NOW. The corrected mode coverage (52 traps, computed from the registry, not drafted by hand) puts
# M2 at TEN families -- ecology, clinical survival, healthcare quality, epidemiology twice and genomics three
# times -- the widest cross-domain surface the registry has. M5, the mode the running held-out test uses, has six.
# If "an agent can be taught a failure MODE rather than a family" is true at all, M2 is where it has the most room
# to show, and where a null is hardest to explain away as too few domains.
#
# It is also the mode the contentless scaffold already REDUCES (mode_sweep), which makes the placebo arm
# indispensable rather than decorative: on M2 a content arm that merely matches its placebo will still look good
# against control, and that is precisely the mistake the 28/0 tally made.
#
# DE-LEAKED BY CONSTRUCTION, applying the M5 review's lesson before the fact rather than after it: this stage
# names no grouping that any evaluation family uses. No hospitals, no sequencing batches, no sexes, no weekdays,
# no ancestry, no sites, no sessions, no species. The vocabulary is "a column whose labels split the units", and
# the conditions are stated so that they can be checked against any table.
# ---------------------------------------------------------------------------- E3: MODE RECOGNITION
#
# THE QUESTION THIS ANSWERS, which no other arm in this programme asks. Every test so far measures REMEDIATION:
# does the agent avoid the mistake. Remediation has transferred nowhere outside causal_confounded. But an agent
# that cannot fix an error it has never seen might still RECOGNISE it -- and recognition and remediation are
# different capabilities that call for different interventions. If recognition is at chance, the mode layer is a
# labelling convenience for the analyst and nothing in the model corresponds to it. If recognition is good while
# remediation is null, the gap is knowing-what-to-do, which is a far more tractable target than knowing-what-is-
# wrong, and it is the single most useful thing this programme could report.
#
# Deliberately MINIMAL and attached to the plain agent, not to the self-verify loop: it adds a menu and one line
# of output and changes nothing else, so `diag` vs `control` isolates the naming. The menu itself is a known
# quantity -- the prospective menu ablation on the causal recipe was null at three scales (+55.0->+54.2pt,
# +62.1->+67.1pt, +16.2->+13.7pt) -- so a performance change here would be the naming, not the list.
#
# The seven descriptions are the MODES dict in scripts/trap_modes.py verbatim: abstract, and naming no family's
# variables, so a correct answer cannot be read off the task text's vocabulary.
STAGE_DIAGNOSE = (
    "Before you analyse anything, say what could go wrong. Here are seven ways an analysis of data like this is "
    "commonly wrong:\n"
    "  M1 censoring -- a value observed only partially is treated as complete\n"
    "  M2 heterogeneity -- units that are not exchangeable are pooled\n"
    "  M3 functional -- a nonlinear process is forced into a linear fit\n"
    "  M4 measurement -- the measuring process (units, calibration, drift, ascertainment) is not modelled\n"
    "  M5 selection -- conditioning on a post-exposure variable or on a selected sample\n"
    "  M6 identification -- an available exogenous source of variation is ignored\n"
    "  M7 entity -- distinct entities are counted or matched wrongly\n"
    "Look at the data and the question, then print exactly one line before anything else:\n"
    "  MODES <comma-separated codes, e.g. M2,M5 -- those that genuinely threaten THIS analysis, or none>\n"
    "followed by one line per code you named:  WHY <code> <the specific thing in these data that makes it apply>\n"
    "Name a mode only if you can point at what in these data triggers it; naming all seven is the same as naming "
    "none. Then carry out the analysis and answer the question as you normally would.\n\n"
)

STAGE_REPLICATE_HETERO = (
    "You have not seen this kind of problem before. Before estimating anything, decide WHICH UNITS ARE "
    "COMPARABLE WITH WHICH, and print your decision first.\n"
    "For every column, print one line:  COL <name> GROUPS <yes|no|unclear> because <reason> -- GROUPS means its "
    "values are labels that split the rows into sets produced, observed or handled under different conditions, "
    "so that two rows agreeing on everything else could still differ because they carry different labels here. "
    "Columns with far fewer distinct values than rows, and columns recording where, when, by whom, with which "
    "device or under which protocol a row was obtained, are candidates.\n"
    "Then print two lines about what pooling those sets would do:\n"
    "  BASELINE <differs|same|unclear> because <reason> -- would the outcome sit at a different level in "
    "different sets even with the quantity in ESTIMAND at zero?\n"
    "  EFFECT <differs|same|unclear> because <reason> -- could that quantity itself differ between sets, making "
    "one pooled number an average over unlike things?\n"
    "Analyse accordingly: when BASELINE differs the labelling enters the analysis -- a level per set, a "
    "within-set comparison, or a random effect -- and pooling is wrong even when the sets look balanced; when "
    "EFFECT differs, say which average you are reporting and how it is weighted; when rows are repeats of one "
    "underlying thing they are not independent and the uncertainty must say so. Treat unclear as differs, and "
    "say that you did.\n"
    "Write a simulator of these data with the quantity in ESTIMAND set to a value you plant, generate the "
    "differences between sets that you described, and confirm your analysis recovers the planted value there "
    "before running it on the real data. Put synthetic files under ./sim/, never in data_files/."
)

STAGE_REPLICATE_CAUSAL = (
    "You have not seen this kind of problem before, so make yourself a test you can grade. Write a simulator of "
    "these data, with the quantity in ESTIMAND set to a value you plant. Write it in THREE BLOCKS, in this order, "
    "and keep them separate in the code:\n"
    "  BLOCK 1 -- BEFORE: every variable that was already fixed before the exposure/treatment/intervention occurred. "
    "Generate these first, from their own distributions.\n"
    "  BLOCK 2 -- THE EXPOSURE: generate the exposure as a function of the BLOCK 1 variables only. If it was "
    "randomised, say so and generate it independently.\n"
    "  BLOCK 3 -- AFTER: every variable whose value could have been influenced by the exposure, including the "
    "outcome, and including anything recorded later in time. Generate each as a function of the exposure and of "
    "BLOCK 1. Who is observed, who is censored, who is referred and what is recorded belong here if the exposure "
    "could have changed them.\n"
    "Before the code, print the assignment as one line per variable:  VAR <name> BLOCK <1|2|3> because <reason>. "
    "If you are unsure whether a variable belongs in 1 or 3, put it in 3 and say why it was unclear.\n"
    "Fit the nuisance parameters to the observed data so a draw resembles it, keep the simulator the smallest that "
    "produces the columns your analysis reads, and write every synthetic file under ./sim/ -- never into "
    "data_files/. Print the planted value and a summary of one draw beside the same summary of the real data."
)

# v20: the REPLICATE-stage PLACEBO, and the control v12 has never had. v16 showed that adding a contentless
# instruction to READ buys ~0.07 on DABench and nothing on causal_confounded -- but v16 is a READ-stage placebo and
# v12 changes REPLICATE, so it cannot speak to v12 at all. This matches STAGE_REPLICATE_CAUSAL clause for clause:
# three named blocks kept separate in the code, an explicit ordering rule, one printed line per variable with a
# reason, a rule for the unsure case, and the same closing instructions. The partition is by how a value is
# RECORDED rather than by when it occurred, so it demands the same enumeration and the same per-variable judgement
# while carrying nothing about causal order. If causal_confounded under v20 lands near the loop, v12's gain is in
# the causal content; if it lands near v12, then imposing ANY structure on the simulator is what works.
STAGE_REPLICATE_PLACEBO = (
    "You have not seen this kind of problem before, so make yourself a test you can grade. Write a simulator of "
    "these data, with the quantity in ESTIMAND set to a value you plant. Write it in THREE BLOCKS, in this order, "
    "and keep them separate in the code:\n"
    "  BLOCK 1 -- COUNTS: every variable recorded as a whole number -- counts, tallies, integer codes, anything "
    "that cannot take a fractional value. Generate these first.\n"
    "  BLOCK 2 -- MEASUREMENTS: every variable recorded on a continuous scale -- lengths, times, rates, "
    "concentrations, money, anything that can take a fractional value.\n"
    "  BLOCK 3 -- LABELS: every variable recorded as a category or a name -- group labels, yes/no indicators, "
    "identifiers, strings, and anything whose values are a fixed set of choices rather than a magnitude.\n"
    "Before the code, print the assignment as one line per variable:  VAR <name> BLOCK <1|2|3> because <reason>. "
    "If you are unsure whether a variable belongs in 1 or 3, put it in 3 and say why it was unclear.\n"
    "Fit the nuisance parameters to the observed data so a draw resembles it, keep the simulator the smallest that "
    "produces the columns your analysis reads, and write every synthetic file under ./sim/ -- never into "
    "data_files/. Print the planted value and a summary of one draw beside the same summary of the real data."
)

# v21: ROUTE, then build. Everything measured says no fixed device is best everywhere: causal ordering buys +29.7
# on causal_confounded against its matched placebo and COSTS -10.8 on causal_did; the plain loop wins causal_did
# and surv_cox outright. The dissociation is by FAMILY, and the family is visible in the TASK TEXT -- which is a
# different situation from the one that killed self-checking (the model provably cannot tell which ANALYSIS its
# data needs, 4.2% precision, but "does this task ask for an adjusted effect from one cross-section?" is a reading
# question). So the agent states a ROUTE first, in one printed line, and the route picks which simulator it builds.
# The printed line makes routing accuracy measurable per family, separately from the outcome.
STAGE_REPLICATE_ROUTED = (
    "You have not seen this kind of problem before, so make yourself a test you can grade: a simulator of these "
    "data, with the quantity in ESTIMAND set to a value you plant. HOW to write it depends on what the task asks, "
    "so first print exactly one line:\n"
    "  ROUTE order because <reason>  -- if the task asks for the effect of an exposure/treatment/intervention "
    "estimated from ONE observed cross-section, so that the analysis must choose which recorded variables to "
    "adjust for.\n"
    "  ROUTE plain because <reason>  -- for everything else: before/after or grouped-period comparisons, "
    "instruments, cutoffs and discontinuities, survival or censoring times, rates, curve fits, and any task that "
    "is not a causal-effect estimate.\n"
    "If your ROUTE is order, write the simulator in THREE BLOCKS, in this order, and keep them separate in the "
    "code:\n"
    "  BLOCK 1 -- BEFORE: every variable that was already fixed before the exposure/treatment/intervention "
    "occurred. Generate these first, from their own distributions.\n"
    "  BLOCK 2 -- THE EXPOSURE: generate the exposure as a function of the BLOCK 1 variables only. If it was "
    "randomised, say so and generate it independently.\n"
    "  BLOCK 3 -- AFTER: every variable whose value could have been influenced by the exposure, including the "
    "outcome, and including anything recorded later in time. Generate each as a function of the exposure and of "
    "BLOCK 1. Who is observed, who is censored, who is referred and what is recorded belong here if the exposure "
    "could have changed them.\n"
    "  Before the code, print the assignment as one line per variable:  VAR <name> BLOCK <1|2|3> because "
    "<reason>. If you are unsure whether a variable belongs in 1 or 3, put it in 3 and say why it was unclear.\n"
    "If your ROUTE is plain, skip the blocks: just write the smallest simulator that produces the columns your "
    "analysis reads, with the planted value where the estimand should be.\n"
    "Either way, fit the nuisance parameters to the observed data so a draw resembles it, write every synthetic "
    "file under ./sim/ -- never into data_files/, and print the planted value and a summary of one draw beside "
    "the same summary of the real data."
)

# v26: the GATED router. v21/:general print one free-form `ROUTE order|plain because ...` line and then -- measurably --
# say "order" in 240/240 causal_iv rollouts at 8B AND 14B, although the stated criterion already lists instruments
# under "plain". The criterion is not wrong; the ELICITATION is. The declaration is a single judgement the model makes
# about a causal-flavoured task, and it echoes the flavour. That mis-route is the mechanism of the programme's only
# replicated harm: at 14B the causal-order content costs 7 points on causal_iv (8.1% vs control 15.4%, p = 1e-04)
# while its contentless scaffold sits at control (14.2%, p = 0.74).
# v26 replaces the judgement with four concrete checks against the data, each requiring the column or phrase that
# justifies it, and DERIVES the route from them rather than asking for it. Everything after the route is v21 verbatim,
# so v26 - v21 isolates the elicitation.
STAGE_REPLICATE_GATED = (
    "You have not seen this kind of problem before, so make yourself a test you can grade: a simulator of these "
    "data, with the quantity in ESTIMAND set to a value you plant. HOW to write it depends on the STUDY DESIGN, and "
    "you will decide that from the data rather than from how the question sounds.\n"
    "First print exactly these four lines, each naming the column or the phrase in the task that decides it -- "
    "write `none` when there is nothing:\n"
    "  DESIGN instrument=<yes|no> because <column or phrase>   -- is some variable used to move the exposure "
    "without acting on the outcome directly: an assignment, a lottery, an encouragement, a distance or a rule that "
    "shifts who is treated?\n"
    "  DESIGN periods=<yes|no> because <column or phrase>      -- are the same units observed in more than one "
    "period, or are there before/after or grouped-period labels?\n"
    "  DESIGN cutoff=<yes|no> because <column or phrase>       -- is treatment assigned by a threshold on a running "
    "variable, so that comparison is local to a cutoff?\n"
    "  DESIGN duration=<yes|no> because <column or phrase>     -- is the outcome a time-to-event with follow-up or "
    "censoring, rather than a value measured once?\n"
    "Then print the route, and derive it -- do not judge it:\n"
    "  ROUTE order   -- ONLY if all four answers are `no`, so the exposure is compared across one observed "
    "cross-section and the analysis must choose which recorded variables to adjust for.\n"
    "  ROUTE plain   -- if ANY of the four is `yes`. The design already identifies the effect; imposing an "
    "adjustment ordering on it is wrong.\n"
    "If your ROUTE is order, write the simulator in THREE BLOCKS, in this order, and keep them separate in the "
    "code:\n"
    "  BLOCK 1 -- BEFORE: every variable that was already fixed before the exposure/treatment/intervention "
    "occurred. Generate these first, from their own distributions.\n"
    "  BLOCK 2 -- THE EXPOSURE: generate the exposure as a function of the BLOCK 1 variables only. If it was "
    "randomised, say so and generate it independently.\n"
    "  BLOCK 3 -- AFTER: every variable whose value could have been influenced by the exposure, including the "
    "outcome, and including anything recorded later in time. Generate each as a function of the exposure and of "
    "BLOCK 1. Who is observed, who is censored, who is referred and what is recorded belong here if the exposure "
    "could have changed them.\n"
    "  Before the code, print the assignment as one line per variable:  VAR <name> BLOCK <1|2|3> because "
    "<reason>. If you are unsure whether a variable belongs in 1 or 3, put it in 3 and say why it was unclear.\n"
    "If your ROUTE is plain, skip the blocks: just write the smallest simulator that produces the columns your "
    "analysis reads, with the planted value where the estimand should be.\n"
    "Either way, fit the nuisance parameters to the observed data so a draw resembles it, write every synthetic "
    "file under ./sim/ -- never into data_files/, and print the planted value and a summary of one draw beside "
    "the same summary of the real data."
)

# v22: v21's replicated +13 on causal_confounded (50.8% vs v12's 37.6%, p=7.8e-06) confounds two things: the
# route-CRITERIA text adds causal vocabulary to the prompt (content), and the printed ROUTE line makes the model
# COMMIT to a reading (commitment). v22 keeps v12's causal-order simulator and adds the same criteria as a purely
# DESCRIPTIVE note in READ -- no choice requested, nothing printed. v22 ~= v21 -> the content did it; v22 ~= v12
# -> the printed commitment is the active part. Registered before any v22 result exists.
ROUTE_CONTEXT_NOTE = (
    "\nA note on what such tasks can ask. Some ask for the effect of an exposure/treatment/intervention estimated "
    "from one observed cross-section, so that the analysis must choose which recorded variables to adjust for. "
    "Others are before/after or grouped-period comparisons, instruments, cutoffs and discontinuities, survival or "
    "censoring times, rates, curve fits, or other quantities that are not causal-effect estimates."
)

# v24: the COMMITMENT-PLACEBO. v22 showed the active part of v21 is the act of printing a committed reading, not
# the criteria text (v22 42.1% ~= v12; v21 50.8% >> both). Every effect in this programme gets a matched placebo,
# so the commitment claim needs one: a printed commitment with NO causal content. v24 = v20's recorded-type
# scaffold + one committed line about that partition. If v24 > v20 on causal_confounded, commitment amplifies
# anything and the mechanism is content-free; if v24 = v20 (~10%), commitment only deepens content that already
# bites, which is the reading v22 suggests. Registered before any result.
STAGE_REPLICATE_PLACEBO_COMMIT = (
    "Before anything else, print exactly one line committing to how you will organise the simulator:\n"
    "  PARTITION counts-then-measurements-then-labels because <reason>\n"
    "Then proceed exactly as follows.\n" + STAGE_REPLICATE_PLACEBO
)

# v14: the ACTION half of v7's sentence -- enumerate the variables and their timing, with no rule attached.
CAUTION_TIMING_ONLY = (
    "\nOne thing to do while you write the PLAN. List every variable you intend to use, and for each one say "
    "whether its value was already fixed before the exposure/treatment/intervention occurred, or could have been "
    "influenced by it, or whether you cannot tell from what you were given. Just state the list; no rule follows "
    "from it here."
)
# v15: the CAUTION half -- the rule, with no instruction to enumerate anything.
CAUTION_RULE_ONLY = (
    "\nOne caution while you write the PLAN. Adjusting for more is not always safer. Adjust for what could confound "
    "the comparison -- anything that influences both the exposure and the outcome and was fixed before the exposure. "
    "Do NOT adjust for a variable that is measured after the exposure, or that the exposure could itself have "
    "changed: conditioning on a consequence of the treatment distorts the estimate exactly as omitting a confounder "
    "does."
)

# v16: a length- and specificity-matched instruction about nothing that can bear on which analysis is right. The
# control for "would ANY concrete added instruction help?", which the v7 result needs and does not have.
CAUTION_PLACEBO = (
    "\nOne thing to do while you write the PLAN. Give each step of the plan a short lower-case label, number the "
    "steps consecutively from one, and repeat the numbered list of labels at the end of the plan in the same order. "
    "Keep each label to at most four words, and use the same label for a step if you refer back to it later."
)

# v25: the FALSIFY-matched placebo. v9's stages (READ -> DESCRIBE -> FALSIFY) with the falsification stage replaced by
# one of the same length, register and printed format that checks a BOOKKEEPING fact about the file instead of the
# claim that decides the analysis. v20 plays this role for v12's simulator; nothing played it for v9 until 2026-09-20,
# so the falsify device's ownership of surv_tvc / arr_kinetics was matched only against a different scaffold.
STAGE_FALSIFY_PLACEBO = (
    "Now check that what you just said is CONSISTENT with the file.\n"
    "Pick the one statement in your description that is easiest to verify by counting -- how many rows there are, "
    "how many columns, how many distinct values the first categorical column takes, how many rows have no missing "
    "entries, whatever it is for these data. State what your description implies you should COUNT in the file you "
    "were given: a number of units, a number of variables, a number of groups, a number of complete rows.\n"
    "Then compute it, on the real data, and print the number beside what your description predicted.\n"
    "Finish with one line, exactly:  FALSIFY claim=<short name> predicted=<value> observed=<value> holds=<yes|no>\n"
    "If it does not hold, correct the description and proceed with the analysis."
)

# v18: falsify a consequence against the REAL file, used ALONGSIDE a simulator rather than instead of one.
STAGE_FALSIFY_WITH_SIM = (
    "Your simulator tests whether your analysis recovers a value you planted. It cannot test whether the FILE YOU "
    "WERE GIVEN is what you think it is, because you fitted the simulator to that file -- if the units are wrong, "
    "or the clock starts at the wrong event, or the signal was never detrended, your simulator inherited that and "
    "will confirm your analysis anyway.\n"
    "So make one check the simulator cannot make. Name a property the REAL data must have if your reading of it is "
    "right -- a plausible range for a physical quantity, a rate that should match a known one, a distribution that "
    "should be centred somewhere, a total that should reconcile -- and compute it on the real file.\n"
    "Print exactly one line:  FILECHECK property=<short name> expected=<value or range> observed=<value> "
    "holds=<yes|no>\n"
    "If it does not hold, the file is not what you assumed: say what that implies and revise before you answer."
)

STAGE_REPLICATE = (
    "You have not seen this kind of problem before, so make yourself a test you can grade.\n"
    "Write a simulator that generates data files of the SAME shape and columns as the ones you were given, from "
    "parameters you choose, with the quantity in ESTIMAND set to a value you plant. Fit its nuisance parameters "
    "(sizes, noise, rates, covariate distributions) to the observed data so a draw resembles it. Keep it to the "
    "SMALLEST simulator that produces the columns your analysis reads, with the same shapes and ranges: you are "
    "making a test, not a faithful model of the system. Print the planted "
    "value and a summary of one draw beside the same summary of the real data.\n"
    "Write every synthetic file under ./sim/ -- never into data_files/, which holds the real data you must not "
    "alter and must not confuse with your own draws.")
# v6: the observation process, made explicit. The loop crosses on surv_cox, whose trap (censoring) CANNOT be left out
# of a survival simulator; it does nothing where the trap is equally a property of how the data were collected but is
# easy to omit (hc_dx referral bias, eco_size catchability). A simulator with no selection step cannot exhibit a
# selection bias, so no amount of checking against it will find one.
STAGE_REPLICATE_2 = (
    "You have not seen this kind of problem before, so make yourself a test you can grade. Write the simulator in TWO "
    "steps, and keep them separate in the code:\n"
    "  (1) THE PROCESS: generate the underlying units -- every patient, animal, cell or star that could have been "
    "measured -- with the quantity in ESTIMAND set to a value you plant.\n"
    "  (2) THE OBSERVATION: decide which of those units reach the file you were given, and what is recorded for each. "
    "Who is measured and who is missed, who is followed to the end and who is censored, what is truncated, what is "
    "recorded with error or through a proxy. If who gets observed depends on what happened to them, your code must say "
    "so in step (2), because that is what the data you were handed look like.\n"
    "Write files of the SAME shape and columns as the ones you were given, from step (2) only. Print the planted "
    "value, how many units step (1) generated, how many reached the file, and a summary of one draw beside the same "
    "summary of the real data.\n"
    "If those two counts are equal, say in one line why nothing is missing here -- in real data of this kind something "
    "usually is, and a simulator in which nothing is missing cannot show you a bias that comes from what is missing.\n"
    "Write every synthetic file under ./sim/ -- never into data_files/, which holds the real data you must not alter."
)

# v2 adds the stage the v1 diagnosis asks for. In 288 rollouts v1 wrote a working simulator 94% of the time and
# emitted a readable check 46% of the time, but of the 16 rollouts whose own check passed, NONE passed the real task:
# the simulators were too easy, planting a parameter that the obvious estimator recovers. This is invariant I2 of the
# engine (sage/audit.py) asked of the agent at test time: a synthetic instance is only a test if a tempting wrong
# analysis FAILS on it.
SEPARATION_RE = re.compile(r"SEPARATION\s+planted\s*=\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s+"
                           r"right\s*=\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s+"
                           r"wrong\s*=\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
STAGE_TRAP = (
    "A synthetic data set is only a test if a WRONG analysis fails on it. Name the most tempting wrong analysis for "
    "this task -- the obvious estimator, or yours with one correction skipped -- and run both it and your intended "
    "analysis on one synthetic draw. Print, computed by the code itself:\n"
    "    print(f'SEPARATION planted={{p:.6g}} right={{r:.6g}} wrong={{w:.6g}}')\n"
    "If the wrong analysis lands as close to the planted value as yours does, your simulator is missing the feature "
    "the correction exists for: add it and run this again.")
# v3: v2 showed the model can plant a parameter (94% wrote a working simulator) but almost never builds an instance on
# which a wrong analysis FAILS (4%). It is not the programming that is missing but the knowledge of what goes wrong in
# data like this. This names the usual culprits without naming the science, so the model chooses rather than invents.
TRAP_MENU = (
    "\nIn real data of this kind the tempting analysis is usually wrong for one of these reasons. Say which applies "
    "here, and build THAT into your simulator so the wrong analysis fails on it:\n"
    "  - a confounder that moves both the exposure and the outcome;\n"
    "  - selection, detection or ascertainment bias: who enters the data depends on what happened to them;\n"
    "  - censoring or truncation: the observation stops before the event, or the range is cut;\n"
    "  - contamination or mixture: some of the signal comes from another population or source;\n"
    "  - measurement error, or a proxy standing in for the quantity of interest;\n"
    "  - repeated measures or clustering that makes observations dependent;\n"
    "  - exposure or covariates that change during follow-up.")
STAGE_TRAP_RETRY = (
    "Your simulator does not separate them: the wrong analysis gives {wrong:.6g} against the planted {planted:.6g}, "
    "inside the {tol:.0%} you need your own analysis to hit. Add whatever the skipped correction is for -- the "
    "confounding, the censoring, the detection bias, the contamination -- and print SEPARATION again.")

# v5: the v1-v4 diagnosis is that the loop fails at its own first substantive step. Across 12 arm-families the agent
# ran a simulator cell successfully in 100% of rollouts and emitted a readable check in 40-77%, but the check met the
# 10% tolerance in only 2-21% -- and a failed check is ambiguous three ways (the simulator is unrealistic, the
# analysis is wrong, or the noise is too large to resolve the effect). This stage removes the ambiguity by asking the
# easy question first: on data with the nuisance turned off, only a wrong analysis can miss the planted value.
FLOOR_RE = re.compile(r"FLOOR\s+rel_err\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
STAGE_FLOOR = (
    "Before you test your analysis against a realistic draw, test it where it MUST work. Run your simulator once with "
    "the nuisance turned off -- no measurement noise, no missingness, no censoring, and a large sample -- so the "
    "planted value is exactly recoverable in principle, then run your intended analysis on that draw. Print, computed "
    "by the code itself:\n"
    "    print(f'FLOOR rel_err={{r:.6g}}')\n"
    "If your analysis cannot recover the planted value on noiseless data, the analysis is wrong and no amount of "
    "realism in the simulator will fix it.")
STAGE_FLOOR_REPAIR = (
    "On noiseless data your analysis is off by {err:.3f} relative error, more than the {tol:.3f} this case allows. "
    "Nothing here is a matter of noise or bias: either the quantity your simulator plants is not the one your "
    "analysis estimates, or a step of the analysis is wrong. Fix whichever it is, then print FLOOR rel_err=<number> "
    "again.")

STAGE_CALIBRATE = (
    "Now grade your PLAN against your own simulator. Run the analysis exactly as you intend to run it on the real "
    "data, on {n} fresh draws with different planted values, and print for each: planted, estimated, relative error. "
    "The last line your code prints must be the largest relative error over the draws, computed and formatted by the "
    "code itself, like this:\n"
    "    print(f'SELFCHECK worst_rel_err={{worst:.4f}}')\n"
    "A line that still contains braces, or any text in place of a number, does not count as a check.")
STAGE_REPAIR = (
    "Your analysis recovers the planted value to only {err:.3f} relative error, worse than the {tol:.3f} you need. "
    "It is wrong in a specific way: find the step responsible (the estimator, a correction you skipped, a bias you "
    "did not remove), fix it, and re-run the same check. End again with SELFCHECK worst_rel_err=<number>.")
STAGE_ABLATE = (
    "Your analysis passes. Confirm each step earns its place: for every step of PLAN, disable that step alone, "
    "re-run on synthetic data and print the estimate. A step whose removal leaves the estimate unchanged is doing "
    "nothing -- say so.")
STAGE_APPLY = (
    "Now run the analysis you verified -- unchanged -- on the REAL files in data_files/, and give the answer in the "
    "format the task requires.")
STAGE_APPLY_UNVERIFIED = (
    "Your check did not pass, so treat the synthetic work as finished and set it aside. Run your best analysis of the "
    "REAL files in data_files/ now -- start a fresh cell, read the real data, and give the answer in the format the "
    "task requires.")


@dataclass
class SelfVerifyAgent(LLMAgent):
    """LLMAgent driven through the stages above instead of a free-running loop."""

    name: str = "selfver"
    n_sim: int = field(default_factory=lambda: int(os.environ.get("SELFVER_N", "6")))
    tol: float = field(default_factory=lambda: float(os.environ.get("SELFVER_TOL", "0.10")))
    max_repair: int = field(default_factory=lambda: int(os.environ.get("SELFVER_REPAIRS", "3")))
    stage_turns: int = field(default_factory=lambda: int(os.environ.get("SELFVER_STAGE_TURNS", "4")))
    # writing the simulator is the hard cell and the one worth extra turns: in the first probe an 8B model spent
    # four turns on array-shape errors and never reached a check.
    replicate_turns: int = field(default_factory=lambda: int(os.environ.get("SELFVER_REPLICATE_TURNS", "7")))
    # v2: require the synthetic instance to defeat a wrong analysis before the self-check is trusted.
    require_trap: bool = False
    trap_menu: bool = False          # v3: offer the usual failure modes rather than asking the model to invent one
    # v4 (an oracle upper bound, not a deployable agent): $SELFVER_TRAP_HINT names THIS family's documented wrong
    # analyses, the ones our own generators were built from published case-study reports to catch. If the loop still
    # fails when the failure mode is handed over, the approach is dead; if it works, what test-time acquisition lacks
    # is exactly this knowledge.
    trap_hint: str = ""              # set ONLY for the v4 oracle arm, so the hint cannot leak into v2/v3 runs
    trap_turns: int = field(default_factory=lambda: int(os.environ.get("SELFVER_TRAP_TURNS", "5")))
    # v5: verify on a noiseless draw before the realistic one, so a failed check says which half is wrong.
    require_floor: bool = False
    # plan_only: READ then APPLY, no simulator and no check -- the control for "is it the verification or the
    # staged prompting?" (surv_cox: 0.615 pass with 1 of 96 rollouts verifying).
    plan_only: bool = False
    # v6: REPLICATE becomes process + observation, so a selection bias can exist in the synthetic data at all.
    two_stage: bool = False
    # v7: warn against over-adjustment in READ, where the plan is formed.
    no_overadjust: bool = False
    # v8: READ -> DESCRIBE -> APPLY. The generating process stated in prose, never run.
    describe_only: bool = False
    # v9: DESCRIBE then FALSIFY against the real file -- falsifiability without a simulator.
    check_claim: bool = False
    # v25: v9's scaffold with the falsification content removed (bookkeeping check of the same shape).
    placebo_claim: bool = False
    # v11: PROPOSE three candidate analyses, then SELECT by which recovers the simulator's planted value.
    select_candidates: bool = False
    # v12: REPLICATE written in causal order, so a post-exposure variable is GENERATED as one.
    causal_order: bool = False
    # M6: identification named at the MODE level, not in the family's surface (see the stage above).
    ident_source: bool = False
    # M5: selection/post-exposure conditioning named at the MODE level (see the stage above).
    select_mode: bool = False
    # M2: unmodelled heterogeneity named at the MODE level (see the stage above).
    hetero_mode: bool = False
    # v20: same simulator scaffolding as causal_order, partitioned by recorded type instead of time.
    placebo_order: bool = False
    # v21: print a ROUTE line from the task text, then build the causal-order or the plain simulator accordingly.
    routed_order: bool = False
    # v26: derive the route from four printed design checks instead of asking for a judgement.
    gated_order: bool = False
    # v22: the route criteria as a descriptive READ note; no choice, no printing.
    route_context: bool = False
    # v24: v20's scaffold with a printed, causally-empty commitment line.
    placebo_commit: bool = False
    # v14/v15: the two halves of v7's sentence, to find which one carries its effect.
    timing_only: bool = False
    rule_only: bool = False
    placebo: bool = False
    # v18: causal-order simulator AND a check against the real file -- both axes at once.
    file_check: bool = False
    floor_tol: float = field(default_factory=lambda: float(os.environ.get("SELFVER_FLOOR_TOL", "0.02")))
    # THE ANSWERING TAX, measured 2026-09-22. Splitting the paired external-benchmark problems by whether the
    # loop answered shows the whole of its QRData deficit is non-answers: where it answers it ties the plain
    # agent (41.2 v 40.7, p = 0.93), where it does not it scores 0.0 against 39.0 (p = 3e-05). Those runs burn
    # the ENTIRE budget -- median 40 turns of 40, against a median of 8 when they answer -- and every one of
    # them dies in the `apply` stage. The cause is in the apply loop below: it only sends the force-an-answer
    # message when the reply contained NO code, so a model that keeps writing code is never told to stop.
    # force_tail > 0 sends it anyway once this many apply turns remain. DEFAULT 0 = OFF, because changing the
    # agent silently would break comparability with every arm already measured; `:ftail` opts in.
    force_tail: int = field(default_factory=lambda: int(os.environ.get("SELFVER_FORCE_TAIL", "0")))

    def solve(self, cfg: dict, sandbox: Sandbox) -> tuple[dict | None, list]:
        task = build_prompt(cfg["task"], sandbox.listing(), self.prompt_style)
        messages = [{"role": "system", "content": SYSTEMS[os.environ.get("GBP_SYSTEM", "default")]},
                    {"role": "user", "content": task}]
        transcript: list[dict] = []
        turn = 0
        # An answer offered before the verification is finished is not acted on -- the point of the loop is to
        # check first -- but it is kept: if the verified analysis never yields one, a scoreable early answer is
        # strictly more informative than none.
        early: dict | None = None
        _rep = STAGE_REPLICATE_PLACEBO_COMMIT if self.placebo_commit else (
            STAGE_REPLICATE_SELECT if self.select_mode else (
            STAGE_REPLICATE_HETERO if self.hetero_mode else (
            STAGE_REPLICATE_IDENT if self.ident_source else (
            STAGE_REPLICATE_GATED if self.gated_order else (
            STAGE_REPLICATE_ROUTED if self.routed_order else (
            STAGE_REPLICATE_CAUSAL if self.causal_order else (
                STAGE_REPLICATE_PLACEBO if self.placebo_order else (
                    STAGE_REPLICATE_2 if self.two_stage else STAGE_REPLICATE))))))))
        _read = STAGE_READ + (ROUTE_CONTEXT_NOTE if self.route_context else "") \
                           + (CAUTION_OVERADJUST if self.no_overadjust else "") \
                           + (CAUTION_TIMING_ONLY if self.timing_only else "") \
                           + (CAUTION_RULE_ONLY if self.rule_only else "") \
                           + (CAUTION_PLACEBO if self.placebo else "")
        # Three shapes, not two: the full loop, plan-only (READ alone), and v8's READ -> DESCRIBE. The last two run
        # no simulator, so neither can reach a stage that checks one -- floor, trap and calibrate all compare an
        # analysis against a planted value that only a simulator creates.
        _nosim = self.plan_only or self.describe_only or self.check_claim or self.placebo_claim
        if self.plan_only:
            stages = [("read", _read)]
        elif self.check_claim:
            stages = [("read", _read), ("describe", STAGE_DESCRIBE), ("falsify", STAGE_FALSIFY)]
        elif self.placebo_claim:
            stages = [("read", _read), ("describe", STAGE_DESCRIBE), ("falsify", STAGE_FALSIFY_PLACEBO)]
        elif self.describe_only:
            stages = [("read", _read), ("describe", STAGE_DESCRIBE)]
        else:
            stages = [("read", _read), ("replicate", _rep)]
        if self.file_check and not _nosim:
            stages.append(("filecheck", STAGE_FALSIFY_WITH_SIM))
        if self.select_candidates and not _nosim:
            # between building the simulator and using it: name the alternatives, then let the planted value rank them
            stages.append(("propose", STAGE_PROPOSE))
            stages.append(("select", STAGE_SELECT))
        if self.require_floor and not _nosim:
            stages.append(("floor", STAGE_FLOOR))
        if self.require_trap and not _nosim:
            hint = f"\nFor data of this kind the documented wrong analyses are: {self.trap_hint}." if self.trap_hint else ""
            stages.append(("trap", STAGE_TRAP + (TRAP_MENU if self.trap_menu else "") + hint))
        if not _nosim:
            stages.append(("calibrate", STAGE_CALIBRATE.format(n=self.n_sim)))
        traps, trap_ok = [], not self.require_trap
        repairs, checks = 0, []
        floors, floor_repairs = [], 0
        stage_i, sent = 0, False
        while stage_i < len(stages) and turn < self.max_turns:
            name, prompt = stages[stage_i]
            if not sent:
                messages.append({"role": "user", "content": prompt})
                transcript.append({"turn": turn, "role": "tool", "content": f"[stage {name}]"})
                sent = True
            used = 0
            advance = False
            budget = self.replicate_turns if name == "replicate" else self.stage_turns
            while used < budget and turn < self.max_turns:
                messages = self._trim(messages)
                reply, _ = self._chat(messages)
                messages.append({"role": "assistant", "content": reply})
                transcript.append({"turn": turn, "role": "assistant", "content": reply, "stage": name})
                turn += 1
                used += 1
                if early is None:
                    early = extract_answer(reply)
                code = extract_code(reply)
                out = ""
                if code:
                    res = sandbox.run(code)
                    out = res.as_feedback()
                    messages.append({"role": "user", "content": out})
                    transcript.append({"turn": turn, "role": "tool", "content": out, "stage": name})
                if name == "trap":
                    m = SEPARATION_RE.search(out) or SEPARATION_RE.search(reply)
                    if m:
                        pl, rt, wr = (float(m.group(i)) for i in (1, 2, 3))
                        denom = abs(pl) if pl else 1.0
                        sep = abs(wr - pl) / denom > self.tol and abs(rt - pl) / denom <= self.tol
                        traps.append((pl, rt, wr, sep))
                        if sep or len(traps) >= self.max_repair:
                            trap_ok = sep
                            advance = True
                            break
                        messages.append({"role": "user", "content": STAGE_TRAP_RETRY.format(
                            wrong=wr, planted=pl, tol=self.tol)})
                        transcript.append({"turn": turn, "role": "tool",
                                           "content": f"[stage trap retry {len(traps)}]"})
                        used = 0
                elif name == "floor":
                    m = FLOOR_RE.search(out) or FLOOR_RE.search(reply)
                    if m:
                        ferr = float(m.group(1))
                        floors.append(ferr)
                        if ferr <= self.floor_tol or floor_repairs >= self.max_repair:
                            advance = True
                            break
                        floor_repairs += 1
                        messages.append({"role": "user", "content": STAGE_FLOOR_REPAIR.format(
                            err=ferr, tol=self.floor_tol)})
                        transcript.append({"turn": turn, "role": "tool",
                                           "content": f"[stage floor repair {floor_repairs}]"})
                        used = 0
                elif name == "calibrate":
                    m = SELFCHECK_RE.search(out) or SELFCHECK_RE.search(reply)
                    if m:
                        err = float(m.group(1))
                        checks.append(err)
                        if err <= self.tol or repairs >= self.max_repair:
                            advance = True
                            break
                        repairs += 1
                        messages.append({"role": "user", "content": STAGE_REPAIR.format(err=err, tol=self.tol)})
                        transcript.append({"turn": turn, "role": "tool", "content": f"[stage repair {repairs}]"})
                        used = 0
                elif code and not (res.stderr or "").strip():
                    advance = True
                    break
                elif not code:
                    advance = True                 # a stage that wants prose (READ) is done when it answers
                    break
            stage_i += 1
            sent = False
            if not advance and name == "calibrate":
                break                              # never produced a readable check: go straight to the answer
        # v2: a check is only trusted when the instance it was measured on defeated a wrong analysis.
        verified = bool(checks) and checks[-1] <= self.tol and trap_ok
        if verified and turn < self.max_turns - 2:
            messages.append({"role": "user", "content": STAGE_ABLATE})
            transcript.append({"turn": turn, "role": "tool", "content": "[stage ablate]"})
            messages = self._trim(messages)
            reply, _ = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            transcript.append({"turn": turn, "role": "assistant", "content": reply, "stage": "ablate"})
            turn += 1
            code = extract_code(reply)
            if code:
                out = sandbox.run(code).as_feedback()
                messages.append({"role": "user", "content": out})
                transcript.append({"turn": turn, "role": "tool", "content": out, "stage": "ablate"})
        transcript.append({"turn": turn, "role": "system",
                           "content": f"[selfcheck {'passed' if verified else 'not passed'}; "
                                      f"errors {', '.join(f'{c:.3f}' for c in checks) or 'none'}; repairs {repairs}"
                                      + (f"; trap separated {trap_ok} after {len(traps)} attempt(s)]" if self.require_trap else "")
                                      + (f"; floor {min(floors):.3f} after {floor_repairs} repair(s)]"
                                         if floors else ("; floor none]" if self.require_floor else "]"))})
        messages.append({"role": "user", "content": STAGE_APPLY if verified else STAGE_APPLY_UNVERIFIED})
        apply_budget = max(2, self.max_turns - turn)
        for i in range(apply_budget):
            messages = self._trim(messages)
            reply, _ = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            transcript.append({"turn": turn, "role": "assistant", "content": reply, "stage": "apply"})
            turn += 1
            ans = extract_answer(reply)
            if ans is not None:
                return ans, transcript
            code = extract_code(reply)
            if not code:
                messages.append({"role": "user", "content": self._force_msg(1)})
                continue
            out = sandbox.run(code).as_feedback()
            messages.append({"role": "user", "content": out})
            transcript.append({"turn": turn, "role": "tool", "content": out, "stage": "apply"})
            left = apply_budget - i - 1
            if self.force_tail and left < self.force_tail:      # the LAST force_tail apply turns, not one more
                # See force_tail above: the code branch used to fall straight through to the next turn, so a run
                # that kept producing code was never asked for an answer and simply ran out of budget.
                messages.append({"role": "user", "content": self._force_msg(max(left, 1))})
                transcript.append({"turn": turn, "role": "system",
                                   "content": f"[force_tail: {left} apply turn(s) left; answer demanded]"})
        if early is not None:
            transcript.append({"turn": turn, "role": "system",
                               "content": "[no answer after verification; falling back to the one offered earlier]"})
        return early, transcript


def make_agent(spec: str) -> Any:
    if spec.startswith("claude:"):   # 2026-10-25: the Anthropic Messages API, LLMAgent's loop otherwise (harness/anthropic_agent.py)
        from anthropic_agent import AnthropicAgent
        return AnthropicAgent(model=spec.split(":", 1)[1])
    if spec == "oracle":
        return OracleAgent()
    if spec == "null":
        return NullAgent()
    if spec.startswith("llm:"):
        model = spec.split(":", 1)[1]
        # "llm:MODEL:nothink" disables the reasoning trace. Thinking burns
        # thousands of tokens per turn and risks truncation in a tool loop;
        # the diagnostic showed Qwen3 emits a correct code block in 54 tokens
        # with thinking off.
        # PROMPTING BASELINES. A reviewer will ask whether the fine-tuning gain
        # is something a prompt buys. These modifiers answer that with the same
        # base weights, same harness, same grader -- only the prompt differs.
        #   :cot      -- explicit plan-then-verify instruction
        #   :fewshot  -- :cot plus a worked example of the answer contract
        # Both compose with :nothink, which is applied last.
        #   :train    -- render with the SFT training template (fine-tuned adapters)
        #   :rl       -- render with the RL template (rl/grpo_onpolicy.py policies)
        # Either bypasses the server's chat template; see render_prompt.
        #   :selfver  -- the self-verifying agent: it writes a simulator of the data it was
        #                given, checks its own analysis against a value it planted there, repairs
        #                until the check passes, and only then answers. Composes with the others.
        # `:ftail` composes with every other modifier, so it is stripped BEFORE the dispatch chain rather
        # than added as another elif -- the chain takes the first match and would swallow it.
        ftail = 0
        if ":ftail" in model:
            ftail, model = 3, model.replace(":ftail", "")
        cls, kw = LLMAgent, {}
        if ":general" in model:            # THE DELIVERABLE: the evidence-based default science agent.
            # Composition chosen cell-by-cell from the 2026-09-18/20 grid (docs/GENERALIZATION_PLAN.md):
            #  * small models (<20B): causal-order content + BOTH commitment devices (v23) -- 65.8% on the
            #    trap family vs 18% control, two draws, every component placebo-matched; costs under test.
            #  * large models (>=20B): bare causal-order (v12) -- the caution and (pending 277) the declaration
            #    wash out at 32B, and v12 there is a free fix: +18 on the trap family, no cost anywhere measured.
            # Scale-routing is OUR decision from measurements, not the model's self-diagnosis, which fails.
            _big = any(x in model for x in ("32B", "70B", "72B"))
            cls = SelfVerifyAgent
            kw = {"causal_order": True} if _big else {"routed_order": True, "no_overadjust": True}
            model = model.replace(":general", "")
        elif ":select" in model:             # M5: mode-level selection instruction (cross-domain test)
            cls, kw, model = SelfVerifyAgent, {"select_mode": True}, model.replace(":select", "")
        elif ":diag" in model:               # E3: the plain agent, plus a mode menu and one printed MODES line
            cls, kw, model = LLMAgent, {"diagnose": True}, model.replace(":diag", "")
        elif ":hetero" in model:             # M2: mode-level heterogeneity instruction (cross-domain test)
            cls, kw, model = SelfVerifyAgent, {"hetero_mode": True}, model.replace(":hetero", "")
        elif ":ident" in model:              # M6: mode-level identification instruction (cross-domain test)
            cls, kw, model = SelfVerifyAgent, {"ident_source": True}, model.replace(":ident", "")
        elif ":selfver26" in model:          # v26: GATED router -- four design checks decide the route, not a judgement
            cls, kw, model = SelfVerifyAgent, {"gated_order": True, "no_overadjust": True}, model.replace(":selfver26", "")
        elif ":selfver25" in model:          # v25: FALSIFY-stage placebo -- v9's scaffolding, no falsification content
            cls, kw, model = SelfVerifyAgent, {"placebo_claim": True}, model.replace(":selfver25", "")
        elif ":selfver24" in model:          # v24: commitment-placebo -- v20 scaffold + a causally-empty printed commitment
            cls, kw, model = SelfVerifyAgent, {"placebo_commit": True}, model.replace(":selfver24", "")
        elif ":selfver23" in model:          # v23: the routed declaration AND the caution -- additive or same mechanism?
            cls, kw, model = SelfVerifyAgent, {"routed_order": True, "no_overadjust": True}, model.replace(":selfver23", "")
        elif ":selfver22" in model:          # v22: causal-order + the route criteria as descriptive content only
            cls, kw, model = SelfVerifyAgent, {"causal_order": True, "route_context": True}, model.replace(":selfver22", "")
        elif ":selfver21" in model:          # v21: route from the task text, then causal-order or plain simulator
            cls, kw, model = SelfVerifyAgent, {"routed_order": True}, model.replace(":selfver21", "")
        elif ":selfver20" in model:          # v20: REPLICATE-stage placebo -- v12's scaffolding, no causal content
            cls, kw, model = SelfVerifyAgent, {"placebo_order": True}, model.replace(":selfver20", "")
        elif ":selfver19" in model:          # v19: candidate selection AND the over-adjustment caution
            cls, kw, model = SelfVerifyAgent, {"select_candidates": True, "no_overadjust": True}, model.replace(":selfver19", "")
        elif ":selfver18" in model:          # v18: causal order + a check against the real file
            cls, kw, model = SelfVerifyAgent, {"causal_order": True, "file_check": True}, model.replace(":selfver18", "")
        elif ":selfver17" in model:          # v17: causal-order simulator AND the timing sentence
            cls, kw, model = SelfVerifyAgent, {"causal_order": True, "no_overadjust": True}, model.replace(":selfver17", "")
        elif ":selfver16" in model:          # v16: placebo instruction
            cls, kw, model = SelfVerifyAgent, {"placebo": True}, model.replace(":selfver16", "")
        elif ":selfver14" in model:          # v14: enumerate variable timing, no rule
            cls, kw, model = SelfVerifyAgent, {"timing_only": True}, model.replace(":selfver14", "")
        elif ":selfver15" in model:        # v15: the rule, no enumeration
            cls, kw, model = SelfVerifyAgent, {"rule_only": True}, model.replace(":selfver15", "")
        elif ":selfver13" in model:          # v13: causal-order simulator AND candidate selection
            cls, kw, model = SelfVerifyAgent, {"causal_order": True, "select_candidates": True}, model.replace(":selfver13", "")
        elif ":selfver12" in model:          # v12: the simulator written in causal order
            cls, kw, model = SelfVerifyAgent, {"causal_order": True}, model.replace(":selfver12", "")
        elif ":selfver11" in model:          # v11: propose three analyses, select by the planted value
            cls, kw, model = SelfVerifyAgent, {"select_candidates": True}, model.replace(":selfver11", "")
        elif ":selfver10" in model:          # v10: falsify the description AND warn against over-adjustment
            cls, kw, model = SelfVerifyAgent, {"check_claim": True, "no_overadjust": True}, model.replace(":selfver10", "")
        elif ":selfver9" in model:           # v9: state the process, then try to falsify it on the real data
            cls, kw, model = SelfVerifyAgent, {"check_claim": True}, model.replace(":selfver9", "")
        elif ":selfver8" in model:           # v8: the generating process in prose, never executed
            cls, kw, model = SelfVerifyAgent, {"describe_only": True}, model.replace(":selfver8", "")
        elif ":selfver7" in model:           # v7: the loop plus a caution against over-adjustment
            cls, kw, model = SelfVerifyAgent, {"no_overadjust": True}, model.replace(":selfver7", "")
        elif ":selfver6" in model:           # v6: the simulator must model the observation process, not just values
            cls, kw, model = SelfVerifyAgent, {"require_trap": True, "trap_menu": True, "two_stage": True}, \
                model.replace(":selfver6", "")
        elif ":selfver0" in model:           # plan-only: the READ stage alone, no simulator and no check
            cls, kw, model = SelfVerifyAgent, {"plan_only": True}, model.replace(":selfver0", "")
        elif ":selfver5" in model:           # v5: v3 plus a noiseless floor check before the realistic one
            cls, kw, model = SelfVerifyAgent, {"require_trap": True, "trap_menu": True, "require_floor": True}, \
                model.replace(":selfver5", "")
        elif ":selfver4" in model:            # v4: the oracle arm -- this family's documented wrong analyses are handed over
            cls, model = SelfVerifyAgent, model.replace(":selfver4", "")
            kw = {"require_trap": True, "trap_menu": True, "trap_hint": os.environ.get("SELFVER_TRAP_HINT", "")}
        elif ":selfver3" in model:          # v3: v2 plus a menu of the usual failure modes
            cls, kw, model = SelfVerifyAgent, {"require_trap": True, "trap_menu": True}, model.replace(":selfver3", "")
        elif ":selfver2" in model:          # v2: the synthetic instance must defeat a wrong analysis first
            cls, kw, model = SelfVerifyAgent, {"require_trap": True}, model.replace(":selfver2", "")
        elif ":selfver" in model:
            cls, model = SelfVerifyAgent, model.replace(":selfver", "")
        render = None
        for tag in (":train", ":rl", ":mtrain", ":ltrain"):
            if model.endswith(tag) or f"{tag}:" in model:
                render = tag[1:]
                model = model.replace(tag, "")
                break
        if render:
            model = model.replace(":nothink", "")
            return cls(model=model, render=render, **kw)
        if ftail and cls is SelfVerifyAgent:
            kw["force_tail"] = ftail
        style = None
        for tag in (":fewshot", ":cot"):
            if tag in model:
                style = tag[1:]
                model = model.replace(tag, "")
                break
        if model.endswith(":nothink"):
            return cls(model=model[: -len(":nothink")],
                       enable_thinking=False, prompt_style=style, **kw)
        if style:
            return cls(model=model, prompt_style=style, **kw)
        return cls(model=model, **kw)
    raise ValueError(f"unknown agent: {spec!r} (use oracle | null | llm:<model>)")
