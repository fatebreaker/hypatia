"""On-policy GRPO for the agentic sandbox, on ONE GPU beside a vLLM server.

WHY THIS FILE EXISTS
--------------------
rl/grpo_train.py sampled every step from the adapter the server started with and
never pushed updated weights back (RESULTS §18 retracted its flat curve for
exactly that). The paper's RL used verl instead. This is the same algorithm as
the verl runs -- episode-level GRPO, group-standardised advantages, clipped
ratio, no KL, the step-attributed reward of rl/reward.py, the same sandbox and
turn budget, 8 prompts x 6 rollouts per step, actor LR 1e-6 -- made on-policy by
hot-loading each new adapter into the running vLLM server (runtime LoRA
updating) before the next step's rollouts:

    step t:  rollouts from served adapter pol<t>  ->  update  ->  save  ->  load pol<t+1>

WHAT IT IS FOR
--------------
Whether RL can start from zero success. Runs from the untrained base (a fresh,
zero-initialised LoRA, so pol0 == base) on a family the base never solves, and
from a family where it sometimes does, plus a positive control from an SFT
initialisation that verl is known to improve -- without which a null here would
only show that this trainer does not work.

    venv_vllm/bin/python rl/grpo_onpolicy.py --family popgen_pulse --port 18731 \
        --out $DEEP_DATA_ROOT/ckpt/rl-popgen-base-s1 --steps 200 --seed 1
    ... --init-adapter /path/to/sft/adapter          # start from an SFT adapter
    ... --smoke                                     # 2 tiny steps + assertions
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from paths import data_root  # noqa: E402
from rl.dispersion import instance_tol, one_rollout  # noqa: E402
from rl.grpo_train import advantages, encode, seq_logprobs  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from rl.sft_train import resolve_model  # noqa: E402
from rl.chat_formats import LLAMA3_TRAIN as LLAMA3, MISTRAL_TRAIN as MISTRAL  # noqa: E402

# The episode is sampled through harness render mode "rl" (agents.render_prompt):
# every assistant turn, past and current, carries Qwen3's empty no-think block.
# Scoring the episode under the SAME rendering makes the single training pass
# exact: each turn's log-probs are computed in precisely the context it was
# sampled in. (Scoring it under sft_train's TRAIN_TEMPLATE, as the first
# version did, drops the block the policy saw right before every reply.)
RL_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}"
    "{% generation %}{{- message['content'] + '<|im_end|>' + '\n' }}{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{%- endif %}"
)


def encode_rl(tok, messages: list[dict], max_len: int):
    """Token ids + assistant-content mask under RL_TEMPLATE (cf. grpo_train.encode)."""
    enc = tok.apply_chat_template(messages, chat_template=RL_TEMPLATE, tokenize=True,
                                  return_dict=True, return_assistant_tokens_mask=True,
                                  truncation=True, max_length=max_len)
    ids = torch.tensor(enc["input_ids"][:max_len], dtype=torch.long)
    mask = torch.tensor(enc["assistant_masks"][:max_len], dtype=torch.long)
    return ids, mask


def mask_prefix(tok, messages: list[dict], ids: torch.Tensor, mask: torch.Tensor, n_prefix: int, max_len: int):
    """Zero the gradient mask over the first n_prefix messages (system + an in-context library): the policy did not
    write the library's assistant turns, so crediting them would train on tokens it never sampled. RL_TEMPLATE renders
    every message independently and concatenates, so the prefix's tokens are exactly the episode's first tokens --
    checked, not assumed; a mismatch stops the run rather than training on a misaligned mask."""
    if n_prefix <= 0:
        return mask
    pre = tok.apply_chat_template(messages[:n_prefix], chat_template=RL_TEMPLATE, tokenize=True,
                                  return_dict=True, truncation=True, max_length=max_len)["input_ids"]
    n = min(len(pre), len(ids))
    if ids[:n].tolist() != list(pre[:n]):
        raise SystemExit("library prefix does not tokenise to the episode's first tokens -- refusing to mask blindly")
    mask = mask.clone()
    mask[:n] = 0
    return mask


def encode_with(tok, messages: list[dict], max_len: int, template: str):
    """Token ids + assistant mask under an arbitrary template (cf. encode_rl)."""
    enc = tok.apply_chat_template(messages, chat_template=template, tokenize=True,
                                  return_dict=True, return_assistant_tokens_mask=True,
                                  truncation=True, max_length=max_len)
    return (torch.tensor(enc["input_ids"][:max_len], dtype=torch.long),
            torch.tensor(enc["assistant_masks"][:max_len], dtype=torch.long))


def check_rl_parity(tok) -> None:
    """RL_TEMPLATE must render exactly what the harness sends in mode "rl"."""
    from agents import render_prompt
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
            {"role": "user", "content": "STDOUT:\n1"}]
    a = tok.apply_chat_template(msgs, chat_template=RL_TEMPLATE, tokenize=False,
                                add_generation_prompt=True)
    if a != render_prompt(msgs, "rl"):
        raise SystemExit("RL_TEMPLATE and agents.render_prompt('rl') disagree -- the "
                         "policy would be scored in a context it was not sampled in")
    print("template parity: RL_TEMPLATE == harness render mode 'rl'", flush=True)


AFTER_ROLLOUTS: list = []                      # callables run once a step's rollouts are done, before the gradient step
BEFORE_ROLLOUTS: list = []                     # callables run before a step's rollouts (e.g. the second server's health)


def post(url: str, payload: dict) -> None:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 200:
            raise RuntimeError(f"{url} -> HTTP {r.status}")


class Server:
    """The vLLM server's runtime-LoRA endpoints, used to publish each new policy."""

    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}/v1"
        self.root = f"http://127.0.0.1:{port}"
        self.live: str | None = None
        self.asleep = False

    # 2026-09-29 'RP_1 on both GPUs': a second server on the TRAINER's GPU sleeps (weights to host RAM, KV cache freed)
    # during the gradient step and wakes before the next policy is published, so it can hold a large KV cache during
    # rollouts without competing with the backward pass for memory.
    def _wait_state(self, want: bool, limit: float = 120.0) -> None:
        # vllm/entrypoints/serve/dev/sleep: "the sleep command is sent but does not finish yet when we return a
        # response" -- poll /is_sleeping so the trainer never allocates into memory the server has not released yet
        t_end = time.time() + limit
        while time.time() < t_end:
            with urllib.request.urlopen(f"{self.root}/is_sleeping", timeout=30) as r:
                if bool(json.loads(r.read()).get("is_sleeping")) == want:
                    return
            time.sleep(1)
        raise RuntimeError(f"{self.root} did not reach is_sleeping={want} within {limit:.0f} s")

    def sleep(self) -> None:
        post(f"{self.root}/sleep?level=1", {})
        self._wait_state(True)
        self.asleep = True

    def wake(self) -> None:
        post(f"{self.root}/wake_up", {})
        self._wait_state(False)
        self.asleep = False

    def publish(self, name: str, path: Path) -> None:
        post(f"{self.base}/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        if self.live and self.live != name:
            try:
                post(f"{self.base}/unload_lora_adapter", {"lora_name": self.live})
            except Exception as exc:                                   # noqa: BLE001
                print(f"  warn: unload {self.live}: {exc}", flush=True)
        self.live = name


def lean_logprobs(model, ids: torch.Tensor, mask: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Log-probs of the realised next tokens at the positions `mask` selects.

    grpo_train.seq_logprobs materialises the full-vocabulary logits of the
    whole episode in fp32: for a 16k-token episode that is ~10 GB, and ~40 GB
    across log_softmax and its backward -- next to a vLLM server holding half
    the GPU, the first long episode would OOM the run. Here the head is applied
    only to the positions that carry loss, a chunk at a time, under activation
    checkpointing, so at most one chunk's logits are ever alive. Same values:
    the smoke test asserts agreement with seq_logprobs.
    """
    inner = model.get_base_model()
    h = inner.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    sel = mask.bool()
    h, tgt = h[sel], ids[0, 1:][sel]

    def piece(hc, tc):
        lg = inner.lm_head(hc).float()
        return lg.gather(-1, tc[:, None]).squeeze(-1) - torch.logsumexp(lg, dim=-1)

    return torch.cat([checkpoint(piece, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
                      for i in range(0, len(tgt), chunk)])


HOST_CKPT_MIN_TOKENS = 8192


def host_checkpoint(fn, *args, **kw):
    """Activation checkpointing that keeps each layer's saved INPUT in host memory between the forward and backward pass
    (10-15). Plain non-reentrant checkpointing keeps every decoder layer's input on the GPU: for a 24k-token 27B episode
    that is ~64 x 24k x 5120 x 2 bytes (~16 GB) next to 54 GB of weights, and 4 of the 8 sequences of the first ASTRO27
    smoke step were skipped for OOM. The input is copied to the host, and the recomputation copies it back; the copies
    are exact, so values and gradients are unchanged -- only where the saved input lives changes. Short episodes, and
    passes without gradients, take the plain path."""
    x = args[0] if args else None
    if not (torch.is_grad_enabled() and torch.is_tensor(x) and x.is_cuda and x.dim() == 3
            and x.shape[1] >= HOST_CKPT_MIN_TOKENS):
        return checkpoint(fn, *args, use_reentrant=False, **kw)
    dev = x.device

    def run(xh, *rest):
        return fn(xh.to(dev), *rest)
    return checkpoint(run, x.to("cpu"), *args[1:], use_reentrant=False, **kw)


def use_host_checkpoint(model, max_len: int) -> bool:
    """On for models above 20B parameters trained at a context above 16k (ASTRO27) on a GPU under 100 GB; every other
    run keeps the plain path (10-16: a 141 GB H200 holds the inputs, so the host copies only cost time there).
    GRPO_HOST_CKPT=1/0 overrides."""
    env = os.environ.get("GRPO_HOST_CKPT")
    if env in ("0", "1"):
        return env == "1"
    return (max_len > 16384 and sum(p.numel() for p in model.parameters()) > 20e9
            and torch.cuda.get_device_properties(0).total_memory < 100e9)


def lora_norm(model) -> float:
    """||B|| over all LoRA B matrices: 0 at a fresh init, so it reads directly as
    how far the policy has moved from the base (a null needs this to be > 0)."""
    return float(sum(p.detach().float().pow(2).sum() for n, p in model.named_parameters()
                     if "lora_B" in n) ** 0.5)


def reward_steps(cfg: dict, default: int = 5) -> int:
    """The attributed reward's step scale S for ONE instance -- the same formula main() applies to a family's first
    instance. Mixed-family pools (RL_PLAN.md R1/R2) need it per instance: families differ in S."""
    return max([v.get("fails_at_step") for v in (cfg.get("_provenance", {}).get("ablations") or {}).values()
                if v.get("fails_at_step")], default=default)


def family_of(inst: Path) -> str:
    """generated/<family>/train/<instance> -- resolved, so mixed pools built from symlinks (rlmix6) still know it."""
    return inst.resolve().parent.parent.name


def transfer_checks(groups, chosen, pools, spec, args, rng):
    """RL_PLAN.md R2, the GENERALITY reward. A passing rollout keeps its reward only in proportion to how many of
    args.transfer_check checks pass: its own transcript (system message dropped) is placed in context as a worked
    example, and the CURRENT policy solves another train instance of the same family with it. A lucky program --
    right on its own instance, wrong elsewhere (8 of 19 correct attempts, item 450) -- earns little; a general one
    keeps its full reward. Failed rollouts keep their proximity credit unchanged. Returns (checked, passed)."""
    by_name = {i.name: i for i in chosen}
    jobs = []
    for name, outs in groups.items():
        src = by_name[name]; fam = family_of(src)
        others = [p for p in pools[fam] if p.name != name]
        for o in outs:
            if o.get("passed") and o.get("messages"):
                demo = [m for m in o["messages"] if m.get("role") != "system"]
                for tgt in rng.sample(others, min(args.transfer_check, len(others))):
                    jobs.append((o, demo, tgt))
    if not jobs:
        return 0, 0

    def go(job):
        o, demo, tgt = job
        cfg = json.loads((tgt / "eval_config.json").read_text())
        r = one_rollout(cfg, tgt / "data_files", spec, args.temperature, instance_tol(cfg, 0.025), args.timeout,
                        reward_steps(cfg), partial_ceiling=args.c, demo=demo)
        return id(o), bool(r.get("passed"))

    res = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for oid, ok in ex.map(go, jobs):
            res.setdefault(oid, []).append(ok)
    for outs in groups.values():
        for o in outs:
            if id(o) in res:
                frac = sum(res[id(o)]) / len(res[id(o)])
                o["transfer_frac"] = frac
                o["reward"] = o["reward"] * frac
    return len(jobs), sum(ok for v in res.values() for ok in v)


def rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
    """k rollouts per instance, all in flight at once (bounded by `workers`).
    n_steps=None -> each instance's own reward step scale (reward_steps), for mixed-family pools.
    demo_for(inst) -> message list: a worked-example LIBRARY placed in context AND kept in the stored episode
    (RL_PLAN.md R3); its turns are masked out of the gradient in train()."""
    jobs = [(i, r) for i in insts for r in range(k)]

    def go(job):
        inst, _ = job
        cfg = json.loads((inst / "eval_config.json").read_text())
        t = time.time()
        demo = demo_for(inst) if demo_for else None
        out = one_rollout(cfg, inst / "data_files", spec, temperature,
                          instance_tol(cfg, 0.025), timeout,
                          reward_steps(cfg) if n_steps is None else n_steps, partial_ceiling=c,
                          demo=demo, demo_in_messages=demo is not None)
        out["seconds"] = round(time.time() - t, 1)
        if require_code and not (out.get("n_cells") or 0):
            # No credit without analysis: RL from the base otherwise learns to
            # answer without executing a cell (a guess earns the proximity
            # credit and never overflows the context). "passed" is left as
            # graded, so the logged solve rate stays honest.
            out["reward"] = 0.0
        return inst.name, out

    groups: dict[str, list[dict]] = {i.name: [] for i in insts}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, out in ex.map(go, jobs):
            groups[name].append(out)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="directory under generated/ (uses its train/)")
    ap.add_argument("--init-adapter", default=None, help="SFT adapter to start from; omit = base")
    ap.add_argument("--port", type=int, required=True, help="the pod's vLLM server")
    ap.add_argument("--port2", type=int, default=0,
                    help="a SECOND vLLM server of the same base model (docs/RL_PLAN.md 2026-09-29 'RP_1 on both GPUs'): "
                         "every new policy is published to both and episodes are spread over them (harness/agents.py "
                         "round-robin per agent). Sampling is unchanged -- same policy, same parameters; only throughput")
    ap.add_argument("--port2-sleep", action="store_true",
                    help="the --port2 server shares the trainer's GPU: put it to sleep after each step's rollouts and "
                         "wake it before the next publish. If it ever fails (sleep, wake, publish), it is dropped from "
                         "the rollout rotation and the run continues on --port alone -- never a crash, never errored "
                         "rollouts from a dead server")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--render", choices=["rl", "train", "mtrain", "ltrain"], default="rl",
                    help="rl: base-model no-think format (fresh adapter); train: the SFT format "
                         "(an --init-adapter was trained in it and must be sampled and scored in it)")
    ap.add_argument("--require-code", action="store_true",
                    help="zero reward for any rollout that executed no cell")
    ap.add_argument("--c", type=float, default=0.5,
                    help="reward partial-credit ceiling c (paper: 0.5; 0 = binary)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=48, help="episodes in flight (48 = the whole step)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=25, help="adapter kept for evaluation")
    ap.add_argument("--resume-every", type=int, default=5, help="rolling adapter+optimizer")
    ap.add_argument("--live-dir", default="/tmp/grpo_live", help="fast disk for per-step adapters")
    ap.add_argument("--kl", type=float, default=0.0,
                    help="beta of a per-token k3 KL penalty to the untrained model (DeepSeekMath GRPO), whose "
                         "log-probs are the policy's with its LoRA switched off; 0 = none, the paper's recipe. "
                         "Runs from the base only")
    ap.add_argument("--overflow-neutral", action="store_true",
                    help="leave errored rollouts (context overflow, sandbox failure) out of the group's mean and std "
                         "as well as the gradient; by default their reward 0 enters the group statistics")
    ap.add_argument("--library", default=None,
                    help="R3: JSON manifest {family: code-only worked-example file}; each prompt carries the programs of "
                         "every OTHER family in it (never its own), kept in context and masked out of the gradient")
    ap.add_argument("--transfer-check", type=int, default=0,
                    help="R2 generality reward: a passing rollout's reward x the fraction of this many transfer checks "
                         "(its transcript as a worked example on other same-family train instances) that pass; 0 = off")
    ap.add_argument("--per-instance-steps", action="store_true",
                    help="reward step scale S per instance, not from the pool's first instance (mixed-family pools)")
    ap.add_argument("--wall-budget", type=float, default=0,
                    help="seconds of wall clock per rollout before it is made to answer (harness/agents.py "
                         "wall_budget; RL_PLAN.md 2026-09-24 straggler cap); 0 = none")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.wall_budget:                  # read by every agent the rollout threads construct (make_agent)
        os.environ["GBP_WALL_BUDGET"] = str(args.wall_budget)
    if args.kl and args.init_adapter:
        ap.error("--kl penalises distance from the untrained model; it is for runs from the base")
    if args.smoke:
        args.steps, args.prompts_per_step, args.k = 2, 2, 4
    # The agent reads its endpoint from GBP_BASE_URL and otherwise defaults to a
    # public API -- where every rollout fails instantly, scores 0, and the run
    # "trains" on nothing. Point it at this pod's server, always.
    os.environ["GBP_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"
    if args.port2:
        os.environ["GBP_BASE_URL"] += f",http://127.0.0.1:{args.port2}/v1"
        print(f"rollout servers: {os.environ['GBP_BASE_URL']}", flush=True)
    # This trainer runs in venv_vllm (torch + vLLM client); the analysis stack
    # (scipy, statsmodels, sklearn, ...) lives in venv, where every evaluation
    # runs its cells. Cells must run THERE, or training rollouts execute a
    # different Python from the one they are scored under -- venv_vllm has no
    # scipy at all, so the first two launches of this trainer were void.
    os.environ.setdefault("GBP_SANDBOX_PYTHON",
                          str(Path(__file__).resolve().parent.parent / "venv" / "bin" / "python"))
    from sandbox import preflight, sandbox_python  # noqa: E402
    preflight(strict=True)
    print(f"sandbox interpreter: {sandbox_python()} (preflight OK)", flush=True)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    live = Path(args.live_dir) / out.name; live.mkdir(parents=True, exist_ok=True)
    hist_f = out / "history.jsonl"

    base = resolve_model()
    tok = AutoTokenizer.from_pretrained(base)
    if args.render == "rl":
        check_rl_parity(tok)
    elif args.render == "mtrain":
        from rl.chat_formats import MISTRAL_TRAIN, render_mistral
        _m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
              {"role": "assistant", "content": "a"}, {"role": "user", "content": "o"}]
        assert tok.apply_chat_template(_m, chat_template=MISTRAL_TRAIN, tokenize=False,
                                       add_generation_prompt=True) == "<s>" + render_mistral(_m)
        print("template parity: MISTRAL_TRAIN == harness render mode 'mtrain'", flush=True)
    elif args.render == "ltrain":
        from rl.chat_formats import render_llama3
        _m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
              {"role": "assistant", "content": "a"}, {"role": "user", "content": "o"}]
        assert tok.apply_chat_template(_m, chat_template=LLAMA3, tokenize=False,
                                       add_generation_prompt=True) == "<|begin_of_text|>" + render_llama3(_m)
        print("template parity: LLAMA3_TRAIN == harness render mode 'ltrain'", flush=True)
    else:
        from rl.sft_train import TRAIN_TEMPLATE, check_template_parity
        check_template_parity(tok, TRAIN_TEMPLATE)
        from agents import render_prompt
        _m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        assert tok.apply_chat_template(_m, chat_template=TRAIN_TEMPLATE, tokenize=False,
                                       add_generation_prompt=True) == render_prompt(_m, "train")
        print("template parity: TRAIN_TEMPLATE == harness render mode 'train'", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")

    # Resume from the newest checkpoint if one exists (pods end; long runs must survive).
    # Resume point: out/resume (rolling, every --resume-every steps, with the
    # optimizer), else the newest out/stepN. A crash costs at most a few steps.
    ckpts = sorted((p for p in out.glob("step*") if (p / "adapter_model.safetensors").exists()),
                   key=lambda p: int(p.name[4:]))
    res = out / "resume"
    if (res / "state.json").exists():
        ckpts.append(res)
    start = 0
    if ckpts:
        src = ckpts[-1]
        model = PeftModel.from_pretrained(model, str(src), is_trainable=True)
        start = (json.loads((src / "state.json").read_text())["step"] if src == res
                 else int(src.name[4:]))
        print(f"resumed from {src} at step {start}")
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"init adapter: {args.init_adapter}")
    else:
        # Fresh LoRA with B = 0: the first policy is exactly the base model.
        model = get_peft_model(model, LoraConfig(
            r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        print("init: fresh zero-initialised LoRA (policy 0 == base)")
    # Dropout would make the recomputed log-probs noisy; the objective assumes
    # the trainer's policy is the sampled one.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if use_host_checkpoint(model, args.max_len):
        n_host = 0
        for mod in model.modules():
            if getattr(mod, "gradient_checkpointing", False) and hasattr(mod, "_gradient_checkpointing_func"):
                mod._gradient_checkpointing_func = host_checkpoint
                n_host += 1
        print(f"host checkpointing: {n_host} modules keep their saved inputs in host memory for episodes of "
              f">= {HOST_CKPT_MIN_TOKENS} tokens", flush=True)
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    if ckpts and (ckpts[-1] / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(ckpts[-1] / "optimizer.pt", map_location="cuda"))
    elif ckpts:
        print("  warn: no optimizer state at the resume point; Adam moments restart")
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M params")

    root = data_root() / "generated" / args.family / "train"
    insts = sorted(p for p in root.iterdir() if (p / "eval_config.json").is_file())
    probe = json.loads((insts[0] / "eval_config.json").read_text())
    n_steps_fam = max([v.get("fails_at_step") for v in
                       (probe.get("_provenance", {}).get("ablations") or {}).values()
                       if v.get("fails_at_step")], default=5)
    args.pools = {}
    for p_ in insts:
        args.pools.setdefault(family_of(p_), []).append(p_)
    args.library_for = None
    if args.library:
        if args.render != "rl":
            raise SystemExit("--library masks the library by token prefix under RL_TEMPLATE; use --render rl")
        from agents import ICL_BRIDGE
        man = json.loads(Path(args.library).read_text())
        progs = {f: json.loads(Path(v).read_text()) for f, v in sorted(man.items())}
        libs = {}
        for fam in args.pools:
            parts = [progs[f] for f in sorted(progs) if f != fam]
            stacked = []
            for i, m in enumerate(parts):
                stacked += m if i == 0 else [{**m[0], "content": ICL_BRIDGE + m[0]["content"]}] + m[1:]
            libs[fam] = stacked
        args.library_for = lambda inst: libs[family_of(inst)]
        print(f"library ON (R3): each prompt carries {len(progs) - 1} other families' programs; "
              f"~{max(sum(len(x['content']) for x in v) for v in libs.values()) // 1000}k chars max")
    if args.transfer_check:
        print(f"generality reward ON: {args.transfer_check} transfer checks per passing rollout; families "
              f"{ {f: len(v) for f, v in args.pools.items()} }")
    if args.per_instance_steps:
        fams = sorted({p.name.rsplit("_", 1)[0] for p in insts})
        print(f"train pool: {len(insts)} instances from {len(fams)} families {fams}; reward steps S PER INSTANCE")
        n_steps_fam = None
    else:
        print(f"train pool: {len(insts)} {args.family} instances; reward steps S={n_steps_fam}")
    for _ in range(start):                          # keep the instance stream reproducible on resume
        rng.sample(insts, args.prompts_per_step)

    servers = [Server(args.port)] + ([Server(args.port2)] if args.port2 else [])

    def drop_second(why: str) -> None:
        if len(servers) > 1:
            servers.pop()
            os.environ["GBP_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"
            print(f"  WARN: second rollout server dropped ({why}); continuing on port {args.port} alone", flush=True)

    def sleep_second() -> None:
        if args.port2_sleep and len(servers) > 1:
            try:
                servers[1].sleep()
            except Exception as exc:                                   # noqa: BLE001
                drop_second(f"sleep failed: {exc}")
    AFTER_ROLLOUTS[:] = [sleep_second]

    def check_second() -> None:
        if len(servers) > 1:
            try:
                with urllib.request.urlopen(f"{servers[1].root}/health", timeout=30) as r:
                    if r.status != 200:
                        raise RuntimeError(f"HTTP {r.status}")
            except Exception as exc:                                   # noqa: BLE001
                drop_second(f"health check failed: {exc}")
    BEFORE_ROLLOUTS[:] = [check_second]
    # Names must be unique per run INSTANCE: the server outlives runs, and
    # re-registering a name it already holds (a restarted or resumed run) is an
    # HTTP 400. The tag also makes every served policy traceable to its run.
    tag = f"{out.name}-{int(time.time()) % 100000:05d}"

    def publish(step: int) -> None:
        d = live / f"pol{step}"
        if d.exists():
            shutil.rmtree(d)
        model.save_pretrained(str(d))
        for i, server in enumerate(list(servers)):
            if i == 0:
                server.publish(f"{tag}-pol{step}", d)        # the primary server: a failure here stops the run
                continue
            try:
                if server.asleep:
                    server.wake()
                server.publish(f"{tag}-pol{step}", d)
            except Exception as exc:                                   # noqa: BLE001
                drop_second(f"wake/publish failed: {exc}")
        stale = live / f"pol{step - 2}"
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    publish(start)
    try:
        train(args, model, tok, opt, trainable, insts, rng, start, n_steps_fam, tag,
              publish, hist_f, out)
    finally:
        # Leave the shared server as we found it: resident adapters are not free
        # (the paper's BixBench base score moved 0.006 -> 0.301 with their number).
        for server in servers:
            if server.live:
                try:
                    post(f"{server.base}/unload_lora_adapter", {"lora_name": server.live})
                except Exception as exc:                               # noqa: BLE001
                    print(f"  warn: final unload {server.live}: {exc}", flush=True)
    if args.smoke:
        print("smoke: OK (advantages centred, mask sane, loss finite, lean logprobs exact, "
              "policy published and served)")
    return 0


def train(args, model, tok, opt, trainable, insts, rng, start, n_steps_fam, tag,
          publish, hist_f, out) -> None:
    for step in range(start + 1, args.steps + 1):
        t0 = time.time()
        chosen = rng.sample(insts, args.prompts_per_step)
        spec = f"llm:{tag}-pol{step - 1}:{args.render}"
        for hook in BEFORE_ROLLOUTS:
            hook()
        groups = rollouts(chosen, spec, args.k, args.temperature, args.timeout,
                          n_steps_fam, args.workers, args.c, args.require_code,
                          demo_for=getattr(args, "library_for", None))
        tc_checked = tc_passed = 0
        if args.transfer_check:
            tc_checked, tc_passed = transfer_checks(groups, chosen, args.pools, spec, args, rng)
        t_roll = time.time() - t0
        for hook in AFTER_ROLLOUTS:                    # e.g. put the trainer-GPU rollout server to sleep (--port2-sleep)
            hook()

        allouts = [o for outs in groups.values() for o in outs]
        errored = sum(str(o.get("attribution") or "").startswith("error") for o in allouts)
        cells = sum(int(o.get("n_cells") or 0) for o in allouts)
        # Rule 1: a step in which nothing actually ran is a broken instrument, not
        # a policy that scores zero. Stop rather than log a plausible flat curve.
        # But "no cells" with answers given is the POLICY answering without
        # looking at the data -- RL from the base on wf got there in 20 steps,
        # because a guess beat the base's own analyses. That is a result, and the
        # first version of this guard stopped the run on it.
        answered = sum(bool(o.get("answered")) for o in allouts)
        if errored == len(allouts) or (cells == 0 and answered == 0):
            kinds = sorted({str(o.get("attribution")) for o in allouts})[:5]
            raise SystemExit(f"step {step}: {errored}/{len(allouts)} rollouts errored, "
                             f"{cells} cells executed ({kinds}) -- instrument failure")
        batch, rewards, passes, dead = [], [], [], 0
        prefixes = []                                  # R3: leading messages (system + library) not written by the policy
        for outs in groups.values():
            rs = [o["reward"] for o in outs]
            rewards += rs
            passes += [bool(o.get("passed")) for o in outs]
            # An errored rollout has no messages, so it never enters the gradient; by default its reward 0 still
            # enters the group's mean and std, which scores an analysis that overflowed the context as a failed
            # one. --overflow-neutral leaves errored rollouts out of the group statistics too.
            scored = [o for o in outs
                      if not (args.overflow_neutral and str(o.get("attribution") or "").startswith("error"))]
            adv = advantages([o["reward"] for o in scored])
            dead += all(abs(a) < 1e-12 for a in adv)
            for o, a in zip(scored, adv):
                if abs(a) > 1e-9 and o.get("messages"):
                    batch.append((o["messages"], a))
                    prefixes.append(int(o.get("n_prefix") or 0))
            if args.smoke and len(adv) > 1 and statistics.pstdev([o["reward"] for o in scored]) > 0:
                assert abs(sum(adv)) < 1e-6, f"advantages sum to {sum(adv)}"

        loss_sum, nseq, mask_frac, nll_sum, gnorm, oom, kl_sum = 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0
        if batch:
            opt.zero_grad(set_to_none=True)
            enc = [encode_rl(tok, m, args.max_len) if args.render == "rl"
                   else encode_with(tok, m, args.max_len, MISTRAL) if args.render == "mtrain"
                   else encode_with(tok, m, args.max_len, LLAMA3) if args.render == "ltrain"
                   else encode(tok, m, args.max_len) for m, _ in batch]
            if any(prefixes):                          # R3: no gradient through the library's turns
                enc = [(ids, mask_prefix(tok, m, ids, mk, npre, args.max_len))
                       for (ids, mk), (m, _), npre in zip(enc, batch, prefixes)]
            tot_tok = sum(len(ids) for ids, _ in enc)
            mask_frac = sum(int(mk.sum()) for _, mk in enc) / max(tot_tok, 1)
            for (ids, mk), (_, a) in zip(enc, batch):
                ids = ids.unsqueeze(0).cuda()
                m = mk[1:].cuda().float()
                if m.sum() < 1:
                    continue
                # One update per step on freshly sampled data, as in the verl runs
                # (mini-batch = batch): the behaviour policy IS the current one, so
                # the ratio is exactly 1 and new - new.detach() gives it without a
                # second forward pass. The clip is kept for fidelity to the objective.
                try:
                    new = lean_logprobs(model, ids, m)  # assistant positions only
                    if args.kl:
                        with torch.no_grad(), model.disable_adapter():
                            ref_lp = lean_logprobs(model, ids, m)
                except torch.cuda.OutOfMemoryError:
                    # The GPU is shared with a vLLM server and, at times, an SFT
                    # job. Skip this episode rather than lose the run; counted.
                    torch.cuda.empty_cache(); oom += 1
                    continue
                if args.smoke and nseq == 0 and ids.shape[1] > 8192:
                    # 10-14: skip the full-logit check for long sequences -- the full-vocabulary logits of a 24-32k-token 27B sequence do
                    # not fit next to the weights on an 80 GB trainer GPU (smoke OOMs at 16:31/19:58), and a PREFIX comparison is not exact on
                    # the linear-attention (fla) layers (0.68 max diff at 22:00). The check still runs whenever the step's first episode is short.
                    print(f"  smoke: lean-vs-full check skipped for a {ids.shape[1]}-token episode (memory)", flush=True)
                elif args.smoke and nseq == 0:
                    with torch.no_grad():
                        ref = seq_logprobs(model, ids, torch.ones_like(ids)).squeeze(0)[m.bool()]
                    err = float((ref - new.detach()).abs().max())
                    assert err < 0.05, f"lean logprobs disagree with seq_logprobs by {err}"
                    print(f"  smoke: lean vs full logprobs max |diff| = {err:.2e}", flush=True)
                    del ref
                ratio = torch.exp(new - new.detach())
                obj = torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a)
                loss = -obj.mean()                     # token mean within an episode
                if args.kl:
                    d = ref_lp - new                   # k3: exp(d) - d - 1 >= 0, unbiased for KL(policy || base)
                    kl = (torch.exp(d) - d - 1).mean()
                    if args.smoke and step == 1:
                        assert float(kl) < 1e-3, f"fresh adapter should equal the base, KL {float(kl):.2e}"
                    loss = loss + args.kl * kl
                    kl_sum += float(kl.detach())
                    del ref_lp, d, kl
                try:
                    (loss / len(batch)).backward()    # mean over episodes
                except torch.cuda.OutOfMemoryError:
                    del new, ratio, obj, loss
                    torch.cuda.empty_cache(); oom += 1
                    continue
                loss_sum += float(loss.detach()); nseq += 1
                nll_sum += float(-new.detach().mean())
                del new, ratio, obj, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            # Hand the backward pass's peak back to the driver: the next ~6
            # minutes are rollouts, and the GPU is shared with vLLM and, at
            # times, an SFT job that needs that memory more than our cache does.
            torch.cuda.empty_cache()
            if args.smoke:
                assert 0.02 < mask_frac < 0.98, f"assistant mask covers {mask_frac:.1%}"
                assert torch.isfinite(torch.tensor(loss_sum)), "non-finite loss"

        publish(step)                                  # next step samples the updated policy
        rec = {"step": step, "mean_reward": statistics.fmean(rewards), "wall_budget": args.wall_budget,
               "transfer_checked": tc_checked, "transfer_passed": tc_passed,
               "pass_rate": sum(passes) / len(passes), "dead_groups": dead,
               "groups": len(groups), "live_seqs": len(batch), "mask_frac": round(mask_frac, 4),
               "errored": errored, "cells_per_rollout": round(cells / len(allouts), 2),
               "error_kinds": sorted({str(o.get("attribution")) for o in allouts
                                      if str(o.get("attribution") or "").startswith("error")}),
               "error_msgs": sorted({str(o.get("error_msg", ""))[:120] for o in allouts
                                     if o.get("error_msg")})[:3],
               "loss": loss_sum / max(nseq, 1), "grad_norm": round(gnorm, 4),
               # sampled-token NLL under the policy: an entropy proxy, to see collapse
               "nll": round(nll_sum / max(nseq, 1), 4), "lora_B_norm": round(lora_norm(model), 4),
               "oom_skipped": oom,
               **({"kl": round(kl_sum / max(nseq, 1), 5)} if args.kl else {}),
               "rollout_s": [round(float(x), 1) for x in       # p50, p90, max episode
                             np.percentile([o.get("seconds", 0) for o in allouts], [50, 90, 100])],
               "t_rollout": round(t_roll, 1),
               "t_step": round(time.time() - t0, 1)}
        with open(hist_f, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # Every rollout, compactly: what the policy answered and which documented
        # wrong analysis (if any) the answer matched. A reward that rises while
        # pass@1 stays at zero is only interpretable with this -- it separates
        # "closer to the right analysis" from "converging on a trap's answer".
        with open(out / "rollouts.jsonl", "a") as f:
            for inst, outs in groups.items():
                for o in outs:
                    f.write(json.dumps({"step": step, "inst": inst, "reward": o["reward"],
                                        "passed": bool(o.get("passed")),
                                        "attr": o.get("attribution"),
                                        "fails_at_step": o.get("fails_at_step"),
                                        "answered": o.get("answered"), "answer": o.get("answer"),
                                        "cells": o.get("n_cells"), "sec": o.get("seconds")},
                                       default=str) + "\n")
        print(json.dumps(rec), flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0 or step == args.steps:
            d = out / f"step{step}"
            model.save_pretrained(str(d))              # adapter only: for evaluation
            print(f"  checkpoint -> {d}", flush=True)
        if args.resume_every and step % args.resume_every == 0 or step == args.steps:
            tmp, res = out / "resume.tmp", out / "resume"
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(str(tmp))
            torch.save(opt.state_dict(), tmp / "optimizer.pt")
            (tmp / "state.json").write_text(json.dumps({"step": step}))
            shutil.rmtree(res, ignore_errors=True)
            tmp.rename(res)


if __name__ == "__main__":
    raise SystemExit(main())
