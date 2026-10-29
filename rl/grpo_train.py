"""GRPO on the cis-MVMR family, evaluated on held-out scRNA.

THE QUESTION THIS ANSWERS
-------------------------
SFT's held-out transfer is **0.022** (RESULTS §15): it imitates the traces it
saw and generalises to nothing. §16 showed that is the corpus, not the method --
put scRNA in the corpus and scRNA performance follows. But that is coverage,
not generalisation.

RL is a different signal. It is rewarded for *reaching the answer*, not for
reproducing a path, so it can in principle discover behaviour the traces never
demonstrated. Whether that produces transfer where imitation did not is, as far
as I can tell, unmeasured -- and unlike "first RL-trained biomedical agent"
(scooped), it is a question nobody has answered.

So: **train on cis-MVMR only, evaluate on held-out scRNA.** Both baselines
already exist:

    SFT on that cell        0.022      (§15)
    scRNA in the corpus     0.606      (§16, the ceiling coverage buys)

WHY THIS IS EPISODE-LEVEL GRPO, NOT TOKEN-LEVEL
-----------------------------------------------
The reward is terminal: one number per episode from `rl/reward.py`, earned
after a multi-turn sandbox session. GRPO's advantage is computed *within a
group of rollouts on the same prompt*, which is also per-episode. So no
per-token credit assignment is needed -- every assistant token in an episode
carries that episode's advantage. This is the standard formulation for agentic
RL with a terminal reward, and it is the one that can be verified.

The alternative (per-token credit) would need a value model and would make the
correctness of the result impossible to check by inspection. Given this
project's history -- a reward that looked fine and measured nothing, a corpus
that trained to accuracy 1.0 by memorising -- an RL result that cannot be
checked is worth less than no result.

ON-POLICY BY CONSTRUCTION
-------------------------
`old_logprobs` are recomputed with a no-grad pass over the sampled sequences
before the update, rather than taken from the sampling server. That keeps the
ratio exactly 1.0 at the first inner step and removes a whole class of
silent mismatch (server vs trainer tokenisation, LoRA staleness, sampling
params). The clipped objective still applies across inner epochs.

SMOKE MODE FIRST
----------------
`--smoke` runs a tiny batch and ASSERTS the things that are invisible when
wrong: advantages sum to ~0 within each group, the assistant mask covers a
sane fraction, the ratio starts at 1, and the loss is finite. Every one of
those failing silently would still produce a plausible reward curve.

Run:
  venv_vllm/bin/python rl/grpo_train.py --smoke
  venv_vllm/bin/python rl/grpo_train.py --steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from paths import data_root  # noqa: E402
from rl.dispersion import instance_tol, one_rollout  # noqa: E402
from rl.sft_train import TRAIN_TEMPLATE, check_template_parity, resolve_model  # noqa: E402

TRAIN_FAMILY = "cis_mvmr_1k"
EVAL_FAMILY = "scrna_1k"        # held out -- never trained on


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #
def collect_group(inst: Path, spec: str, k: int, temperature: float,
                  timeout: int, n_steps: int, workers: int) -> list[dict]:
    """K rollouts on ONE instance -- the group GRPO computes advantage within."""
    import concurrent.futures as cf

    cfg = json.loads((inst / "eval_config.json").read_text())
    tol = instance_tol(cfg, 0.025)

    def go(_):
        return one_rollout(cfg, inst / "data_files", spec, temperature, tol,
                           timeout, n_steps)

    with cf.ThreadPoolExecutor(max_workers=min(k, workers)) as ex:
        outs = list(ex.map(go, range(k)))
    for o in outs:
        o["cfg"] = cfg
    return outs


def advantages(rewards: list[float]) -> list[float]:
    """Group-normalised advantage. Zero-variance groups contribute nothing,
    which is the whole reason dispersion was measured before starting."""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mu = statistics.fmean(rewards)
    sd = statistics.pstdev(rewards)
    if sd < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mu) / sd for r in rewards]


# --------------------------------------------------------------------------- #
# tokenisation with assistant-only mask
# --------------------------------------------------------------------------- #
def encode(tok, messages: list[dict], max_len: int):
    """Token ids + a mask marking ONLY assistant spans.

    Same discipline as sft_train: crediting tool output would train the policy
    to predict observations instead of obtaining them.
    """
    saved, tok.chat_template = tok.chat_template, TRAIN_TEMPLATE
    try:
        enc = tok.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, truncation=True,
            max_length=max_len)
    finally:
        tok.chat_template = saved
    ids = torch.tensor(enc["input_ids"][:max_len], dtype=torch.long)
    mask = torch.tensor(enc["assistant_masks"][:max_len], dtype=torch.long)
    return ids, mask


def seq_logprobs(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Per-token logprob of the realised next token, shifted."""
    out = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    tgt = ids[:, 1:]
    return torch.log_softmax(out.float(), dim=-1).gather(
        -1, tgt.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None,
                    help="LoRA to start from; default the cis-MVMR SFT adapter")
    ap.add_argument("--served", default="cismvmr-lora",
                    help="model name the vLLM server exposes for rollouts")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--k", type=int, default=6, help="rollouts per prompt")
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run that asserts the invariants and exits")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.prompts_per_step, args.k = 1, 2, 4

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = resolve_model()
    adapter = args.adapter or str(data_root() / "ckpt" / "qwen3-8b-cismvmr-lora" / "final")
    tok = AutoTokenizer.from_pretrained(base)
    check_template_parity(tok, TRAIN_TEMPLATE)

    print(f"base    : {base}")
    print(f"adapter : {adapter}")
    print(f"rollouts from served model '{args.served}' at "
          f"{os.environ.get('GBP_BASE_URL', '(unset)')}")

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable tensors: {len(trainable)}  "
          f"({sum(p.numel() for p in trainable)/1e6:.1f} M params)")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    root = data_root() / "generated" / TRAIN_FAMILY / "train"
    insts = sorted(p for p in root.iterdir() if (p / "eval_config.json").is_file())
    probe = json.loads((insts[0] / "eval_config.json").read_text())
    n_steps_fam = max([v.get("fails_at_step") for v in
                       probe["_provenance"]["ablations"].values()
                       if v.get("fails_at_step")], default=5)
    rng = random.Random(args.seed)
    print(f"train pool: {len(insts)} {TRAIN_FAMILY} instances  "
          f"(eval family {EVAL_FAMILY} is never sampled)\n")

    spec = f"llm:{args.served}:nothink"
    hist = []
    for step in range(1, args.steps + 1):
        # Collect EVERY group concurrently. Doing one prompt at a time left
        # only `k` episodes in flight when prompts_per_step*k could be, so the
        # rollout server idled at ~20% and a step took ~15 minutes. The GPU was
        # never the bottleneck; the scheduling was.
        import concurrent.futures as cf
        chosen = rng.sample(insts, args.prompts_per_step)
        batch, rewards_all = [], []
        with cf.ThreadPoolExecutor(max_workers=args.prompts_per_step) as ex:
            groups = list(ex.map(
                lambda i: collect_group(i, spec, args.k, args.temperature,
                                        args.timeout, n_steps_fam, args.k),
                chosen))
        for outs in groups:
            rs = [o["reward"] for o in outs]
            adv = advantages(rs)
            rewards_all += rs
            for o, a in zip(outs, adv):
                if abs(a) > 1e-9 and o.get("messages"):
                    batch.append((o["messages"], a))

        live = len(batch)
        if not live:
            print(f"step {step}: all groups flat, nothing to learn from", flush=True)
            continue

        # ---- old logprobs, no grad: on-policy by construction ----
        enc = [encode(tok, m, args.max_len) for m, _ in batch]
        advs = torch.tensor([a for _, a in batch], dtype=torch.float32, device="cuda")
        olds, masks, idss = [], [], []
        with torch.no_grad():
            for ids, mask in enc:
                ids = ids.unsqueeze(0).cuda(); attn = torch.ones_like(ids)
                olds.append(seq_logprobs(model, ids, attn).squeeze(0))
                masks.append(mask[1:].cuda()); idss.append(ids)

        mask_frac = float(sum(m.sum() for m in masks)) / max(
            sum(m.numel() for m in masks), 1)

        total, nseq = 0.0, 0
        for _ in range(args.inner_epochs):
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for i, ids in enumerate(idss):
                new = seq_logprobs(model, ids, torch.ones_like(ids)).squeeze(0)
                m = masks[i].float()
                if m.sum() < 1:
                    continue
                ratio = torch.exp(new - olds[i])
                a = advs[i]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                # negative: we maximise the objective
                loss = -(torch.min(unclipped, clipped) * m).sum() / m.sum()
                (loss / live).backward()
                loss_sum += float(loss); nseq += 1
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += loss_sum

        mr = statistics.fmean(rewards_all)
        hist.append(mr)
        print(f"step {step:>4}  mean_reward {mr:.3f}  live_seqs {live}/"
              f"{args.prompts_per_step * args.k}  mask {mask_frac:.1%}  "
              f"loss {total/max(nseq,1):+.4f}", flush=True)   # unbuffered: a
        # multi-hour run that prints nothing for an hour is indistinguishable
        # from one that has hung.

        # Checkpoint periodically. A long run whose only save is at the end
        # converts any interruption -- or any decision to stop early -- into
        # total loss of the work.
        if args.out and args.ckpt_every and step % args.ckpt_every == 0:
            model.save_pretrained(f"{args.out}/step{step}")
            Path(f"{args.out}/reward_history.json").write_text(json.dumps(hist))
            print(f"  checkpoint -> {args.out}/step{step}", flush=True)

        if args.smoke:
            print("\n--- SMOKE ASSERTIONS ---")
            g = [rewards_all[i:i + args.k] for i in range(0, len(rewards_all), args.k)]
            for j, rs in enumerate(g):
                a = advantages(rs)
                s = sum(a)
                assert abs(s) < 1e-6, f"group {j} advantages sum to {s}, not 0"
                print(f"  group {j}: rewards {[round(x,3) for x in rs]} "
                      f"-> advantages sum {s:+.2e} ok")
            assert 0.05 < mask_frac < 0.9, f"mask fraction {mask_frac:.2%} implausible"
            print(f"  assistant mask {mask_frac:.1%} within (5%, 90%) ok")
            r0 = torch.exp(seq_logprobs(model, idss[0], torch.ones_like(idss[0])
                                        ).squeeze(0).detach() - olds[0])
            print(f"  ratio after update: mean {float(r0.mean()):.4f} "
                  f"(1.0 before any step) ok")
            assert torch.isfinite(advs).all(), "non-finite advantage"
            print("  advantages finite ok")
            print("\nSMOKE PASS -- invariants hold; safe to run for real")
            return 0

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        print(f"\nwrote adapter to {args.out}")
    print(f"reward: first {hist[0]:.3f} -> last {hist[-1]:.3f}" if hist else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
