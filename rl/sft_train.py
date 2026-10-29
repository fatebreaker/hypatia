"""LoRA SFT of Qwen3-8B on the verified cis-MVMR traces.

Purpose: teach the policy the output protocol and the shape of the five-step
pipeline, so that the dense step reward has something to discriminate. Before
SFT the model runs 17 cells and submits {0.0, 0.0} -- reward 0.012, no step
attribution, and therefore no GRPO gradient on any rollout.

TWO CHOICES THAT DECIDE WHETHER THIS WORKS
------------------------------------------
1. LOSS IS MASKED TO ASSISTANT TURNS ONLY (`assistant_only_loss=True`).
   Tool outputs live in `user` turns. Training on them teaches the model to
   PREDICT observations instead of obtaining them, and it will then fabricate
   plausible stdout at inference -- the same failure the trace generator was
   built to avoid on the data side. Qwen3's stock template has NO
   {% generation %} markers -- TRL rejects it outright -- so we substitute
   TRAIN_TEMPLATE, which reproduces the stock history rendering byte-for-byte
   and marks assistant spans. Both the parity and the resulting mask fraction
   are asserted: a silent fallback to full-sequence loss would be invisible in
   the loss curve but ruinous at eval.

2. LoRA, NOT FULL FINE-TUNE.
   Full bf16 fine-tuning of 8B needs ~16 GB weights + 16 GB grads + ~64 GB
   AdamW state, far past a 48 GB A40. LoRA keeps the base frozen and is
   sufficient for teaching format and procedure, which is all that is being
   taught here -- we are not trying to install new statistics knowledge.

Traces are ~3.1k tokens (p95 3095), so seq_len 4096 truncates nothing.

Run:
  venv_vllm/bin/python rl/sft_train.py --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

MODEL_KEY = os.environ.get("SFT_MODEL_KEY", "Qwen3-8B")


def resolve_model(key: str = MODEL_KEY) -> str:
    """Local snapshot directory for the policy.

    This was a hardcoded Unity checkpoint path, which fails anywhere else with
    an unhelpful "Repo id must be in the form 'repo_name'" from transformers --
    it treats a non-existent absolute path as a hub id. Resolution order:

        $SFT_MODEL              explicit path or hub id, wins outright
        $SHARED_LLM_CACHE/hub   a site-wide shared cache, if the site has one
        ~/.cache/huggingface    the ordinary per-user cache

    Mirrors scripts/serve_local.sh so training and serving find the same
    weights.
    """
    explicit = os.environ.get("SFT_MODEL")
    if explicit:
        return explicit
    cache = Path(os.environ.get("SHARED_LLM_CACHE",
                                Path.home() / ".cache" / "huggingface"))
    for pattern in (f"hub/models--*--{key}/snapshots/*", f"models--*--{key}/snapshots/*"):
        snaps = sorted(cache.glob(pattern))
        # A snapshot dir can exist while weights are still downloading
        # (FINDINGS D5), so require the weights, not just the directory.
        snaps = [s for s in snaps if any(s.glob("*.safetensors"))]
        if snaps:
            return str(snaps[-1])
    raise FileNotFoundError(
        f"no local snapshot for {key!r} under {cache}. Either download it, or "
        f"set SFT_MODEL=/path/to/snapshot (or a hub id)."
    )


MODEL = resolve_model()

# Qwen3's stock template is not training-compatible: TRL rejects it for lacking
# {% generation %} markers, and it cannot be patched automatically because its
# assistant branch carries multi-way reasoning_content logic.
#
# This replacement reproduces the stock HISTORY rendering byte-for-byte -- note
# Qwen3 emits the empty <think></think> block ONLY on the final assistant
# message, so history turns are plain '<|im_start|>assistant\n...' -- and wraps
# assistant content in generation markers so loss can be masked to it.
#
# im_end is inside the markers deliberately: the model must learn to STOP, and
# an unsupervised terminator is how you get a policy that runs on past the end
# of its turn.
TRAIN_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{% generation %}{{- message['content'] + '<|im_end|>' + '\n' }}{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content']"
    " + '<|im_end|>' + '\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def check_template_parity(tok, train_tpl: str) -> None:
    """The training template must render history exactly as inference does.

    A silent divergence here trains the model on a prompt format it will never
    see again, and nothing downstream would reveal it.
    """
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
    stock = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    saved, tok.chat_template = tok.chat_template, train_tpl
    trained = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok.chat_template = saved
    # Inference adds the empty think block to the generation prefix only; strip
    # it before comparing the shared history portion.
    norm = stock.replace("<think>\n\n</think>\n\n", "")
    if norm != trained:
        raise AssertionError(
            "training template diverges from inference rendering:\n"
            f"  inference (think stripped): {norm!r}\n"
            f"  training                  : {trained!r}")
    print("template parity: history rendering matches inference")


def load(path: Path):
    from datasets import Dataset
    rows = [json.loads(l) for l in open(path)]
    # SFTTrainer consumes the `messages` column for conversational data.
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def verify_masking(trainer, tok) -> None:
    """Confirm loss really is restricted to assistant turns.

    If assistant_only_loss silently degrades to full-sequence loss, the loss
    curve looks completely normal while the model learns to hallucinate tool
    output. Checking the label mask is the only way to see it.
    """
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    labels = batch["labels"]
    total = int((labels != -100).sum() + (labels == -100).sum())
    supervised = int((labels != -100).sum())
    frac = supervised / max(total, 1)
    print(f"\nloss mask: {supervised:,}/{total:,} tokens supervised ({frac:.1%})")
    if frac > 0.75:
        raise AssertionError(
            f"{frac:.0%} of tokens are supervised -- assistant-only masking is "
            f"not in effect. The model would be trained to predict tool output.")
    if frac < 0.05:
        raise AssertionError(
            f"only {frac:.1%} of tokens supervised -- masking is too aggressive; "
            f"check the chat template's {{% generation %}} markers.")
    # Show what is actually being learned, as a sanity read.
    ids = batch["input_ids"][0]
    lab = labels[0]
    kept = tok.decode([int(i) for i, m in zip(ids, lab) if m != -100][:60])
    print(f"first supervised tokens: {kept[:220]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--corpus", default="cis_mvmr",
                    help="stem under sft/: cis_mvmr | scrna_ambient | mixed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and verify masking, then stop")
    # Continuing from an existing adapter is how the sample-efficiency question
    # gets asked: from base, the demonstration threshold is a cliff (25 -> 0.000,
    # 50 -> 0.000, 100 -> 1.000). If a model that already knows nine families
    # clears the same family on 25, the transfer that leave-one-family-out
    # measured at zero shot is real -- it just needs a few demonstrations to
    # surface.
    ap.add_argument("--init-adapter", default=None,
                    help="continue training an existing LoRA instead of a fresh one")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only N training traces (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="full-parameter fine-tuning in bf16 (the LoRA-vs-full control; needs ~70 GB, an H200) "
                         "instead of LoRA; pass a full-FT learning rate, e.g. --lr 1e-5")
    ap.add_argument("--base-bf16", action="store_true",
                    help="hold the frozen base weights of a LoRA run in bf16 instead of TRL's default fp32 (matmuls run "
                         "in bf16 under autocast either way); needed to fit Qwen3-32B (131 GB in fp32) on one H200")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint-N in --out (optimizer, scheduler and "
                         "RNG state included), e.g. after moving a run to another pod")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft = data_root() / "sft"
    out = (Path(args.out) if args.out
           else data_root() / "ckpt" / f"qwen3-8b-{args.corpus.replace('_','')}-lora")
    out.mkdir(parents=True, exist_ok=True)

    train = load(sft / f"{args.corpus}_train.jsonl")
    val = load(sft / f"{args.corpus}_val.jsonl")
    if args.limit:
        # Shuffled before truncation: the corpora are written in generation
        # order, so the first N traces are not a random N.
        train = train.shuffle(seed=args.seed).select(
            range(min(args.limit, len(train))))
    print(f"train {len(train)} traces | val {len(val)} traces")
    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    from rl.chat_formats import LLAMA3_TRAIN, MISTRAL_TRAIN, family
    if family(MODEL) == "mistral":
        # Mistral's own template moves the system message to the last user turn;
        # rl/chat_formats.py folds it into the first, and inference renders the same.
        tok.chat_template = MISTRAL_TRAIN
        print("template: rl.chat_formats.MISTRAL_TRAIN (Mistral family)")
    elif family(MODEL) == "llama":
        tok.chat_template = LLAMA3_TRAIN
        if tok.pad_token is None:
            tok.pad_token = "<|finetune_right_pad_id|>"
        print("template: rl.chat_formats.LLAMA3_TRAIN (Llama-3 family)")
    else:
        check_template_parity(tok, TRAIN_TEMPLATE)
        tok.chat_template = TRAIN_TEMPLATE

    # transformers >= 5 removed `warmup_ratio` and reads a float `warmup_steps`
    # below 1 as a ratio; pass whichever this install accepts so a rebuilt
    # environment trains exactly as the original (5% linear warm-up).
    import inspect
    warm = ({"warmup_ratio": 0.05}
            if "warmup_ratio" in inspect.signature(SFTConfig.__init__).parameters
            else {"warmup_steps": 0.05})
    cfg = SFTConfig(
        **warm,
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=args.batch,
        save_strategy="steps",
        save_steps=100 if args.full else 50,          # a full checkpoint is ~50 GB with optimizer state
        save_total_limit=1 if args.full else 3,
        bf16=True,
        max_length=args.seq_len,
        packing=False,                 # multi-turn traces must stay intact
        assistant_only_loss=True,      # see verify_masking
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        **({"model_init_kwargs": {"dtype": torch.bfloat16}} if args.full or args.base_bf16 else {}),
    )
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    if args.init_adapter:
        # PeftModel carries its own r/alpha/target_modules, so peft_config is
        # NOT passed -- handing SFTTrainer both would wrap a second adapter
        # around the first and train that instead of the one being continued.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, args.init_adapter,
                                          is_trainable=True)
        print(f"continuing from adapter: {args.init_adapter}")
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=train,
                             eval_dataset=val, processing_class=tok)
    elif args.full:
        trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=train,
                             eval_dataset=val, processing_class=tok)
        n = sum(p_.numel() for p_ in trainer.model.parameters() if p_.requires_grad)
        print(f"full fine-tuning: {n / 1e9:.2f} B trainable parameters, dtype {next(trainer.model.parameters()).dtype}")
    else:
        trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=train,
                             eval_dataset=val, processing_class=tok,
                             peft_config=peft_cfg)
    if not args.full:
        trainer.model.print_trainable_parameters()
    verify_masking(trainer, tok)

    if args.dry_run:
        print("\ndry run complete -- masking verified, not training")
        return 0

    ckpts = sorted(out.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1])) if args.resume else []
    if ckpts:
        print(f"resuming from {ckpts[-1]}")
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"\n{'model' if args.full else 'adapter'} written to {out / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
