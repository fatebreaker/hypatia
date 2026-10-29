"""Bake a LoRA adapter into its base model and write a standalone checkpoint.

    venv/bin/python scripts/merge_lora.py \
        --adapter $DEEP_DATA_ROOT/ckpt/qwen3-8b-cismvmr-lora/final \
        --out     $DEEP_DATA_ROOT/ckpt/qwen3-8b-sft-merged

WHY THIS IS NEEDED FOR THE KL RUN
---------------------------------
verl sets `ref_in_actor = True` whenever LoRA is on: the reference policy is
not a separate model but the actor with the adapter switched off
(`engine.disable_adapter()`). That is the right default when the adapter starts
from zero, because "adapter off" is then exactly the initial policy.

It is the WRONG reference for our setup. RL here starts from the cis-MVMR SFT
adapter loaded via `model.lora_adapter_path`, so "adapter off" is base
Qwen3-8B, and a KL penalty would pull the policy toward the base model --
actively undoing SFT, which is the opposite of the intent. The KL is supposed
to stop the policy drifting away from the SFT init while it learns.

Merging fixes this by moving the SFT weights from the adapter into the base:

    model.path            = this merged checkpoint  (= the SFT policy)
    lora_adapter_path     = unset, so the LoRA starts at zero
    initial policy        = merged + zero LoRA      = the SFT policy
    reference policy      = merged, adapter off     = the SFT policy

Now the run starts where the old one started, and the KL reference is the
thing we actually want to stay near.

The merge is exact, not an approximation: PEFT folds B@A*(alpha/r) into each
target weight, which is the same arithmetic the adapter performs at inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", default=None,
                    help="base model; default is the adapter's own record of it")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    base = args.base or cfg["base_model_name_or_path"]
    print(f"base    : {base}")
    print(f"adapter : {args.adapter}  (r={cfg['r']}, alpha={cfg['lora_alpha']})")
    print(f"out     : {args.out}")

    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty {args.out}", file=sys.stderr)
        return 1

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # The tokenizer must travel with the weights: vLLM and the trainer both
    # load it from model.path, and a checkpoint without one fails at startup
    # rather than at first use.
    AutoTokenizer.from_pretrained(base).save_pretrained(args.out)

    n = len(list(args.out.glob("*.safetensors")))
    print(f"wrote {n} safetensors shards to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
