#!/usr/bin/env python
"""Pull a PEFT LoRA adapter out of a verl FSDP checkpoint.

WHY NOT `verl.model_merger`. That writes a full merged model -- ~16 GB per
checkpoint, and this run left eight checkpoints on a filesystem already at 97 %.
The adapter itself is ~400 MB, and every earlier RL evaluation in this project
served `base + adapter` through `serve_local.sh`'s LORAS path, so extracting one
keeps the new numbers on exactly the same serving path as the old ones.

The LoRA tensors are NOT sharded across ranks -- rank 0 holds them at full shape
(checked: q_proj lora_A is (128, 4096), lora_B is (4096, 128) for r=128) -- so a
single shard is enough. The base weights in the same file are ignored.

Key renaming is the whole trick: verl stores
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
and PEFT expects the same path without the `.default` infix.

  venv_verl/bin/python scripts/extract_verl_lora.py \\
      --ckpt $DEEP_DATA_ROOT/ckpt/verl-grpo-scrna_np3/global_step_40/actor \\
      --out  $DEEP_DATA_ROOT/ckpt/snp3-step40-lora
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

REF_CONFIG = (data_root() / "ckpt" / "qwen3-8b-fewshot-np-lora" / "final"
              / "adapter_config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the .../actor directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-file", default="model_world_size_2_rank_0.pt")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    src = ck / args.rank_file
    if not src.exists():
        cands = sorted(ck.glob("model_world_size_*_rank_0.pt"))
        if not cands:
            raise SystemExit(f"no rank-0 model shard under {ck}")
        src = cands[0]

    # The adapter IS sharded: rank 0 holds (64, 4096) of a (128, 4096) lora_A
    # at r=128 on 2 ranks. Writing rank 0 alone would have produced a
    # half-adapter that loads without complaint and evaluates as noise, so every
    # rank shard is read and reassembled, and the result is checked against the
    # DTensor's own global shape.
    shards = sorted(ck.glob(args.rank_file.replace("rank_0", "rank_*")))
    if not shards:
        shards = sorted(ck.glob("model_world_size_*_rank_*.pt"))
    print(f"reading {len(shards)} rank shard(s)")
    per_rank = []
    for f in shards:
        d_ = torch.load(f, map_location="cpu", weights_only=False)
        per_rank.append(d_ if isinstance(d_, dict) else d_.state_dict())

    def shard_dim(t) -> int | None:
        for pl in getattr(t, "placements", ()) or ():
            if getattr(pl, "is_shard", lambda: False)():
                return int(getattr(pl, "dim", 0))
        return None

    out: dict[str, torch.Tensor] = {}
    for k, v in per_rank[0].items():
        if "lora_" not in k or not hasattr(v, "shape"):
            continue
        nk = k.replace(".default.weight", ".weight")
        full = tuple(v.shape)
        if hasattr(v, "to_local"):
            dim = shard_dim(v)
            locals_ = [r[k].to_local() for r in per_rank]
            t = locals_[0] if dim is None else torch.cat(locals_, dim=dim)
        else:
            t = v
        if tuple(t.shape) != full:
            raise SystemExit(
                f"{k}: reassembled {tuple(t.shape)} != global {full} -- the "
                f"adapter would be wrong and the evaluation meaningless")
        # PEFT keeps adapters in the dtype it will apply them in; verl stores
        # them fp32. bf16 matches the base model and halves the file.
        out[nk] = t.detach().to(torch.bfloat16).clone().contiguous()

    if not out:
        raise SystemExit(f"no LoRA tensors found in {src}")

    meta_path = ck / "lora_train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    save_file(out, str(d / "adapter_model.safetensors"))

    cfg = json.loads(REF_CONFIG.read_text())
    # Trust the checkpoint's own record over the reference file: a run trained
    # at a different rank would otherwise be served with the wrong scaling.
    if meta.get("r"):
        cfg["r"] = int(meta["r"])
    if meta.get("lora_alpha"):
        cfg["lora_alpha"] = int(meta["lora_alpha"])
    cfg["inference_mode"] = True
    (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    size_mb = (d / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"wrote {len(out)} tensors ({size_mb:.0f} MB) to {d}")
    print(f"  r={cfg['r']} alpha={cfg['lora_alpha']} from {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
