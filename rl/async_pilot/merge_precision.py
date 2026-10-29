"""How much of a LoRA update survives verl's merge-and-sync path?  (CPU only; evidence for docs/ASYNC_PILOT.md)

verl's fully-async mode ships weights to the rollout GPU through the checkpoint engine, and for a LoRA actor that path
works only with `model.lora.merge=True`: the adapter is merged into the base weights on the trainer and the merged
tensors are sent as bf16 (verl/workers/engine/fsdp/transformer_impl.py `_merged_lora_per_tensor_param`,
`.to(torch.bfloat16)`). The base weights are themselves bf16 values, so an entry of the merged matrix changes only if
its delta is larger than half a bf16 ulp of that entry. A fresh LoRA trained at LR 1e-5 moves slowly; this script
measures, on a REAL R1 checkpoint, how much of the update the rollout policy would actually receive:

    retained = || bf16(W + dW) - W ||_F / || dW ||_F          (1.0 = nothing lost)
    changed  = fraction of entries of W that change at all
    cos      = cosine(bf16(W + dW) - W, dW)                 (direction of what survives)
    out_ret  = || x (Wm - W)^T || / || x dW^T ||  for random unit-scale inputs x (effect on layer outputs)

vLLM's LoRA path (what R1 uses) never forms W + dW: it adds x A^T B^T scale in activation space, so none of this is lost.

    venv_vllm/bin/python rl/async_pilot/merge_precision.py --adapter $DEEP_DATA_ROOT/ckpt/<R1 run>/step50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

BASE = "/scratch4/workspace/hieutran_umass_edu-deeprs-bench/models/Qwen3-8B"


def base_tensor(base: Path, index: dict, name: str) -> torch.Tensor:
    with safe_open(str(base / index["weight_map"][name]), framework="pt") as f:
        return f.get_tensor(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--layers", default="0,12,24,35")
    ap.add_argument("--modules", default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,"
                                         "mlp.gate_proj,mlp.up_proj,mlp.down_proj")
    args = ap.parse_args()
    ad = Path(args.adapter)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    base = Path(args.base)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    torch.manual_seed(0)
    rows, tot = [], {"dw2": 0.0, "eff2": 0.0, "dot": 0.0, "changed": 0, "n": 0, "out_d": 0.0, "out_e": 0.0}
    with safe_open(str(ad / "adapter_model.safetensors"), framework="pt") as fa:
        keys = set(fa.keys())
        for layer in [int(x) for x in args.layers.split(",")]:
            for mod in args.modules.split(","):
                stem = f"base_model.model.model.layers.{layer}.{mod}"
                ka, kb = f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"
                if ka not in keys:
                    continue
                A, B = fa.get_tensor(ka).float(), fa.get_tensor(kb).float()
                W = base_tensor(base, index, f"model.layers.{layer}.{mod}.weight")      # bf16, as stored
                dW = scale * (B @ A)
                Wm = (W.float() + dW).to(torch.bfloat16)                                  # verl merge -> bf16 sync
                eff = Wm.float() - W.float()
                x = torch.randn(64, W.shape[1])
                od, oe = (x @ dW.T).norm().item(), (x @ eff.T).norm().item()
                r = {"layer": layer, "module": mod,
                     "retained": round(eff.norm().item() / max(dW.norm().item(), 1e-30), 4),
                     "changed": round((eff != 0).float().mean().item(), 4),
                     "cos": round(torch.nn.functional.cosine_similarity(eff.flatten(), dW.flatten(), dim=0).item(), 4),
                     "out_ret": round(oe / max(od, 1e-30), 4),
                     "dW_rms": float(f"{dW.pow(2).mean().sqrt().item():.3e}"),
                     "W_rms": float(f"{W.float().pow(2).mean().sqrt().item():.3e}")}
                rows.append(r)
                print(json.dumps(r), flush=True)
                tot["dw2"] += dW.pow(2).sum().item(); tot["eff2"] += eff.pow(2).sum().item()
                tot["dot"] += (eff * dW).sum().item(); tot["changed"] += int((eff != 0).sum().item())
                tot["n"] += eff.numel(); tot["out_d"] += od ** 2; tot["out_e"] += oe ** 2
    summ = {"adapter": str(ad), "modules": len(rows),
            "retained_norm": round((tot["eff2"] / tot["dw2"]) ** 0.5, 4),
            "cos": round(tot["dot"] / (tot["eff2"] * tot["dw2"]) ** 0.5, 4),
            "changed_frac": round(tot["changed"] / tot["n"], 4),
            "output_retained": round((tot["out_e"] / tot["out_d"]) ** 0.5, 4)}
    print("SUMMARY " + json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
