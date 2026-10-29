"""Convert a Qwen3.5/3.6 vision-language checkpoint (Qwen3_5ForConditionalGeneration) into a TEXT-ONLY bf16 checkpoint
(Qwen3_5ForCausalLM) that both the RL trainer (transformers AutoModelForCausalLM) and the rollout server (vLLM, LoRA on)
load with the same parameter names (2026-10-01; RL at 27B, docs/PAPER_PLAN.md).

  weights: 'model.language_model.X' -> 'model.X'; 'lm_head.*' kept; 'model.visual.*' (and any 'mtp.*') dropped
  config:  the text_config, with architectures = ['Qwen3_5ForCausalLM'] and its own model_type
  tokenizer / generation / chat-template files copied unchanged.
    venv_vllm/bin/python scripts/make_text_only_qwen35.py Qwen/Qwen3.6-27B <REVISION> OUT_DIR
"""
import json, os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

repo, rev, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
src = Path(snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"]))
tmp = out.with_name(out.name + ".part"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
cfg = json.load(open(src / "config.json"))
tc = dict(cfg.get("text_config", cfg))
tc["architectures"] = ["Qwen3_5ForCausalLM"]
tc.setdefault("torch_dtype", "bfloat16")
for k in ("tie_word_embeddings",):
    if k in cfg and k not in tc: tc[k] = cfg[k]
json.dump(tc, open(tmp / "config.json", "w"), indent=2)
idx = json.load(open(src / "model.safetensors.index.json"))
new_map, kept, dropped = {}, 0, 0
for shard in sorted(set(idx["weight_map"].values())):
    tensors = {}
    with safe_open(src / shard, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."): nk = "model." + k[len("model.language_model."):]
            elif k.startswith("lm_head."): nk = k
            else: dropped += 1; continue
            tensors[nk] = f.get_tensor(k); new_map[nk] = shard; kept += 1
    if tensors: save_file(tensors, str(tmp / shard), metadata={"format": "pt"})
    print(f"{shard}: {len(tensors)} tensors", flush=True)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_map}, open(tmp / "model.safetensors.index.json", "w"), indent=1)
for p in src.iterdir():
    if p.name in ("config.json", "model.safetensors.index.json") or p.suffix == ".safetensors": continue
    if p.is_file(): shutil.copy2(p, tmp / p.name)
(tmp / ".complete").write_text(f"{repo}@{rev} text-only: kept {kept}, dropped {dropped}\n")
shutil.rmtree(out, ignore_errors=True); os.replace(tmp, out)
print(f"wrote {out}: kept {kept} tensors, dropped {dropped}")
