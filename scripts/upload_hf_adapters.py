"""Publish all 120 LoRA adapters to one public HF model repo.

WHY PUBLIC, WHEN THE PROBLEM DATA IS NOT
----------------------------------------
These are weights we trained. The generated instances are a different matter --
they encode data-generating processes transcribed from OpenAI's GeneBench-Pro
case-study reports, so they stay private (scripts/upload_hf.py). The split
happens to match the storage limits: 124 GB of adapters exceeds the private
quota, and none of it is anyone else's data to redistribute.

WHY ALL 120 AND NOT JUST THE HEADLINE FOUR
------------------------------------------
The paper's load-bearing claims are comparisons, not single models: the
leave-one-family-out transfer nulls, the four control corpora, and the seed
variance table (identical corpora, 0.029 to 0.341 depending only on seed).
Checking any of those requires the specific adapter for each arm. Publishing
only the winner would make the positive result reproducible and every negative
result unverifiable.

Each adapter uploads under its own directory; `final/` is flattened away so a
path is just the run name. Intermediate training checkpoints are excluded --
only the weights actually evaluated in the paper.

  export HF_TOKEN=hf_...
  venv/bin/python scripts/upload_hf_adapters.py --user hieutran81
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

D = data_root() / "ckpt"

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags: [lora, peft, statistical-genomics, agents, reinforcement-learning]
---

# deeprs — LoRA adapters

Every adapter trained for *Amplification without Acquisition*: what
reinforcement learning buys when training a language model to analyse
scientific data, and which capability the training spends.

All adapters are LoRA on **Qwen/Qwen3-8B** unless the name says `14b`.

## The headline model

`tenrl-step150-lora` scores **0.994** across all ten public GeneBench-Pro
problems — 8 of 10 at exactly 1.000 over 256 rollouts each — where the same
base model untrained scores **0.014**, and the best published frontier system
reaches 36% on that split.

| adapter | what it is |
|---|---|
| `qwen3-8b-tenfam-lora` | ten-family SFT, the base of the headline |
| `qwen3-8b-tenfam2-lora` | independent second seed (0.963 vs 0.959) |
| `tenrl-step150-lora` | **+RL, the 0.994 model** |
| `qwen3-8b-lofov-*-lora` | leave-one-family-out, for the transfer nulls |
| `qwen3-8b-wrangle-*`, `plainqa-*`, `causalonly-*` | the four control corpora |
| `qwen3-8b-*-s2/-s3/...` | replicate seeds for the variance table |

## Why all 120 are here

The paper's load-bearing claims are comparisons. The transfer result is that
every leave-one-out model scores *below* the untrained base; the forgetting
result compares four corpora; the stability result is that identical corpora
give 0.029 or 0.341 depending only on seed. Verifying any of those needs the
specific adapter for that arm, so publishing only the winner would leave every
negative result unverifiable.

## Reproducing

Code, paper draft, and a results index that recomputes every published number:
https://github.com/fatebreaker/deeprs — start at `CONTINUITY.md`.

The problem instances these were trained on are in a separate, access-gated
dataset repository; note that re-running the generators does **not** reproduce
them, as the generators were edited after the corpora were built.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--repo", default="deeprs-adapters")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, whoami
    tok = os.environ.get("HF_TOKEN")
    try:
        whoami(token=tok)
    except Exception as e:
        print(f"not authenticated: {e}", file=sys.stderr)
        return 1

    items = []
    for d in sorted(D.glob("*-lora")):
        src = d / "final" if (d / "final").is_dir() else d
        if not any(f.suffix == ".safetensors" for f in src.iterdir() if f.is_file()):
            continue
        sz = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
        items.append((d.name, src, sz))
    total = sum(s for _, _, s in items)
    repo = f"{a.user}/{a.repo}"
    print(f"{'[dry-run] ' if a.dry_run else ''}{len(items)} adapters, {total/1e9:.1f} GB "
          f"-> {repo} ({'private' if a.private else 'PUBLIC'})")
    if a.dry_run:
        for n, _, s in items[:5]:
            print(f"    {s/1e6:7.0f} MB  {n}")
        return 0

    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    manifest = [{"adapter": n, "bytes": s} for n, _, s in items]
    api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(),
                    path_in_repo="manifest.json", repo_id=repo, repo_type="model")

    done = 0
    for name, src, sz in items:
        try:
            api.upload_folder(folder_path=str(src), path_in_repo=name,
                              repo_id=repo, repo_type="model",
                              ignore_patterns=["**/checkpoint-*/**", "**/__pycache__/**",
                                               "*.bin", "README.md"])
            done += 1
            print(f"  [{done}/{len(items)}] {name}  ({sz/1e6:.0f} MB)", flush=True)
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"done: {done}/{len(items)} -> https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
