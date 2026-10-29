"""Upload the artifacts that cannot be regenerated, to private HF repos.

WHY THESE AND NOT OTHERS
------------------------
Re-running a generator with the same seed no longer reproduces the stored
instances: the generators were edited after the corpora were built (instances
2026-08-10, generator 2026-09-01). The generated problems are therefore an
artefact, not a cache, and every measurement in the paper was taken on them.

  data/generated/  2.3 GB  12,862 instances, 18 families -- NOT regenerable
  data/sft/        390 MB  110 corpora; each trace executes real analysis code
  4 adapters       4.0 GB  the headline model, its second seed, two RL steps

Deliberately excluded:
  verl RL checkpoints   2.1 TB of FSDP shards + optimiser state. The useful
                        667 MB of each is already extracted as a LoRA adapter.
  SFT checkpoint-*/     intermediate epochs; only final/ is used anywhere.
  benchmarks/           17 GB, re-fetchable, and other people's data to
                        redistribute -- see scripts/fetch_*.py.
  data/rl/ transcripts  3.9 GB, already distilled into results_index/, which
                        reproduces every published number.

PRIVATE BY DEFAULT. The instances encode data-generating processes transcribed
from OpenAI's GeneBench-Pro case-study reports. The code is ours and the data is
synthetic, but the derivation is theirs; publishing is a judgement call the
project owner should make deliberately. Pass --public to override.

  export HF_TOKEN=hf_...            # needs WRITE scope
  venv/bin/python scripts/upload_hf.py --user hieutran81
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

DATA = data_root()
ADAPTERS = [
    ("qwen3-8b-tenfam-lora/final", "tenfam-sft"),          # headline SFT model
    ("qwen3-8b-tenfam2-lora/final", "tenfam-sft-seed2"),   # its independent replicate
    ("tenrl-step150-lora", "tenfam-rl-step150"),           # the 0.994 model
    ("tenrl-step100-lora", "tenfam-rl-step100"),
]

CARD = """---
license: apache-2.0
tags: [statistical-genomics, agents, reinforcement-learning, genebench-pro]
---

# {name}

Artifacts from *Amplification without Acquisition* — what reinforcement
learning buys when training a language model to analyse scientific data.

One Qwen3-8B model scores **0.994** across all ten public GeneBench-Pro
problems (8/10 at exactly 1.000 over 256 rollouts each), where the same model
untrained scores **0.014** and the best published frontier system reaches 36%
on that split.

Code, paper draft and the full results index:
https://github.com/fatebreaker/deeprs — start at `CONTINUITY.md`.

{extra}

## Reproducing

The generated instances here are **not** reproducible by re-running the
generators: those were edited after these corpora were built. Use these files
rather than regenerating, or exact numbers will not match.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--public", action="store_true",
                    help="publish publicly; default is private (see module docstring)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, whoami
    tok = os.environ.get("HF_TOKEN")
    try:
        who = whoami(token=tok)
    except Exception as e:
        print(f"not authenticated: {e}", file=sys.stderr)
        return 1
    role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role")
    if role not in ("write", "fineGrained") and not a.dry_run:
        print(f"token role is {role!r}; a WRITE token is required.\n"
              f"  create one at https://huggingface.co/settings/tokens\n"
              f"  then:  export HF_TOKEN=hf_...", file=sys.stderr)
        return 1

    api = HfApi(token=tok)
    private = not a.public
    plan = []

    # Adapters are published together by upload_hf_adapters.py; separate
    # per-model repos duplicated them and failed anyway -- PEFT writes a
    # README.md whose `base_model` is the local snapshot PATH, which the Hub
    # rejects as model-card metadata when it lands at a repo root.
    ds = f"{a.user}/deeprs-problems"
    plan.append(("dataset", ds, [(DATA / "generated", "generated"),
                                 (DATA / "sft", "sft")]))

    for kind, repo, items in plan:
        total = sum(sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
                    for s, _ in items if s.exists())
        print(f"{'[dry-run] ' if a.dry_run else ''}{kind:7s} {repo:42s} {total/1e9:6.2f} GB "
              f"{'private' if private else 'PUBLIC'}")
        if a.dry_run:
            continue
        api.create_repo(repo, repo_type=kind, private=private, exist_ok=True)
        extra = ("This repository holds the generated problem instances and SFT corpora."
                 if kind == "dataset" else
                 "This repository holds a LoRA adapter for Qwen3-8B.")
        api.upload_file(path_or_fileobj=CARD.format(name=repo.split("/")[-1], extra=extra).encode(),
                        path_in_repo="README.md", repo_id=repo, repo_type=kind)
        for src, dest in items:
            if not src.exists():
                continue
            api.upload_folder(folder_path=str(src), path_in_repo=dest,
                              repo_id=repo, repo_type=kind,
                              ignore_patterns=["**/checkpoint-*/**", "**/__pycache__/**",
                                               "README.md", "**/README.md"])
        print(f"    -> https://huggingface.co/{'datasets/' if kind=='dataset' else ''}{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
