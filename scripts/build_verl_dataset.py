"""Turn generated instances into a verl RL dataset.

verl's RLHFDataset passes every parquet column through to `non_tensor_batch`,
and the agent-loop worker turns each column into a kwarg:

    kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
    ... _run_agent_loop(sampling_params, ..., **kwargs)

So `instance_dir` on the row is what `rl/verl_sandbox_loop.py` reads to know
which problem a rollout is solving, and `agent_name` is what routes it to our
loop rather than verl's single-turn default.

TRAIN AND EVAL ARE DIFFERENT FAMILIES ON PURPOSE
------------------------------------------------
Train is cis-MVMR, validation is scRNA — held out and never trained on. That is
the whole experiment (RESULTS §15/§16): SFT's transfer to an unseen family is
0.022, and the question is whether RL, which is rewarded for reaching the
answer rather than for reproducing a path, does better. Mixing the families
here would silently answer a different, already-answered question.

The `prompt` column carries the real system+task messages so verl's length
filtering is meaningful, but the agent loop rebuilds them itself: it needs a
live Sandbox to enumerate data_files, and the prompt must match what the policy
was trained on.

Run:
  venv_verl/bin/python scripts/build_verl_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from paths import data_root  # noqa: E402

AGENT = "genebench_sandbox_agent"


def rows(family: str, split: str, limit: int | None) -> list[dict]:
    from agents import SYSTEM

    root = data_root() / "generated" / family / split
    insts = sorted(p for p in root.iterdir() if (p / "eval_config.json").is_file())
    if limit:
        insts = insts[:limit]
    out = []
    for p in insts:
        cfg = json.loads((p / "eval_config.json").read_text())
        listing = "\n".join(
            f"  data_files/{f.name}  ({f.stat().st_size:,} bytes)"
            for f in sorted((p / "data_files").iterdir()))
        out.append({
            "data_source": family,
            "agent_name": AGENT,
            "instance_dir": str(p),
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"{cfg['task']}\n\nFiles in your working directory:\n{listing}\n"},
            ],
            "ability": "statistical-genomics",
            # Reward comes from rl/reward.py inside the agent loop, not from a
            # verl reward function -- it needs the instance's ablation table,
            # which is what makes the score step-attributed rather than binary.
            "reward_model": {"style": "rule", "ground_truth": json.dumps(cfg["ground_truth"])},
            "extra_info": {"id": cfg["id"], "split": split, "family": family},
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-split", default="rl",
                    help="split to draw RL training instances from. Defaults to "
                         "'rl', NOT 'train': the SFT corpus is built from "
                         "'train', and RL on those same instances optimises "
                         "recall of answers the model was already shown. That "
                         "leak raised training reward 0.751->0.938 while "
                         "held-out solve rate FELL 0.500->0.395 (RESULTS).")
    ap.add_argument("--sft-corpus", default=None,
                    help="jsonl of SFT traces; if given, the build FAILS on any "
                         "instance-id overlap with the RL training set")
    ap.add_argument("--train-family", default="cis_mvmr_1k")
    ap.add_argument("--eval-family", default="scrna_1k",
                    help="HELD OUT -- never trained on; this is the experiment")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--allow-same-family", action="store_true",
                    help="evaluate on the TRAINED family on purpose. The guard "
                         "below exists because a silent family match measures "
                         "coverage while looking like transfer (RESULTS §15/§16). "
                         "But 'does RL make this agent better at the task it is "
                         "trained on' is a different and equally real question, "
                         "and answering it requires exactly this. Explicit only.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else data_root() / "verl_data"
    out.mkdir(parents=True, exist_ok=True)

    tr = rows(args.train_family, args.train_split, args.train_limit or None)

    # The leak this exists to prevent is silent and looks like success: the
    # policy scores higher every step on instances whose answers were in its
    # supervised corpus, and the training curve rises beautifully while the
    # model gets worse at the task.
    if args.sft_corpus:
        import json as _json
        seen = set()
        for line in Path(args.sft_corpus).read_text().splitlines():
            got = _json.loads(line).get("id")
            if got:
                seen.add(got)
        clash = {Path(r["instance_dir"]).name for r in tr} & seen
        if clash:
            raise SystemExit(
                f"{len(clash)} of {len(tr)} RL training instances also appear in "
                f"{args.sft_corpus}. RL would be optimising recall, not method. "
                f"Mint a disjoint split (generators/emit.py --seed0 ...).")
        print(f"leak check: 0 overlap with {Path(args.sft_corpus).name}  (OK)")
    va = rows(args.eval_family, "val", args.eval_limit or None)
    if args.train_family == args.eval_family and not args.allow_same_family:
        raise SystemExit("train and eval families are the same -- that measures "
                         "coverage, not transfer (see RESULTS §16). Pass "
                         "--allow-same-family if in-family improvement is the "
                         "measurement you actually want.")

    pd.DataFrame(tr).to_parquet(out / "train.parquet")
    pd.DataFrame(va).to_parquet(out / "val.parquet")
    print(f"train {len(tr):>5}  {args.train_family}")
    print(f"val   {len(va):>5}  {args.eval_family}   (held out)")
    print(f"wrote {out}/train.parquet and val.parquet")
    print(f"agent_name column -> {AGENT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
