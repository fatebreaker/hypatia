"""Entry point: R1 on AReaL v2.0.0 (single-controller, local scheduler; one vLLM GPU + one FSDP trainer GPU).

    venv_async/bin/python -m rl.async_pilot.train --out RUN_DIR --seed 1 [--dry-run] -- \
        --config rl/async_pilot/config.yaml [areal.key=value ...]

Everything before `--` is ours; everything after goes to AReaL's own parser (load_expr_config, GRPOConfig). Launch via
rl/async_pilot/launch.sh, which sets the environment the harness needs (GBP_ECHO_LAST, GBP_SANDBOX_PYTHON, ...).

Differences from PPOTrainer, all by subclassing (AReaL's files are not edited):
  * the actor is rl.async_pilot.actor.DeeprsFSDPPPOActor (R1 loss aggregation, empty-batch skip, history.jsonl);
  * after update N in --ckpt-steps the LoRA adapter is saved to RUN_DIR/stepN in PEFT layout
    (adapter_model.safetensors + adapter_config.json), which experiments/rl_plan/eval.sh hot-loads unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse(argv: list[str]):
    if "--" in argv:
        i = argv.index("--")
        ours, theirs = argv[:i], argv[i + 1:]
    else:
        ours, theirs = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory (history.jsonl, rollouts.jsonl, stepN/)")
    ap.add_argument("--family", default="rlmix6")
    ap.add_argument("--k", type=int, default=6, help="rollouts per group (R1: 6)")
    ap.add_argument("--c", type=float, default=0.5, help="reward partial-credit ceiling (R1: 0.5)")
    ap.add_argument("--timeout", type=int, default=180, help="sandbox cell timeout (R1: 180)")
    ap.add_argument("--wall-budget", type=float, default=600.0, help="R1 straggler cap, seconds")
    ap.add_argument("--no-require-code", action="store_true", help="R1 uses --require-code; this turns it off")
    ap.add_argument("--ckpt-steps", default="25,50,75,100,125,150")
    ap.add_argument("--threads", type=int, default=256, help="episode threads in the rollout worker")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check everything that does not need a GPU, print the plan, and stop")
    return ap.parse_args(ours), theirs


def workflow_kwargs(args, config) -> dict:
    return dict(tokenizer_path=config.tokenizer_path or config.actor.path, lora_name=config.gconfig.lora_name,
                k=args.k, temperature=config.gconfig.temperature, top_p=config.gconfig.top_p,
                max_model_len=config.vllm.max_model_len, max_new_tokens=config.gconfig.max_new_tokens,
                c=args.c, timeout=args.timeout, require_code=not args.no_require_code,
                wall_budget=args.wall_budget, log_dir=str(args.out), threads=args.threads)


def check_config(args, config) -> list[str]:
    """R1 invariants the config must satisfy. Returns the list of violations (empty = OK)."""
    a, g, v, r = config.actor, config.gconfig, config.vllm, config.rollout
    bad = []

    def want(name, got, exp):
        if got != exp:
            bad.append(f"{name} = {got!r}, R1 needs {exp!r}")
    want("actor.use_lora", a.use_lora, True)
    want("rollout.use_lora", r.use_lora, True)
    want("actor.lora_rank", a.lora_rank, 128)
    want("actor.lora_alpha", a.lora_alpha, 256)
    want("actor.target_modules", sorted(a.target_modules or []),
         sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    want("actor.optimizer.lr", a.optimizer.lr, 1e-5)
    want("actor.optimizer.weight_decay", a.optimizer.weight_decay, 0.01)     # torch AdamW default, R1's optimizer
    want("actor.optimizer.lr_scheduler_type", a.optimizer.lr_scheduler_type, "constant")
    want("actor.optimizer.warmup_steps_proportion", a.optimizer.warmup_steps_proportion, 0.0)
    want("actor.optimizer.gradient_clipping", a.optimizer.gradient_clipping, 1.0)
    want("actor.eps_clip", a.eps_clip, 0.2)
    want("actor.kl_ctl", a.kl_ctl, 0.0)
    want("actor.discount", a.discount, 1.0)
    want("actor.gae_lambda", a.gae_lambda, 1.0)
    want("actor.reward_norm", a.reward_norm, None)
    want("actor.adv_norm", a.adv_norm, None)
    want("actor.reward_scaling", a.reward_scaling, 1.0)
    want("actor.reward_bias", a.reward_bias, 0.0)
    want("actor.ppo_n_minibatches", a.ppo_n_minibatches, 1)                 # one optimizer step per update, as R1
    want("actor.temperature", a.temperature, g.temperature)                # logits scaled like the sampler
    want("actor.disable_dropout", a.disable_dropout, True)
    want("actor.weight_update_mode", a.weight_update_mode, "disk")          # AReaL: LoRA + vLLM goes through disk
    want("gconfig.n_samples", g.n_samples, 1)                               # the workflow runs the group itself
    want("gconfig.temperature", g.temperature, 1.0)
    want("gconfig.top_p", g.top_p, 0.95)                                    # harness LLMAgent.top_p
    want("gconfig.max_new_tokens", g.max_new_tokens, 8192)                  # harness LLMAgent.max_tokens
    want("vllm.max_model_len", v.max_model_len, 16384)                      # R1 server MAX_LEN and trainer --max-len
    want("vllm.enable_lora", v.enable_lora, True)
    want("vllm.max_lora_rank", v.max_lora_rank, 128)
    want("train_dataset.shuffle", config.train_dataset.shuffle, False)      # R1's stream, in order
    want("train_dataset.batch_size", config.train_dataset.batch_size, 8)    # 8 groups per update
    want("rollout.consumer_batch_size", r.consumer_batch_size, 8)
    rows = config.train_dataset.batch_size * args.k
    want("actor.mb_spec.n_mbs", a.mb_spec.n_mbs, rows)                      # one sequence per micro-batch
    if (a.mb_spec.max_tokens_per_mb or 0) < v.max_model_len:
        bad.append(f"actor.mb_spec.max_tokens_per_mb {a.mb_spec.max_tokens_per_mb} < max_model_len {v.max_model_len}")
    if a.attn_impl == "sdpa" and a.mb_spec.n_mbs != rows:
        bad.append("sdpa with packed micro-batches would let sequences attend to each other")
    if r.max_head_offpolicyness < 1:
        bad.append("rollout.max_head_offpolicyness < 1 is synchronous RL, not the pilot")
    if not a.use_decoupled_loss:
        bad.append("actor.use_decoupled_loss must be on for stale samples")
    return bad


def main(argv: list[str]) -> int:
    args, theirs = parse(argv)
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.api.io_struct import SaveLoadMeta
    from areal.utils.environ import is_single_controller

    from rl.async_pilot.dataset import R1InstanceStream

    config, _ = load_expr_config(theirs, GRPOConfig)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPRS_ASYNC_OUT"] = str(out)          # read by the actor worker (history.jsonl)
    ckpt_steps = sorted({int(s) for s in args.ckpt_steps.split(",") if s})
    bad = check_config(args, config)
    for b in bad:
        print(f"[async_pilot] CONFIG VIOLATION: {b}", flush=True)
    if bad:
        return 2
    steps = int(config.total_train_steps or 150)
    ds = R1InstanceStream(seed=config.seed, steps=steps, per_step=config.train_dataset.batch_size, family=args.family)
    wkw = workflow_kwargs(args, config)
    meta = {"engine": "AReaL v2.0.0 (single-controller, local scheduler)", "r1_condition": "R1 (rlmix6, outcome reward)",
            "seed": config.seed, "steps": steps, "ckpt_steps": ckpt_steps, "workflow_kwargs": wkw,
            "max_head_offpolicyness": config.rollout.max_head_offpolicyness,
            "max_concurrent_rollouts": config.rollout.max_concurrent_rollouts,
            "decoupled_loss": config.actor.use_decoupled_loss, "prox_logp_method": config.actor.prox_logp_method,
            "attn_impl": config.actor.attn_impl, "model": config.actor.path, "dataset_items": len(ds),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "render.txt").write_text("rl\n")             # experiments/rl_foothold/eval.sh reads the render from here
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    if args.dry_run:
        # Everything short of starting workers: the workflow and actor classes import, the workflow constructs
        # (tokenizer, sandbox preflight), the vLLM command AReaL would run, and the first update's instances.
        from areal.api.alloc_mode import ModelAllocation
        from areal.api.cli_args import vLLMConfig
        from areal.utils.dynamic_import import import_from_string
        wf_cls = import_from_string("rl.async_pilot.workflow.DeeprsGroupWorkflow")
        wf_cls(**wkw)
        import_from_string("rl.async_pilot.actor.DeeprsFSDPPPOActor")
        ra = ModelAllocation.from_str(config.rollout.backend, name="rollout")
        aa = ModelAllocation.from_str(config.actor.backend, name="actor")
        cmd = vLLMConfig.build_cmd(config.vllm, tp_size=ra.parallel.tp_size, pp_size=ra.parallel.pp_size,
                                   host="127.0.0.1", port=0)
        print("[dry-run] config OK against every R1 invariant in check_config()")
        print(f"[dry-run] allocation: rollout {config.rollout.backend} ({ra.parallel.world_size} GPU), "
              f"actor {config.actor.backend} ({aa.parallel.world_size} GPU); cluster.n_gpus_per_node "
              f"{config.cluster.n_gpus_per_node}")
        print(f"[dry-run] vLLM server command (port/LoRA module filled at launch):\n  {cmd}")
        print(f"[dry-run] dataset: {len(ds)} items; update 1 draws {[Path(x['inst']).name for x in ds.items[:8]]}")
        print(f"[dry-run] workflow kwargs: {json.dumps(wkw)}")
        print(f"[dry-run] single-controller: {is_single_controller()} -- next step would start the local scheduler "
              f"and GPU workers; stopping here")
        return 0

    class DeeprsPPOTrainer(PPOTrainer):
        def _create_train_engine(self, actor_config, alloc):
            if alloc.backend != "fsdp":
                return super()._create_train_engine(actor_config, alloc)
            from rl.async_pilot.actor import DeeprsFSDPPPOActor
            actor = (DeeprsFSDPPPOActor.as_controller(actor_config, self.scheduler) if is_single_controller()
                     else DeeprsFSDPPPOActor(config=actor_config))
            actor.create_process_group(parallel_strategy=alloc.parallel)
            return actor

        def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
            super()._save_hf(epoch, epoch_step, global_step)
            step = global_step + 1                       # after update g the policy has had g+1 updates = R1 "step"
            if step in ckpt_steps:
                d = out / f"step{step}"
                self.actor.save(SaveLoadMeta(path=str(d), weight_format="hf", with_optim=False, tokenizer=None,
                                             base_model_path=config.actor.path))
                ok = (d / "adapter_model.safetensors").is_file() and (d / "adapter_config.json").is_file()
                print(f"[async_pilot] checkpoint step{step} -> {d} ({'OK' if ok else 'MISSING adapter files'})",
                      flush=True)
                if not ok:
                    raise RuntimeError(f"adapter checkpoint {d} is not in PEFT layout")

    with DeeprsPPOTrainer(config, train_dataset=ds, valid_dataset=None) as trainer:
        trainer.train(workflow="rl.async_pilot.workflow.DeeprsGroupWorkflow", workflow_kwargs=wkw)
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
