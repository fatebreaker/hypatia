"""The TRAINER side on CPU: AReaL's real FSDP engine + rl/async_pilot/actor.py on a tiny random Qwen3 (no GPU).

AReaL ships a CPU platform (gloo), so its FSDP2 engine, PEFT wrapping, micro-batching, decoupled loss and HF save run
here unchanged. With a 2-layer Qwen3 of the same architecture and the real Qwen3 tokenizer:

  1. DeeprsFSDPPPOActor builds with the pilot's LoRA (7 target modules) and only LoRA is trainable;
  2. one-sequence-per-micro-batch (mb_spec.n_mbs = rows) + sdpa gives each sequence exactly the log-probs of a plain
     HF forward over that sequence alone -- and packing several sequences with sdpa does NOT (the reason for n_mbs);
  3. compute_logp -> compute_advantages -> ppo_update runs on a workflow-shaped batch (multi-turn masks, stubs),
     the loss is finite, LoRA-B moves off zero, history.jsonl gets a row;
  4. a batch with no loss tokens (every group flat) is skipped, not crashed on (R1: no optimizer step);
  5. the checkpoint save writes PEFT layout (adapter_model.safetensors + adapter_config.json, PEFT key names) that
     PEFT reloads onto the base to the same outputs -- the layout experiments/rl_plan/eval.sh hot-loads into vLLM.

    venv_async/bin/python rl/async_pilot/tests/test_trainer_cpu.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
TOKENIZER = "/scratch4/workspace/hieutran_umass_edu-deeprs-bench/models/Qwen3-8B"
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def tiny_model(d: Path) -> Path:
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=False, torch_dtype="float32")
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def row(ids, loss, ver, adv):
    n = len(ids)
    return {"input_ids": torch.tensor(ids, dtype=torch.int32), "loss_mask": torch.tensor(loss, dtype=torch.int32),
            "logprobs": torch.tensor([(-0.5 if m else 0.0) for m in loss], dtype=torch.float32),
            "versions": torch.tensor(ver, dtype=torch.int32), "attention_mask": torch.ones(n, dtype=torch.bool),
            "rewards": torch.tensor(float(adv), dtype=torch.float32)}


def group(tok, advs, seed):
    """Six workflow-shaped rows: [prompt | gen | tool | gen] with masks, plus stubs where adv == 0."""
    from areal.utils.data import concat_padded_tensors
    g = torch.Generator().manual_seed(seed)
    rows = []
    for a in advs:
        if a == 0.0:
            ids = [tok.convert_tokens_to_ids("<|im_start|>")] * 2
            r = row(ids, [0, 0], [-1, -1], 0.0)
        else:
            p, g1, t, g2 = (int(torch.randint(20, 60, (1,), generator=g)) for _ in range(4))
            ids = torch.randint(100, 5000, (p + g1 + t + g2,), generator=g).tolist()
            loss = [0] * p + [1] * g1 + [0] * t + [1] * g2
            ver = [-1] * p + [3] * g1 + [-1] * t + [4] * g2
            r = row(ids, loss, ver, a)
        rows.append({k: v.unsqueeze(0) for k, v in r.items()})
    return concat_padded_tensors(rows)


def main() -> int:
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
                 "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())}.items():
        os.environ.setdefault(k, v)
    tmp = Path(tempfile.mkdtemp(prefix="apilot_cpu_", dir=os.environ.get("TMPDIR")))
    os.environ["DEEPRS_ASYNC_OUT"] = str(tmp / "run")
    model = tiny_model(tmp / "tiny_qwen3")
    from areal.api import FinetuneSpec
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.api.io_struct import SaveLoadMeta
    from rl.async_pilot.actor import DeeprsFSDPPPOActor
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ROWS = 12                                         # 2 groups x 6
    cfg, _ = load_expr_config(["--config", str(REPO / "rl/async_pilot/config.yaml"), f"actor.path={model}",
                               "actor.dtype=float32", "actor.lora_rank=8", "actor.lora_alpha=16",
                               f"actor.mb_spec.n_mbs={ROWS}", "actor.mb_spec.max_tokens_per_mb=4096",
                               "actor.optimizer.lr=1e-2", "actor.gradient_checkpointing=false",
                               f"cluster.fileroot={tmp}/exp", f"cluster.name_resolve.nfs_record_root={tmp}/nr"],
                              GRPOConfig)
    print("[1] build DeeprsFSDPPPOActor on CPU (gloo)")
    eng = DeeprsFSDPPPOActor(cfg.actor)
    eng.create_process_group(parallel_strategy=ModelAllocation.from_str(cfg.actor.backend).parallel)
    eng.initialize(None, FinetuneSpec(total_train_epochs=1, dataset_size=64, train_batch_size=8))
    tr = [n for n, p in eng.model.named_parameters() if p.requires_grad]
    mods = sorted({n.split(".lora_")[0].rsplit(".", 1)[-1] for n in tr})
    check("only LoRA parameters train, on the 7 R1 target modules", all("lora_" in n for n in tr) and
          mods == sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]), str(mods))

    print("[2] log-probs: one sequence per micro-batch == plain HF forward per sequence")
    batch = [group(tok, [0.9, -0.4, 0.0, 1.1, -1.6, 0.0], 1), group(tok, [0.5, 0.0, -0.5, 0.0, 0.0, 0.0], 2)]
    lp = eng.compute_logp(batch)
    ref = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, attn_implementation="sdpa").eval()
    worst = 0.0
    flat_lp = [x for t in lp for x in (t if t.dim() == 2 else [t])]
    rows_all = [(b, i) for b in batch for i in range(b["input_ids"].shape[0])]
    for (b, i), got in zip(rows_all, flat_lp):
        n = int(b["attention_mask"][i].sum())
        ids = b["input_ids"][i][:n].long()[None]
        with torch.no_grad():
            lg = ref(input_ids=ids).logits[0].float()
        want = torch.log_softmax(lg, -1)[:-1].gather(-1, ids[0, 1:, None])[:, 0]
        worst = max(worst, float((got[:n - 1] - want).abs().max()))
    check("per-sequence log-probs match an unpacked HF forward (LoRA B = 0: policy == base)", worst < 1e-4,
          f"max |diff| {worst:.2e} over {len(rows_all)} rows")
    # what the actor's guard prevents: pack everything into one micro-batch under sdpa (guard bypassed by relabelling
    # attn_impl for this call only; the model itself still runs sdpa)
    eng.config.attn_impl, eng.config.mb_spec.n_mbs = "sdpa-guard-off", 1
    lp_packed = eng.compute_logp(batch)
    eng.config.attn_impl, eng.config.mb_spec.n_mbs = "sdpa", ROWS
    flat_p = [x for t in lp_packed for x in (t if t.dim() == 2 else [t])]
    diff = max(float((a - b).abs().max()) for a, b in zip(flat_lp, flat_p))
    check("packing several sequences under sdpa changes them (cross-sequence attention) -> n_mbs must = rows",
          diff > 1e-3, f"max |diff| packed vs one-per-mb {diff:.2e}")

    print("[3] compute_advantages + ppo_update on a workflow-shaped batch")
    for t, l in zip(batch, lp):
        t["prox_logp"] = l
    adv = eng.compute_advantages(batch)
    eng.ppo_update(adv)
    hist = [json.loads(x) for x in open(Path(os.environ["DEEPRS_ASYNC_OUT"]) / "history.jsonl")]
    h = hist[-1]
    check("update ran; LoRA-B moved off zero; history row written", h.get("updated") and h.get("lora_B_norm", 0) > 0,
          json.dumps({k: h[k] for k in ("step", "live_seqs", "loss_tokens", "staleness_head", "lora_B_norm",
                                        "gae_check_max_err")}))
    check("live episodes = rows with loss tokens and non-zero advantage", h["live_seqs"] == 6, str(h["live_seqs"]))

    print("[4] a batch with no loss tokens (every group flat) is skipped, not crashed on")
    flat = [group(tok, [0.0] * 6, 3), group(tok, [0.0] * 6, 4)]
    for t, l in zip(flat, eng.compute_logp(flat)):
        t["prox_logp"] = l
    eng.ppo_update(eng.compute_advantages(flat))
    h = json.loads(open(Path(os.environ["DEEPRS_ASYNC_OUT"]) / "history.jsonl").read().splitlines()[-1])
    check("empty batch -> no optimizer step, logged", h.get("updated") is False)

    print("[5] checkpoint in PEFT layout, reloadable")
    out = tmp / "run" / "step1"
    eng.save(SaveLoadMeta(path=str(out), weight_format="hf", with_optim=False, tokenizer=None,
                          base_model_path=str(model)))
    files = sorted(p.name for p in out.iterdir())
    from safetensors import safe_open
    with safe_open(str(out / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    check("adapter_model.safetensors + adapter_config.json with PEFT key names",
          "adapter_model.safetensors" in files and "adapter_config.json" in files
          and all(k.startswith("base_model.model.model.layers.") and (".lora_A.weight" in k or ".lora_B.weight" in k)
                  for k in keys), f"{files}; e.g. {keys[0]}")
    acfg = json.loads((out / "adapter_config.json").read_text())
    check("adapter_config: r, alpha, target modules as trained", acfg["r"] == 8 and acfg["lora_alpha"] == 16
          and sorted(acfg["target_modules"]) == sorted(cfg.actor.target_modules), str({k: acfg[k] for k in ("r", "lora_alpha", "target_modules")}))
    from peft import PeftModel
    pm = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32), str(out)).eval()
    b = batch[0]
    n = int(b["attention_mask"][0].sum())
    ids = b["input_ids"][0][:n].long()[None]
    with torch.no_grad():
        want = torch.log_softmax(pm(input_ids=ids).logits[0].float(), -1)[:-1].gather(-1, ids[0, 1:, None])[:, 0]
    got = eng.compute_logp([b])[0]               # a 6-row batch: the actor re-sets n_mbs to the row count itself
    got = got[0] if got.dim() == 2 else got
    d = float((got[:n - 1] - want).abs().max())
    check("PEFT reload of the saved adapter reproduces the trainer's policy", d < 1e-3, f"max |diff| {d:.2e}")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
