"""AReaL FSDP actor with R1's loss aggregation, the empty-batch skip, and a per-update history.

LOSS AGGREGATION. rl/grpo_onpolicy.py averages the clipped surrogate over each episode's assistant tokens, then over
the episodes in its batch (`(loss / len(batch)).backward()` per episode; the batch holds only episodes with a non-zero
advantage). AReaL averages over every loss token of the batch at once, which weights an episode by its length. With
per-sequence weights

    A'_it = A_i * N / (n_live * L_i)        N = ALL loss tokens in the batch (AReaL's denominator),
                                            L_i = loss tokens of episode i,
                                            n_live = episodes with L_i > 0 and |A_i| > 1e-9 (R1's len(batch))

AReaL's token mean  (1/N) sum_i sum_t A'_it f_it  equals R1's  (1/n_live) sum_i (1/L_i) sum_t A_i f_it  exactly, for
the clipped objective too (the weight is a positive per-sequence constant, so the clip sides are unchanged).
A_i arrives as the trajectory "reward": rl/async_pilot/workflow.py computed it with R1's own advantages().

EMPTY BATCH. R1 skips the optimizer step when no episode has a non-zero advantage. AReaL asserts a positive loss
weight and would stop the run, so an update with no loss tokens is skipped here (version still advances, as R1 still
publishes an unchanged policy).

HISTORY. One JSON line per update in $DEEPRS_ASYNC_OUT/history.jsonl: wall time, live episodes, tokens, the staleness
of every trained episode (trainer version minus the version that sampled its first / last token), the LoRA-B norm
(R1's "has the policy moved" number), and whether the update ran.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist

from areal.engine.fsdp_engine import FSDPPPOActor
from areal.trainer.ppo.actor import PPOActor


def r1_sequence_weights(loss_mask: torch.Tensor, adv: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    """Per-sequence multipliers turning AReaL's batch token-mean into R1's per-episode-mean-then-mean (see module doc).
    loss_mask [B, S] (any dtype), adv [B]. Returns (weights [B], n_live, N)."""
    lm = loss_mask.float()
    L = lm.sum(-1)
    live = (L > 0) & (adv.abs() > 1e-9)
    n_live = int(live.sum().item())
    if n_live == 0:
        return torch.zeros_like(L), 0, 0.0
    N = float(L.sum().item())          # AReaL divides by every loss token of the batch, live or not
    w = torch.where(live, N / (n_live * L.clamp(min=1.0)), torch.zeros_like(L))
    return w, n_live, N


def _lora_b_norm(model) -> float:
    tot = 0.0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            t = p.detach()
            if hasattr(t, "full_tensor"):
                t = t.full_tensor()
            tot += float(t.float().pow(2).sum())
    return tot ** 0.5


class DeeprsPPOActor(PPOActor):
    def _one_sequence_per_micro_batch(self, data) -> None:
        """sdpa (no flash-attn build) is only correct if a micro-batch holds ONE sequence: AReaL packs a micro-batch
        into one row with attention_mask=None, and sdpa then lets later sequences attend to earlier ones
        (tests/test_trainer_cpu.py shows it). Set the micro-batch count to the batch's row count on every call, so the
        invariant holds whatever the batch size."""
        if self.engine.config.attn_impl == "sdpa":
            self.engine.config.mb_spec.n_mbs = int(data["input_ids"].shape[0])

    def _compute_logp(self, data):
        self._one_sequence_per_micro_batch(data)
        return super()._compute_logp(data)

    def _history(self, row: dict) -> None:
        out = os.environ.get("DEEPRS_ASYNC_OUT")
        if not out or (dist.is_initialized() and dist.get_rank() != 0):
            return
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "history.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def _compute_advantages(self, data, meta=None):
        data = super()._compute_advantages(data, meta)
        lm = data["loss_mask"]                              # rolled float mask: position t predicts token t+1
        adv = data["rewards"].float().to(lm.device)         # R1 group advantage, one per sequence
        w, n_live, N = r1_sequence_weights(lm, adv)
        # AReaL's GAE with gamma = lambda = 1, no critic, no KL puts the terminal reward on every loss token; R1 puts
        # the episode advantage there. Check that, then apply R1's aggregation weights.
        live = w > 0
        gae_err = (float(((data["advantages"] - adv[:, None]).abs() * lm)[live].max()) if n_live else 0.0)
        data["advantages"] = lm * (adv * w)[:, None]
        cur = int(self.engine.get_version())
        vers = data.get("versions")
        head, tail = [], []
        if vers is not None:
            att = data["attention_mask"].bool()
            for i in range(vers.shape[0]):
                v = vers[i][(vers[i] >= 0) & att[i]]           # sampled tokens (context is -1; padding is outside att)
                if bool(live[i]) and v.numel():
                    head.append(cur - int(v.min()))
                    tail.append(cur - int(v.max()))
        self._pending = {
            "step": cur + 1, "t_adv": round(time.time(), 1), "rows": int(lm.shape[0]), "live_seqs": n_live,
            "loss_tokens": int(N), "gae_check_max_err": round(gae_err, 6),
            "staleness_head": dict(sorted(Counter(head).items())), "staleness_tail": dict(sorted(Counter(tail).items())),
            "frac_head_gt2": round(sum(h > 2 for h in head) / len(head), 4) if head else None,
            "mean_abs_adv": round(float(adv[live].abs().mean()), 4) if n_live else 0.0}
        return data

    def _ppo_update(self, data) -> None:
        row = getattr(self, "_pending", {}) or {}
        self._pending = None
        if float(data["loss_mask"].sum()) == 0.0:
            # R1: `if batch:` -- no live episode, no optimizer step
            row.update({"updated": False, "t_end": round(time.time(), 1)})
            self._history(row)
            return
        self._one_sequence_per_micro_batch(data)
        super()._ppo_update(data)
        row.update({"updated": True, "t_end": round(time.time(), 1),
                    "lora_B_norm": round(_lora_b_norm(self.engine.model), 4)})
        self._history(row)


class DeeprsFSDPPPOActor(FSDPPPOActor):
    """FSDPPPOActor whose PPO logic is DeeprsPPOActor. Imported by class path in AReaL's actor worker process."""

    def __init__(self, config):
        super().__init__(config)
        self.actor = DeeprsPPOActor(config, self)
