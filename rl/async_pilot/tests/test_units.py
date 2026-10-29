"""CPU unit checks for the async pilot (no GPU, no model weights):

  1. dataset parity  -- rl/async_pilot/dataset.py reproduces the instances R1 seed s drew at every logged step
                        (read back from the R1 run's rollouts.jsonl);
  2. loss parity     -- AReaL's own ppo_actor_loss_fn, aggregated the way FSDPEngine.train_batch aggregates micro-batches,
                        with rl/async_pilot/actor.py's per-sequence weights, equals rl/grpo_onpolicy.py's objective
                        (value AND gradient), on-policy and with clipping active;
  3. advantage path  -- DeeprsPPOActor._compute_advantages on a synthetic batch: AReaL's GAE puts the group advantage on
                        every loss token, the R1 weights are applied, stubs get nothing, staleness is logged.

    venv_async/bin/python rl/async_pilot/tests/test_units.py [--run-dir R1_RUN_DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R1_S1 = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/ckpt/rl-czero-rlmix6-base-echo-rc-lr1e-5-s1"
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def test_dataset(run_dir: Path) -> None:
    print("[1] dataset parity with R1's instance stream")
    from rl.async_pilot.dataset import R1InstanceStream, r1_stream
    seed = int(run_dir.name.rsplit("-s", 1)[1])
    logged: dict[int, set] = defaultdict(set)
    for line in open(run_dir / "rollouts.jsonl"):
        r = json.loads(line)
        logged[int(r["step"])].add(r["inst"])
    steps = sorted(logged)
    draws = r1_stream(seed, max(steps))
    mism = [s for s in steps if {p.name for p in draws[s - 1]} != logged[s]]
    check(f"seed {seed}: steps 1..{max(steps)} draw the same 8 instances as the R1 run logged",
          not mism, f"{len(steps)} steps compared, mismatches at {mism[:5]}")
    ds = R1InstanceStream(seed=seed, steps=len(steps), margin_steps=0)
    flat = [Path(x["inst"]).name for x in ds.items]
    check("flat dataset order = step order", flat[:8] == [p.name for p in draws[0]] and len(flat) == 8 * len(steps))


def r1_objective(new: list[torch.Tensor], old: list[torch.Tensor], adv: list[float], clip: float) -> torch.Tensor:
    """rl/grpo_onpolicy.py train(): per live episode, -mean_t min(ratio*A, clip(ratio)*A); summed / len(batch)."""
    batch = [(n, o, a) for n, o, a in zip(new, old, adv) if abs(a) > 1e-9 and n.numel()]
    tot = torch.zeros(())
    for n, o, a in batch:
        ratio = torch.exp(n - o)
        obj = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
        tot = tot + (-obj.mean()) / len(batch)
    return tot


def test_loss(seed: int = 0) -> None:
    print("[2] loss parity: AReaL loss + R1 weights == R1 objective")
    from areal.api.cli_args import RejectionSamplingConfig
    from areal.utils.functional import ppo_actor_loss_fn
    from rl.async_pilot.actor import r1_sequence_weights
    g = torch.Generator().manual_seed(seed)
    B, S, clip = 7, 40, 0.2
    lens = [0, 5, 17, 33, 9, 0, 21]                      # rows with 0 loss tokens are stubs
    adv = torch.tensor([0.0, 1.3, -0.4, 0.9, -1.7, 0.0, 0.0])   # last row: live length but zero advantage (R1 drops it)
    lm = torch.zeros(B, S)
    for i, L in enumerate(lens):
        lm[i, 3:3 + L] = 1
    for label, perturb in (("on-policy (ratio 1)", 0.0), ("off-policy, clipping active", 0.6)):
        old = -torch.rand(B, S, generator=g) * 3
        base = old + perturb * torch.randn(B, S, generator=g)
        # AReaL side: one sequence per micro-batch; loss_mb = sum(l)/n_mb ; total = sum_mb n_mb/N * loss_mb
        w, n_live, N = r1_sequence_weights(lm, adv)
        new_a = base.clone().requires_grad_(True)
        A = lm * (adv * w)[:, None]
        total_w = lm.sum()
        loss_a = torch.zeros(())
        for i in range(B):
            n_mb = lm[i].sum()
            if n_mb == 0:
                continue
            li, _ = ppo_actor_loss_fn(logprobs=new_a[i], proximal_logprobs=old[i], old_logprobs=old[i],
                                      advantages=A[i], eps_clip=clip, loss_mask=lm[i].bool(),
                                      rejection_sampling=RejectionSamplingConfig(metric="ratio", upper=5.0))
            loss_a = loss_a + li * (n_mb / total_w)
        loss_a.backward()
        # R1 side
        new_r = base.clone().requires_grad_(True)
        loss_r = r1_objective([new_r[i][lm[i].bool()] for i in range(B)], [old[i][lm[i].bool()] for i in range(B)],
                              adv.tolist(), clip)
        loss_r.backward()
        dv = abs(loss_a.item() - loss_r.item())
        dg = float((new_a.grad - new_r.grad).abs().max())
        check(f"{label}: value {loss_a.item():+.6f} vs R1 {loss_r.item():+.6f}; max |grad diff| {dg:.2e}",
              dv < 1e-6 and dg < 1e-7, f"n_live={n_live}, N={N:.0f}")
    # without the weights AReaL's token mean differs (the reason the weights exist)
    A0 = lm * adv[:, None]
    val0 = float((-(A0 * lm)).sum() / lm.sum())
    val_r1 = float(r1_objective([torch.zeros(int(L)) for L in lens], [torch.zeros(int(L)) for L in lens],
                                adv.tolist(), clip))
    check("unweighted AReaL token-mean is NOT R1's objective (weights are needed)", abs(val0 - val_r1) > 1e-3,
          f"{val0:+.4f} vs {val_r1:+.4f}")


class _Engine:
    def __init__(self, v):
        self.v = v

    def get_version(self):
        return self.v


def test_advantage_path() -> None:
    print("[3] DeeprsPPOActor._compute_advantages on a synthetic batch")
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from rl.async_pilot.actor import DeeprsPPOActor, r1_sequence_weights
    tmp = Path(tempfile.mkdtemp(prefix="apilot_", dir=os.environ.get("TMPDIR")))
    cfg, _ = load_expr_config(["--config", str(REPO / "rl/async_pilot/config.yaml"),
                               f"cluster.fileroot={tmp}/exp", f"cluster.name_resolve.nfs_record_root={tmp}/nr"],
                              GRPOConfig)
    os.environ["DEEPRS_ASYNC_OUT"] = str(tmp / "run")
    actor = DeeprsPPOActor(cfg.actor, _Engine(5))
    B, S = 4, 12
    lm0 = torch.zeros(B, S, dtype=torch.int32)
    lm0[0, 4:8] = 1; lm0[0, 9:11] = 1              # two turns, sampled tokens end at position 10
    lm0[1, 5:9] = 1                                 # one turn, ends at 8
    lm0[3, 2:12] = 1
    seqlen = [11, 9, 2, 12]                         # row 2 is a stub
    att = torch.zeros(B, S, dtype=torch.bool)
    for i, L in enumerate(seqlen):
        att[i, :L] = True
    ver = torch.full((B, S), -1, dtype=torch.int32)
    ver[0][lm0[0].bool()] = torch.tensor([3, 3, 3, 3, 4, 4], dtype=torch.int32)
    ver[1][lm0[1].bool()] = 5
    ver[3][lm0[3].bool()] = 1
    adv = torch.tensor([0.8, -1.2, 0.0, 0.4])
    data = {"input_ids": torch.randint(0, 100, (B, S), dtype=torch.int32), "attention_mask": att,
            "loss_mask": lm0, "logprobs": -torch.rand(B, S) * lm0, "versions": ver, "rewards": adv.clone()}
    out = actor._compute_advantages(data)
    lm = out["loss_mask"]                            # rolled
    w, n_live, N = r1_sequence_weights(lm, adv)
    exp = lm * (adv * w)[:, None]
    check("advantages = rolled loss mask x A_i x R1 weight", torch.allclose(out["advantages"], exp),
          f"n_live={n_live}, N={N:.0f}, weights={[round(float(x), 3) for x in w]}")
    row = actor._pending
    check("AReaL GAE put A_i on every loss token (gamma=lambda=1, no critic)", row["gae_check_max_err"] < 1e-6,
          f"max err {row['gae_check_max_err']}")
    check("staleness = trainer version (5) - sampling version, per live row",
          row["staleness_head"] == {0: 1, 2: 1, 4: 1} and row["staleness_tail"] == {0: 1, 1: 1, 4: 1}, json.dumps(row))
    check("stub and zero-advantage rows carry no advantage", float(out["advantages"][2].abs().sum()) == 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=R1_S1)
    a = ap.parse_args()
    test_dataset(Path(a.run_dir))
    test_loss()
    test_advantage_path()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
