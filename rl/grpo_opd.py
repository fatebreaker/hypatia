"""ON-POLICY DISTILLATION TRAINER -- H-OPD (docs/RL_PLAN.md, registered 2026-10-09): rl/grpo_v3.py installed exactly as it
installs itself (--process-lambda 0 --process-sham off), with rl.grpo_onpolicy.train replaced by opd_train below, which is
train() line for line except in two places:

  1. EVERY rollout that has a conversation enters the update, not only those of groups whose rewards vary: the signal no
     longer comes from the group's reward, so no group is "dead" (GR9-MC trains on ~2 of 8 groups a step late in training).
  2. The ADVANTAGE is per token: a_t = clamp(lp_T(x_t | x_<t) - lp_S(x_t | x_<t), -C, C), where lp_S is the policy's own
     log-probability of the token it sampled (the trainer's forward pass, detached) and lp_T the TEACHER's for the same
     token after the same prefix -- a vLLM server (--teacher-port, --teacher-model) scoring the trainer's exact token ids
     with prompt_logprobs. With this trainer's ratio of 1, -mean_t(a_t * log pi(x_t)) is the per-token reverse-KL gradient
     of on-policy distillation: the student learns on the trajectories it actually produces, and the teacher marks each
     token. --opd-reward-weight W adds W x the group-normalised TASK advantage to every token (default 0 = pure
     distillation: the task reward is still computed and logged, and never trained on).

The student and teacher must share the tokenizer (checked at start: equal tokenizer.json hashes), because the teacher
scores the student's token ids. Everything else -- LoRA, render, masking (assistant turns only), max length, optimizer,
checkpoints, resume -- is the RL trainer's, so runs compare with the RL runs step for step.

    venv_vllm/bin/python rl/grpo_opd.py --teacher-port P --teacher-model NAME --teacher-tokenizer T/tokenizer.json \
        --student-tokenizer S/tokenizer.json [--opd-clip 10] [--opd-reward-weight 0] --process-lambda 0 --process-sham off \
        <every rl/grpo_onpolicy.py argument>              (experiments/harness_v3/run_v3opd.sh launches it)

LOGS (in --out): harness_opd.json -- this file's sha256, the teacher (model name, adapter, tokenizer hash), clip and
weight; resuming a run dir under another teacher or objective is refused. history.jsonl gains opd_rkl (mean lp_S - lp_T
over the scored tokens: the reverse-KL estimate), opd_adv (mean a_t), opd_clip (fraction of tokens clipped), opd_tokens,
opd_missing (tokens without a teacher value; their a_t is 0) and t_teacher (seconds spent scoring).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl import grpo_onpolicy as G  # noqa: E402

# set by the __main__ block (or a test) before opd_train runs
OPD = argparse.Namespace(port=None, model=None, clip=10.0, weight=0.0, workers=8, timeout=900.0)


def parse_prompt_logprobs(ids: list[int], pl: list) -> list:
    """vLLM's prompt_logprobs (one entry per prompt token, None for the first) -> the log-probability of each ACTUAL
    token, None where it is missing. In the OpenAI-compatible JSON the keys are token ids as strings (ints accepted)."""
    if len(pl) != len(ids):
        raise RuntimeError(f"teacher returned {len(pl)} prompt log-probs for {len(ids)} tokens")
    out = []
    for i, d in enumerate(pl):
        if not d or not isinstance(d, dict):
            out.append(None)
            continue
        e = d.get(str(ids[i]), d.get(ids[i]))
        if e is None:
            out.append(None)
        else:
            out.append(float(e["logprob"]) if isinstance(e, dict) else float(e))
    return out


def teacher_logprobs(ids: list[int]) -> list:
    """The teacher's log-probability of every token of `ids` after its prefix (vLLM completions, prompt_logprobs=0:
    only the actual token's value is returned). Three attempts, then the step fails rather than train on nothing."""
    payload = {"model": OPD.model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{OPD.port}/v1/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=OPD.timeout) as r:
                body = json.loads(r.read())
            return parse_prompt_logprobs(ids, body["choices"][0]["prompt_logprobs"])
        except Exception as e:  # noqa: BLE001 -- retried, then raised
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"teacher scoring failed 3 times ({len(ids)} tokens): {last}")


def select_teacher(tl: list, mk: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The teacher's log-probs at exactly the positions lean_logprobs(model, ids, mk[1:]) returns -- the tokens ids[i],
    i >= 1, with mk[i] = 1, in order. Missing values become nan (their advantage is set to 0); returns (values, n_missing)."""
    idx = ((mk[1:] == 1).nonzero().squeeze(-1) + 1).tolist()
    vals = [tl[i] if tl[i] is not None else float("nan") for i in idx]
    return torch.tensor(vals, dtype=torch.float32), sum(tl[i] is None for i in idx)


def token_advantages(t: torch.Tensor, new_detached: torch.Tensor, clip: float, extra: float = 0.0):
    """a_t = clamp(lp_T - lp_S, -clip, clip) + extra; a token without a teacher value gets only `extra`.
    Returns (a, n_clipped, n_finite, sum of lp_S - lp_T over the finite tokens)."""
    d = t - new_detached
    ok = torch.isfinite(d)
    a = torch.where(ok, d.clamp(-clip, clip), torch.zeros_like(d)) + extra
    n_clip = int(((d.abs() > clip) & ok).sum())
    rkl = float((-d[ok]).sum()) if bool(ok.any()) else 0.0
    return a, n_clip, int(ok.sum()), rkl


def opd_train(args, model, tok, opt, trainable, insts, rng, start, n_steps_fam, tag,
              publish, hist_f, out) -> None:
    if OPD.port is None or OPD.model is None:
        raise SystemExit("rl/grpo_opd.py: no teacher configured (OPD.port / OPD.model)")
    for step in range(start + 1, args.steps + 1):
        t0 = time.time()
        chosen = rng.sample(insts, args.prompts_per_step)
        spec = f"llm:{tag}-pol{step - 1}:{args.render}"
        for hook in G.BEFORE_ROLLOUTS:
            hook()
        groups = G.rollouts(chosen, spec, args.k, args.temperature, args.timeout,
                            n_steps_fam, args.workers, args.c, args.require_code,
                            demo_for=getattr(args, "library_for", None))
        t_roll = time.time() - t0
        for hook in G.AFTER_ROLLOUTS:
            hook()

        allouts = [o for outs in groups.values() for o in outs]
        errored = sum(str(o.get("attribution") or "").startswith("error") for o in allouts)
        cells = sum(int(o.get("n_cells") or 0) for o in allouts)
        answered = sum(bool(o.get("answered")) for o in allouts)
        if errored == len(allouts) or (cells == 0 and answered == 0):
            kinds = sorted({str(o.get("attribution")) for o in allouts})[:5]
            raise SystemExit(f"step {step}: {errored}/{len(allouts)} rollouts errored, "
                             f"{cells} cells executed ({kinds}) -- instrument failure")
        batch, rewards, passes, dead = [], [], [], 0
        prefixes = []
        for outs in groups.values():
            rs = [o["reward"] for o in outs]
            rewards += rs
            passes += [bool(o.get("passed")) for o in outs]
            adv = G.advantages(rs)                     # the TASK advantage: logged (dead groups) and, x W, added
            dead += all(abs(a) < 1e-12 for a in adv)
            for o, a in zip(outs, adv):
                if o.get("messages"):                  # OPD: every rollout with a conversation (change 1)
                    batch.append((o["messages"], a))
                    prefixes.append(int(o.get("n_prefix") or 0))

        loss_sum, nseq, mask_frac, nll_sum, gnorm, oom = 0.0, 0, 0.0, 0.0, 0.0, 0
        n_tok = n_clip = n_miss = 0
        rkl_sum = adv_sum = 0.0
        t_teacher = 0.0
        if batch:
            opt.zero_grad(set_to_none=True)
            enc = [G.encode_rl(tok, m, args.max_len) if args.render == "rl"
                   else G.encode_with(tok, m, args.max_len, G.MISTRAL) if args.render == "mtrain"
                   else G.encode_with(tok, m, args.max_len, G.LLAMA3) if args.render == "ltrain"
                   else G.encode(tok, m, args.max_len) for m, _ in batch]
            if any(prefixes):
                enc = [(ids, G.mask_prefix(tok, m, ids, mk, npre, args.max_len))
                       for (ids, mk), (m, _), npre in zip(enc, batch, prefixes)]
            tot_tok = sum(len(ids) for ids, _ in enc)
            mask_frac = sum(int(mk.sum()) for _, mk in enc) / max(tot_tok, 1)
            # the teacher scores every episode's exact token ids before the backward pass (change 2)
            t1 = time.time()
            with cf.ThreadPoolExecutor(max(1, OPD.workers)) as ex:
                tls = list(ex.map(lambda e: teacher_logprobs(e[0].tolist()), enc))
            t_teacher = time.time() - t1
            for (ids, mk), (_, a_ep), tl in zip(enc, batch, tls):
                ids_c = ids.unsqueeze(0).cuda()
                m = mk[1:].cuda().float()
                if m.sum() < 1:
                    continue
                try:
                    new = G.lean_logprobs(model, ids_c, m)  # assistant positions only
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); oom += 1
                    continue
                t, nm = select_teacher(tl, mk)
                if t.numel() != new.numel():
                    raise SystemExit(f"step {step}: the teacher gave {t.numel()} log-probs for {new.numel()} policy "
                                     "tokens -- misaligned, refusing to train")
                a, ncl, nfin, rkl = token_advantages(t.cuda(), new.detach(), OPD.clip, OPD.weight * float(a_ep))
                if args.smoke and nseq == 0:
                    with torch.no_grad():
                        ref = G.seq_logprobs(model, ids_c, torch.ones_like(ids_c)).squeeze(0)[m.bool()]
                    err = float((ref - new.detach()).abs().max())
                    assert err < 0.05, f"lean logprobs disagree with seq_logprobs by {err}"
                    assert nm <= 0.01 * max(1, t.numel()), f"teacher values missing for {nm}/{t.numel()} tokens"
                    tf = t[torch.isfinite(t)]
                    assert tf.numel() and float(tf.max()) <= 1e-3, "teacher log-probs must be <= 0"
                    print(f"  smoke: lean vs full logprobs max |diff| = {err:.2e}; teacher tokens {t.numel()}, "
                          f"missing {nm}, mean lp_T {float(tf.mean()):.3f} vs lp_S {float(new.detach().mean()):.3f}, "
                          f"clipped {ncl}", flush=True)
                    del ref
                ratio = torch.exp(new - new.detach())  # 1: one update on fresh samples, as train()
                loss = -(ratio * a).mean()             # token mean within an episode
                try:
                    (loss / len(batch)).backward()     # mean over episodes
                except torch.cuda.OutOfMemoryError:
                    del new, ratio, a, loss
                    torch.cuda.empty_cache(); oom += 1
                    continue
                loss_sum += float(loss.detach()); nseq += 1
                nll_sum += float(-new.detach().mean())
                n_tok += nfin; n_clip += ncl; n_miss += nm; rkl_sum += rkl; adv_sum += float(a.detach().sum())
                del new, ratio, a, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if args.smoke:
                assert 0.02 < mask_frac < 0.98, f"assistant mask covers {mask_frac:.1%}"
                assert torch.isfinite(torch.tensor(loss_sum)), "non-finite loss"
                assert n_tok > 0, "no token was scored by the teacher"

        publish(step)                                  # next step samples the updated policy
        rec = {"step": step, "mean_reward": statistics.fmean(rewards), "wall_budget": args.wall_budget,
               "transfer_checked": 0, "transfer_passed": 0,
               "pass_rate": sum(passes) / len(passes), "dead_groups": dead,
               "groups": len(groups), "live_seqs": len(batch), "mask_frac": round(mask_frac, 4),
               "errored": errored, "cells_per_rollout": round(cells / len(allouts), 2),
               "error_kinds": sorted({str(o.get("attribution")) for o in allouts
                                      if str(o.get("attribution") or "").startswith("error")}),
               "error_msgs": sorted({str(o.get("error_msg", ""))[:120] for o in allouts
                                     if o.get("error_msg")})[:3],
               "loss": loss_sum / max(nseq, 1), "grad_norm": round(gnorm, 4),
               "nll": round(nll_sum / max(nseq, 1), 4), "lora_B_norm": round(G.lora_norm(model), 4),
               "oom_skipped": oom,
               "opd_rkl": round(rkl_sum / max(n_tok, 1), 5), "opd_adv": round(adv_sum / max(n_tok, 1), 5),
               "opd_clip": round(n_clip / max(n_tok, 1), 5), "opd_tokens": n_tok, "opd_missing": n_miss,
               "t_teacher": round(t_teacher, 1),
               "rollout_s": [round(float(x), 1) for x in
                             np.percentile([o.get("seconds", 0) for o in allouts], [50, 90, 100])],
               "t_rollout": round(t_roll, 1),
               "t_step": round(time.time() - t0, 1)}
        with open(hist_f, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(out / "rollouts.jsonl", "a") as f:
            for inst, outs in groups.items():
                for o in outs:
                    f.write(json.dumps({"step": step, "inst": inst, "reward": o["reward"],
                                        "passed": bool(o.get("passed")),
                                        "attr": o.get("attribution"),
                                        "fails_at_step": o.get("fails_at_step"),
                                        "answered": o.get("answered"), "answer": o.get("answer"),
                                        "cells": o.get("n_cells"), "sec": o.get("seconds")},
                                       default=str) + "\n")
        print(json.dumps(rec), flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0 or step == args.steps:
            d = out / f"step{step}"
            model.save_pretrained(str(d))
            print(f"  checkpoint -> {d}", flush=True)
        if args.resume_every and step % args.resume_every == 0 or step == args.steps:
            tmp, res = out / "resume.tmp", out / "resume"
            G.shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(str(tmp))
            torch.save(opt.state_dict(), tmp / "optimizer.pt")
            (tmp / "state.json").write_text(json.dumps({"step": step}))
            G.shutil.rmtree(res, ignore_errors=True)
            tmp.rename(res)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def guard(out: Path | None, rec: dict) -> None:
    """harness_opd.json: written at the start; a resume must match its teacher and objective."""
    if out is None:
        return
    f = out / "harness_opd.json"
    keys = ("teacher_model", "teacher_adapter", "teacher_tokenizer_sha256", "opd_clip", "opd_reward_weight")
    if f.exists():
        old = json.loads(f.read_text())
        if any(old.get(k) != rec.get(k) for k in keys):
            raise SystemExit(f"rl/grpo_opd.py: {out} was trained with {[(k, old.get(k)) for k in keys]} -- refusing to "
                             f"resume it with {[(k, rec.get(k)) for k in keys]}")
    else:
        if (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_opd.py: {out} has training history but no harness_opd.json -- refusing to resume "
                             "another objective as on-policy distillation")
        f.write_text(json.dumps(rec, indent=1))


def adopt(grpo_mod) -> None:
    grpo_mod.train = opd_train


if __name__ == "__main__":
    from rl import dispersion
    from rl import grpo_v3 as V3
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--teacher-port", type=int, required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--teacher-tokenizer", required=True)
    ap.add_argument("--student-tokenizer", required=True)
    ap.add_argument("--teacher-adapter", default="")
    ap.add_argument("--opd-clip", type=float, default=10.0)
    ap.add_argument("--opd-reward-weight", type=float, default=0.0)
    ap.add_argument("--opd-workers", type=int, default=8)
    oa, rest = ap.parse_known_args(sys.argv[1:])
    v3args, rest = V3.parse_v3_args(rest)
    if v3args.process_lambda != 0.0 or v3args.process_sham != "off":
        raise SystemExit("rl/grpo_opd.py: the teacher is the signal -- run with --process-lambda 0 --process-sham off")
    for bad in ("--kl", "--overflow-neutral", "--transfer-check", "--dyn-sample"):
        if any(a == bad or a.startswith(bad + "=") for a in rest):
            raise SystemExit(f"rl/grpo_opd.py: {bad} is not combined with on-policy distillation")
    if not oa.opd_clip > 0:
        raise SystemExit("rl/grpo_opd.py: --opd-clip must be > 0")
    th, sh = sha(Path(oa.teacher_tokenizer)), sha(Path(oa.student_tokenizer))
    if th != sh:
        raise SystemExit(f"rl/grpo_opd.py: teacher tokenizer {th} != student tokenizer {sh} -- the teacher cannot score "
                         "the student's token ids")
    OPD.port, OPD.model, OPD.clip, OPD.weight, OPD.workers = (oa.teacher_port, oa.teacher_model, oa.opd_clip,
                                                              oa.opd_reward_weight, oa.opd_workers)
    # fail fast if the teacher is not serving (a two-token probe)
    probe = teacher_logprobs([9707, 11, 1879])
    if len(probe) != 3 or probe[1] is None:
        raise SystemExit(f"rl/grpo_opd.py: the teacher probe returned {probe}")
    sys.argv = [sys.argv[0]] + rest
    V3.install(G, dispersion, v3args, rest)
    po = argparse.ArgumentParser(add_help=False)
    po.add_argument("--out")
    g, _ = po.parse_known_args(rest)
    out = Path(g.out) if g.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    guard(out, {"objective": "on-policy distillation (per-token a_t = clamp(lp_T - lp_S) + W x task advantage)",
                "trainer_opd_sha256": sha(Path(__file__)), "teacher_model": oa.teacher_model,
                "teacher_adapter": oa.teacher_adapter, "teacher_tokenizer_sha256": th,
                "opd_clip": oa.opd_clip, "opd_reward_weight": oa.opd_reward_weight})
    adopt(G)
    print(f"grpo_opd: ON-POLICY DISTILLATION from teacher {oa.teacher_model} (port {oa.teacher_port}"
          f"{', adapter ' + oa.teacher_adapter if oa.teacher_adapter else ''}); per-token advantage clamp +-{oa.opd_clip}, "
          f"task-reward weight {oa.opd_reward_weight}; probe lp {probe[1:]}", flush=True)
    raise SystemExit(G.main())
