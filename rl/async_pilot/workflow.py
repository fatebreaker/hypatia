"""AReaL RolloutWorkflow: one R1 GRPO group (k episodes of one instance) -> a training batch slice.

Runs inside AReaL's rollout worker, in its event loop. Each episode is the synchronous harness loop
(rl/async_pilot/episode.py run_episode), so it runs on a worker thread; its model calls come back to the event loop
as engine.agenerate() coroutines (asyncio.run_coroutine_threadsafe), which is where AReaL's versioning, pause/resume
around weight updates and interruptible generation live.

Why the workflow runs the whole group instead of AReaL's group_size: the GRPO advantage is R1's own function
(rl/grpo_train.advantages: group mean, POPULATION std, zero for a flat group, errored rollouts entering as reward 0),
computed here over the finished group and handed to the trainer as the per-sequence "reward"; the trainer's reward
and advantage normalisation are off (config.yaml) and rl/async_pilot/actor.py turns it into R1's
per-episode-token-mean / mean-over-live-episodes loss. A sequence that R1 would not put in its batch (errored, zero
advantage, desynchronised) is returned as a 2-token stub with no loss tokens, so every group is exactly k rows and the
one-sequence-per-micro-batch layout (mb_spec.n_mbs = batch rows) holds.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextvars
import functools
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch

from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils.data import concat_padded_tensors

from rl.async_pilot.episode import OVERFLOW_TAG, is_infrastructure_error, run_episode
from rl.grpo_train import advantages as r1_advantages          # R1's group advantage, unchanged

_EXEC: cf.ThreadPoolExecutor | None = None
_EXEC_LOCK = threading.Lock()
_TOK: dict[str, Any] = {}
_LOG_LOCK = threading.Lock()
_PREFLIGHT = {"done": False}
_FAIL = {"consecutive": 0}
FATAL_AFTER = 3          # consecutive groups lost to infrastructure errors before the run is stopped


def _executor(n: int) -> cf.ThreadPoolExecutor:
    global _EXEC
    with _EXEC_LOCK:
        if _EXEC is None:
            _EXEC = cf.ThreadPoolExecutor(max_workers=n, thread_name_prefix="r1-episode")
        return _EXEC


def _tokenizer(path: str):
    if path not in _TOK:
        from transformers import AutoTokenizer
        _TOK[path] = AutoTokenizer.from_pretrained(path)
    return _TOK[path]


def _stub(tok) -> dict[str, torch.Tensor]:
    ids = tok.encode("<|im_start|>system", add_special_tokens=False)[:2]
    ids = (ids + ids)[:2]
    return {"input_ids": torch.tensor(ids, dtype=torch.int32),
            "loss_mask": torch.zeros(2, dtype=torch.int32),
            "logprobs": torch.zeros(2, dtype=torch.float32),
            "versions": torch.full((2,), -1, dtype=torch.int32),
            "attention_mask": torch.ones(2, dtype=torch.bool),
            "rewards": torch.tensor(0.0, dtype=torch.float32)}


def group_tensors(results: list[tuple[dict, Any]], advs: list[float], tok) -> tuple[dict, list[bool]]:
    """k (record, trajectory) pairs + their advantages -> AReaL's padded tensor dict [k, L]; and which rows train."""
    rows, trained = [], []
    for (out, tr), a in zip(results, advs):
        sl = tr.trainable_slice() if (out.get("messages") and abs(a) > 1e-9) else None
        if sl is None:
            d = _stub(tok)
        else:
            n = len(sl["ids"])
            d = {"input_ids": torch.tensor(sl["ids"], dtype=torch.int32),
                 "loss_mask": torch.tensor(sl["loss"], dtype=torch.int32),
                 "logprobs": torch.tensor(sl["logp"], dtype=torch.float32),
                 "versions": torch.tensor(sl["ver"], dtype=torch.int32),
                 "attention_mask": torch.ones(n, dtype=torch.bool),
                 "rewards": torch.tensor(float(a), dtype=torch.float32)}
        trained.append(sl is not None)
        rows.append({k: v.unsqueeze(0) for k, v in d.items()})
    return concat_padded_tensors(rows), trained


class DeeprsGroupWorkflow(RolloutWorkflow):
    """workflow_kwargs (rl/async_pilot/train.py fills them from config.yaml):
    k, tokenizer_path, lora_name, temperature, top_p, max_model_len, max_new_tokens, c, timeout, require_code,
    wall_budget, log_dir, threads."""

    def __init__(self, tokenizer_path: str, lora_name: str, k: int = 6, temperature: float = 1.0, top_p: float = 0.95,
                 max_model_len: int = 16384, max_new_tokens: int = 8192, c: float = 0.5, timeout: int = 180,
                 require_code: bool = True, wall_budget: float = 600.0, log_dir: str = "", threads: int = 256):
        self.tok = _tokenizer(tokenizer_path)
        self.lora_name = lora_name
        self.k, self.temperature, self.top_p = int(k), float(temperature), float(top_p)
        self.max_model_len, self.max_new_tokens = int(max_model_len), int(max_new_tokens)
        self.c, self.timeout, self.require_code = float(c), int(timeout), bool(require_code)
        self.wall_budget = float(wall_budget)
        self.log_dir = Path(log_dir) if log_dir else None
        self.threads = int(threads)
        self.stop_ids = sorted({self.tok.convert_tokens_to_ids("<|im_end|>"),
                                self.tok.convert_tokens_to_ids("<|endoftext|>")})
        if not _PREFLIGHT["done"]:
            # rl/grpo_onpolicy.py main(): cells must run in the analysis venv, or rollouts score under a different Python
            from sandbox import preflight, sandbox_python
            if os.environ.get("GBP_ECHO_LAST") != "1":
                raise RuntimeError("GBP_ECHO_LAST=1 is part of R1's setup (notebook display); refusing to run without it")
            preflight(strict=True)
            print(f"[async_pilot] sandbox interpreter {sandbox_python()} (preflight OK), echo on, "
                  f"wall budget {self.wall_budget}s", flush=True)
            _PREFLIGHT["done"] = True

    # ------------------------------------------------------------------------------------------------------------
    def _gen_fn(self, engine: InferenceEngine, loop: asyncio.AbstractEventLoop):
        def gen(prompt_ids, max_new_tokens, temperature, top_p):
            async def _call():
                req = ModelRequest(
                    rid=uuid.uuid4().hex, input_ids=list(prompt_ids), tokenizer=self.tok,
                    gconfig=GenerationHyperparameters(
                        n_samples=1, max_new_tokens=int(max_new_tokens), max_tokens=self.max_model_len,
                        temperature=float(temperature), top_p=float(top_p), greedy=False,
                        stop_token_ids=list(self.stop_ids), lora_name=self.lora_name))
                resp = await engine.agenerate(req)
                return resp.output_tokens, resp.output_logprobs, resp.output_versions, resp.stop_reason
            return asyncio.run_coroutine_threadsafe(_call(), loop).result()
        return gen

    def _log(self, rows: list[dict]) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(self.log_dir / "rollouts.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def _fatal(self, msg: str) -> None:
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "FATAL").write_text(msg + "\n")
        print(f"[async_pilot] FATAL: {msg}", flush=True)
        os._exit(3)          # an engine that cannot generate must stop the run, not feed it zeros

    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]) -> dict | None:
        inst = Path(data["inst"])
        loop = asyncio.get_running_loop()
        gen = self._gen_fn(engine, loop)
        v0, t0, gid = engine.get_version(), time.time(), uuid.uuid4().hex[:12]
        job = functools.partial(run_episode, inst, gen, self.tok, temperature=self.temperature, timeout=self.timeout,
                                c=self.c, require_code=self.require_code, wall_budget=self.wall_budget,
                                max_model_len=self.max_model_len, max_tokens=self.max_new_tokens)
        ex = _executor(self.threads)
        try:
            results = await asyncio.gather(*[loop.run_in_executor(ex, contextvars.copy_context().run, job)
                                             for _ in range(self.k)])
        except Exception as exc:                                         # noqa: BLE001
            self._log([{"t": time.time(), "group": gid, "inst": inst.name, "workflow_error": repr(exc),
                        "trace": traceback.format_exc()[-2000:]}])
            raise
        self.last_results = results           # (record, trajectory) per row, in row order -- read by the CPU tests
        outs = [o for o, _ in results]
        infra = sum(is_infrastructure_error(o) for o in outs)
        if infra == len(outs):
            _FAIL["consecutive"] += 1
            msgs = sorted({str(o.get("error_msg", ""))[:200] for o in outs})
            self._log([{"t": time.time(), "group": gid, "inst": inst.name, "instrument_failure": msgs}])
            if _FAIL["consecutive"] >= FATAL_AFTER:
                self._fatal(f"{_FAIL['consecutive']} consecutive groups lost to infrastructure errors: {msgs}")
            raise RuntimeError(f"instrument failure on {inst.name}: every rollout errored ({msgs})")
        _FAIL["consecutive"] = 0
        rewards = [float(o["reward"]) for o in outs]
        advs = r1_advantages(rewards)
        batch, trained = group_tensors(results, advs, self.tok)
        v1, t1 = engine.get_version(), time.time()
        rows = []
        for (o, tr), a, tflag in zip(results, advs, trained):
            vs = [v for v, m in zip(tr.ver, tr.loss) if m]
            rows.append({"t": round(t1, 1), "group": gid, "inst": inst.name, "reward": o["reward"], "adv": round(a, 6),
                         "passed": bool(o.get("passed")), "attr": o.get("attribution"),
                         "fails_at_step": o.get("fails_at_step"), "answered": o.get("answered"),
                         "answer": o.get("answer"), "cells": o.get("n_cells"), "sec": o.get("seconds"),
                         "turns": tr.turns, "gen_tokens": tr.gen_tokens, "seq_len": len(tr.ids),
                         "head_version": min(vs) if vs else None, "tail_version": max(vs) if vs else None,
                         "version_at_submit": v0, "version_at_finish": v1, "group_sec": round(t1 - t0, 1),
                         "trained": tflag, "desync": tr.desync,
                         "overflow": OVERFLOW_TAG in str(o.get("error_msg", "")),
                         "error_msg": (str(o.get("error_msg"))[:200] if o.get("error_msg") else None)})
        self._log(rows)
        return batch
