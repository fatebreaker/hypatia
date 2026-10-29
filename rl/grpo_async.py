"""ASYNC-1: GROUP-STREAMING rollouts for the harness-v3 GRPO trainer (docs/RL_PLAN.md 2026-09-29 23:35 UTC, registered).

WHY. A synchronous step (rl/grpo_onpolicy.train) waits for its slowest episode: on HV3_119 (causal_rdd, LR 3e-6) the
median episode took 35 s, the slowest of the step's 48 took 410 s on average (up to the 600-s wall budget), so the step
was 497 s, of which 413 s were rollouts and 83 s the gradient step + publish -- the GPU idles through the tail.

WHAT (flag-gated: --async-inflight 0, the default, is rl/grpo_v3.py EXACTLY -- this file then only runs grpo_v3.install
and grpo_onpolicy.main(), the path grpo_v3's __main__ runs; grpo_onpolicy.train, Server and the rollout chain untouched):

    --async-inflight N     prompt GROUPS kept in flight (submitted, not yet trained on); N >= --prompts-per-step P.
                           The pilot: N = 16 for P = 8. Under the trainer's --smoke (P = 2) the same N:P ratio (4).
    --async-max-stale S    staleness cap (default 2): a group sampled from a policy more than S updates older than the
                           policy being updated is DROPPED (never waited for, counted, logged) instead of trained on.
    --async-is-cap C       truncation of the per-token importance weight (default 2.0; see CORRECTION).
    --async-correction     tis (default) | none (ablation: stale groups get the plain on-policy objective).
    --async-sample-s T     seconds between GPU-utilisation / vLLM-queue samples (default 10; 0 = off).

  STREAMING. The trainer draws prompts from its OWN seeded stream (rng.sample(insts, P) per batch, exactly the
  synchronous trainer's draws, in order) and submits them one group (k episodes) at a time. A group samples from the
  LATEST published policy at the moment its thread starts (the version is read at start, not at submission; the pool
  has 2N + 4 threads so a submitted group starts at once). In-flight = submitted and not yet consumed (running OR
  finished and waiting), so staleness is bounded by construction.
  REFILL happens right after each publish (and at start): the queue is topped up to N with groups on the NEW policy --
  never to more than the remaining steps can consume, so nothing is left running after the last step.
  FIFO BY SUBMISSION. Step t takes the P OLDEST-SUBMITTED groups, waiting for the head of the queue even when later
  groups are already complete: a slow analysis is waited for, never skipped, so the run does not select against slow
  (long, careful) episodes. The only exception is a head group that is ALREADY staler than the cap: it is dropped at
  once (it could never be trained on) and the next one is taken.
  STALENESS of a group trained at step t (which updates policy t-1 -> t) = (t - 1) - v, v = the version it sampled
  from. With refill-after-publish, N = P reproduces the synchronous trainer exactly (every group has staleness 0; the
  CPU parity test checks bit-identical weights); N = 2P gives staleness 1 for every group in steady state; N = 3P gives
  2; beyond that the cap drops groups. Step 1 (and the first step after a resume) always has staleness 0.
  RETIREMENT. A published adapter stays loaded in vLLM while a running group still samples from it (the synchronous
  Server.publish unloads the previous one at once; this file's AsyncServer does not) and is unloaded as soon as no
  running group needs it. Versions older than latest - S are unloaded even if a DROPPED group still runs on them (its
  results are discarded anyway), so at most S + 1 adapters are ever resident (<= pod_serve.sh's --max-loras 6, hence
  vLLM never evicts one and never re-reads a deleted live/pol<N> directory).

CORRECTION (decoupled-PPO / truncated importance sampling). The synchronous objective per episode is
    L = -mean_j min(r_j a, clip(r_j, 1-eps, 1+eps) a),   r_j = exp(logp_theta(y_j) - logp_theta(y_j).detach()) == 1,
one update per batch, so its gradient is -a mean_j grad logp_theta(y_j). For a group of staleness s >= 1 each token's
term is multiplied by the detached, truncated behaviour weight
    w_j = min( exp(logp_theta(y_j) - logp_beta(y_j)), C ),   C = --async-is-cap (2.0)
where beta is the adapter the group sampled from: L = -mean_j w_j min(r_j a, clip(r_j) a). This is Hilton et al.'s
decoupled PPO objective (pi_prox / pi_behav) x PPO-clip(pi_theta / pi_prox) with the proximal policy = the current one
(one update per batch, as today, so the inner ratio is 1 and the clip inactive), with the behaviour weight truncated as
in IMPALA / TIS (Yao et al. 2025, C = 2) / AReaL's behaviour-weight cap. Per TOKEN, not a sequence product: episodes are
thousands of tokens and a product of per-token ratios has unusable variance (PPO/GRPO's own convention). At s = 0 the
weight is not computed at all and the loss is the synchronous expression, bit for bit.
  WHERE logp_beta COMES FROM: the TRAINER recomputes it under the behaviour adapter (kept as an exact CPU copy of the
  LoRA parameters for the last S + 1 versions, swapped into the model in place for one no-grad forward, then restored
  bit-exactly), not from vLLM's sampled-token logprobs, because (1) the synchronous trainer already treats its own
  recomputation as the behaviour policy (ratio == 1; the vLLM-vs-HF numerical mismatch is uncorrected there) --
  correcting that mismatch only in the async arm would make HV3_151 differ from HV3_119 in two ways, not one;
  (2) the harness exchanges TEXT with vLLM and the trainer re-tokenises the conversation under RL_TEMPLATE; the ids vLLM
  sampled need not equal that re-tokenisation, and aligning them would mean editing harness/agents.py (pinned);
  (3) with it, s = 0 is exactly today's objective. Cost: one no-grad forward per stale episode (~+25% of the gradient
  step), 4.2 GB of host RAM for three fp32 LoRA copies of the 8B.

RESUME. The rolling resume/ point (every --resume-every steps, as today) is the only state: at a restart every in-flight
group is DISCARDED (never logged, never trained on); grpo_onpolicy.main replays `start` draws of the prompt stream, so
the resumed run re-draws, in order, the prompts that were in flight (with N <= 2P and no drops, exactly the prompts the
uninterrupted run would have trained on); the resumed groups sample from the resumed policy, so the first step after a
restart has staleness 0 and no pipelining (it is as slow as a synchronous step). harness_async.json pins the settings: a
resume with other settings, or of a directory trained synchronously, is refused.

LOGS. history.jsonl keeps every field of the synchronous trainer (t_rollout = the time the trainer WAITED for groups)
and adds: t_wait, t_behav, t_train, t_publish, t_wall; async_inflight (N), async_queue_at_take, async_ready_at_take
(groups already complete when the step started waiting), async_staleness_hist {"0": n, ...} over the trained groups,
async_stale_mean, async_dropped (this step), async_dropped_total, async_dropped_stale, async_submitted,
async_adapters_loaded, group_s [p50, max] (group latency), group_age_s [p50, max] (group start -> taken),
async_is_tokens, async_is_w_mean, async_is_raw_max (the largest UNtruncated weight), async_is_trunc_frac, async_is_kl
(mean logp_beta - logp_theta over stale tokens, a KL(beta || theta) estimate), async_behav_skipped, gpu_util_mean /
gpu_idle_frac (torch.cuda.utilization samples; idle = util < 5%), vllm_running_mean / vllm_idle_frac
(vllm:num_requests_running from /metrics), n_util_samples.
process_history.jsonl: grpo_v3's summary over the TRAINED groups (groups_alive_total etc. keep their meaning) + the
staleness histogram and drop counts. process.jsonl / rollouts.jsonl carry "step" = the step that TRAINED on the rollout
plus async_behav / async_stale / async_seq. async_dropped.jsonl: one line per dropped group.

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_async.py --async-inflight 16 \
        --process-lambda 0.3 <every rl/grpo_onpolicy.py argument>
Launch through experiments/harness_v3/run_v3async.sh (run dir -hv3pr<lambda>-async1-...). Tests (CPU, fakes):
experiments/harness_v3/tests/test_async.py.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_v3 as V3  # noqa: E402

ASYNC_VERSION = 1
DEFAULT_P = 8                 # rl/grpo_onpolicy.py's --prompts-per-step default
IDLE_UTIL = 5.0               # a GPU-utilisation sample below this (percent) counts as idle
_TLS = threading.local()      # per group thread: where grpo_v3's per-call process log is held
_SCORE_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------------------- configuration
def n_inflight(n: int, p: int, p_cfg: int) -> int:
    """Groups in flight when the trainer asks for p per step: n at the configured batch; under the trainer's --smoke
    (p = 2) the same n:p ratio (16 at 8 -> 4 at 2), never fewer than p."""
    return max(int(p), math.ceil(int(n) * int(p) / int(p_cfg)))


def parse_async_args(argv: list) -> tuple:
    # allow_abbrev=False: no trainer option may be read as an abbreviation of an --async-* option
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--async-inflight", type=int, default=0)
    ap.add_argument("--async-max-stale", type=int, default=2)
    ap.add_argument("--async-is-cap", type=float, default=2.0)
    ap.add_argument("--async-correction", choices=("tis", "none"), default="tis")
    ap.add_argument("--async-sample-s", type=float, default=10.0)
    return ap.parse_known_args(argv)


class AsyncConfig:
    def __init__(self, inflight: int, p_cfg: int, max_stale: int = 2, is_cap: float = 2.0, correction: str = "tis",
                 sample_s: float = 10.0):
        self.inflight, self.p_cfg, self.max_stale = int(inflight), int(p_cfg), int(max_stale)
        self.is_cap, self.correction, self.sample_s = float(is_cap), str(correction), float(sample_s)

    def record(self) -> dict:
        return {"version": ASYNC_VERSION, "inflight": self.inflight, "prompts_per_step": self.p_cfg,
                "max_stale": self.max_stale, "is_cap": self.is_cap, "correction": self.correction,
                "refill": "after publish (N = P is the synchronous trainer)", "select": "FIFO by submission",
                "behaviour_logprobs": "trainer recomputation under the behaviour adapter",
                "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


PINNED_KEYS = ("inflight", "prompts_per_step", "max_stale", "is_cap", "correction")


# --------------------------------------------------------------------------------------------------- the objective
def is_weights(new_det, behav, cap: float):
    """Per-token truncated importance weight w = min(exp(logp_theta - logp_beta), cap), detached, and its statistics."""
    import torch
    logr = (new_det - behav).float()
    raw = torch.exp(logr)
    w = torch.clamp(raw, max=cap).to(new_det.dtype).detach()
    st = {"n": int(raw.numel()), "w_sum": float(w.float().sum()), "w_max": float(raw.max()) if raw.numel() else 0.0,
          "trunc": int((raw > cap).sum()), "kl_sum": float((-logr).sum())}
    return w, st


def episode_objective(new, a: float, clip: float, w=None):
    """-mean_j [w_j] min(r_j a, clip(r_j) a), r_j = exp(new - new.detach()). w None (staleness 0): the synchronous
    trainer's expression exactly (rl/grpo_onpolicy.train); w given: the decoupled-PPO / TIS term (module docstring)."""
    import torch
    ratio = torch.exp(new - new.detach())
    obj = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    if w is not None:
        obj = obj * w
    return -obj.mean()


# --------------------------------------------------------------------------------------------------- policy copies
class PolicyBank:
    """Exact CPU copies of the trainable (LoRA) parameters per published version; load(v) copies one back in place (the
    optimizer keeps its references). load(current) after a behaviour pass restores the model bit-exactly, because the
    current parameters ARE the copy taken right after they were published."""

    def __init__(self, params):
        self.params = list(params)
        self.snaps: dict = {}

    def snapshot(self, version: int) -> None:
        self.snaps[int(version)] = [p.detach().to("cpu", copy=True) for p in self.params]

    def prune(self, keep_from: int) -> None:
        for v in [v for v in self.snaps if v < keep_from]:
            del self.snaps[v]

    def load(self, version: int) -> None:
        import torch
        with torch.no_grad():
            for p, s in zip(self.params, self.snaps[int(version)]):
                p.copy_(s)

    def equals(self, version: int) -> bool:
        import torch
        return all(torch.equal(p.detach().cpu(), s) for p, s in zip(self.params, self.snaps[int(version)]))


# --------------------------------------------------------------------------------------------------- the stream
class PromptStream:
    """The synchronous trainer's instance stream, one prompt at a time: batches of rng.sample(insts, P), in order."""

    def __init__(self, rng, insts, p: int):
        self.rng, self.insts, self.p = rng, insts, int(p)
        self.buf: collections.deque = collections.deque()
        self.batches = 0

    def next(self):
        if not self.buf:
            self.buf.extend(self.rng.sample(self.insts, self.p))
            self.batches += 1
        return self.buf.popleft()


class Group:
    __slots__ = ("seq", "inst", "version", "spec", "t_submit", "t_start", "t_done", "future", "outs", "proc",
                 "started", "stale", "error")

    def __init__(self, seq: int, inst):
        self.seq, self.inst = seq, inst
        self.version = self.spec = self.t_start = self.t_done = self.future = self.outs = self.proc = None
        self.t_submit = time.time()
        self.started, self.stale, self.error = False, None, None

    @property
    def done(self) -> bool:
        return self.future is not None and self.future.done()


class Scheduler:
    """Submitted-not-consumed groups in FIFO order (module docstring: STREAMING / FIFO / STALENESS).
    runner(group) fills group.outs / group.proc; it runs in a pool thread with group.version / group.spec set."""

    def __init__(self, runner, n: int, p: int, max_stale: int, stream: PromptStream, spec_of, latest: int,
                 before_submit=None, log=print):
        self.runner, self.n, self.p, self.max_stale = runner, int(n), int(p), int(max_stale)
        self.stream, self.spec_of, self.before_submit, self.log = stream, spec_of, before_submit, log
        self.q: collections.deque = collections.deque()
        self.dropped_running: list = []
        self.lock = threading.Lock()
        self.latest = int(latest)
        self.seq = 0
        self.dropped_total = 0
        self.ex = cf.ThreadPoolExecutor(max_workers=2 * self.n + 4, thread_name_prefix="async-group")

    def set_latest(self, version: int) -> None:
        with self.lock:
            self.latest = int(version)

    def _run(self, g: Group):
        with self.lock:                      # the LATEST published policy when the group starts
            g.version = self.latest
            g.spec = self.spec_of(g.version)
            g.t_start = time.time()
            g.started = True
        try:
            self.runner(g)
        finally:
            g.t_done = time.time()
        return g

    def submit(self) -> Group:
        g = Group(self.seq, self.stream.next())
        self.seq += 1
        g.future = self.ex.submit(self._run, g)
        self.q.append(g)
        return g

    def refill(self, need: int) -> int:
        """Top the queue up to min(N, need) groups (need = what the remaining steps can still consume)."""
        target = min(self.n, max(int(need), 0))
        if len(self.q) >= target:
            return 0
        if self.before_submit:
            self.before_submit()
        n = 0
        while len(self.q) < target:
            self.submit()
            n += 1
        return n

    def take(self, step: int) -> dict:
        """The P oldest-submitted groups whose staleness at this step is within the cap; waits for the head."""
        cur = int(step) - 1                  # the policy this step updates
        kept, dropped, t_wait = [], [], 0.0
        q0 = len(self.q)
        ready0 = sum(1 for g in self.q if g.done)
        while len(kept) < self.p:
            if not self.q:                   # drops emptied the queue: one more group on the latest policy
                self.submit()
            g = self.q[0]
            if g.started and cur - g.version > self.max_stale:
                self.q.popleft()             # can never be trained on: dropped without waiting
                g.stale = cur - g.version
                dropped.append(g)
                self.dropped_running.append(g)
                continue
            t = time.time()
            g.future.result()                # re-raises a failed group (the run stops, as a failed rollout step does)
            t_wait += time.time() - t
            self.q.popleft()
            g.stale = cur - g.version
            if g.stale > self.max_stale:     # started after the check above and turned out stale (cannot happen:
                dropped.append(g)            # a group starts on the latest policy); kept for safety
                continue
            kept.append(g)
        self.dropped_total += len(dropped)
        self.dropped_running = [g for g in self.dropped_running if not g.done]
        return {"kept": kept, "dropped": dropped, "t_wait": t_wait, "queue_at_take": q0, "ready_at_take": ready0}

    def versions_in_use(self, latest: int) -> set:
        """Versions a RUNNING group still samples from, that a future step could still train on (>= latest - S)."""
        live = [g for g in list(self.q) + self.dropped_running if g.started and not g.done]
        return {g.version for g in live if g.version >= latest - self.max_stale} | {int(latest)}

    def shutdown(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------------------------------- server + reward
def make_server_class(G):
    """grpo_onpolicy.Server whose publish keeps the previous adapters loaded (running groups still sample from them);
    the trainer unloads a version once nothing needs it (retire)."""
    base = G.Server

    class AsyncServer(base):
        instances: list = []

        def __init__(self, port: int):
            super().__init__(port)
            self.loaded: list = []
            AsyncServer.instances.append(self)

        def publish(self, name: str, path) -> None:
            G.post(f"{self.base}/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            if name not in self.loaded:
                self.loaded.append(name)
            self.live = name

        def unload(self, name: str) -> None:
            try:
                G.post(f"{self.base}/unload_lora_adapter", {"lora_name": name})
            except Exception as exc:                                    # noqa: BLE001
                print(f"  warn: unload {name}: {exc}", flush=True)
            if name in self.loaded:
                self.loaded.remove(name)

    AsyncServer.__name__ = AsyncServer.__qualname__ = "AsyncServer"
    AsyncServer.async_base = base
    return AsyncServer


def version_of(name: str) -> int | None:
    m = re.search(r"-pol(\d+)$", name or "")
    return int(m.group(1)) if m else None


def retire(servers, needed: set) -> int:
    """Unload every adapter no running group needs (never the live one); returns the number still loaded (max)."""
    n = 0
    for s in servers:
        for name in list(getattr(s, "loaded", [])):
            v = version_of(name)
            if name != s.live and v is not None and v not in needed:
                s.unload(name)
        n = max(n, len(getattr(s, "loaded", [])))
    return n


class AsyncReward(V3.ProcessReward):
    """grpo_v3's ProcessReward (re-classed after grpo_v3.install): its per-call log is HELD in the calling group thread
    (the trainer writes the step's log once it knows which groups it trained on); scoring is serialised."""

    def log(self, step: int, lines: list, stats: dict, n_groups: int) -> None:
        held = getattr(_TLS, "held", None)
        if held is None:
            return super().log(step, lines, stats, n_groups)
        held.update(step=step, lines=lines, stats=stats)

    def score_group(self, *a, **kw):
        with _SCORE_LOCK:
            return super().score_group(*a, **kw)


# --------------------------------------------------------------------------------------------------- utilisation
class UtilSampler(threading.Thread):
    """Every `period` s: torch.cuda.utilization() (the one GPU vLLM and the trainer share) and vLLM's running requests.
    Daemon, best effort: a failed probe records None."""

    def __init__(self, period: float, metrics_url: str | None, util_fn=None):
        super().__init__(daemon=True, name="async-util")
        self.period, self.url = float(period), metrics_url
        self.util_fn = util_fn
        self.samples: list = []
        self.stop_ev = threading.Event()

    def _util(self):
        try:
            if self.util_fn is not None:
                return float(self.util_fn())
            import torch
            return float(torch.cuda.utilization()) if torch.cuda.is_available() else None
        except Exception:                                               # noqa: BLE001
            return None

    def _running(self):
        if not self.url:
            return None
        try:
            with urllib.request.urlopen(self.url, timeout=5) as r:
                txt = r.read().decode("utf-8", "replace")
            vals = [float(line.rsplit(" ", 1)[1]) for line in txt.splitlines()
                    if line.startswith("vllm:num_requests_running")]
            return sum(vals) if vals else None
        except Exception:                                               # noqa: BLE001
            return None

    def run(self):
        while not self.stop_ev.wait(self.period):
            self.samples.append((time.time(), self._util(), self._running()))

    def stop(self):
        self.stop_ev.set()

    def stats(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        u = [x[1] for x in s if x[1] is not None]
        r = [x[2] for x in s if x[2] is not None]
        return {"gpu_util_mean": round(statistics.fmean(u), 1) if u else None,
                "gpu_idle_frac": round(sum(x < IDLE_UTIL for x in u) / len(u), 3) if u else None,
                "vllm_running_mean": round(statistics.fmean(r), 2) if r else None,
                "vllm_idle_frac": round(sum(x == 0 for x in r) / len(r), 3) if r else None,
                "n_util_samples": len(s)}


def _p(xs, qs):
    import numpy as np
    return [round(float(x), 1) for x in np.percentile(xs, qs)] if xs else None


# --------------------------------------------------------------------------------------------------- the trainer
def make_async_train(cfg: AsyncConfig, G):
    """A drop-in for grpo_onpolicy.train (same signature). The per-step block reproduces the synchronous one line for
    line (advantages, batch, encoding, clip, the record's fields); what differs is where the groups come from and the
    behaviour weight on stale groups."""

    def async_train(args, model, tok, opt, trainable, insts, rng, start, n_steps_fam, tag, publish, hist_f, out):
        import numpy as np
        import torch
        from rl.grpo_train import advantages, encode, seq_logprobs

        if args.kl or args.transfer_check or getattr(args, "library_for", None):
            raise SystemExit("rl/grpo_async.py: --kl / --transfer-check / --library are not built for ASYNC-1")
        P, k = args.prompts_per_step, args.k
        N = n_inflight(cfg.inflight, P, cfg.p_cfg)
        S = cfg.max_stale
        servers = [s for s in getattr(G.Server, "instances", [])]
        bank = PolicyBank(trainable)
        bank.snapshot(start)
        stream = PromptStream(rng, insts, P)

        def runner(g: Group):
            _TLS.held = {}
            try:
                res = G.rollouts([g.inst], g.spec, k, args.temperature, args.timeout, n_steps_fam, k, args.c,
                                 args.require_code, demo_for=None)
                g.proc = dict(_TLS.held) if _TLS.held else None
            finally:
                _TLS.held = None
            g.outs = res[g.inst.name]

        def before_submit():
            for hook in G.BEFORE_ROLLOUTS:
                hook()

        sched = Scheduler(runner, N, P, S, stream, lambda v: f"llm:{tag}-pol{v}:{args.render}", start,
                          before_submit=before_submit)
        metrics_url = f"http://127.0.0.1:{args.port}/metrics" if getattr(args, "port", None) else None
        sampler = None
        if cfg.sample_s > 0 and torch.cuda.is_available():
            dev = torch.cuda.current_device()          # resolved here, so the sampler thread only calls NVML
            sampler = UtilSampler(cfg.sample_s, metrics_url, util_fn=lambda: torch.cuda.utilization(dev))
        if sampler:
            sampler.start()
        pr = getattr(G.rollouts, "process_reward", None)
        print(f"ASYNC-1: {N} groups in flight (P={P}, k={k}; --workers is not used: each group runs its k episodes at "
              f"once), staleness cap {S}, correction {cfg.correction} (IS cap {cfg.is_cap}); starting at step {start}"
              + (" -- a RESUME: groups that were in flight when the previous process stopped are discarded (they were "
                 "never logged); the first step has staleness 0 and no pipelining" if start else ""), flush=True)
        sched.refill((args.steps - start) * P)
        smoke = {"stale_trained": 0, "restore_exact": None, "same_policy_dev": None, "behav_dev": None}
        try:
            for step in range(start + 1, args.steps + 1):
                t0 = time.time()
                took = sched.take(step)
                kept, dropped = took["kept"], took["dropped"]
                t_took = time.time()
                if dropped:
                    with open(out / "async_dropped.jsonl", "a") as f:
                        for g in dropped:
                            f.write(json.dumps({"step": step, "seq": g.seq, "inst": g.inst.name, "behav": g.version,
                                                "stale": g.stale, "done": g.done}) + "\n")
                    print(f"async step {step}: dropped {len(dropped)} group(s) staler than {S} "
                          f"({sorted({g.stale for g in dropped})})", flush=True)
                groups = [(g.inst.name, g.outs) for g in kept]
                allouts = [o for _, outs in groups for o in outs]
                errored = sum(str(o.get("attribution") or "").startswith("error") for o in allouts)
                cells = sum(int(o.get("n_cells") or 0) for o in allouts)
                answered = sum(bool(o.get("answered")) for o in allouts)
                if errored == len(allouts) or (cells == 0 and answered == 0):
                    kinds = sorted({str(o.get("attribution")) for o in allouts})[:5]
                    raise SystemExit(f"step {step}: {errored}/{len(allouts)} rollouts errored, "
                                     f"{cells} cells executed ({kinds}) -- instrument failure")
                batch, rewards, passes, dead, prefixes, stales = [], [], [], 0, [], []
                for g, (_, outs) in zip(kept, groups):
                    rs = [o["reward"] for o in outs]
                    rewards += rs
                    passes += [bool(o.get("passed")) for o in outs]
                    scored = [o for o in outs
                              if not (args.overflow_neutral and str(o.get("attribution") or "").startswith("error"))]
                    adv = advantages([o["reward"] for o in scored])
                    dead += all(abs(a) < 1e-12 for a in adv)
                    for o, a in zip(scored, adv):
                        if abs(a) > 1e-9 and o.get("messages"):
                            batch.append((o["messages"], a))
                            prefixes.append(int(o.get("n_prefix") or 0))
                            stales.append((g.stale, g.version))
                    if args.smoke and len(adv) > 1 and statistics.pstdev([o["reward"] for o in scored]) > 0:
                        assert abs(sum(adv)) < 1e-6, f"advantages sum to {sum(adv)}"

                loss_sum, nseq, mask_frac, nll_sum, gnorm, oom = 0.0, 0, 0.0, 0.0, 0.0, 0
                isd = {"n": 0, "w_sum": 0.0, "w_max": 0.0, "trunc": 0, "kl_sum": 0.0}
                behav_skipped, t_behav = 0, 0.0
                if args.smoke and smoke["restore_exact"] is None and cfg.correction == "tis":
                    stale_g = [g for g in kept if g.stale]
                    if stale_g:
                        smoke.update(smoke_behaviour_check(G, model, tok, args, bank, stale_g, step - 1))
                if batch:
                    opt.zero_grad(set_to_none=True)
                    enc = [G.encode_rl(tok, m, args.max_len) if args.render == "rl"
                           else G.encode_with(tok, m, args.max_len, G.MISTRAL) if args.render == "mtrain"
                           else G.encode_with(tok, m, args.max_len, G.LLAMA3) if args.render == "ltrain"
                           else encode(tok, m, args.max_len) for m, _ in batch]
                    if any(prefixes):
                        enc = [(ids, G.mask_prefix(tok, m, ids, mk, npre, args.max_len))
                               for (ids, mk), (m, _), npre in zip(enc, batch, prefixes)]
                    tot_tok = sum(len(ids) for ids, _ in enc)
                    mask_frac = sum(int(mk.sum()) for _, mk in enc) / max(tot_tok, 1)
                    # BEHAVIOUR log-probs of every stale episode, under its own adapter (no grad; restored after)
                    behav: dict = {}
                    need = {i for i, (s, _) in enumerate(stales) if s and cfg.correction == "tis"}
                    if need:
                        tb = time.time()
                        cur = step - 1
                        for v in sorted({stales[i][1] for i in need}):
                            bank.load(v)
                            try:
                                with torch.no_grad():
                                    for i in need:
                                        if stales[i][1] != v:
                                            continue
                                        ids_, mk_ = enc[i]
                                        m_ = mk_[1:].cuda().float()
                                        if m_.sum() < 1:
                                            continue
                                        try:
                                            behav[i] = G.lean_logprobs(model, ids_.unsqueeze(0).cuda(), m_).detach()
                                        except torch.cuda.OutOfMemoryError:
                                            torch.cuda.empty_cache()
                                            behav[i] = None
                            finally:
                                bank.load(cur)
                        t_behav = time.time() - tb
                    for i, ((ids, mk), (_, a)) in enumerate(zip(enc, batch)):
                        ids = ids.unsqueeze(0).cuda()
                        m = mk[1:].cuda().float()
                        if m.sum() < 1:
                            continue
                        w = None
                        if i in need:
                            if behav.get(i) is None:
                                behav_skipped += 1          # no behaviour log-probs (OOM): no unweighted stale term
                                continue
                        try:
                            new = G.lean_logprobs(model, ids, m)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache(); oom += 1
                            continue
                        if args.smoke and nseq == 0:       # the synchronous smoke's check, kept
                            with torch.no_grad():
                                ref = seq_logprobs(model, ids, torch.ones_like(ids)).squeeze(0)[m.bool()]
                            err = float((ref - new.detach()).abs().max())
                            assert err < 0.05, f"lean logprobs disagree with seq_logprobs by {err}"
                            print(f"  smoke: lean vs full logprobs max |diff| = {err:.2e}", flush=True)
                            del ref
                        if i in need:
                            w, st = is_weights(new.detach(), behav[i], cfg.is_cap)
                            isd["n"] += st["n"]; isd["w_sum"] += st["w_sum"]; isd["trunc"] += st["trunc"]
                            isd["kl_sum"] += st["kl_sum"]; isd["w_max"] = max(isd["w_max"], st["w_max"])
                            smoke["stale_trained"] += 1
                        loss = episode_objective(new, a, args.clip, w)
                        try:
                            (loss / len(batch)).backward()
                        except torch.cuda.OutOfMemoryError:
                            del new, loss
                            torch.cuda.empty_cache(); oom += 1
                            continue
                        loss_sum += float(loss.detach()); nseq += 1
                        nll_sum += float(-new.detach().mean())
                        del new, loss, w
                    behav.clear()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if args.smoke:
                        assert 0.02 < mask_frac < 0.98, f"assistant mask covers {mask_frac:.1%}"
                        assert torch.isfinite(torch.tensor(loss_sum)), "non-finite loss"
                t_pub0 = time.time()
                publish(step)                                  # groups submitted from now on sample the new policy
                sched.set_latest(step)
                bank.snapshot(step)
                bank.prune(step - S)
                loaded = retire(servers, sched.versions_in_use(step))
                submitted = sched.refill((args.steps - step) * P)
                t_pub = time.time() - t_pub0
                hist = collections.Counter(str(g.stale) for g in kept)
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
                       "rollout_s": [round(float(x), 1) for x in
                                     np.percentile([o.get("seconds", 0) for o in allouts], [50, 90, 100])],
                       "t_rollout": round(took["t_wait"], 1),
                       "t_step": round(time.time() - t0, 1),
                       # ---- ASYNC-1 (module docstring: LOGS)
                       "async": 1, "t_wait": round(took["t_wait"], 1), "t_behav": round(t_behav, 1),
                       "t_train": round(t_pub0 - t_took, 1), "t_publish": round(t_pub, 1), "t_wall": round(time.time(), 1),
                       "async_inflight": N, "async_queue_at_take": took["queue_at_take"],
                       "async_ready_at_take": took["ready_at_take"],
                       "async_staleness_hist": dict(sorted(hist.items(), key=lambda x: int(x[0]))),
                       "async_stale_mean": round(statistics.fmean([g.stale for g in kept]), 3),
                       "async_dropped": len(dropped), "async_dropped_total": sched.dropped_total,
                       "async_dropped_stale": sorted(g.stale for g in dropped),
                       "async_submitted": submitted, "async_adapters_loaded": loaded,
                       "group_s": _p([g.t_done - g.t_start for g in kept], [50, 100]),
                       "group_age_s": _p([t_took - g.t_start for g in kept], [50, 100]),
                       "async_is_tokens": isd["n"],
                       "async_is_w_mean": round(isd["w_sum"] / isd["n"], 5) if isd["n"] else None,
                       "async_is_raw_max": round(isd["w_max"], 4) if isd["n"] else None,
                       "async_is_trunc_frac": round(isd["trunc"] / isd["n"], 5) if isd["n"] else None,
                       "async_is_kl": round(isd["kl_sum"] / isd["n"], 6) if isd["n"] else None,
                       "async_behav_skipped": behav_skipped,
                       **(sampler.stats(t0, time.time()) if sampler else
                          {"gpu_util_mean": None, "gpu_idle_frac": None, "vllm_running_mean": None,
                           "vllm_idle_frac": None, "n_util_samples": 0})}
                with open(hist_f, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                with open(out / "rollouts.jsonl", "a") as f:
                    for g, (inst, outs) in zip(kept, groups):
                        for o in outs:
                            f.write(json.dumps({"step": step, "inst": inst, "reward": o["reward"],
                                                "passed": bool(o.get("passed")),
                                                "attr": o.get("attribution"),
                                                "fails_at_step": o.get("fails_at_step"),
                                                "answered": o.get("answered"), "answer": o.get("answer"),
                                                "cells": o.get("n_cells"), "sec": o.get("seconds"),
                                                "async_behav": g.version, "async_stale": g.stale,
                                                "async_seq": g.seq}, default=str) + "\n")
                if pr is not None:
                    log_process(pr, step, kept, rec)
                print(json.dumps(rec), flush=True)
                if args.ckpt_every and step % args.ckpt_every == 0 or step == args.steps:
                    d = out / f"step{step}"
                    model.save_pretrained(str(d))
                    print(f"  checkpoint -> {d}", flush=True)
                if args.resume_every and step % args.resume_every == 0 or step == args.steps:
                    import shutil
                    tmp, res = out / "resume.tmp", out / "resume"
                    shutil.rmtree(tmp, ignore_errors=True)
                    model.save_pretrained(str(tmp))
                    torch.save(opt.state_dict(), tmp / "optimizer.pt")
                    (tmp / "state.json").write_text(json.dumps({"step": step, "async": cfg.record(),
                                                                "note": "in-flight groups are discarded at a restart"}))
                    shutil.rmtree(res, ignore_errors=True)
                    tmp.rename(res)
            if args.smoke:
                if cfg.correction == "tis":
                    assert smoke["restore_exact"] is True, \
                        "async smoke: no stale group reached the trainer, or the behaviour swap was not exact"
                print(f"async smoke: OK (behaviour swap restored the policy bit-exactly: {smoke['restore_exact']}; "
                      f"same-policy recomputation max |dlogp| = {smoke['same_policy_dev']}; behaviour vs current "
                      f"max |dlogp| = {smoke.get('behav_dev')}; stale episodes trained with IS weights: "
                      f"{smoke['stale_trained']})", flush=True)
        finally:
            sched.shutdown()
            if sampler:
                sampler.stop()
            for s in servers:                  # main() unloads each server's live adapter; the older ones here
                for name in list(getattr(s, "loaded", [])):
                    if name != s.live:
                        s.unload(name)

    async_train.async_config = cfg
    return async_train


def smoke_behaviour_check(G, model, tok, args, bank, stale_groups: list, cur: int) -> dict:
    """--smoke only, on the first step with a stale group (whatever its advantages): recompute one of its episodes under
    the CURRENT policy twice (must agree: the no-grad pass is deterministic enough for w == 1 at staleness 0) and under
    its BEHAVIOUR adapter, then check the swap restored the current policy bit-exactly."""
    import torch
    o = next((o for g in stale_groups for o in g.outs if o.get("messages")), None)
    if o is None:
        raise SystemExit("async smoke: the stale groups have no episode with messages -- cannot check the behaviour pass")
    v = next(g.version for g in stale_groups if any(x is o for x in g.outs))
    ids, mk = G.encode_rl(tok, o["messages"], args.max_len) if args.render == "rl" else (None, None)
    if ids is None:
        return {"restore_exact": bank.equals(cur), "same_policy_dev": None, "behav_dev": None}
    ids, m = ids.unsqueeze(0).cuda(), mk[1:].cuda().float()
    with torch.no_grad():
        a1 = G.lean_logprobs(model, ids, m)
        a2 = G.lean_logprobs(model, ids, m)
        bank.load(v)
        try:
            b = G.lean_logprobs(model, ids, m)
        finally:
            bank.load(cur)
    exact = bank.equals(cur)
    same = float((a1 - a2).abs().max()) if a1.numel() else 0.0
    dev = float((a1 - b).abs().max()) if a1.numel() else 0.0
    assert exact, "async smoke: the behaviour swap did not restore the current policy bit-exactly"
    assert same < 0.05, f"async smoke: two no-grad passes under the same policy differ by {same}"
    assert torch.isfinite(b).all(), "async smoke: non-finite behaviour log-probs"
    print(f"  async smoke: behaviour check on a staleness-{cur - v} episode ({int(m.sum())} tokens): same-policy "
          f"max |dlogp| {same:.2e}, behaviour vs current max |dlogp| {dev:.2e}, restore exact {exact}", flush=True)
    return {"restore_exact": exact, "same_policy_dev": same, "behav_dev": dev}


def log_process(pr, step: int, kept: list, rec: dict) -> dict | None:
    """grpo_v3's per-step process log over the TRAINED groups (their held per-group logs), with the step rewritten."""
    lines, stats = [], {"answer_alive": 0, "total_alive": 0}
    for g in kept:
        h = g.proc or {}
        for x in h.get("lines") or []:
            lines.append({**x, "step": step, "async_behav": g.version, "async_stale": g.stale, "async_seq": g.seq})
        st = h.get("stats") or {}
        stats["answer_alive"] += int(st.get("answer_alive", 0))
        stats["total_alive"] += int(st.get("total_alive", 0))
    summ = pr.summary(step, lines, stats, len(kept))
    summ.update({k: rec[k] for k in ("async_staleness_hist", "async_stale_mean", "async_dropped", "async_dropped_total",
                                     "async_inflight")})
    print("process " + json.dumps(summ), flush=True)
    if pr.out is not None:
        with pr.lock:
            with open(pr.out / "process.jsonl", "a") as f:
                for x in lines:
                    f.write(json.dumps(x, default=str) + "\n")
            with open(pr.out / "process_history.jsonl", "a") as f:
                f.write(json.dumps(summ) + "\n")
    return summ


# --------------------------------------------------------------------------------------------------- install
def _trainer_args(rest: list) -> argparse.Namespace:
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    pa.add_argument("--prompts-per-step", type=int, default=DEFAULT_P)
    pa.add_argument("--port2", type=int, default=0)
    pa.add_argument("--port2-sleep", action="store_true")
    pa.add_argument("--kl", type=float, default=0.0)
    g, _ = pa.parse_known_args(rest)
    return g


def install(grpo_mod, dispersion_mod, v3args, aargs, rest: list):
    """Flag off: grpo_v3.install only (+ a refusal to continue an ASYNC-1 directory synchronously). Flag on: grpo_v3's
    install, then the async trainer, server and held process log. Returns (process reward, AsyncConfig | None)."""
    g = _trainer_args(rest)
    out = Path(g.out) if g.out else None
    cfgf = out / "harness_async.json" if out is not None else None
    if not aargs.async_inflight:
        if cfgf is not None and cfgf.exists():
            raise SystemExit(f"rl/grpo_async.py: {out} was trained with ASYNC-1 ({cfgf.name}) -- refusing to resume it "
                             "synchronously (pass its --async-* settings)")
        return V3.install(grpo_mod, dispersion_mod, v3args, rest), None
    cfg = AsyncConfig(aargs.async_inflight, g.prompts_per_step, aargs.async_max_stale, aargs.async_is_cap,
                      aargs.async_correction, aargs.async_sample_s)
    if cfg.inflight < cfg.p_cfg:
        raise SystemExit(f"rl/grpo_async.py: --async-inflight {cfg.inflight} must be >= --prompts-per-step {cfg.p_cfg}")
    if cfg.max_stale < 0 or cfg.is_cap < 1.0:
        raise SystemExit("rl/grpo_async.py: --async-max-stale must be >= 0 and --async-is-cap >= 1")
    if g.port2 or g.port2_sleep:
        raise SystemExit("rl/grpo_async.py: --port2 / --port2-sleep are not built for ASYNC-1 (a sleeping server would "
                         "strand in-flight groups); one server")
    if g.kl:
        raise SystemExit("rl/grpo_async.py: --kl is not built for ASYNC-1")
    rec = cfg.record()
    if cfgf is not None:
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if any(old.get(k) != rec[k] for k in PINNED_KEYS):
                raise SystemExit(f"rl/grpo_async.py: {out} was trained with {({k: old.get(k) for k in PINNED_KEYS})} "
                                 f"-- refusing to resume it as {({k: rec[k] for k in PINNED_KEYS})}")
        elif (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_async.py: {out} has steps trained WITHOUT ASYNC-1 (no harness_async.json) -- "
                             "refusing to resume it with --async-inflight")
    pr = V3.install(grpo_mod, dispersion_mod, v3args, rest)
    if type(pr) is not V3.ProcessReward:
        raise SystemExit(f"rl/grpo_async.py: grpo_v3.install returned {type(pr).__name__}, not grpo_v3's ProcessReward")
    if cfgf is not None and not cfgf.exists():
        out.mkdir(parents=True, exist_ok=True)
        tmp = cfgf.with_name(cfgf.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(cfgf)
    pr.__class__ = AsyncReward
    grpo_mod.Server = make_server_class(grpo_mod)
    grpo_mod.train = make_async_train(cfg, grpo_mod)
    print(f"ASYNC-1 (RL_PLAN 2026-09-29 23:35): {cfg.inflight} groups in flight for {cfg.p_cfg} per step, FIFO by "
          f"submission, staleness cap {cfg.max_stale}, correction {cfg.correction} (truncated IS, cap {cfg.is_cap}; "
          f"behaviour log-probs recomputed under the behaviour adapter)", flush=True)
    return pr, cfg


def run(argv: list) -> int:
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(argv)
    aargs, rest = parse_async_args(rest)
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, aargs, rest)
    return grpo_onpolicy.main()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
