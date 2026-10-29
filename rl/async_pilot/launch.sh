#!/bin/bash
# Async-RL pilot (docs/ASYNC_PILOT.md): R1 on AReaL v2.0.0, one vLLM GPU + one trainer GPU of a double A100-80G pod.
#
#   rl/async_pilot/launch.sh SEED [STEPS]                 inside a pod (scripts/pod_exec.sh), both GPUs visible
#   DRY_RUN=1 rl/async_pilot/launch.sh SEED               anywhere (login node): checks and prints the plan, no GPU
#
# Env knobs: STALENESS (rollout.max_head_offpolicyness, default 2), GPUS (default "1,0": AReaL hands the FIRST listed
# GPU to the trainer, which it creates first, and the second to vLLM -> trainer on GPU 1, vLLM on GPU 0),
# MODEL (default /tmp/deeprs_models/Qwen3-8B, staged by the pod queue), OUT (run dir), AREAL_ROOT (AReaL's fileroot:
# logs + the per-version LoRA copies vLLM loads; default pod-private /tmp), CONCURRENCY (groups in flight, default 16),
# EXTRA (more AReaL overrides, space-separated key=value).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
SEED="${1:?SEED}"; STEPS="${2:-150}"
# shellcheck source=/dev/null
source scripts/env.sh >/dev/null || exit 1
STALE="${STALENESS:-2}"
OUT="${OUT:-$DEEP_DATA_ROOT/ckpt/rl-async-rlmix6-base-echo-rc-wb600-lr1e-5-st$STALE-s$SEED}"
MODEL="${MODEL:-/tmp/deeprs_models/Qwen3-8B}"
AREAL_ROOT="${AREAL_ROOT:-/tmp/deeprs_areal_s$SEED}"
PYA="$P/venv_async/bin/python"
[ -x "$PYA" ] || { echo "[async_pilot] FATAL: $PYA missing (venv_async -> /scratch4/.../async_pilot/venv_async)" >&2; exit 2; }

# The harness environment R1 runs in (rl/grpo_onpolicy.py main + the pod item's env):
export GBP_ECHO_LAST=1                                   # notebook display (RL_PLAN s.1)
export GBP_WALL_BUDGET=600                               # straggler cap (RL_PLAN 2026-09-24); also passed explicitly
export GBP_SANDBOX_PYTHON="$P/venv/bin/python"           # cells run in the analysis venv, as every evaluation
export PYTHONPATH="$P:$P/harness${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# AReaL starts the vLLM server as a literal `python3 -m areal.engine.vllm_ext.areal_vllm_server` (cli_args.get_py_cmd):
# venv_async must come first on PATH, or pod_exec.sh's venv_vllm (vLLM 0.28, no AReaL) would be picked up.
export PATH="$P/venv_async/bin:$PATH"
export DEEPRS_ASYNC_OUT="$OUT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc CXX=/usr/bin/g++                  # FINDINGS D2: Triton JITs with $CC
export VLLM_USE_FLASHINFER_SAMPLER=0                     # as pod_serve.sh
export AREAL_CACHE_DIR="$AREAL_ROOT/cache"               # AReaL's triton/vllm/torch kernel caches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPUS:-1,0}"
# No resume (AReaL recover is off): a restarted pilot must not append to a dead run's logs. Move it aside instead.
if [ "${DRY_RUN:-0}" != 1 ] && [ -s "$OUT/rollouts.jsonl" ]; then
  mv "$OUT" "$OUT.aborted-$(date +%Y%m%d-%H%M%S)" && echo "[async_pilot] previous attempt moved aside: $OUT.aborted-*"
fi
mkdir -p "$AREAL_ROOT" "$OUT"

N_MBS=$(( 8 * 6 ))                                       # rows per update: one sequence per micro-batch
ARGS=(--out "$OUT" --k 6 --c 0.5 --timeout 180 --wall-budget 600 --ckpt-steps 25,50,75,100,125,150)
[ "${DRY_RUN:-0}" = 1 ] && ARGS+=(--dry-run)
AREAL=(--config "$P/rl/async_pilot/config.yaml"
       seed="$SEED" trial_name="s$SEED" total_train_steps="$STEPS"
       actor.path="$MODEL"
       rollout.max_head_offpolicyness="$STALE"
       rollout.max_concurrent_rollouts="${CONCURRENCY:-16}"
       actor.mb_spec.n_mbs="$N_MBS"
       cluster.fileroot="$AREAL_ROOT/experiments"
       cluster.name_resolve.nfs_record_root="$AREAL_ROOT/name_resolve")
[ -n "${EXTRA:-}" ] && read -r -a _x <<< "$EXTRA" && AREAL+=("${_x[@]}")

echo "[async_pilot] $(date -Is) seed=$SEED steps=$STEPS staleness=$STALE out=$OUT model=$MODEL gpus=$CUDA_VISIBLE_DEVICES"
echo "[async_pilot] areal root $AREAL_ROOT ($(df -Ph "$AREAL_ROOT" | awk 'NR==2{print $4}') free)"
exec "$PYA" -m rl.async_pilot.train "${ARGS[@]}" -- "${AREAL[@]}"
