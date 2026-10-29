#!/bin/bash
# Serve Qwen3-8B (+ optional LoRA adapters) with vLLM in the FOREGROUND, inside a
# held SLURM pod step. The counterpart of serve_local.sh for pods: that script
# backgrounds vLLM and exits, which inside `srun --overlap` ends the step and the
# step's cgroup takes the server down with it. Here the server IS the step.
#
#   srun --jobid=$J --overlap -n1 -c3 scripts/pod_serve.sh PORT [name=/adapter/dir ...]
#
# Pods on one node share a network namespace, so give each server its own PORT.
# GPU_UTIL (default 0.50) leaves room on a 141 GB H200 for an SFT job beside it.
# Carries serve_local.sh's D1-D3 fixes (venv_vllm on PATH, real compiler).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:?usage: pod_serve.sh PORT [name=adapter_dir ...]}"; shift
# shellcheck source=/dev/null
source "$P/scripts/env.sh" >/dev/null
MODEL_DIR="${POD_MODEL_DIR:-$(ws_find deeprs-bench | grep '^/' | head -1)/models/Qwen3-8B}"
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: no model at $MODEL_DIR" >&2; exit 1; }

LMOD="${LMOD_CMD:-/usr/share/lmod/lmod/libexec/lmod}"
[ -x "$LMOD" ] && eval "$("$LMOD" bash load cuda/13.1 2>/dev/null)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=WARNING
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-x}_$PORT"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID:-x}_$PORT"
# 10-09: vLLM's own cache (VLLM_CACHE_ROOT, default ~/.cache/vllm: compile + flashinfer autotune caches) filled the 100 GB
# /home quota with the pip cache and a 27B server failed to start on the H200 ("Disk quota exceeded"); keep it on /scratch4.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch4/workspace/hieutran_umass_edu-deeprs-bench/vllm_cache}"
mkdir -p "$VLLM_CACHE_ROOT"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PATH="$P/venv_vllm/bin:$PATH"            # D3: flashinfer's JIT needs ninja

# LoRA is always on, and adapters can be added to the RUNNING server (so a newly
# trained adapter costs a POST, not a 2-3 minute restart):
# SERVE_HOST (default 127.0.0.1) -- set to 0.0.0.0 to accept workers on OTHER NODES. Added 2026-09-21 after
# measuring that the agent loop is two halves, model call (GPU) then sandbox execution (CPU), and that a pod can
# sit at 0 GPU-busy samples out of 105 over 15 minutes while legitimately working. Inter-node HTTP between Unity
# compute nodes is open (tested: gpu023 -> gpu024, HTTP 200 in 0.54s), so one GPU pod can serve sandbox workers
# running on CPU-only pods instead of idling between its own eight. The DEFAULT IS UNCHANGED, so every item now in
# flight behaves exactly as before; only a caller that sets SERVE_HOST is affected.
#   curl -s localhost:PORT/v1/load_lora_adapter -H 'Content-Type: application/json' \
#        -d '{"lora_name":"x","lora_path":"/abs/adapter/dir"}'
# Every reported adapter is rank 128; vLLM's default max rank (16) would refuse them.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
LORA_ARGS=(--enable-lora --max-lora-rank 128 --max-loras "${MAX_LORAS:-6}")
# NO_LORA=1 (2026-09-30): serve a reference model with LoRA OFF (untrained frontier-class models, H-FRONTIER-OPEN --
# FP8 / MoE / MXFP4 checkpoints need no adapters and some LoRA paths refuse them). Unset = LoRA on, as always.
[ "${NO_LORA:-0}" = 1 ] && LORA_ARGS=()
if [ "$#" -gt 0 ]; then
  for spec in "$@"; do
    [ -f "${spec#*=}/adapter_model.safetensors" ] || { echo "FATAL: no adapter in ${spec#*=}" >&2; exit 1; }
  done
  LORA_ARGS+=(--lora-modules "$@")
fi

echo "[pod_serve] $(hostname) job=${SLURM_JOB_ID:-?} gpu=$CUDA_VISIBLE_DEVICES port=$PORT model=$MODEL_DIR"
printf '[pod_serve] lora %s\n' "$@"
exec "$P/venv_vllm/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "${SERVED_NAME:-Qwen3-8B}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_LEN:-16384}" \
    --max-num-seqs "${MAX_SEQS:-64}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.50}" \
    --host "${SERVE_HOST:-127.0.0.1}" --port "$PORT" \
    "${LORA_ARGS[@]}" ${EXTRA_SERVE_ARGS:-}   # e.g. Mistral: --tokenizer-mode hf --config-format hf
