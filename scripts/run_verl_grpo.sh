#!/bin/bash
# GRPO on cis-MVMR with verl, evaluated on HELD-OUT scRNA, across GPUs 4 and 5.
#
# WHY verl RATHER THAN rl/grpo_train.py
# --------------------------------------
# The hand-rolled pilot had no weight synchronisation: the trainer updated its
# LoRA while the rollout server kept serving the frozen SFT adapter, so every
# step sampled from the same unchanging policy and the reward curve was flat BY
# CONSTRUCTION. Nothing surfaced it -- live_seqs 48/48, mask 52%, checkpoints
# written, 29 clean steps -- and the flat curve was briefly reported as "no
# learning at this scale", which was wrong.
#
# Rollout/update synchronisation is the central piece of RL infrastructure, and
# verl owns it. That is the whole reason to be here.
#
# THE EXPERIMENT
# --------------
#   train  cis-MVMR          (800 instances)
#   eval   scRNA, HELD OUT   (32, never trained on)
# Standing baselines to beat, both from RESULTS:
#   §15  SFT transfer to this cell        0.022
#   §16  ceiling when scRNA is in corpus  0.606
#
# Both GPUs train; verl colocates the rollout engine with the actor, so there
# is no separate serving GPU to keep in sync by hand. The sync itself is
# engine_workers.update_weights(), which pushes per-tensor LoRA params plus the
# peft_config into the live vLLM engine every step -- the exact thing the
# hand-rolled loop was missing.
#
# HOW THE ADAPTER REACHES THE ROLLOUT: merged, not as a vLLM LoRA slot
# ---------------------------------------------------------------------
# verl's default is to keep vLLM's LoRA slots live and push only the ~175 MB of
# adapter tensors each step. On this stack (verl + vLLM 0.26) that path dies in
# vllm/lora/layers/column_parallel_linear.py::set_lora with
#   IndexError: tuple index out of range   (lora_a[i] on a 2-D tensor)
# -- vLLM indexes lora_a per output slice, so it wants a list of per-slice
# tensors for the packed projections, and gets one stacked tensor.
#
# model.lora.merge=True takes the other route verl already supports: merge the
# adapter into the base weights on the actor and sync full weights. It also
# makes lora_as_adapter False, so the vLLM engine is built with no LoRA slots
# at all and the broken code never runs. The cost is ~16 GB per weight sync
# instead of 175 MB, which is seconds against rollout steps measured in
# minutes. Training is unchanged -- only the LoRA parameters get gradients
# either way.
#
# THROUGHPUT: token budgets, not sequence counts
# --------------------------------------------------
# These are H200 NVL (141 GB), and the first configuration used 72 GB of it.
# Where a step actually went, at micro-batch 1:
#   update_actor 116s | old_log_prob 46s | gen 45s | update_weights 27s
# so the two forward/backward phases were 69% of the step and both were
# processing ONE sequence at a time on a half-empty GPU.
#
# The lever is use_dynamic_bsz: verl then fills each micro-batch up to a token
# budget instead of a fixed sequence count. That matters here because episode
# length is not fixed -- responses run from ~1k to the 8k cap, so any fixed
# count is either unsafe for the long tail or wasteful for everything else.
# Budgets are in tokens per GPU and are the memory knob; they do not change the
# math. Micro-batching is gradient accumulation, so the update is identical
# whatever the split -- these settings buy wall-clock, not a different result.
#
# The ceiling is the vocabulary, not the model. Qwen3's vocab is 151936, so a
# micro-batch's logits are tokens x 151936: bf16 to materialise them, then
# entropy_from_logits upcasts to fp32 for both logsumexp and softmax. The first
# attempt at a 32768-token budget died in exactly that call, asking for 34 GiB
# on top of 121 GiB already held -- and it was the forward-only log-prob pass,
# which stores no activations at all. The logits WERE the memory.
#
# use_fused_kernels replaces that with a chunked linear-cross-entropy path that
# never holds the full logit tensor (impl_backend "torch" -- the triton one is
# unavailable here for the same reason flash-attn is). It is what makes a
# multi-sequence micro-batch fit on a 151936-vocab model.
#
# Two things to keep in mind when retuning these budgets:
#   - The budget counts REAL tokens, but with use_remove_padding=False the
#     forward runs on the padded rectangle, num_seqs x 12288. Roughly 1.6x the
#     budget at current episode lengths -- so the number is a floor on cost,
#     not the cost.
#   - vLLM's share is NOT what limits training. It sleeps during the update
#     (measured: 1.89 GiB resident while the actor held 121 GiB), so
#     gpu_memory_utilization buys KV cache for the rollout without taxing the
#     phases that dominate the step.
#
# RAY WORKERS DO NOT INHERIT THE DRIVER'S ENVIRONMENT.
# The head is started separately (see below), so workers inherit the CLUSTER's
# env, not this script's. Anything the agent loop reads from os.environ has to
# be passed through ray_init.runtime_env.env_vars or it silently is not there --
# which is how the first attempt at rollout tracing produced zero trace files.
#
# EPISODE BUDGETS MUST MATCH THE EVAL HARNESS
# ----------------------------------------------
# MAX_RESP caps CUMULATIVE tokens across the whole episode, not per turn, and
# the agent loop stops once it is reached. harness/agents.py has no such cap,
# so a value that truncates episodes makes training and evaluation score
# different things. At 12288 on scRNA, 40% of episodes were cut off before
# answering (response_length/clip_ratio 0.396) and step-1 reward read 0.002
# against the 0.298 the same adapter scores under the harness. Likewise
# max_turns had to go 12 -> 25 to match GBP_MAX_TURNS.
#
# cis-MVMR never surfaced either problem: it finishes in 4-6 turns and ~4k
# tokens, comfortably inside both caps. A second family is what exposed them.
#
# STARTING POINT: the SFT adapter, not a fresh LoRA
# -------------------------------------------------
# model.lora_adapter_path makes verl PeftModel.from_pretrained(..., is_trainable=True)
# instead of get_peft_model() on a random init. Without it this would be RL from
# the base model, which is a different (and much weaker) experiment: the SFT
# policy is what §15's 0.022 was measured on, so that is what RL has to improve.
# When the path is set, the adapter's own r/alpha/target_modules win; the flags
# below are set to match it (r=32, alpha=64, the 7 attention+MLP projections)
# so that rollout-side vLLM sizes its LoRA slots identically.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P"

# ONE RUN PER GPU SET. Two invocations on the SAME GPUs both reserve them; the
# loser dies with "Total available GPUs 0 is less than total desired GPUs 2"
# after several minutes of model loading, and if they share a log file the
# survivor's output is interleaved with the corpse's. Fail fast instead.
#
# The lock is keyed by GPUS rather than global, because running independent
# SEEDS concurrently -- one per GPU -- is the cheapest way to answer the
# single-seed caveat, and a global lock forbids exactly that. Disjointness is
# the caller's responsibility; the key only stops an identical re-launch.
_LOCK_KEY="$(echo "${GPUS:-4,5}" | tr -c 'a-zA-Z0-9' '_')"
exec 9>"${TMPDIR:-/tmp}/deeprs-verl-grpo-${_LOCK_KEY}.lock"
if ! flock -n 9; then
  echo "another run_verl_grpo.sh already holds GPUs ${GPUS:-4,5} -- refusing" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$P/scripts/env.sh"      # resolves DEEP_DATA_ROOT the one supported way
# VERL_DATA lets a second family train from its own parquets. Without it every
# run reads $DEEP_DATA_ROOT/verl_data/{train,val}.parquet, so setting up a
# rare_burden run means overwriting the scRNA pair that every published number
# in RESULTS §24 was trained on -- and the overwrite is silent.
DATA="${VERL_DATA:-$DEEP_DATA_ROOT/verl_data}"

# One checkpoint directory per experiment. verl's resume_mode defaults to
# "auto", so a run whose default_local_dir already holds a checkpoint silently
# continues from it -- a 2-step smoke run at batch 4x4 left global_step_2
# behind and the following 40-step run at 8x6 resumed straight into it,
# splicing two configurations into one trajectory and throwing away the step-0
# baseline. Give smoke runs their own EXP so they cannot contaminate a real
# one, and say out loud when a resume is about to happen.
EXP="${EXP:-cismvmr_to_scrna}"
CKPT="$DEEP_DATA_ROOT/ckpt/verl-grpo-$EXP"
# TWO WAYS TO START FROM THE SFT POLICY, and they differ in one thing that
# matters only once a KL term is switched on.
#
#   MERGED=0 (default): base model + the SFT adapter loaded into the trainable
#     LoRA. Simple, but verl sets ref_in_actor for LoRA runs, so its reference
#     policy is "the actor with the adapter disabled" = BASE Qwen3-8B. A KL
#     penalty would pull toward the base model and undo SFT.
#
#   MERGED=1: SFT baked into the weights (scripts/merge_lora.py) and a fresh
#     zero-init LoRA on top. Initial policy is identical -- merged + zero LoRA
#     is exactly the SFT policy -- but now "adapter disabled" is the SFT policy
#     too, so KL pulls toward the right reference.
#
# Use MERGED=1 whenever KL_COEF is non-zero; the check below enforces it.
SFT_ADAPTER="${SFT_ADAPTER:-$DEEP_DATA_ROOT/ckpt/qwen3-8b-cismvmr-lora/final}"
MERGED_MODEL="$DEEP_DATA_ROOT/ckpt/qwen3-8b-sft-merged"
KL_COEF="${KL_COEF:-0}"

if [ "${MERGED:-0}" = "1" ]; then
  MODEL="$MERGED_MODEL"
  ADAPTER_ARGS=()            # fresh zero-init LoRA; SFT is already in MODEL
  START_DESC="$MERGED_MODEL (SFT merged) + zero-init LoRA"
else
  MODEL=$("$P/venv/bin/python" -c "
import sys; sys.path.insert(0,'$P')
from rl.sft_train import resolve_model; print(resolve_model())")
  ADAPTER_ARGS=(actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER")
  START_DESC="base + SFT adapter"
fi

if [ "$KL_COEF" != "0" ] && [ "${MERGED:-0}" != "1" ]; then
  echo "KL_COEF=$KL_COEF needs MERGED=1: otherwise verl's reference policy is" >&2
  echo "base Qwen3-8B, and the KL would pull the policy away from SFT." >&2
  exit 1
fi

# SEED is OPT-IN. verl's rollout.seed already defaults to 42, but data.seed
# defaults to NULL -- so passing a value unconditionally would change the batch
# ORDER of every existing recipe, and silently splice a different data stream
# into a RESUMED trajectory. Only override when the caller actually asks, which
# is what a replication seed does.
SEED_ARGS=()
if [ -n "${SEED:-}" ]; then
  SEED_ARGS=(actor_rollout_ref.rollout.seed="$SEED" data.seed="$SEED")
fi

export CUDA_VISIBLE_DEVICES="${GPUS:-4,5}"
# Derive the GPU COUNT from the GPU LIST so the two can never disagree. This
# used to be a hardcoded 2 in both `ray start --num-gpus` and
# trainer.n_gpus_per_node, which meant GPUS=5 asked Ray for two GPUs it did not
# have and hung.
NGPU="${NGPU:-$(echo "${GPUS:-4,5}" | tr ',' '\n' | grep -c .)}"
export PYTHONPATH="$P:$P/harness:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
# NO FLASH-ATTENTION ON THE TRAINING SIDE.
# This venv has torch 2.11+cu130 while the system nvcc is 12.8, so flash-attn
# refuses to build and publishes no matching wheel. verl defaults to
# flash_attention_2 and to use_remove_padding=True, whose packing helpers
# (verl/utils/attention_utils.py) import flash_attn.bert_padding with no CUDA
# fallback -- so both have to be turned off together, below.
#
# The cost is small here: sequence packing mainly saves padding waste when a
# micro-batch holds several unequal-length sequences, and ours holds one
# (ppo_micro_batch_size_per_gpu=1), so there is nothing to pack. What remains
# is sdpa's attention kernel being slower than FA2. Rollout -- the part that
# dominates wall-clock in agentic RL, since every episode is 12 turns of
# generate-then-execute -- is unaffected: vLLM uses FlashInfer, which is
# installed.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export HF_HOME="${SHARED_LLM_CACHE:-$HOME/.cache/huggingface}"
# Ray workers inherit the driver environment, which is how they find our
# harness/ and rl/ packages -- hydra instantiates the agent loop by dotted path
# in the worker, so a wrong PYTHONPATH there is a ModuleNotFoundError per
# rollout. Some site configs install a runtime-env hook that rewrites the
# worker env; clear it so ours survives. It must be UNSET, not set to "" --
# Ray checks presence, then tries to import a class named "" and dies.
unset RAY_RUNTIME_ENV_HOOK

# PRIVATE RAY HEAD.  This box is shared, and /tmp/ray belongs to another user
# who runs Ray here too. Letting verl call a bare ray.init() made it try to
# attach to their (dead) cluster and spin forever on
#   "Failed to connect to GCS at 172.16.34.19:6379"
# without ever creating a session dir. Our own temp dir and port keep the two
# clusters from seeing each other. num-gpus=2 under CUDA_VISIBLE_DEVICES=4,5
# is also what pins Ray to the two GPUs we are allowed to use -- the raylet
# hands out logical GPU 0,1 which map to physical 4,5.
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6390}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$USER}"
# The guard checks for a live GPU node, not merely a reachable GCS: killing
# raylets leaves a GCS that still answers `ray status` while owning no
# resources, and verl then fails with "No node info found matching attributes".
if ! "$P/venv_verl/bin/ray" status 2>/dev/null | grep -q "GPU"; then
  echo "no usable ray head at $RAY_ADDRESS -- starting one"
  "$P/venv_verl/bin/ray" stop --force >/dev/null 2>&1
  mkdir -p "$RAY_TMPDIR"
  "$P/venv_verl/bin/ray" start --head \
    --port="${RAY_ADDRESS##*:}" --temp-dir="$RAY_TMPDIR" \
    --num-gpus="$NGPU" --dashboard-host=127.0.0.1 >/dev/null 2>&1 \
    || { echo "ray head failed to start"; exit 1; }
fi

mkdir -p "$P/logs"
echo "model   : $MODEL"
echo "start   : $START_DESC"
echo "kl_coef : $KL_COEF"
echo "gpus    : $CUDA_VISIBLE_DEVICES  (n_gpus_per_node=$NGPU)"
echo "seed    : ${SEED:-<verl default: rollout 42, data unset>}"
if [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  echo "RESUMING from $CKPT at step $(cat "$CKPT/latest_checkpointed_iteration.txt")"
  echo "  (verl resume_mode=auto; use a different EXP= for a fresh trajectory)"
fi
echo "ckpt    : $CKPT"
# Read the families out of the parquets. These two lines used to print
# "(cis-MVMR)" and "(scRNA, HELD OUT)" unconditionally, so a run whose data had
# since been rebuilt for a different family still announced the old one.
_fam() { "$P/venv/bin/python" -c "
import sys,pandas as pd
print('+'.join(sorted(set(pd.read_parquet(sys.argv[1],columns=['data_source'])['data_source']))))
" "$1" 2>/dev/null || echo "?"; }
echo "train   : $DATA/train.parquet   ($(_fam "$DATA/train.parquet"))"
echo "val     : $DATA/val.parquet     ($(_fam "$DATA/val.parquet"), HELD OUT)"
echo

"$P/venv_verl/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/val.parquet" \
  data.train_batch_size="${TRAIN_BS:-8}" \
  data.max_prompt_length=4096 \
  data.max_response_length="${MAX_RESP:-8192}" \
  data.filter_overlong_prompts=True \
  data.continuous_token.enable=True \
  data.continuous_token.model_family=qwen3 \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  "${ADAPTER_ARGS[@]}" \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.model.use_fused_kernels="${FUSED:-True}" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS:-8}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${DYN_BSZ:-True}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_TOKENS:-32768}" \
  actor_rollout_ref.actor.use_kl_loss=$([ "$KL_COEF" != "0" ] && echo True || echo False) \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${DYN_BSZ:-True}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGP_TOKENS:-32768}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL:-0.80}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-16384}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-6}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS:-8}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CFG:-$P/scripts/verl_agent_loop.yaml}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${DYN_BSZ:-True}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGP_TOKENS:-32768}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.logger=console \
  trainer.project_name=deeprs-grpo \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.total_epochs="${EPOCHS:-1}" \
  trainer.total_training_steps="${STEPS:-40}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.GBP_TRACE_DIR="${GBP_TRACE_DIR:-}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.GBP_TRACE_FRAC="'${GBP_TRACE_FRAC:-0}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.GBP_SANDBOX_THREADS="'${GBP_SANDBOX_THREADS:-1}'" \
  "${SEED_ARGS[@]}" \
  "$@"
