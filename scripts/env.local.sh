# Machine-local overrides for UMass Unity, sourced by scripts/env.sh.
# Everything here is a path that exists on this cluster and nowhere else.

BENCH_WS="$(ws_find deeprs-bench 2>/dev/null | grep '^/' | head -1)"

# Hard links, not copies, for BixBench data. Sandbox.link_data hard-links the
# per-rollout data_files; os.link across filesystems raises EXDEV and the
# sandbox silently falls back to copying 1.1 GB per rollout. benchmarks/*/
# problems lives on scratch, so $TMPDIR must too. See docs/CLUSTER.md.
# 2026-09-26: episode workdirs on NODE-LOCAL /tmp, not NFS. With ~200 concurrent sandbox episodes fleet-wide each
# copying data to and running from $BENCH_WS/tmp, the /scratch4 filer stalled (every episode in rpc_wait; evals frozen
# 50 min, RL steps crawling) and the directory had leaked 11k entries. The sandbox COPIES its data into the workdir, so
# the workdir need not be on scratch. NFS path kept only as a fallback for a node with no writable /tmp.
_ltmp="/tmp/deeprs_${USER}"
if mkdir -p "$_ltmp" 2>/dev/null && [ -w "$_ltmp" ]; then export TMPDIR="$_ltmp"
elif [ -n "$BENCH_WS" ]; then export TMPDIR="$BENCH_WS/tmp"; mkdir -p "$TMPDIR"; fi
unset _ltmp

# Keep HF and uv caches off /project (NFS, shared with the whole PI group).
[ -n "$BENCH_WS" ] && export HF_HOME="$BENCH_WS/hf" UV_CACHE_DIR="$BENCH_WS/uv"
export UV_LINK_MODE=copy          # cache and venvs are on different filesystems
export UV_CONCURRENT_INSTALLS=1   # uv's parallel installer races NFS (CLUSTER.md)

# serve_local.sh looks for $SHARED_LLM_CACHE/hub/models--Qwen--<model>, but the
# shared checkpoint tree IS the hub directory. llm_cache/hub symlinks onto it.
[ -n "$BENCH_WS" ] && export SHARED_LLM_CACHE="$BENCH_WS/llm_cache"

# 2026-10-09: $HOME hit its quota. vLLM's torch.compile cache (~/.cache/vllm, 7.7 GB) and its usage-stats file
# go to scratch instead. Literal fallback: ws_find can fail on compute nodes, leaving BENCH_WS empty.
export VLLM_CACHE_ROOT="${BENCH_WS:-/scratch4/workspace/hieutran_umass_edu-deeprs-bench}/vllm_cache"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

unset BENCH_WS
