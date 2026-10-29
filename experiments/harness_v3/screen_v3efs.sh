#!/bin/bash
# HARNESS v4 = v3 + THE SOFT EXECUTION GATE (EFS) EVALUATION (docs/RL_PLAN.md 2026-09-29 04:40 UTC, registered before any build
# or run). screen_v3ef.sh's harness and gate -- KERNEL + science prompt + notebook echo, NO library,
# experiments/harness_v2/screen_any.sh; an answer is accepted only if its numbers were printed by the agent's own executed code,
# else one fixed reply and the episode continues under the same limits -- with ONE change at the end of an episode without an
# accepted answer: the last grounded offered answer if any (as EF), ELSE the LAST answer the model offered (logged
# "soft_fallback"), instead of none (experiments/harness_v3/exec_forcing_soft.py; its docstring has the rule and how it reuses
# exec_forcing.py unchanged).
#   [ORDER=test] [WORKERS=16] experiments/harness_v3/screen_v3efs.sh PORT                 -> <root>/V3EFS/    (custom:kernel:V3EFS)
#   EFS_EXT=1 [BENCHES="qrdata dabench bixbench"] experiments/harness_v3/screen_v3efs.sh PORT -> <root>/V3EFSX/ext/<bench>/
#                                                                                                          (custom:kernel:V3EFSX)
# <root> = results/pods/harness_v3 (V3EFS_RESULTS overrides it: tests only), beside -- never inside -- V3/, V3EF/, V3EFX/ (the
# plain and hard-gate cells experiments/harness_v3/read_efs.py pairs it with). A wrapper, not an edit: screen_any.sh,
# screen_v3.sh, screen_v3ef.sh, harness/*.py, exec_forcing.py, exec_forcing_ext.py, ef_site/ and ef2_site/ are running / pinned
# files. EFS reaches the harness through experiments/harness_v3/efs_site/sitecustomize.py (this directory first on PYTHONPATH,
# GBP_EFS=1, + GBP_EFS_EXT=1 for the externals): in every python process of the screen it installs the soft gate when
# harness/agents.py / harness/v2.py (/ harness/runner.py) are imported, and removes itself and its flags from the environment
# the sandbox cells inherit. A pre-flight in the same environment checks that the SOFT gate (not EF's hard one) is installed
# and recorded before any rollout, so a V3EFS row can never be a plain V3 or a hard-gate V3EF row. Resumable as every
# screen_any.sh screen: rerun to finish (families / benchmarks already written are skipped; config.json refuses a mix).
#
# SCOPES (EF's guards, kept): with no flag, the UNTRAINED Qwen3-8B only (no MTAG, no CKPT), stages design / test (default
# ORDER "test": the registered read is TEST-all + HO6). Each flag below is OFF unless exactly 1, at most one per call; the
# registered v4 read uses only the default scope and EFS_EXT:
#   EFS_EXT=1         the UNTRAINED 8B on the external benchmarks only (ORDER must be "ext"; BENCHES within dabench qrdata
#                     bixbench), with exec_forcing_ext.py's answer formats on the SAME gate -- EF_3's externals path.
#   EFS_ALLOW_14B=1   the untrained Qwen3-14B, MODEL=Qwen3-14B MTAG=q14 required -> results/pods/harness_v3_q14/V3EFS
#                     (NOT registered: explicit only).
#   EFS_ALLOW_CKPT=1  an 8B checkpoint, CKPT=<run dir> STEP=<n> required (no MTAG) -> <root>/V3EFS__<run>_s<step>
#                     (NOT registered: the 04:50 EFPR entry names "EFS on PR s1 @100" as the follow-up IF v4 is adopted).
# screen_v3ef.sh's own flags (EF_EXT, EF_ALLOW_14B, EF_ALLOW_CKPT) are refused here, and so is a PYTHONPATH already holding
# another harness_v3 hook directory: a hard-gate or v3.1 hook can never ride along.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
PORT="${1:?PORT}"
HOOK="$P/experiments/harness_v3/efs_site"
[ -f "$HOOK/sitecustomize.py" ] && [ -f "$P/experiments/harness_v3/exec_forcing.py" ] \
  && [ -f "$P/experiments/harness_v3/exec_forcing_soft.py" ] && [ -f "$P/harness/process_signals.py" ] \
  || { echo "[efs] missing $HOOK/sitecustomize.py, experiments/harness_v3/exec_forcing{,_soft}.py or harness/process_signals.py" >&2; exit 2; }
for fl in EF_EXT EF_ALLOW_14B EF_ALLOW_CKPT; do
  [ -z "${!fl:-}" ] || { echo "[efs] $fl is screen_v3ef.sh's (hard gate) flag -- refused here; the EFS flags are EFS_EXT / EFS_ALLOW_14B / EFS_ALLOW_CKPT" >&2; exit 2; }
done
case ":${PYTHONPATH:-}:" in
  *harness_v3/ef_site*|*harness_v3/ef2_site*|*harness_v3/v31_site*)
    echo "[efs] PYTHONPATH already holds another harness_v3 hook directory ($PYTHONPATH) -- refused" >&2; exit 2 ;;
esac
SC=""
for fl in EFS_ALLOW_14B EFS_ALLOW_CKPT EFS_EXT; do
  case "${!fl:-}" in
    "") ;;
    1) [ -z "$SC" ] || { echo "[efs] $SC=1 and $fl=1 together: one scope per call -- refused" >&2; exit 2; }
       SC=$fl ;;
    *) echo "[efs] $fl must be 1 or unset, got '${!fl}'" >&2; exit 2 ;;
  esac
done
CONFIG=custom:kernel:V3EFS
case "$SC" in
"")
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[efs] v4 is registered for the UNTRAINED base -- CKPT/STEP refused (EFS_ALLOW_CKPT=1 is the explicit checkpoint scope)" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[efs] v4 is registered for Qwen3-8B (no MTAG) only (EFS_ALLOW_14B=1 is the explicit 14B scope)" >&2; exit 2; }
export ORDER="${ORDER:-test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[efs] ORDER stage '$st' is outside the EFS synthetic read (design test; the externals are EFS_EXT=1)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP
;;
EFS_ALLOW_14B)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[efs] EFS_ALLOW_14B is the UNTRAINED Qwen3-14B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-}" = Qwen3-14B ] && [ "${MTAG:-}" = q14 ] \
  || { echo "[efs] EFS_ALLOW_14B=1 needs MODEL=Qwen3-14B MTAG=q14 (the 14B plain base's conventions, harness_v3_q14/V3)" >&2; exit 2; }
export ORDER="${ORDER:-test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-14B MTAG=q14
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[efs] ORDER stage '$st' is outside the 14B scope (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL CKPT STEP
;;
EFS_ALLOW_CKPT)
[ -n "${CKPT:-}" ] && [ -n "${STEP:-}" ] || { echo "[efs] EFS_ALLOW_CKPT=1 needs CKPT=<run dir> and STEP=<n>" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] \
  || { echo "[efs] EFS_ALLOW_CKPT is for an 8B checkpoint (MODEL Qwen3-8B, no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-design test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B CKPT STEP
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[efs] ORDER stage '$st' is outside the checkpoint scope (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG
;;
EFS_EXT)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[efs] EFS_EXT is the UNTRAINED 8B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[efs] EFS_EXT is registered for Qwen3-8B (no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-ext}" MODEL=Qwen3-8B
[ "$ORDER" = ext ] || { echo "[efs] EFS_EXT=1 runs the external benchmarks only (ORDER=ext), got '$ORDER'" >&2; exit 2; }
for b in ${BENCHES-dabench qrdata bixbench}; do case "$b" in dabench|qrdata|bixbench) ;; *) echo "[efs] BENCHES '$b' is outside the registered read (dabench qrdata bixbench)" >&2; exit 2 ;; esac; done
[ -f "$P/experiments/harness_v3/exec_forcing_ext.py" ] || { echo "[efs] missing experiments/harness_v3/exec_forcing_ext.py" >&2; exit 2; }
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP DESIGN_FAMS
CONFIG=custom:kernel:V3EFSX
;;
esac
unset GBP_EF GBP_EF_EXT GBP_EFS_EXT
export HV2_RESULTS="${V3EFS_RESULTS:-$P/results/pods/harness_v3}"
export PYTHONPATH="$HOOK${PYTHONPATH:+:$PYTHONPATH}"
export GBP_EFS=1
[ "$SC" = EFS_EXT ] && export GBP_EFS_EXT=1
# pre-flight: the efs hook is loaded, installs the SOFT gate (not EF's hard one), records exec_forcing_soft (and NOT
# exec_forcing), logs per rollout / per problem, and leaves no trace for children
EFS_SCOPE="$SC" venv/bin/python - <<'PY' || { echo "[efs] FATAL pre-flight: the EFS hook did not install -- nothing run" >&2; exit 3; }
import os, sys
ext = os.environ.pop("EFS_SCOPE", "") == "EFS_EXT"
sc = sys.modules.get("sitecustomize")
assert sc is not None and "efs_site" in (getattr(sc, "__file__", "") or ""), "sitecustomize is not efs_site's"
assert getattr(sc, "ON", False), "GBP_EFS=1 did not reach the hook"
assert getattr(sc, "EXT", None) is ext, f"GBP_EFS_EXT reached the hook as {getattr(sc, 'EXT', None)}, expected {ext}"
sys.path.insert(0, "harness")
import agents, v2
S = sys.modules["_gbp_exec_forcing_soft"]
EF = S.current()
solve = agents.LLMAgent.solve
assert getattr(solve, "_ef", False) and getattr(solve, "_efs", False), "LLMAgent.solve is not the EFS loop"
assert solve.__globals__ is vars(EF) and getattr(EF.Gate, "_efs", False), "the loop's gate is not the SOFT gate"
d = v2.HarnessConfig.from_env().as_dict()
rec = d.get("exec_forcing_soft") or {}
assert rec.get("harness") == "exec_forcing_soft" and rec.get("on") and "exec_forcing" not in d, "as_dict does not record EFS (only)"
assert bool((rec.get("gate") or {}).get("ext")) is ext, "the recorded gate's external formats do not match the scope"
assert getattr(v2.V2Sandbox.stats, "_ef", False), "V2Sandbox.stats does not log the gate"
if ext:
    import runner
    assert getattr(EF.grounded, "_ef_ext", False) and getattr(EF.Gate, "_ef_ext", False), "the gate has no external formats"
    assert getattr(runner.run_one, "_ef_ext", False), "runner.run_one does not record the gate log"
assert not {"GBP_EF", "GBP_EF_EXT", "GBP_EFS", "GBP_EFS_EXT"} & set(os.environ) and "efs_site" not in os.environ.get("PYTHONPATH", ""), \
    "the hook left a trace for children"
print("[efs] pre-flight OK" + (" (EFSX)" if ext else "") + ": LLMAgent.solve = " + solve.__module__ + "." + solve.__qualname__
      + "; gate = " + EF.Gate.__module__ + "." + EF.Gate.__qualname__ + "; harness recorded as exec_forcing_soft")
PY
exec experiments/harness_v2/screen_any.sh "$CONFIG" "$PORT"
