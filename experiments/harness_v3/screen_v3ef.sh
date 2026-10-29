#!/bin/bash
# THE EXECUTION-FORCING (EF) EVALUATION (docs/RL_PLAN.md 2026-09-28 14:40 UTC, registered before any build or run): screen_v3.sh's
# harness -- KERNEL + science prompt + notebook echo, NO library, experiments/harness_v2/screen_any.sh -- for the UNTRAINED
# Qwen3-8B, with ONE change: an answer is accepted only if its numbers were printed by the agent's own executed code
# (harness/process_signals.py's 'grounded' signal); otherwise the harness replies with one fixed message and the episode
# continues under the same turn / wall limits (experiments/harness_v3/exec_forcing.py; the docstring has the rule).
#   [DESIGN_FAMS=causal_rdd] [ORDER="design test"] [WORKERS=16] experiments/harness_v3/screen_v3ef.sh PORT
# -> results/pods/harness_v3/V3EF/ (config custom:kernel:V3EF), beside -- never inside -- V3/ (the untrained base read it is
#    paired with by experiments/harness_v3/read_ef.py). V3EF_RESULTS overrides the root (tests only).
# A wrapper, not an edit: screen_any.sh, screen_v3.sh, harness/agents.py and harness/v2.py are running / pinned files. EF
# reaches the harness through experiments/harness_v3/ef_site/sitecustomize.py (this directory first on PYTHONPATH, GBP_EF=1):
# in every python process of the screen it installs the gate when harness/agents.py / harness/v2.py are imported, and it
# removes itself and GBP_EF from the environment the sandbox cells inherit.
# The registered scope is enforced: MODEL Qwen3-8B only (no MTAG), NO checkpoint (the read is the untrained base), stages
# design / test only (the read is TEST + HO6 + DESIGN causal_rdd; externals are not part of it). A pre-flight in the same
# environment checks that the hook actually installs the gate before any rollout, so a V3EF row can never be a plain V3 row.
#
# EF2 SCOPES (docs/RL_PLAN.md 2026-09-28 23:00 UTC, "EF2 REGISTERED (before any build or run)"). Each is OFF unless its flag
# is exactly 1, at most one per call; with none of them set this script is the EF_1 wrapper above, unchanged
# (experiments/harness_v3/tests/test_ef2.py runs the pre-EF2 copy and this one on the same scripted episodes: identical).
# The gate is the SAME in all three (exec_forcing.py, unchanged).
#   EF_ALLOW_14B=1   EF14: the UNTRAINED Qwen3-14B, MODEL=Qwen3-14B MTAG=q14 required (HV3_0's conventions: the 14B plain
#                    base harness_v3_q14/V3) -> results/pods/harness_v3_q14/V3EF; no CKPT; stages design / test (default
#                    ORDER "test": the read is HO6 + TEST-all).
#   EF_ALLOW_CKPT=1  EFPR: an 8B checkpoint, CKPT=<run dir> STEP=<n> required (no MTAG); screen_any.sh hot-loads the LoRA and
#                    sets the run's system prompt exactly as for every checkpoint eval (HV3_22) ->
#                    results/pods/harness_v3/V3EF__<run>_s<step>; stages design / test.
#   EF_EXT=1         EFX: the UNTRAINED 8B on the external benchmarks only (ORDER must be "ext"; BENCHES within dabench qrdata
#                    bixbench) through experiments/harness_v3/ef2_site/ (ef_site's hook + experiments/harness_v3/
#                    exec_forcing_ext.py: the externals' answer formats mapped onto the same gate; GBP_EF=1 and
#                    GBP_EF_EXT=1) -> results/pods/harness_v3/V3EFX/ext/<bench>/ (config custom:kernel:V3EFX).
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
PORT="${1:?PORT}"
HOOK="$P/experiments/harness_v3/ef_site"
[ -f "$HOOK/sitecustomize.py" ] && [ -f "$P/experiments/harness_v3/exec_forcing.py" ] && [ -f "$P/harness/process_signals.py" ] \
  || { echo "[ef] missing $HOOK/sitecustomize.py, experiments/harness_v3/exec_forcing.py or harness/process_signals.py" >&2; exit 2; }
EF2=""
for fl in EF_ALLOW_14B EF_ALLOW_CKPT EF_EXT; do
  case "${!fl:-}" in
    "") ;;
    1) [ -z "$EF2" ] || { echo "[ef] $EF2=1 and $fl=1 together: each EF2 scope is its own registered read -- refused" >&2; exit 2; }
       EF2=$fl ;;
    *) echo "[ef] $fl must be 1 or unset, got '${!fl}'" >&2; exit 2 ;;
  esac
done
CONFIG=custom:kernel:V3EF
case "$EF2" in
"")
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[ef] EF is registered for the UNTRAINED base only -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[ef] EF is registered for Qwen3-8B (no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-design test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[ef] ORDER stage '$st' is outside the registered EF read (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP
;;
EF_ALLOW_14B)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[ef] EF14 is the UNTRAINED Qwen3-14B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-}" = Qwen3-14B ] && [ "${MTAG:-}" = q14 ] \
  || { echo "[ef] EF_ALLOW_14B=1 needs MODEL=Qwen3-14B MTAG=q14 (the 14B plain base's conventions, harness_v3_q14/V3)" >&2; exit 2; }
export ORDER="${ORDER:-test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-14B MTAG=q14
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[ef] ORDER stage '$st' is outside the registered EF14 read (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL CKPT STEP
;;
EF_ALLOW_CKPT)
[ -n "${CKPT:-}" ] && [ -n "${STEP:-}" ] || { echo "[ef] EF_ALLOW_CKPT=1 needs CKPT=<run dir> and STEP=<n>" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] \
  || { echo "[ef] EFPR is registered for an 8B checkpoint (MODEL Qwen3-8B, no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-design test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B CKPT STEP
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[ef] ORDER stage '$st' is outside the registered EFPR read (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG
;;
EF_EXT)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[ef] EFX is the UNTRAINED 8B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[ef] EFX is registered for Qwen3-8B (no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-ext}" MODEL=Qwen3-8B
[ "$ORDER" = ext ] || { echo "[ef] EF_EXT=1 runs the external benchmarks only (ORDER=ext), got '$ORDER'" >&2; exit 2; }
for b in ${BENCHES-dabench qrdata bixbench}; do case "$b" in dabench|qrdata|bixbench) ;; *) echo "[ef] BENCHES '$b' is outside the registered EFX read (dabench qrdata bixbench)" >&2; exit 2 ;; esac; done
HOOK="$P/experiments/harness_v3/ef2_site"
[ -f "$HOOK/sitecustomize.py" ] && [ -f "$P/experiments/harness_v3/exec_forcing_ext.py" ] \
  || { echo "[ef] missing $HOOK/sitecustomize.py or experiments/harness_v3/exec_forcing_ext.py" >&2; exit 2; }
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP DESIGN_FAMS
CONFIG=custom:kernel:V3EFX
;;
esac
unset GBP_EF_EXT
export HV2_RESULTS="${V3EF_RESULTS:-$P/results/pods/harness_v3}"
export PYTHONPATH="$HOOK${PYTHONPATH:+:$PYTHONPATH}"
export GBP_EF=1
if [ "$EF2" = EF_EXT ]; then
export GBP_EF_EXT=1
# pre-flight (EFX): the ef2 hook is loaded, installs the gate AND the external formats, logs per problem, and leaves no trace
venv/bin/python - <<'PY' || { echo "[ef] FATAL pre-flight: the EF2 external hook did not install -- nothing run" >&2; exit 3; }
import os, sys
sc = sys.modules.get("sitecustomize")
assert sc is not None and "ef2_site" in (getattr(sc, "__file__", "") or ""), "sitecustomize is not ef2_site's"
assert getattr(sc, "ACTIVE", False), "GBP_EF=1 and GBP_EF_EXT=1 did not both reach the hook"
sys.path.insert(0, "harness")
import agents, v2, runner
ef, ext = sys.modules["_gbp_exec_forcing"], sys.modules["_gbp_exec_forcing_ext"]
assert getattr(agents.LLMAgent.solve, "_ef", False), "LLMAgent.solve is not the EF loop"
rec = v2.HarnessConfig.from_env().as_dict().get("exec_forcing") or {}
assert rec.get("on") and rec.get("ext"), "HarnessConfig.as_dict does not record EF with the external formats"
assert getattr(v2.V2Sandbox.stats, "_ef", False), "V2Sandbox.stats does not log EF"
assert getattr(ef.grounded, "_ef_ext", False) and ef.grounded._orig.__module__ == "_gbp_exec_forcing", "the gate has no external formats"
assert getattr(ef.Gate, "_ef_ext", False), "the gate does not log external offers"
assert getattr(runner.run_one, "_ef_ext", False), "runner.run_one does not record the gate log"
assert not {"GBP_EF", "GBP_EF_EXT"} & set(os.environ) and "ef2_site" not in os.environ.get("PYTHONPATH", ""), "the hook left a trace for children"
print("[ef] pre-flight OK (EFX): LLMAgent.solve = " + agents.LLMAgent.solve.__module__ + "." + agents.LLMAgent.solve.__qualname__
      + "; gate = " + ef.grounded.__module__ + "." + ef.grounded.__qualname__ + "; runner.run_one records rec['ef']")
PY
else
# pre-flight: the hook is loaded, installs the gate, records EF, and leaves no trace for children
venv/bin/python - <<'PY' || { echo "[ef] FATAL pre-flight: the EF hook did not install -- nothing run" >&2; exit 3; }
import os, sys
sc = sys.modules.get("sitecustomize")
assert sc is not None and "ef_site" in (getattr(sc, "__file__", "") or ""), "sitecustomize is not ef_site's"
assert getattr(sc, "ON", False), "GBP_EF=1 did not reach the hook"
sys.path.insert(0, "harness")
import agents, v2
assert getattr(agents.LLMAgent.solve, "_ef", False), "LLMAgent.solve is not the EF loop"
assert "exec_forcing" in v2.HarnessConfig.from_env().as_dict(), "HarnessConfig.as_dict does not record EF"
assert getattr(v2.V2Sandbox.stats, "_ef", False), "V2Sandbox.stats does not log EF"
assert "GBP_EF" not in os.environ and "ef_site" not in os.environ.get("PYTHONPATH", ""), "the hook left a trace for children"
print("[ef] pre-flight OK: LLMAgent.solve = " + agents.LLMAgent.solve.__module__ + "." + agents.LLMAgent.solve.__qualname__)
PY
fi
exec experiments/harness_v2/screen_any.sh "$CONFIG" "$PORT"
