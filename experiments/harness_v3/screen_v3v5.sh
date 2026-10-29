#!/bin/bash
# HARNESS v5 "WRITE THE METHOD, TEST IT ON NEW DATA" EVALUATION (docs/RL_PLAN.md 2026-09-29 14:15 UTC, registered before any
# build or run). The v3 harness -- KERNEL + science prompt + notebook echo, NO library, experiments/harness_v2/screen_any.sh --
# with the v5 protocol (experiments/harness_v3/program_first.py; its docstring has the rule and every open detail decided):
# the agent submits `def solve(data_dir, task)` (amended 15:55: each dataset gets its OWN task text); the harness runs it on the target and on K=3 layout-matched unlabelled TRAIN siblings
# (none on the externals), replies with a fixed-format report, allows <= 2 revisions; FINAL ANSWER = solve(target) of the last
# valid submission, else the model's last offered JSON answer (v4's soft fallback). NO execution gate.
#   [ORDER=test] [WORKERS=16] experiments/harness_v3/screen_v3v5.sh PORT                -> <root>/V3V5/     (custom:kernel:V3V5)
#   V5_EXT=1 [BENCHES="qrdata dabench bixbench"] experiments/harness_v3/screen_v3v5.sh PORT -> <root>/V3V5X/ext/<bench>/
#                                                                                                      (custom:kernel:V3V5X)
# <root> = results/pods/harness_v3 (V3V5_RESULTS overrides it: tests only), beside -- never inside -- V3/ (v3), V3EFS/ and
# V3EFSX/ (v4): the cells experiments/harness_v3/read_v5.py pairs it with. A wrapper, not an edit: screen_any.sh, screen_v3.sh,
# screen_v3efs.sh, harness/*.py and every EF / EFS file are running / pinned files. v5 reaches the harness through
# experiments/harness_v3/v5_site/sitecustomize.py (this directory first on PYTHONPATH, GBP_V5=1): in every python process of
# the screen it installs the v5 loop when harness/agents.py / harness/v2.py / harness/runner.py are imported, and removes itself
# and its flag from the environment the sandbox cells inherit. A pre-flight in the same environment checks that the v5 loop
# (and NOT an EF / EFS gate) is installed and recorded as program_first_v5 before any rollout, so a V3V5 row can never be a v3
# or a v4 row (screen_any.sh refuses a config.json written under another harness). Resumable as every screen_any.sh screen.
#
# SCOPES: with no flag, the UNTRAINED Qwen3-8B only (no MTAG, no CKPT), stages design / test (default ORDER "test": the
# registered Phase-1 read is TEST-all + HO6). Each flag below is OFF unless exactly 1, at most one per call:
#   V5_EXT=1         the UNTRAINED 8B on the external benchmarks only (ORDER must be "ext"; BENCHES within dabench qrdata
#                    bixbench) -> <root>/V3V5X/ext (Phase 1's externals).
#   V5_ALLOW_CKPT=1  an 8B checkpoint, CKPT=<run dir> STEP=<n> required (no MTAG) -> <root>/V3V5__<run>_s<step>, default ORDER
#                    "design test" (Phase 2: the v5-trained policy read in v5; HV3_137).
#   V5_ALLOW_14B=1   the untrained Qwen3-14B, MODEL=Qwen3-14B MTAG=q14 required -> results/pods/harness_v3_q14/V3V5
#                    (NOT registered: explicit only).
# The EF / EFS flags (EF_EXT, EF_ALLOW_*, EFS_EXT, EFS_ALLOW_*) are refused here, and so is a PYTHONPATH already holding another
# harness_v3 hook directory: a gate or v3.1 hook can never ride along.
# HARNESS v5.1 (docs/RL_PLAN.md 2026-09-29 21:10 UTC), FLAG-GATED -- OFF unless exactly V51=1 (combinable with every scope):
#   V51=1  v5 + (a) LOOK FIRST: GBP_V51=1 reaches the v5 hook, which calls program_first.configure(look_first=True) (a
#          solve() is run only after an executed cell has read the data); the config is custom:kernel:V3V51 (TEST) /
#          custom:kernel:V3V51X (V5_EXT=1), so the cells are <root>/V3V51, <root>/V3V51X/ext/<bench>, and with V5_ALLOW_CKPT=1
#          <root>/V3V51__<run>_s<step>; the pre-flight requires the look-first loop and the program_first_v51 record (and
#          no program_first_v5 record). Without V51=1 GBP_V51 is unset and everything below is v5 exactly.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
PORT="${1:?PORT}"
HOOK="$P/experiments/harness_v3/v5_site"
[ -f "$HOOK/sitecustomize.py" ] && [ -f "$P/experiments/harness_v3/program_first.py" ] && [ -f "$P/rl/grpo_prog.py" ] \
  || { echo "[v5] missing $HOOK/sitecustomize.py, experiments/harness_v3/program_first.py or rl/grpo_prog.py" >&2; exit 2; }
for fl in EF_EXT EF_ALLOW_14B EF_ALLOW_CKPT EFS_EXT EFS_ALLOW_14B EFS_ALLOW_CKPT; do
  [ -z "${!fl:-}" ] || { echo "[v5] $fl is an execution-gate screen's flag -- refused here; the v5 flags are V5_EXT / V5_ALLOW_CKPT / V5_ALLOW_14B" >&2; exit 2; }
done
case ":${PYTHONPATH:-}:" in
  *harness_v3/ef_site*|*harness_v3/ef2_site*|*harness_v3/efs_site*|*harness_v3/v31_site*)
    echo "[v5] PYTHONPATH already holds another harness_v3 hook directory ($PYTHONPATH) -- refused" >&2; exit 2 ;;
esac
case "${V51:-}" in
  ""|1) ;;
  *) echo "[v5] V51 must be 1 or unset, got '${V51}'" >&2; exit 2 ;;
esac
SC=""
for fl in V5_ALLOW_14B V5_ALLOW_CKPT V5_EXT; do
  case "${!fl:-}" in
    "") ;;
    1) [ -z "$SC" ] || { echo "[v5] $SC=1 and $fl=1 together: one scope per call -- refused" >&2; exit 2; }
       SC=$fl ;;
    *) echo "[v5] $fl must be 1 or unset, got '${!fl}'" >&2; exit 2 ;;
  esac
done
CONFIG=custom:kernel:V3V5
case "$SC" in
"")
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[v5] Phase 1 is registered for the UNTRAINED base -- CKPT/STEP refused (V5_ALLOW_CKPT=1 is the explicit checkpoint scope)" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[v5] Phase 1 is registered for Qwen3-8B (no MTAG) only (V5_ALLOW_14B=1 is the explicit 14B scope)" >&2; exit 2; }
export ORDER="${ORDER:-test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[v5] ORDER stage '$st' is outside the v5 synthetic read (design test; the externals are V5_EXT=1)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP
;;
V5_ALLOW_14B)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[v5] V5_ALLOW_14B is the UNTRAINED Qwen3-14B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-}" = Qwen3-14B ] && [ "${MTAG:-}" = q14 ] \
  || { echo "[v5] V5_ALLOW_14B=1 needs MODEL=Qwen3-14B MTAG=q14 (the 14B plain base's conventions, harness_v3_q14/V3)" >&2; exit 2; }
export ORDER="${ORDER:-test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-14B MTAG=q14
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[v5] ORDER stage '$st' is outside the 14B scope (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL CKPT STEP
;;
V5_ALLOW_CKPT)
[ -n "${CKPT:-}" ] && [ -n "${STEP:-}" ] || { echo "[v5] V5_ALLOW_CKPT=1 needs CKPT=<run dir> and STEP=<n>" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] \
  || { echo "[v5] V5_ALLOW_CKPT is for an 8B checkpoint (MODEL Qwen3-8B, no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-design test}" DESIGN_FAMS="${DESIGN_FAMS:-causal_rdd}" MODEL=Qwen3-8B CKPT STEP
for st in $ORDER; do case "$st" in design|test) ;; *) echo "[v5] ORDER stage '$st' is outside the checkpoint scope (design test)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG
;;
V5_EXT)
[ -z "${CKPT:-}" ] && [ -z "${STEP:-}" ] || { echo "[v5] V5_EXT is the UNTRAINED 8B -- CKPT/STEP refused" >&2; exit 2; }
[ "${MODEL:-Qwen3-8B}" = Qwen3-8B ] && [ -z "${MTAG:-}" ] || { echo "[v5] V5_EXT is registered for Qwen3-8B (no MTAG) only" >&2; exit 2; }
export ORDER="${ORDER:-ext}" MODEL=Qwen3-8B
[ "$ORDER" = ext ] || { echo "[v5] V5_EXT=1 runs the external benchmarks only (ORDER=ext), got '$ORDER'" >&2; exit 2; }
for b in ${BENCHES-dabench qrdata bixbench}; do case "$b" in dabench|qrdata|bixbench) ;; *) echo "[v5] BENCHES '$b' is outside the registered read (dabench qrdata bixbench)" >&2; exit 2 ;; esac; done
unset GBP_LIBRARY GBP_LIBRARY_K GBP_PROFILE GBP_APIDOCS GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_KERNEL MTAG CKPT STEP DESIGN_FAMS
CONFIG=custom:kernel:V3V5X
;;
esac
unset GBP_EF GBP_EF_EXT GBP_EFS GBP_EFS_EXT GBP_V51
if [ "${V51:-}" = 1 ]; then CONFIG="${CONFIG/%:V3V5/:V3V51}"; CONFIG="${CONFIG/%:V3V5X/:V3V51X}"; export GBP_V51=1; fi   # v5.1
export HV2_RESULTS="${V3V5_RESULTS:-$P/results/pods/harness_v3}"
export PYTHONPATH="$HOOK${PYTHONPATH:+:$PYTHONPATH}"
export GBP_V5=1
# pre-flight: the v5 hook is loaded, installs the v5 loop (no gate), records program_first_v5 (and no exec_forcing*), logs per
# rollout / per problem, stores the task with the v5 note, and leaves no trace for children
V5_SCOPE="$SC" V51_ON="${V51:-}" venv/bin/python - <<'PY' || { echo "[v5] FATAL pre-flight: the v5 hook did not install -- nothing run" >&2; exit 3; }
import os, sys
ext = os.environ.pop("V5_SCOPE", "") == "V5_EXT"
look = os.environ.pop("V51_ON", "") == "1"
sc = sys.modules.get("sitecustomize")
assert sc is not None and "v5_site" in (getattr(sc, "__file__", "") or ""), "sitecustomize is not v5_site's"
assert getattr(sc, "ON", False), "GBP_V5=1 did not reach the hook"
sys.path.insert(0, "harness")
import agents, v2, runner
S = sys.modules["_gbp_program_first"]
solve = agents.LLMAgent.solve
assert getattr(solve, "_v5", False) and solve.__globals__ is vars(S), "LLMAgent.solve is not the v5 loop"
assert not getattr(solve, "_ef", False) and not getattr(solve, "_efs", False), "an execution gate rides along"
d = v2.HarnessConfig.from_env().as_dict()
if look:                                                                   # v5.1 (a) look first
    rec = d.get("program_first_v51") or {}
    assert S.LOOK_FIRST is True and "program_first_v5" not in d, "V51=1 but the v5 loop is not in look-first mode"
    assert rec.get("harness") == "program_first_v51" and rec.get("version") == "5.1" and (rec.get("look_first") or {}).get("on"), \
        "as_dict does not record program_first_v51 with look_first on"
else:
    rec = d.get("program_first_v5") or {}
    assert S.LOOK_FIRST is False and "program_first_v51" not in d, "look-first is on without V51=1"
    assert rec.get("harness") == "program_first_v5", "as_dict does not record program_first_v5"
assert rec.get("on") and rec.get("k_siblings") == 3 and rec.get("max_revisions") == 2 \
    and rec.get("signature") == "solve(data_dir, task)", "as_dict does not record program_first_v5 (amended: solve(data_dir, task))"
assert "exec_forcing" not in d and "exec_forcing_soft" not in d, "as_dict records an execution gate"
assert getattr(v2.V2Sandbox.stats, "_v5", False), "V2Sandbox.stats does not log the v5 episode"
assert getattr(v2.wrap_rollout, "_v5", False), "v2.wrap_rollout does not store the v5 task"
assert getattr(runner.run_one, "_v5", False), "runner.run_one does not record the v5 log"
assert not {"GBP_V5", "GBP_V51", "GBP_EF", "GBP_EFS"} & set(os.environ) and "v5_site" not in os.environ.get("PYTHONPATH", ""), \
    "the hook left a trace for children"
print("[v5] pre-flight OK" + ((" (V3V51X)" if look else " (V3V5X)") if ext else "") + ": LLMAgent.solve = " + solve.__module__ + "."
      + solve.__qualname__ + "; harness recorded as " + rec["harness"] + " (solve(data_dir, task), K=3 siblings, <= 2 revisions, no gate"
      + ("; v5.1 LOOK FIRST: a solve() runs only after an executed cell has read the data)" if look else ")"))
PY
exec experiments/harness_v2/screen_any.sh "$CONFIG" "$PORT"
