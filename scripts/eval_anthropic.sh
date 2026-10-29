#!/bin/bash
# Claude (Anthropic Messages API) in OUR harness, cost-capped (2026-10-25; scripts/eval_openai.sh's twin -- same harness, turns,
# sandbox and grader as every other system; agent "claude:MODEL", harness/anthropic_agent.py).
#   BUDGET_USD=<cap> scripts/eval_anthropic.sh MODEL BENCH [OUTTAG]      e.g. scripts/eval_anthropic.sh claude-opus-5-5 qrdata_verified
# KEY: read at run time from $ANTHROPIC_KEY_FILE (default ~/claykey.txt, the user's file); never printed or logged.
# COST: every call is priced into scratchpad/anthropic/ledger.jsonl; calls are refused once the ledger reaches BUDGET_USD, a hard
# stop across ALL runs (the ledger starts empty, so BUDGET_USD is the total new spend). BUDGET_USD has no default.
# Env: EFFORT (default high), MAX_OUT (default 32000), WORKERS (default 4), ONLY (substring filter), BUDGET_USD.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$P"
MODEL="${1:?MODEL}"; BENCH="${2:?BENCH}"; TAG="${3:-$MODEL-${EFFORT:-high}}"
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
KEYF="${ANTHROPIC_KEY_FILE:-$HOME/claykey.txt}"
[ -s "$KEYF" ] || { echo "[eval_anthropic] no key file $KEYF"; exit 2; }
[ -n "${BUDGET_USD:-}" ] || { echo "[eval_anthropic] set BUDGET_USD (total new spend cap across all Claude runs)"; exit 2; }
# $ per 1M tokens: input, cache write (5-minute TTL, 1.25x input), cache read, output (thinking included) -- claude-api skill, 2026-10
case "$MODEL" in
  claude-opus-5-5)   PIN=4;  PW=5;    PC=0.20; POUT=20 ;;
  claude-sonnet-5-5) PIN=2;  PW=2.5;  PC=0.20; POUT=10 ;;
  claude-fable-5-1)  PIN=10; PW=12.5; PC=0.25; POUT=50 ;;
  *) echo "[eval_anthropic] no price for $MODEL -- add it first"; exit 2 ;;
esac
mkdir -p "$S/anthropic"
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$KEYF")"
export GBP_ANTHROPIC_EFFORT="${EFFORT:-high}" GBP_ANTHROPIC_MAX_TOKENS="${MAX_OUT:-32000}"
export GBP_USAGE_LEDGER="$S/anthropic/ledger.jsonl" GBP_BUDGET_USD="$BUDGET_USD" GBP_PRICE_IN=$PIN GBP_PRICE_CACHE_WRITE=$PW GBP_PRICE_CACHED=$PC GBP_PRICE_OUT=$POUT
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 GBP_RESUME=1 PYTHONDONTWRITEBYTECODE=1
unset GBP_PROFILE GBP_APIDOCS GBP_LIBRARY GBP_LIBRARY_K GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_WALL_BUDGET GBP_OPENAI_STYLE GBP_REASONING_EFFORT GBP_SERVICE_TIER
OUT="$P/results/pods/frontier_api/$TAG/$BENCH"; mkdir -p "$OUT"
# MedAgentBench: its FHIR server runs beside the runner (scripts/medagentbench_server.sh), 8 turns as published.
MAB=0
if [[ "$BENCH" == medagentbench* ]]; then
  FENV=$(scripts/medagentbench_server.sh start 2>>"$OUT.server.log" | grep -o 'FHIR_API_BASE=[^ ]*' | tail -1)
  [ -n "$FENV" ] || { echo "[eval_anthropic] MedAgentBench server did not start (see $OUT.server.log)"; exit 3; }
  export "$FENV" GBP_MAX_TURNS=8; MAB=1
fi
t0=$(date +%s); echo "[eval_anthropic] $(date -Is) $MODEL effort=$GBP_ANTHROPIC_EFFORT $BENCH -> $OUT (budget \$$GBP_BUDGET_USD total)"
venv/bin/python experiments/harness_v2/runner_v2.py --agent "claude:$MODEL" --problems "$P/benchmarks/$BENCH" \
    --workers "${WORKERS:-4}" --outdir "$OUT" ${ONLY:+--only "$ONLY"} > "$OUT.log" 2>&1; rc=$?
[ "$MAB" = 1 ] && scripts/medagentbench_server.sh stop >/dev/null 2>&1
[[ "$BENCH" == ehrsql2024* ]] && venv/bin/python scripts/ehrsql2024_score.py "$OUT" --problems "$P/benchmarks/$BENCH" > "$OUT.score.txt" 2>&1 && tail -5 "$OUT.score.txt"
python3 - "$GBP_USAGE_LEDGER" "$t0" <<'PY'
import json, sys
led, t0 = sys.argv[1], float(sys.argv[2]); run = tot = 0.0; n = 0
for l in open(led):
    r = json.loads(l); tot += r.get("usd") or 0
    if r["t"] >= t0: run += r.get("usd") or 0; n += 1
print(f"[eval_anthropic] this run: {n} calls, ${run:.2f}; ledger total ${tot:.2f}")
PY
exit $rc
