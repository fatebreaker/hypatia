#!/bin/bash
# OpenAI frontier model in OUR harness, cost-capped (docs/RL_PLAN.md H-FRONTIER-API, 2026-09-30).
#   scripts/eval_openai.sh MODEL BENCH [OUTTAG]        e.g. scripts/eval_openai.sh gpt-5 qrdata_verified
# The v3 harness exactly as our trained model is scored (GBP_KERNEL=1, GBP_SYSTEM=science, GBP_ECHO_LAST=1, same
# sandbox, turns, grader), but through the model's own chat API (agent "llm:MODEL") with GBP_OPENAI_STYLE=1.
# KEY: read at run time from ~/.config/deeprs/openai_key (created by the user, mode 600); never printed or logged.
# COST: every call's usage is priced into the shared ledger scratchpad/openai/ledger.jsonl; calls are refused once the
# ledger reaches BUDGET_USD (default 100) -- a hard stop across all runs. Prices ($/1M, OpenAI pricing page 2026-09-30)
# are looked up per model below; an unknown model refuses to start.
# Env: EFFORT (reasoning_effort; default medium), WORKERS (default 6), ONLY (substring filter), BUDGET_USD.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$P"
MODEL="${1:?MODEL}"; BENCH="${2:?BENCH}"; TAG="${3:-$MODEL-${EFFORT:-medium}}"
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
KEYF="$HOME/.config/deeprs/openai_key"
[ -s "$KEYF" ] || { echo "[eval_openai] no key file $KEYF -- the user creates it (umask 077)"; exit 2; }
case "$MODEL" in
  gpt-5)        PIN=1.25; PC=0.125; POUT=10 ;;
  gpt-5-mini)   PIN=0.25; PC=0.025; POUT=2 ;;
  gpt-5.1)      PIN=1.25; PC=0.125; POUT=10 ;;
  gpt-6.1-sol)  PIN=2.00; PC=0.10;  POUT=10 ;;
  gpt-6-sol)    PIN=2.00; PC=0.20;  POUT=10 ;;
  gpt-5.6-terra) PIN=2.00; PC=0.20; POUT=12 ;;
  o4-mini)      PIN=1.10; PC=0.275; POUT=4.40 ;;
  gpt-4.1)      PIN=2.00; PC=0.50;  POUT=8 ;;
  gpt-4o)       PIN=2.50; PC=1.25;  POUT=10 ;;
  *) echo "[eval_openai] no price for $MODEL -- add it from the pricing page first"; exit 2 ;;
esac
# non-reasoning models (gpt-4.1, gpt-4o) reject reasoning_effort / need no max_completion_tokens: plain payload
# (max_tokens, the harness temperature 0.2); reasoning models use GBP_OPENAI_STYLE=1.
case "$MODEL" in gpt-4.1|gpt-4o) REASONING=0 ;; *) REASONING=1 ;; esac
mkdir -p "$S/openai"
export OPENAI_API_KEY="$(cat "$KEYF")" GBP_API_KEY="$(cat "$KEYF")"
export GBP_BASE_URL=https://api.openai.com/v1
if [ "$REASONING" = 1 ]; then export GBP_OPENAI_STYLE=1 GBP_REASONING_EFFORT="${EFFORT:-medium}" GBP_MAX_COMPLETION_TOKENS="${MAX_OUT:-8192}"
else unset GBP_OPENAI_STYLE GBP_REASONING_EFFORT GBP_MAX_COMPLETION_TOKENS; fi
# SERVICE_TIER=flex (2026-10-05): OpenAI flex processing -- batch-level price on the synchronous API, slower; the client
# prices a response the API reports as 'flex' at half the standard rate (harness/agents.py _budget_record).
if [ -n "${SERVICE_TIER:-}" ]; then export GBP_SERVICE_TIER="$SERVICE_TIER"; else unset GBP_SERVICE_TIER; fi
export GBP_USAGE_LEDGER="$S/openai/ledger.jsonl" GBP_BUDGET_USD="${BUDGET_USD:-100}" GBP_PRICE_IN=$PIN GBP_PRICE_CACHED=$PC GBP_PRICE_OUT=$POUT
export GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 PYTHONDONTWRITEBYTECODE=1
unset GBP_PROFILE GBP_APIDOCS GBP_LIBRARY GBP_LIBRARY_K GBP_SIBLINGS GBP_CHECKS GBP_ICL_DEMO GBP_WALL_BUDGET
OUT="$P/results/pods/frontier_api/$TAG/$BENCH"; mkdir -p "$OUT"
# MedAgentBench: its FHIR server runs beside the runner (scripts/medagentbench_server.sh), 8 turns as published.
MAB=0
if [[ "$BENCH" == medagentbench* ]]; then
  FENV=$(scripts/medagentbench_server.sh start 2>>"$OUT.server.log" | grep -o 'FHIR_API_BASE=[^ ]*' | tail -1)
  [ -n "$FENV" ] || { echo "[eval_openai] MedAgentBench server did not start (see $OUT.server.log)"; exit 3; }
  export "$FENV" GBP_MAX_TURNS=8; MAB=1
fi
t0=$(date +%s); echo "[eval_openai] $(date -Is) $MODEL effort=${EFFORT:-medium} $BENCH -> $OUT (budget \$$GBP_BUDGET_USD total)"
venv/bin/python experiments/harness_v2/runner_v2.py --agent "llm:$MODEL" --problems "$P/benchmarks/$BENCH" \
    --workers "${WORKERS:-6}" --outdir "$OUT" ${ONLY:+--only "$ONLY"} > "$OUT.log" 2>&1; rc=$?
[ "$MAB" = 1 ] && scripts/medagentbench_server.sh stop >/dev/null 2>&1
[[ "$BENCH" == ehrsql2024* ]] && venv/bin/python scripts/ehrsql2024_score.py "$OUT" --problems "$P/benchmarks/$BENCH" > "$OUT.score.txt" 2>&1 && tail -5 "$OUT.score.txt"
python3 "$S/openai_cost.py" --since "$t0" --model "$MODEL"
exit $rc
