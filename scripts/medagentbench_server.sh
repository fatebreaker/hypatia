#!/bin/bash
# MedAgentBench FHIR server (HAPI FHIR JPA v7.6.0 + the benchmark's preloaded H2 EHR) under apptainer, on THIS node.
#
#   eval "$(scripts/medagentbench_server.sh start)"   # prints `export FHIR_API_BASE=http://localhost:PORT/fhir/`
#   scripts/medagentbench_server.sh status
#   scripts/medagentbench_server.sh stop
#
# Run it INSIDE the pod/job that will run harness/runner.py (the agent's sandbox reaches the server on localhost).
# CPU only: one JVM (heap $MAB_JAVA_HEAP, default 8g) + a small Python proxy; give the job >= 4 CPUs and >= 16 GB.
#
# How it works (tested on cpu057 2026-09-30, apptainer 1.5.4):
#   * image: docker://jyxsu6/medagentbench pinned by digest, pulled once to $(data_root)/cache/medagentbench/*.sif
#     (1.8 GB). Layer cache and build temp go to node-local $TMPDIR, never to the shared /scratch4 quota.
#   * H2 opens its 4.4 GB database /data/test_db.mv.db read-write, so the read-only SIF gets a FRESH writable
#     directory overlay on node-local disk per start (--overlay DIR; the DB is copied up on first write). Every start
#     is therefore the pristine EHR, and stop deletes it. Needs ~5 GB free on $TMPDIR.
#   * HAPI listens on 127.0.0.1:<internal port>; agents get a GET-only proxy (scripts/medagentbench_proxy.py) on
#     another free port, so no agent can write into the shared EHR (official MedAgentBench never executes POSTs).
set -u
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$P/venv/bin/python"
DIGEST="sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
DATA_ROOT="$(cd "$P" && timeout 60 "$PY" -c 'import paths; print(paths.data_root())')"
SIF="$DATA_ROOT/cache/medagentbench/medagentbench_${DIGEST:7:12}.sif"
BASE_TMP="${TMPDIR:-/tmp}"
STATE="${MAB_STATE_DIR:-$BASE_TMP/medagentbench_fhir_${SLURM_JOB_ID:-local}_$(id -u)}"

log() { echo "[medagentbench_server] $*" >&2; }
free_port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }

ensure_sif() {
    [ -s "$SIF" ] && return 0
    log "pulling docker://jyxsu6/medagentbench@$DIGEST -> $SIF (once)"
    mkdir -p "$(dirname "$SIF")" "$BASE_TMP/mab_apptainer_tmp"
    APPTAINER_TMPDIR="$BASE_TMP/mab_apptainer_tmp" apptainer pull --disable-cache "$SIF.part" \
        "docker://jyxsu6/medagentbench@$DIGEST" >&2 && mv "$SIF.part" "$SIF"
    rm -rf "$BASE_TMP/mab_apptainer_tmp"
    [ -s "$SIF" ] || { log "FATAL: pull failed"; exit 1; }
}

start() {
    command -v apptainer >/dev/null || { log "FATAL: apptainer not on PATH (compute nodes only)"; exit 1; }
    if [ -f "$STATE/env" ] && status >/dev/null 2>&1; then log "already running"; cat "$STATE/env"; return 0; fi
    ensure_sif
    avail_kb=$(df -Pk "$BASE_TMP" | awk 'NR==2{print $4}')
    [ "$avail_kb" -gt 6000000 ] || { log "FATAL: <6 GB free on $BASE_TMP (the overlay copies up a 4.4 GB DB)"; exit 1; }
    rm -rf "$STATE"; mkdir -p "$STATE/overlay"
    # MAB_PORT (10-17, H-GEN27): a fixed proxy port, so a watchdog restart keeps FHIR_API_BASE; default: a free port, as before
    HAPI_PORT=$(free_port); PORT=${MAB_PORT:-$(free_port)}
    while [ "$PORT" = "$HAPI_PORT" ]; do HAPI_PORT=$(free_port); done
    setsid apptainer run --containall --pwd /app --overlay "$STATE/overlay" \
        --env "SERVER_PORT=$HAPI_PORT,SERVER_ADDRESS=127.0.0.1,JAVA_TOOL_OPTIONS=-Xmx${MAB_JAVA_HEAP:-1g}" \
        "$SIF" > "$STATE/hapi.log" 2>&1 < /dev/null &
    echo $! > "$STATE/hapi.pid"
    setsid "$PY" "$P/scripts/medagentbench_proxy.py" --port "$PORT" --upstream "http://127.0.0.1:$HAPI_PORT" \
        > "$STATE/proxy.log" 2>&1 < /dev/null &
    echo $! > "$STATE/proxy.pid"
    t0=$(date +%s)
    until curl -sf "http://localhost:$PORT/fhir/metadata" -o /dev/null; do
        if ! kill -0 "$(cat "$STATE/hapi.pid")" 2>/dev/null; then
            log "FATAL: HAPI exited; tail of $STATE/hapi.log:"; tail -20 "$STATE/hapi.log" >&2; stop; exit 1
        fi
        if [ $(( $(date +%s) - t0 )) -gt "${MAB_START_TIMEOUT:-900}" ]; then log "FATAL: not up after timeout"; stop; exit 1; fi
        sleep 3
    done
    n=$(curl -s "http://localhost:$PORT/fhir/Patient?_summary=count&_format=json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["total"])')
    log "up in $(( $(date +%s) - t0 ))s on $(hostname): proxy :$PORT -> HAPI :$HAPI_PORT, $n patients; state $STATE"
    echo "export FHIR_API_BASE=http://localhost:$PORT/fhir/" | tee "$STATE/env"
}

status() {
    [ -f "$STATE/env" ] || { echo "not running (no $STATE/env)"; return 1; }
    # shellcheck disable=SC1090
    . "$STATE/env"
    if curl -sf "${FHIR_API_BASE}metadata" -o /dev/null; then echo "up: FHIR_API_BASE=$FHIR_API_BASE"; return 0; fi
    echo "state file present but server not answering at $FHIR_API_BASE"; return 1
}

stop() {
    for f in proxy hapi; do
        if [ -f "$STATE/$f.pid" ]; then
            pid=$(cat "$STATE/$f.pid")
            # setsid made each a process-group leader: kill the whole group (apptainer starter + java)
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        fi
    done
    sleep 5
    for f in proxy hapi; do
        [ -f "$STATE/$f.pid" ] && kill -KILL -- "-$(cat "$STATE/$f.pid")" 2>/dev/null
    done
    rm -rf "$STATE"
    log "stopped; removed $STATE"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: $0 start|status|stop" >&2; exit 2 ;;
esac
