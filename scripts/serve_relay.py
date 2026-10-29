#!/usr/bin/env python
"""Expose a pod's localhost-bound vLLM server to other nodes, without restarting it.

  serve_relay.py LISTEN_PORT TARGET_PORT [SECONDS]

The servers this programme runs bind 127.0.0.1 (see scripts/pod_serve.sh SERVE_HOST), so a worker on another node
cannot reach them. Restarting a server to rebind would kill the item using it. This relays instead: accept on
0.0.0.0:LISTEN_PORT and forward to 127.0.0.1:TARGET_PORT, so the running server is untouched.

Intended for PROVING the split-architecture path (GPU pod serves, CPU pods run the sandbox half) before committing
to it. For steady use, set SERVE_HOST=0.0.0.0 on a pod dedicated to serving rather than relaying into a pod that is
also running its own workers -- a relay adds remote load to a server that already has eight local clients.
"""
import socket, sys, threading, time

listen_port, target_port = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + (int(sys.argv[3]) if len(sys.argv) > 3 else 300)


def pipe(a, b, state):
    """Forward a->b, then HALF-CLOSE b's write side. Close both only when both directions are done.

    THE BUG THIS REPLACES (2026-09-21, found the hard way). The first version did
    `shutdown(SHUT_RDWR)` on BOTH sockets as soon as EITHER direction ended, and closed them. With a single
    request that is invisible -- the exchange is over anyway. Under 24 concurrent workers on HTTP keep-alive it
    tears down connections that still have a response in flight: every one of 240 rollouts came back
    `error:RemoteDisconnected`, in 134 seconds, and would have been recorded as a measurement if ga_table had not
    refused the file. A half-duplex close is what a proxy must do.
    """
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except Exception:
        pass
    try:
        b.shutdown(socket.SHUT_WR)      # tell the peer this direction is finished; leave the other alone
    except Exception:
        pass
    if state.pop() == 0:                # the second of the two directions to finish closes the pair
        for s in (a, b):
            try:
                s.close()
            except Exception:
                pass


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", listen_port))
srv.listen(256)
srv.settimeout(5)
print(f"relay 0.0.0.0:{listen_port} -> 127.0.0.1:{target_port} for {int(deadline-time.time())}s", flush=True)
while time.time() < deadline:
    try:
        c, _ = srv.accept()
    except socket.timeout:
        continue
    except Exception:
        break
    try:
        u = socket.create_connection(("127.0.0.1", target_port), timeout=10)
    except Exception:
        c.close(); continue
    state = [0, 1]                      # two entries; each direction pops one, the last one closes
    threading.Thread(target=pipe, args=(c, u, state), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c, state), daemon=True).start()
print("relay closed", flush=True)
