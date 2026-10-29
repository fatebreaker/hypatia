#!/bin/bash
# Assert that the endpoint at $1 is OURS and serves exactly the adapters we
# expect ($2... = names). Exit non-zero otherwise.
#
# Written after an eval batch returned 320 RuntimeErrors with 0 cells executed:
# ports 8000/8011 are hardcoded in this repo, and a co-tenant on this shared
# box was already serving port 8011 with THEIR model. Our request failed
# because our adapter names were absent -- but had the names happened to
# match, we would have silently attributed another user's model's outputs to
# our checkpoints. A dead port is recoverable; a wrong-but-live port is a
# fabricated result.
#
# So: never run an eval without asserting the endpoint serves what we think.
URL=$1; shift
BODY=$(curl -s -m 15 "$URL/models" 2>/dev/null) || { echo "assert_served: $URL unreachable"; exit 1; }
for want in "$@"; do
  echo "$BODY" | grep -q "\"id\":\"$want\"" || {
    echo "assert_served: $URL does NOT serve '$want'"
    echo "  it serves: $(echo "$BODY" | grep -oE '"id":"[^"]+"' | cut -d'"' -f4 | tr '\n' ' ')"
    exit 1; }
done
echo "assert_served: $URL OK -> $*"
