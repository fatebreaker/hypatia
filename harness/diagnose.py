#!/usr/bin/env python
"""Diagnose why an agent executes zero code cells.

Two eval runs of Qwen3-8B scored 0/10 with n_cells=0 -- the model never emitted
a ```python block. That is either a model limitation or a harness bug, and the
two need different fixes. This isolates which.

D1  raw response shape -- is the text in message.content, or in a separate
    reasoning_content field the harness never reads? what is finish_reason?
D2  thinking disabled -- Qwen3 supports chat_template_kwargs.enable_thinking.
    If code blocks appear once thinking is off, the protocol is fine.
D3  non-reasoning model -- run the same prompt against Qwen2.5-7B-Instruct. If
    it emits code and Qwen3 does not, the cause is thinking mode, not the loop.

Usage:  diagnose.py --base-url http://127.0.0.1:PORT/v1 --model Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import SYSTEM, extract_code, strip_reasoning  # noqa: E402

PROMPT = """Estimate the mean of column `x` in data_files/toy.tsv.

Files in your working directory:
  data_files/toy.tsv  (40 bytes)

Return the final answer as one JSON object:
{"answer": {"mean_x": <float>}, "reasoning": "<method>"}"""


def call(base_url: str, model: str, *, thinking: bool | None = None,
         max_tokens: int = 4096) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def report(tag: str, resp: dict) -> bool:
    ch = resp["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    code = extract_code(content)

    print(f"\n----- {tag} -----")
    print(f"  message keys      : {sorted(msg)}")
    print(f"  finish_reason     : {ch.get('finish_reason')}")
    print(f"  len(content)      : {len(content)}")
    print(f"  len(reasoning_*)  : {len(reasoning)}")
    print(f"  completion_tokens : {resp.get('usage', {}).get('completion_tokens')}")
    print(f"  content has <think>: {'<think>' in content}")
    print(f"  after strip, chars : {len(strip_reasoning(content).strip())}")
    print(f"  ```python found   : {'YES' if code else 'no'}")
    if code:
        print(f"    code: {code.strip()[:120]!r}")
    else:
        print(f"    content tail: {content[-260:]!r}")
    return bool(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-model", action="store_true",
                    help="also run the enable_thinking=False probe (D2)")
    args = ap.parse_args()

    results = {}
    print("=" * 68)
    print(f"DIAGNOSTICS: {args.model}")
    print("=" * 68)

    results["default"] = report("D1 default (as the harness calls it)",
                                call(args.base_url, args.model))
    results["default_16k"] = report("D1b default, max_tokens=16384",
                                    call(args.base_url, args.model, max_tokens=16384))
    if args.reasoning_model:
        try:
            results["no_thinking"] = report(
                "D2 enable_thinking=False",
                call(args.base_url, args.model, thinking=False))
        except Exception as e:                                   # noqa: BLE001
            print(f"\n----- D2 enable_thinking=False -----\n  FAILED: {str(e)[:200]}")
            results["no_thinking"] = None

    print("\n" + "=" * 68)
    for k, v in results.items():
        print(f"  {k:<14} emitted code block: "
              f"{'YES' if v else ('no' if v is not None else 'n/a')}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
