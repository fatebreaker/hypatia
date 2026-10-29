"""HARNESS V3.1 TRAINER -- ELICIT-THEN-REWARD (docs/RL_PLAN.md s.18, registered 2026-09-26 16:35 UTC): rl/grpo_v3.py
EXACTLY (rl/grpo_onpolicy.py's trainer, the v2 harness installed, the s.15 process reward around its rollouts), with ONE
difference: the system prompt key named by GBP_SYSTEM may be one of rl/systems_extra.py's keys (science_check = the
science prompt + one sentence asking for a robustness re-estimate), registered into harness/agents.py's SYSTEMS here,
before any agent is made. harness/agents.py, rl/grpo_v3.py and rl/grpo_onpolicy.py are not edited.

DETECTOR (v2, 2026-09-26 ~20:30 UTC; docs/RL_PLAN.md s.18 entry below 19:15): under GBP_SYSTEM=science_check the process
reward that grpo_v3.install builds scores rollouts with rl/process_signals_v31.ProcessScorerV31 -- process_signals'
signals with spec := reest (a later-cell re-estimate of the same response under a new estimator / regressor set /
functional form / sample restriction) and the M3 check := max(M3, reest) -- instead of process_signals.ProcessScorer. The
reward composition (compose, the sham permutation, the cap, lambda) is grpo_v3's, unchanged; only the detector differs.
Every rl/grpo_v3.py run path keeps the old scorer (this file is the only place the swap happens).

    GBP_KERNEL=1 GBP_SYSTEM=science_check GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_v31.py \
        --process-lambda 0.3 [--process-sham perm] <every rl/grpo_onpolicy.py argument>

Launch through experiments/harness_v3/run_v31.sh (run dirs ...-sysscicheck-...-hv3pr<lambda>-...).

Written into --out beside grpo_v3's harness_v3.json: harness_v31.json = {system key, sha256 of the full prompt text, the
added sentence, this file's sha256, the detector name + rl/process_signals_v31.py's sha256} and system_prompt.txt = the full text the policy was sampled under (system.txt keeps
the KEY, which is what experiments/harness_v2/screen_any.sh reads). A resume under a different prompt text or a different
detector (including a v3.1 run dir written before the detector existed) is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

V31_VERSION = 2
DETECTOR_FOR = {"science_check": "reest-v2"}      # system key -> the process detector its v3.1 arms are scored with


def install_detector(pr, key: str):
    """Swap the process reward's scorer for the v3.1 detector of this system key (grpo_v3's reward object keeps its
    vocab_override, so the 'cols' sham still applies). Returns the scorer in use."""
    from rl.process_signals_v31 import DETECTOR, ProcessScorerV31
    if DETECTOR_FOR.get(key) != DETECTOR:
        raise SystemExit(f"rl/grpo_v31.py: no v3.1 detector registered for GBP_SYSTEM={key!r}")
    pr.scorer = ProcessScorerV31(vocab_override=pr.scorer.vocab_override)
    print(f"harness v3.1: process detector {DETECTOR} (spec := reest; M3 check := max(M3, reest))", flush=True)
    return pr.scorer


def prepare(out: Path | None) -> dict:
    """Register the extra keys into the `agents` module the trainer's rollouts use; check GBP_SYSTEM resolves; record."""
    from rl import systems_extra
    agents = systems_extra.register()
    key = os.environ.get("GBP_SYSTEM", "default")
    if key not in systems_extra.KEYS:
        raise SystemExit(f"rl/grpo_v31.py: GBP_SYSTEM={key!r} -- v3.1 trains under one of {systems_extra.KEYS} "
                         "(the plain v3 prompt is rl/grpo_v3.py's)")
    from rl import dispersion
    if sys.modules.get("agents") is not agents or dispersion.make_agent.__module__ != "agents":
        raise SystemExit("rl/grpo_v31.py: rl.dispersion does not use the agents module the key was registered in")
    text = agents.SYSTEMS[key]
    from rl.process_signals_v31 import detector_record
    rec = {"version": V31_VERSION, "system": key, "system_sha256": systems_extra.sha(text),
           "base_system": systems_extra.BASE_KEY, "sentence": systems_extra.CHECK_SENTENCE,
           "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], **detector_record()}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        f = out / "harness_v31.json"
        if f.exists():
            old = json.loads(f.read_text())
            if (old.get("system"), old.get("system_sha256")) != (rec["system"], rec["system_sha256"]):
                raise SystemExit(f"rl/grpo_v31.py: {out} was trained under {old.get('system')} "
                                 f"({old.get('system_sha256')}) -- refusing to resume it under {key} "
                                 f"({rec['system_sha256']})")
            if old.get("detector") != rec["detector"]:
                raise SystemExit(f"rl/grpo_v31.py: {out} was trained with process detector {old.get('detector')} "
                                 f"-- refusing to resume it with {rec['detector']}")
            if old.get("detector_sha256") != rec["detector_sha256"]:
                print(f"rl/grpo_v31.py: WARNING rl/process_signals_v31.py changed since this run started "
                      f"({old.get('detector_sha256')} -> {rec['detector_sha256']})", flush=True)
        else:
            f.write_text(json.dumps(rec, indent=1))
        (out / "system_prompt.txt").write_text(text)
    print(f"harness v3.1: system={key} sha256={rec['system_sha256']} (science + 1 sentence); detector "
          f"{rec['detector']} {rec['detector_sha256']}", flush=True)
    return rec


def out_of(rest: list) -> Path | None:
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    g, _ = pa.parse_known_args(rest)
    return Path(g.out) if g.out else None


def main(argv: list) -> int:
    from rl import grpo_v3
    v3args, rest = grpo_v3.parse_v3_args(argv)
    rec = prepare(out_of(rest))
    from rl import dispersion, grpo_onpolicy
    sys.argv = [sys.argv[0]] + rest
    pr = grpo_v3.install(grpo_onpolicy, dispersion, v3args, rest)
    install_detector(pr, rec["system"])
    return grpo_onpolicy.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
