"""rl/grpo_onpolicy.py inside harness v2 (docs/HARNESS_V2.md s.6): the trainer, unchanged, with the components the
GBP_* switches enable installed into its rollouts -- the KERNEL/PROFILE/APIDOCS/CHECKS sandbox (rl.dispersion's Sandbox,
which one_rollout builds), and per episode the LIBRARY (kept in the stored episode with n_prefix, so train() masks it out
of the gradient exactly as R3's --library) and SIBLINGS (one draw per GRPO group and step, as rl/grpo_siblings.py).

    GBP_KERNEL=1 GBP_PROFILE=1 ... venv_vllm/bin/python rl/grpo_v2.py <every rl/grpo_onpolicy.py argument>

A NEW entry point, like rl/grpo_siblings.py: rl/grpo_onpolicy.py (imported by running conditions) is untouched. Launch
through experiments/harness_v2/run_v2.sh, which names the run directory -hv2<tag> and writes harness_v2.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

if __name__ == "__main__":
    import v2
    from rl import dispersion, grpo_onpolicy
    hc = v2.HarnessConfig.from_env()
    if not hc.any():
        print("rl/grpo_v2.py: no harness-v2 switch is set -- this is rl/grpo_onpolicy.py", flush=True)
    if hc.library and "--library" in sys.argv:
        raise SystemExit("rl/grpo_v2.py: GBP_LIBRARY and --library both put a library in context; use one")
    if hc.siblings and any(a.startswith("--siblings") for a in sys.argv):
        raise SystemExit("rl/grpo_v2.py: use GBP_SIBLINGS, not --siblings (that is rl/grpo_siblings.py)")
    v2.install_grpo(grpo_onpolicy, dispersion, hc)
    print(v2.banner(hc), flush=True)
    raise SystemExit(grpo_onpolicy.main())
