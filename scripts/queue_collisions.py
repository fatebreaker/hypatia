#!/usr/bin/env python3
"""Two queue items that collide corrupt results silently. Run this before queueing anything.

On 2026-09-17 two items ran experiments/e1/indist.sh under the same pod_exec step name (`e1_indist`). Step names are
global: pod_exec.sh writes logs/pods/<NAME>.{log,cmd.sh,done,failed} and each pod's srun executes <NAME>.cmd.sh, so the
second item's command file overwrote the first's, both pods ran the second's argument list, and both wrote the same
results/pods/e1/<arm>/in_<fam>.json. The surviving run finally died with

    logs/pods/e1_indist.cmd.sh: line 6: _k7b: command not found

-- bash resuming mid-token in a script another writer had rewritten underneath it. One evaluation read 0.646 and then
0.146 for the same arm and family: about 10 sigma apart for 96 rollouts, so structural, not sampling. Sixteen results
had to be quarantined and re-scored, and a paragraph of the paper had to be rewritten.

Two rules, and the first alone is NOT enough:
  1. every item's pod_exec step names must be unique to it (suffix $JS, as serveQ_$JS does);
  2. no two items may write the same output directory -- that is what actually corrupts results.

  venv/bin/python scripts/queue_collisions.py            # live items only (what matters before queueing)
  venv/bin/python scripts/queue_collisions.py --all      # every item ever queued, for a post-mortem
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

STEP = re.compile(r"pod_exec\.sh\s+\$J\s+\S+\s+([A-Za-z0-9_${}]+)")
# the directory an item writes: results/pods/<dir> plus the <fam>lib_s<seed> / <arm> leaf where the script forms it
OUT = re.compile(r"results/pods/([A-Za-z0-9_]+)")
# Two items may share a parent directory and still be independent, because the scripts form a leaf per arm, corpus or
# model (results/pods/e1/<CORPUS>, results/pods/third_domain_fam/<fam>lib_s<seed>). Capture the leaf the way each
# script forms it -- from the ARM/CORPUS names it is given, NOT from a bare SEED=, which is shared by unrelated items
# and produced a false positive between e1_k1_astb and e2_b5_varb on 2026-09-17.
LEAF_PATTERNS = (
    re.compile(r"experiments/e1/run\.sh\s+(\S+)"),                 # results/pods/e1/<CORPUS>
    re.compile(r"experiments/e1/(?:bench_only|indist)\.sh\s+\$P\s+([^>]+)"),   # one leaf per ARM listed
    re.compile(r"third_domain/sft_family\.sh\s+(\w+)"),            # third_domain_fam/<fam>lib_s<SEED>
    re.compile(r"general_agent/run(?:_model)?\.sh\s+(\S+)"),
    re.compile(r"rl_foothold/(?:run|eval)\.sh\s+(\S+)"),
    re.compile(r"FAMS=\"([^\"]+)\""),
    # qrdata_turns.sh writes one leaf per TAG argument (".../qrdata_<tag>"), so two items sharing its parent
    # directory but naming different tags do not collide -- 41a/41b tripped this on 2026-09-17.
    re.compile(r"qrdata_turns\.sh\s+\w+\s+\S+\s+([^>]+)"),
    re.compile(r"QR_OUT=(\S+)"),
    re.compile(r"SEED=(\d+)"),                                     # only ever a qualifier, never a leaf on its own
)


OUTDIR = re.compile(r"--outdir\s+(\S+)")


def live(items: list[Path]) -> list[Path]:
    return [p for p in items if subprocess.run(["pgrep", "-f", p.name], capture_output=True).returncode == 0]


# ---- what a general_agent item will actually write -------------------------------------------------------------
# experiments/general_agent/run.sh writes results/pods/${OUT_DIR:-general_agent}/<armtag>_<fam>.json, and
# run_model.sh the same with OUT_DIR defaulting to general_agent_<FAM> (its first positional argument). None of that
# appears in the ITEM, which only sets env vars -- so the generic "results/pods/<dir>" regex below matches nothing
# and these items were invisible to the output check. They are also the items most likely to collide, because an arm
# is named by a short tag that several items reuse.
GA_RUN = re.compile(r"experiments/general_agent/run\.sh")
GA_MODEL = re.compile(r"experiments/general_agent/run_model\.sh\s+(\w+)")
GA_OUTDIR = re.compile(r"OUT_DIR=([A-Za-z0-9_]+)")
GA_ARMS = re.compile(r'ARMS="([^"]+)"')
GA_FAMS = re.compile(r'FAMS="([^"]+)"')
DEF_FAMS_RUN = "eco_size osc_damped hc_quality pk_oral eco_mb hc_dx"
DEF_FAMS_MODEL = "eco_size hc_dx"


def general_agent_files(txt: str) -> set[str]:
    """The exact result files a general_agent item will write, so two writers to one file are caught by name."""
    model = GA_MODEL.search(txt)
    if not model and not GA_RUN.search(txt):
        return set()
    od = GA_OUTDIR.search(txt)
    if od:
        out = od.group(1)
    elif model:
        out = f"general_agent_{model.group(1)}"
    else:
        out = "general_agent"
    arms = GA_ARMS.search(txt)
    # an arm is "<tag>:<agent spec>"; the tag alone names the file
    tags = [a.split(":", 1)[0] for a in arms.group(1).split()] if arms else ["selfver", "control"]
    fams = GA_FAMS.search(txt)
    fl = fams.group(1).split() if fams else (DEF_FAMS_MODEL if model else DEF_FAMS_RUN).split()
    return {f"results/pods/{out}/{t}_{f}.json" for t in tags for f in fl}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None, help="pod_queue directory (default: $SCRATCHPAD/pod_queue)")
    ap.add_argument("--all", action="store_true", help="every item, not just the ones running now")
    a = ap.parse_args()
    root = Path(a.queue) if a.queue else next(Path("/scratch/login").glob("claude-*/*/*/scratchpad/pod_queue"), None)
    if root is None or not root.is_dir():
        print("no pod_queue found; pass --queue")
        return 2
    every = [p for p in sorted(root.rglob("*.sh")) if p.name != "_lib.sh"]
    if a.all:
        items = every
        what = "(all, including finished)"
    else:
        # The useful comparison is QUEUED items against the ones RUNNING NOW -- a new item collides with work already
        # in flight, not with work that finished last week. Comparing only the live ones (the old default) could
        # never catch the case the check exists for: on 2026-09-18 a queued item targeted selfver7_causal_rdd.json
        # while a running item was 70/96 rollouts into writing it, and this reported "no collisions".
        started = root / "started"
        items = [p for p in every if started not in p.parents] + live([p for p in every if started in p.parents])
        what = "queued + running now"
    print(f"{len(items)} item(s) {what}")

    steps: dict[str, list[str]] = defaultdict(list)
    outs: dict[str, set[str]] = defaultdict(set)
    files: dict[str, set[str]] = defaultdict(set)
    for p in items:
        # A worker can CLAIM an item between the listing above and this read -- it mv's the file from
        # pod_queue/<kind>/ into pod_queue/started/ -- and the checker then dies with FileNotFoundError on a queue
        # that is simply being drained normally. The tool that guards every queue edit must not crash because the
        # queue moved while it looked (2026-09-18).
        try:
            txt = p.read_text()
        except FileNotFoundError:
            continue
        for m in STEP.finditer(txt):
            n = m.group(1)
            if "$J" in n:                      # already unique per job
                continue
            steps[n].append(p.name)
        # a leaf makes two items writing one parent distinguishable; a SEED alone is not a leaf, so it counts only
        # when some other pattern also matched (otherwise every "SEED=2" item would look like every other one)
        found = [m.group(1).strip() for pat in LEAF_PATTERNS[:-1] for m in pat.finditer(txt)]
        if found:
            found += [m.group(1) for m in LEAF_PATTERNS[-1].finditer(txt)]
        leaf = "|".join(sorted({w for f in found for w in f.split()}))
        precise = bool(OUTDIR.search(txt)) or bool(general_agent_files(txt))
        if not precise:
            for m in OUT.finditer(txt):
                outs[f"{m.group(1)}::{leaf}"].add(p.name)
        for f in general_agent_files(txt):
            files[f].add(p.name)
        # Benchmark items (harness/runner.py) name their destination with --outdir, one directory per arm. Two such
        # items sharing only the PARENT (results/pods/dabench_v8/dabench_selfver7 vs .../dabench_selfver10) do not
        # collide, but the generic results/pods/<dir> regex above sees one shared directory and warns. Capturing the
        # full --outdir makes these items precise instead of noisy -- and catches the real case, two items writing
        # one arm directory.
        for m in OUTDIR.finditer(txt):
            files[m.group(1).strip().rstrip("/")].add(p.name)

    bad = 0
    dup = {n: sorted(set(v)) for n, v in steps.items() if len(set(v)) > 1}
    if dup:
        bad += len(dup)
        print("\nSTEP NAME used by more than one item (shared logs/pods/<NAME>.cmd.sh -- the file srun executes):")
        for n, v in sorted(dup.items()):
            print(f"  {n:<30} {v}")
    ov = {k: sorted(v) for k, v in outs.items() if len(v) > 1}
    if ov:
        bad += len(ov)
        print("\nOUTPUT directory written by more than one item (check the leaf: a shared parent alone is fine):")
        for k, v in sorted(ov.items()):
            print(f"  results/pods/{k.replace('::', '  leaf=')}\n      {v}")
    fv = {k: sorted(v) for k, v in files.items() if len(v) > 1}
    if fv:
        bad += len(fv)
        print("\nRESULT FILE written by more than one item (this is the one that corrupts numbers):")
        for k, v in sorted(fv.items()):
            print(f"  {k}\n      {v}")
    if not bad:
        print("\nno collisions")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
