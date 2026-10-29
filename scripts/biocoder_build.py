"""biocoder_build.py (10-17, H-GEN27 clean training data) -- BioCoder TRAINING tasks from MedAgentGym's BioCoder training split
(external/MedAgentGym/data/biocoder/train_tasks.jsonl, 981 records: problem, signature, context program with the line
<<insert solution here>>, reference solution). BioCoder is not one of our test benchmarks.

A record becomes a task only if it can give a clean reward offline, here:
  * the insertion line stands alone at top level (the solution is a whole function; 685 records), the reference program
    compiles, nothing in it names a web service (requests / urllib / http:// / socket; MedAgentGym's grader would call
    them live), and nothing deletes, moves, renames or re-permissions files or shells out (it runs on a shared cluster);
  * the reference program (context + reference solution) exits 0 under the grader (scripts/biocoder_reference_grader.py:
    our analysis venv, no network, 120 s) FIVE times with identical, non-empty stdout of at most 20,000 characters and no
    memory addresses -- twice as graded, three times with the heap shifted (once with the scratch directory on another filesystem) (v2,
    10-17: v1 checked two plain runs only, and bioc-386, which prints a set of objects in address order, flipped in another
    process; v2's padding allocated only object() instances, a different allocator size class, and kept bioc-386: v3 pads with
    objects of many sizes and a larger environment) -- that stdout is the expected output;
  * a STUB function with the same name (returns None) does NOT reproduce it, and neither does the program with NOTHING
    inserted, and the context does not define the function itself (v4, 10-17: bioc-588's context already held the solution,
    so any insertion passed) -- the output depends on the agent's function.
Output: generated/biocoder4/train/bioc-<idx>/eval_config.json (grader type biocoder_stdout) + MANIFEST.json.
    venv/bin/python scripts/biocoder_build.py [--workers 6]
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
SRC = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/external/MedAgentGym/data/biocoder/train_tasks.jsonl")
OUT = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated/biocoder4")   # v4 (biocoder, biocoder2, biocoder3: weaker checks)
SHM = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
spec = importlib.util.spec_from_file_location("biocoder_grader", R / "scripts" / "biocoder_reference_grader.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
NET = re.compile(r"requests|urllib|https?://|socket")
# third-party programs run on a shared cluster: nothing that deletes, moves, renames, re-permissions files or shells out
DANGER = re.compile(r"subprocess|os\.system|os\.popen|os\.exec|os\.spawn|shutil\.rmtree|shutil\.move|os\.remove|os\.unlink|"
                    r"os\.rmdir|os\.removedirs|os\.rename|os\.replace|os\.chmod|os\.chown|os\.kill|\bctypes\b|\bpty\b")
ADDR = re.compile(r"0x[0-9a-fA-F]{8,}")

TASK = """You are given a bioinformatics coding task (from BioCoder).

Task:
{problem}

Write the function with this signature:
    {signature}

Your function is inserted into the program below in place of the line `<<insert solution here>>`; the program is then run
in a FRESH Python process and graded by what it prints, which must equal what it prints with a correct implementation:
```python
{context}
```

What to deliver. Develop and test your function in the notebook as usual (you can run the program above with your function
inserted). Your FINAL deliverable is the function, with any imports or helpers it needs, saved as the file solution.py in
your working directory, for example with
    SOLUTION = r'''<your code>'''
    open("solution.py", "w").write(SOLUTION)
solution.py is pasted into the program in place of that line, so it must contain only the function (and what it needs), not
the program itself. Available packages include pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib and seaborn.

Return your answer as JSON: {{"answer": {{"{key}": "solution.py"}}}}"""


def check(r):
    ctx, sol = r["context"], r["solution"]
    if len(grader._SLOT.findall(ctx)) != 1:
        return None, "insertion line not alone at top level"
    if NET.search(ctx) or NET.search(sol):
        return None, "names a web service"
    if DANGER.search(ctx) or DANGER.search(sol):
        return None, "deletes/moves files or shells out"
    try:
        compile(grader.program(ctx, sol), "program", "exec")
    except SyntaxError as e:
        return None, f"reference does not compile: {e}"
    m = re.search(r"def\s+(\w+)\s*\(", r["signature"])
    if not m:
        return None, "no function name in the signature"
    if re.search(r"def\s+" + re.escape(m.group(1)) + r"\s*\(", ctx):
        return None, "the context already defines the function"
    a, b = grader.run_program(ctx, sol), grader.run_program(ctx, sol)
    rr = random.Random(r["idx"])
    c = grader.run_program(ctx, sol, pad=rr.randint(1000, 50000), workroot=SHM)
    d = grader.run_program(ctx, sol, pad=rr.randint(1, 999))
    e = grader.run_program(ctx, sol, pad=rr.randint(1000, 9999))
    if a["returncode"] != 0 or a["timeout"]:
        return None, f"reference fails here: rc {a['returncode']} {a['stderr_tail'][-300:]}"
    if len({a["stdout"], b["stdout"], c["stdout"], d["stdout"], e["stdout"]}) > 1:
        return None, "reference output not deterministic"
    out = a["stdout"]
    if not out.strip() or len(out) > 20000 or ADDR.search(out):
        return None, "reference output empty, too long or holds memory addresses"
    stub = grader.run_program(ctx, f"def {m.group(1)}(*args, **kwargs):\n    return None\n")
    if stub["returncode"] == 0 and not stub["timeout"] and stub["stdout"].rstrip() == out.rstrip():
        return None, "a stub function reproduces the output"
    empty = grader.run_program(ctx, "pass\n")
    if empty["returncode"] == 0 and not empty["timeout"] and empty["stdout"].rstrip() == out.rstrip():
        return None, "the program passes with nothing inserted"
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if (OUT / "train").exists():
        sys.exit(f"{OUT / 'train'} exists -- not overwriting")
    recs = [json.loads(line) for line in open(SRC)]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, recs))
    (OUT / "train").mkdir(parents=True)
    kept, why = [], {}
    for r, (out, err) in zip(recs, res):
        if out is None:
            k = err.split(":")[0]
            why[k] = why.get(k, 0) + 1
            continue
        pid = f"bioc-{r['idx']}"
        key = f"{pid}-q1"
        pdir = OUT / "train" / pid
        pdir.mkdir()
        cfg = {"id": pid, "task": TASK.format(problem=r["problem"].strip(), signature=r["signature"].strip(),
                                              context=r["context"].rstrip(), key=key),
               "data_files": [], "ground_truth": {key: r["solution"]},
               "grader": {"type": "biocoder_stdout", "config": {"keys": {key: {"context": r["context"], "expected_stdout": out}}}},
               "_provenance": {"benchmark": "BioCoder TRAINING task (H-GEN27), MedAgentGym's BioCoder training split",
                               "medagentgym_idx": r["idx"], "built": "2026-10-17"}}
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        kept.append(pid)
    json.dump({"built": "2026-10-17", "source": str(SRC), "n_source": len(recs), "n_tasks": len(kept), "excluded": why},
              open(OUT / "MANIFEST.json", "w"), indent=1)
    print(f"{len(recs)} records -> {len(kept)} tasks in {OUT}; excluded: {why}")


if __name__ == "__main__":
    main()
