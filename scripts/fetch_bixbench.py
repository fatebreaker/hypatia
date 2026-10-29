"""Adapt BixBench into harness-compatible problems.

BixBench (FutureHouse, https://arxiv.org/abs/2503.00096) is 205 questions over
59 capsules of real biological data analysis. It is the external benchmark this
project otherwise lacks: GeneBench-Pro releases only 10 of 129 problems, and
everything else here is self-generated, so "we built a benchmark and did well
on it" was the strongest claim available. BixBench has published frontier
numbers to compare against.

THE TRAP -- READ BEFORE CHANGING THE EXTRACTION
----------------------------------------------
Each CapsuleFolder zip contains TWO directories:

    CapsuleData-<uuid>/        the input data
    CapsuleNotebook-<uuid>/    the original analysis, EXECUTED, with outputs

The notebook is the analysis that produced the answers. Unzipping the capsule
wholesale hands the agent the answer key. Measured on one capsule: **7 of 7
answers were recoverable** from numeric literals in the shipped notebook --
e.g. 157912.4 for the ideal range (157500, 158000), and 1.131e-11 for
(1.13e-10, 1.13e-12). It would not look like cheating; it would look like a
model that is unexpectedly good at bioinformatics.

So: only `CapsuleData-*/` is extracted, and `verify()` asserts that no
notebook, checkpoint or executed artefact reached data_files/. That assertion
is the point of this file -- if it ever fires, the scores before it were wrong.

GRADING
-------
Every question carries `ideal` plus exactly three distractors, so we grade the
4-way multiple-choice form with exact letter matching (see
benchmarks/bixbench/reference_grader.py for why not open-answer). Options are
shuffled deterministically from a hash of the question id, so the layout is
reproducible across machines but not guessable from position.

One problem per CAPSULE, not per question: several questions share one capsule,
and a problem per question would copy the same data up to 7 times and let the
agent re-derive the same analysis repeatedly. The grader scores each question
as its own key.

Run:
  venv/bin/python scripts/fetch_bixbench.py            # all 59 capsules (~6 GB)
  venv/bin/python scripts/fetch_bixbench.py --limit 3  # smoke test
  venv/bin/python scripts/fetch_bixbench.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT  # noqa: E402

REPO = "futurehouse/BixBench"
MANIFEST = "BixBench.jsonl"
OUT = PROJECT_ROOT / "benchmarks" / "bixbench"
LETTERS = "ABCD"

# Anything matching these must never reach data_files/. Kept broad on purpose:
# a new artefact name is far more likely than a deliberate change here.
FORBIDDEN = ("CapsuleNotebook", ".ipynb", ".ipynb_checkpoints", "_executed")


def load_manifest(cache: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, MANIFEST, repo_type="dataset", local_dir=str(cache))
    return [json.loads(l) for l in open(p)]


def shuffled_options(row: dict) -> tuple[list[str], str]:
    """Deterministic option order + the correct letter.

    Seeded from the question id so two machines build identical problems, while
    the correct answer is not pinned to a fixed position (always-'A' would be
    trivially farmable).
    """
    opts = [str(row["ideal"])] + [str(d) for d in (row.get("distractors") or [])]
    seed = int(hashlib.sha256(row["question_id"].encode()).hexdigest()[:16], 16)
    order = list(range(len(opts)))
    # Fisher-Yates driven by the seed; stdlib random would also work but this
    # keeps the shuffle independent of Python's RNG implementation.
    for i in range(len(order) - 1, 0, -1):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**64)
        j = seed % (i + 1)
        order[i], order[j] = order[j], order[i]
    shown = [opts[k] for k in order]
    return shown, LETTERS[order.index(0)]


def build_task(rows: list[dict], files: list[str]) -> tuple[str, dict, dict]:
    lines = [
        "You are given the data files from a published biological analysis.",
        "Answer every question below by computing the result from the data.",
        "",
        "Files in data_files/:",
    ]
    lines += [f"  {f}" for f in sorted(files)]
    lines.append("")

    gt: dict[str, str] = {}
    keys: dict[str, dict] = {}
    for i, r in enumerate(rows, 1):
        shown, correct = shuffled_options(r)
        gt[r["question_id"]] = correct
        keys[r["question_id"]] = {"n_options": len(shown)}
        lines.append(f"Q{i} [{r['question_id']}]: {r['question'].strip()}")
        for letter, opt in zip(LETTERS, shown):
            lines.append(f"    {letter}) {opt}")
        lines.append("")

    lines += [
        "Return ONE JSON object mapping every question id to the letter of the",
        "option you chose. Every key is required.",
        "",
        "{",
        '  "answer": {',
    ]
    lines += [f'    "{qid}": "<A|B|C|D>",' for qid in list(gt)[:-1]]
    lines.append(f'    "{list(gt)[-1]}": "<A|B|C|D>"')
    lines += ["  }", "}"]
    return "\n".join(lines), gt, keys


def extract_capsule(zpath: Path, dest: Path) -> list[str]:
    """Extract ONLY CapsuleData-*/ members, flattened into dest."""
    written: list[str] = []
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            if not parts or not parts[0].startswith("CapsuleData-"):
                continue
            if any(bad in member for bad in FORBIDDEN):
                continue
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(str(rel))
    return written


def verify(root: Path, expect_questions: int = 0) -> int:
    """Assert no answer-bearing artefact leaked, and that coverage is complete."""
    probs = sorted(p for p in (root / "problems").glob("*") if (p / "eval_config.json").is_file())
    if not probs:
        print("verify: FAIL -- no problems built")
        return 1
    leaked, empty = [], []
    for p in probs:
        d = p / "data_files"
        files = [f for f in d.rglob("*") if f.is_file()] if d.is_dir() else []
        if not files:
            empty.append(p.name)
        for f in files:
            if any(bad in str(f) for bad in FORBIDDEN):
                leaked.append(str(f.relative_to(root)))
    n_q = sum(len(json.loads((p / "eval_config.json").read_text())["ground_truth"])
              for p in probs)
    print(f"problems: {len(probs)}   questions: {n_q}")
    if empty:
        print(f"  WARN  {len(empty)} problems have no data files: {empty[:5]}")
    if leaked:
        print(f"verify: FAIL -- {len(leaked)} answer-bearing files leaked into data_files/")
        for f in leaked[:10]:
            print(f"    {f}")
        return 1
    # Coverage, not just cleanliness: a name collision silently merged capsules
    # and dropped questions once already, and nothing errored when it did.
    if expect_questions and n_q != expect_questions:
        print(f"verify: FAIL -- {n_q} questions built, manifest has {expect_questions}")
        return 1
    print("verify: PASS -- no notebooks or executed artefacts in any data_files/"
          + (f"; all {expect_questions} manifest questions present" if expect_questions else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only N capsules (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.out)
    if args.verify_only:
        try:
            n = len(load_manifest(root / ".hf"))
        except Exception:                                          # noqa: BLE001
            n = 0            # offline: still check leakage, just not coverage
        return verify(root, n)

    cache = root / ".hf"
    root.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(cache)

    by_capsule: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_capsule[r["capsule_uuid"]].append(r)
    caps = sorted(by_capsule.items(), key=lambda kv: kv[1][0]["short_id"])

    # short_id is NOT unique: in v1.5 `bix-61` maps to SIX distinct capsules.
    # Naming problem dirs by it merged them into one and silently dropped 5
    # questions -- no warning, no error, just 200 of 205. Disambiguate the
    # colliding ones with a uuid prefix and leave the rest readable.
    shared = Counter(qs[0]["short_id"] for _, qs in caps)
    names = {
        uuid: (qs[0]["short_id"] if shared[qs[0]["short_id"]] == 1
               else f"{qs[0]['short_id']}__{uuid[:8]}")
        for uuid, qs in caps
    }
    if args.limit:
        caps = caps[: args.limit]
    print(f"{len(rows)} questions over {len(by_capsule)} capsules; building {len(caps)}")
    dup = {k: v for k, v in shared.items() if v > 1}
    if dup:
        print(f"  note: {len(dup)} short_id(s) map to multiple capsules "
              f"({', '.join(f'{k} x{v}' for k, v in dup.items())}); disambiguated by uuid")

    from huggingface_hub import hf_hub_download

    built = skipped = 0
    for uuid, qs in caps:
        qs.sort(key=lambda r: r["question_id"])
        sid = names[uuid]
        pdir = root / "problems" / sid
        data = pdir / "data_files"
        if (pdir / "eval_config.json").is_file() and any(data.rglob("*")):
            skipped += 1
            continue
        try:
            z = hf_hub_download(REPO, qs[0]["data_folder"], repo_type="dataset",
                                local_dir=str(cache))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  WARN {sid}: capsule download failed ({type(exc).__name__})")
            continue
        if data.exists():
            shutil.rmtree(data)
        data.mkdir(parents=True, exist_ok=True)
        files = extract_capsule(Path(z), data)
        if not files:
            print(f"  WARN {sid}: no CapsuleData-* members; skipping")
            shutil.rmtree(pdir, ignore_errors=True)
            continue
        task, gt, keys = build_task(qs, files)
        cfg = {
            "id": f"bixbench_{sid}",
            "task": task,
            "data_files": sorted(files),
            "grader": {"type": "multi_choice_exact", "config": {"keys": keys}},
            "ground_truth": gt,
            "_provenance": {
                "benchmark": "BixBench",
                "capsule_uuid": uuid,
                "short_id": sid,
                "paper": qs[0].get("paper"),
                "categories": qs[0].get("categories"),
                # BixBench's contamination canary. Kept OUT of `task` so it is
                # never shown to the model, but recorded so a corpus can be
                # grepped for it.
                "canary": qs[0].get("canary"),
                "eval_modes": sorted({q.get("eval_mode", "") for q in qs}),
            },
        }
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=2))
        built += 1
        print(f"  {sid:<10} {len(qs)} questions, {len(files)} data files")

    # Drop problem dirs that no longer correspond to a capsule. Without this a
    # rename leaves the old directory behind and it is still a valid problem to
    # the runner -- which is how a stale `bix-61` survived the collision fix and
    # inflated the question count past the manifest.
    if not args.limit:
        want = set(names.values())
        for stale in sorted((root / "problems").glob("*")):
            if stale.is_dir() and stale.name not in want:
                print(f"  pruning stale problem dir: {stale.name}")
                shutil.rmtree(stale, ignore_errors=True)

    print(f"\nbuilt {built}, skipped {skipped} already present")
    # Coverage is only meaningful over the full set; --limit builds a subset.
    return verify(root, 0 if args.limit else len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
