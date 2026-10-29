"""Multi-dataset episodes: the target instance plus K unlabeled SIBLING instances of the same family in the sandbox.

docs/DIRECTIONS.md A3 (H5 in docs/RL_PLAN.md s.11, 2026-09-24 late). The judge result that motivates it (item 454: an 8B
reading a program's applications to a dozen datasets separates general from lucky programs at AUC 0.94) moved from a
post-hoc pipeline into the episode: the agent can run its analysis on further datasets from the same design and see
whether the results are sensible and consistent before answering. Reward is unchanged -- the grader scores the TARGET
only; siblings differ in truth by construction, so averaging across datasets is not a way to pass.

Nothing in harness/ or rl/dispersion.py changes. `wrap(one_rollout, k)` returns a drop-in replacement for
rl.dispersion.one_rollout that, per call:
  * picks k siblings: other instances of the target's family in the target's own split directory (train while
    training, test while evaluating) -- of the same sub-design (instance-id prefix) when the directory holds several,
    as popgen_pulse's pg_0_* ... pg_4_* do -- by a seeded draw keyed on (target, key) -- `key` is the policy spec during RL, so
    every rollout of a GRPO group sees the same siblings and a new step draws new ones; "eval" in evaluation, so the
    base and every checkpoint see the same siblings for a target;
  * builds a composite data directory (a temp dir of symlinks: the target's own files at the top, each sibling's
    data_files under siblings/sibling_<i>/) -- the sandbox copies it (following the links) as data_files/, so the
    target's files keep their names and paths and no sibling ever exposes an eval_config (no labels);
  * appends SIB_NOTE to the task text (so the stored episode, and the RL gradient, carry the prompt the policy saw);
  * records `siblings` (their instance names) and a usage measure on the rollout: `sib_ref` (the agent's code touched
    data_files/siblings) and `sib_n` (distinct sibling_<i> it named, or k when it iterated over the directory).
"""
from __future__ import annotations

import hashlib
import random
import re
import shutil
import tempfile
from pathlib import Path

SIB_NOTE = ("\n\nFurther datasets. data_files/siblings/sibling_1/ ... data_files/siblings/sibling_{k}/ hold {k} more "
            "datasets from the same study design: the same files and layout as yours, different data (their true values "
            "differ from yours), no answers. You may use them to check your method -- a correct analysis should run "
            "cleanly and give plausible results on each of them. The question is about YOUR dataset (the files directly "
            "in data_files/): answer for it alone.")


def note(k: int) -> str:
    return SIB_NOTE.format(k=k)


def with_note(task: str, k: int) -> str:
    return task + note(k)


_LISTING: dict[Path, list[Path]] = {}      # split dir -> its instances (read once per process: 48 rollouts per step)


def _instances(split_dir: Path) -> list[Path]:
    if split_dir not in _LISTING:
        _LISTING[split_dir] = sorted(p for p in split_dir.iterdir()
                                     if (p / "data_files").is_dir() and (p / "eval_config.json").is_file())
    return _LISTING[split_dir]


def pick(target: Path, k: int, key: str) -> list[Path]:
    """k siblings of `target` (an instance dir, possibly a pool symlink): other instances in its family's split dir."""
    real = Path(target).resolve()
    pool = [p for p in _instances(real.parent) if p.name != real.name]
    # the same sub-design when a family dir holds several (popgen_pulse: pg_0_*, pg_1_*, ...): same id prefix
    same = [p for p in pool if p.name.rsplit("_", 1)[0] == real.name.rsplit("_", 1)[0]]
    pool = same if len(same) >= k else pool
    if len(pool) < k:
        raise RuntimeError(f"{real.parent}: {len(pool)} possible siblings for {real.name}, {k} wanted")
    seed = int(hashlib.sha256(f"{real.parent}|{real.name}|{key}".encode()).hexdigest()[:16], 16)
    return random.Random(seed).sample(pool, k)


def composite(data_dir: Path, sibs: list[Path], root: str | None = None) -> Path:
    """A temp dir: the target's data files (symlinked) + siblings/sibling_<i> -> <sibling>/data_files."""
    d = Path(tempfile.mkdtemp(prefix="sib_", dir=root))
    for p in sorted(Path(data_dir).iterdir()):
        if p.name == "siblings":
            raise RuntimeError(f"{data_dir} already has a 'siblings' entry")
        (d / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    s = d / "siblings"
    s.mkdir()
    for i, sib in enumerate(sibs, 1):
        (s / f"sibling_{i}").symlink_to((Path(sib) / "data_files").resolve(), target_is_directory=True)
    return d


_SIB_ANY = re.compile(r"siblings")
_SIB_ONE = re.compile(r"sibling_(\d+)")


def usage(out: dict, k: int) -> tuple[bool, int]:
    """(touched the siblings at all, how many distinct siblings) from the agent's own turns."""
    code = "\n".join(m.get("content") or "" for m in (out.get("messages") or []) if m.get("role") == "assistant")
    if not _SIB_ANY.search(code):
        return False, 0
    named = {int(x) for x in _SIB_ONE.findall(code) if 1 <= int(x) <= k}
    # a loop over the directory (glob/listdir/iterdir of data_files/siblings) names no index but reaches every sibling
    iterated = re.search(r"(glob|listdir|iterdir|scandir|walk)\([^)]*siblings", code) is not None
    return True, (k if iterated else len(named))


def wrap(one_rollout, k: int, key: str | None = None, tmp_root: str | None = None):
    """A drop-in replacement for rl.dispersion.one_rollout with k siblings mounted. key=None -> the policy spec."""
    if k <= 0:
        return one_rollout

    def sibling_rollout(cfg: dict, data_dir: Path, spec: str, *a, **kw) -> dict:
        target = Path(data_dir).parent
        sibs = pick(target, k, key if key is not None else spec)
        d = composite(Path(data_dir), sibs, tmp_root)
        try:
            out = one_rollout({**cfg, "task": with_note(cfg["task"], k)}, d, spec, *a, **kw)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        out["siblings"] = [s.name for s in sibs]
        out["sib_ref"], out["sib_n"] = usage(out, k)
        return out

    sibling_rollout.siblings_k = k
    return sibling_rollout


def strip_flag(argv: list[str], flag: str = "--siblings") -> tuple[list[str], int]:
    """Remove `--siblings K` / `--siblings=K` from argv (the wrapped script does not know it); returns (argv, K)."""
    out, k, i = [], 0, 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            k = int(argv[i + 1]); i += 2; continue
        if a.startswith(flag + "="):
            k = int(a.split("=", 1)[1]); i += 1; continue
        out.append(a); i += 1
    return out, k
