"""reference_grader.py of the B-QR4 teacher wrappers (benchmarks/qr_teach_1..4, scripts/build_qr_teach.py; docs/RL_PLAN.md
2026-10-24 15:14 UTC). The wrappers mix three training pools, each graded by its own vendored grader -- the same files
rl/reward.py uses for the RL reward -- so this module dispatches on the problem's grader type:
  multi_numeric_tolerance / numeric_tolerance / composite -> benchmarks/genebench-pro-public/reference_grader.py (causal_mix)
  qrdata_official -> benchmarks/qrdata/reference_grader.py (causal_disc; identical to generated/causal_disc/reference_grader.py)
  bdsgen_tests    -> scripts/bdsgen_reference_grader.py (bdsgen; graded on the solution code, hence SUBMISSION_FILES)
harness/runner.load_grader() finds this file as the nearest reference_grader.py above a problem; scripts/build_qr_teach.py's
wrappers get a copy in each tree. Paths resolve in the tree the copy sits in (/project, or /work with its /scratch4 benchmarks).
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent      # the wrapper root, NOT resolved: its parent is this tree's benchmarks directory
TREE = HERE.parent
SCRIPTS = (TREE.parent / "scripts", Path("/work/pi_hongyu_umass_edu/hieutran_umass_edu/deeprs/scripts"),
           Path("/project/pi_hongyu_umass_edu/hieutran/deep/deeprs/scripts"))
SUBMISSION_FILES = ("solution.py",)   # harness/runner._attach_submission_files; only bdsgen_tests reads the files
_PATHS = {"genebench": TREE / "genebench-pro-public" / "reference_grader.py",
          "qrdata": TREE / "qrdata" / "reference_grader.py",
          "bdsgen": next((s / "bdsgen_reference_grader.py" for s in SCRIPTS if (s / "bdsgen_reference_grader.py").is_file()), None)}
_TYPES = {"multi_numeric_tolerance": "genebench", "numeric_tolerance": "genebench", "composite": "genebench",
          "qrdata_official": "qrdata", "bdsgen_tests": "bdsgen"}
_MODS = {}


def _mod(key):
    if key not in _MODS:
        p = _PATHS[key]
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"qr_teach grader: no {key} grader at {p}")
        spec = importlib.util.spec_from_file_location(f"_qr_teach_{key}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[key] = m
    return _MODS[key]


def evaluate(config, submission):
    gtype = (config.get("grader") or {}).get("type")
    key = _TYPES.get(gtype)
    if key is None:
        raise ValueError(f"unsupported grader type: {gtype}")
    if key != "bdsgen" and isinstance(submission, dict) and "_files" in submission:
        submission = {k: v for k, v in submission.items() if k != "_files"}
    return _mod(key).evaluate(config, submission)
