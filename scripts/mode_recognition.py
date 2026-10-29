#!/usr/bin/env python
"""E3 -- CAN THE AGENT NAME A FAILURE MODE IT CANNOT FIX?

Written before the arms landed (2026-09-21 ~23:00). Design: docs/CROSS_DOMAIN_TASK_DESIGN.md.

Every other measurement in this programme scores REMEDIATION -- did the agent avoid the documented mistake -- and
remediation has transferred nowhere outside causal_confounded. Recognition is a different capability. The `:diag`
arm is the plain control agent plus a seven-line menu of abstract modes and one required output line:

    MODES M2,M5
    WHY M2 <the thing in these data that makes it apply>

and then the ordinary analysis, so `diag` vs `control` in the same draw isolates the naming.

GRADING. A family's TRUE modes are the modes of its own documented traps (scripts/trap_modes.py over
sage.registry) -- a documented trap is by definition a way this analysis goes wrong, so its mode genuinely
threatens it. For each rollout and each of the seven modes we observe (mode is present in this family) x (the
agent named it). Pooling those gives one 2x2, and Fisher on it IS the recognition test: does naming track
presence at all? Reading per-mode recall alone would not be a test, because an agent that names M2 on everything
scores perfect recall on M2 and knows nothing.

REGISTERED BEFORE THE DATA:
  RECOGNITION EXISTS    pooled odds ratio > 1 with p < 0.05, AND at least four of the seven modes individually
                        named more often when present than when absent. -> agents can identify unseen failure
                        modes they cannot repair; the gap is knowing-what-to-do, which is the tractable target.
  RECOGNITION AT CHANCE pooled p >= 0.05, or the effect carried by one mode. -> nothing in the model corresponds
                        to the mode layer; it is a labelling convenience for the analyst and the honest write-up
                        says so.
  Secondary, same draw, never read alone: `diag` vs `control` on pass rate and on each family's own trap union.
  Naming the mode without fixing it is the DISSOCIATION the experiment is for, and is reported as such.
  COMPLIANCE is reported first: a rollout with no parsable MODES line is "named nothing", and if compliance is
  below 80% the recognition numbers are reported as a lower bound, not as a measurement.

    venv/bin/python scripts/mode_recognition.py [DRAW]
"""
import sys, os, re, json, glob, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import fisher_exact
from ga_table import score
from trap_modes import MODES, mode_of

DRAW = sys.argv[1] if len(sys.argv) > 1 else "general_agent_E3_diag8"
CODES = [m.split("_")[0] for m in sorted(MODES)]              # M1..M7
LINE = re.compile(r"^\s*MODES\s+(.+)$", re.M | re.I)


def true_modes(fam):
    from sage.registry import families
    f = families()[fam]
    rng = np.random.default_rng(0)
    out = set()
    for t in f.traps(f.draw(rng, f.knobs(rng)).data):
        m = mode_of(t)
        if m:
            out.add(m.split("_")[0])
    return out


def named(msgs):
    """The codes on the FIRST parsable MODES line of the first assistant turn, or None if there is none."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        hit = LINE.search(m.get("content") or "")
        if hit:
            body = hit.group(1)
            if re.search(r"\bnone\b", body, re.I):
                return set()
            return {c.upper() for c in re.findall(r"M[1-7]", body)}
        return None            # the agent's first turn said nothing parsable; later turns are not the question
    return None


def main():
    files = sorted(glob.glob(f"results/pods/{DRAW}/diag_*.json"))
    if not files:
        print(f"no diag arms under results/pods/{DRAW} yet -- nothing to read")
        return
    cell = collections.Counter()                 # (present, named) -> count
    per_mode = collections.defaultdict(collections.Counter)
    compliant = total = 0
    print(f"E3 MODE RECOGNITION -- draw {DRAW}\n")
    print(f"{'family':18s} {'true modes':14s} {'n':>5s} {'compliance':>11s}  most-named")
    for path in files:
        fam = os.path.basename(path)[len("diag_"):-len(".json")]
        try:
            score(path)                          # dead-server gate, as everywhere
            truth = true_modes(fam)
        except Exception as e:
            print(f"{fam:18s} skipped: {str(e)[:60]}")
            continue
        counts = collections.Counter()
        n = c = 0
        for rolls in json.load(open(path))["groups"].values():
            for r in rolls:
                n += 1
                got = named(r.get("messages") or [])
                if got is None:
                    got = set()
                else:
                    c += 1
                counts.update(got)
                for code in CODES:
                    key = (code in truth, code in got)
                    cell[key] += 1
                    per_mode[code][key] += 1
        compliant += c; total += n
        top = ", ".join(f"{k}:{v*100//max(n,1)}%" for k, v in counts.most_common(3)) or "-"
        print(f"{fam:18s} {','.join(sorted(truth)) or '-':14s} {n:5d} {100*c/max(n,1):10.1f}%  {top}")

    if not total:
        return
    print(f"\ncompliance overall: {100*compliant/total:.1f}% of rollouts printed a parsable MODES line")
    if compliant / total < 0.80:
        print("  -> BELOW THE REGISTERED 80% BAR: everything below is a LOWER BOUND, not a measurement.")

    a, b = cell[(True, True)], cell[(True, False)]
    c_, d = cell[(False, True)], cell[(False, False)]
    orr, p = fisher_exact([[a, b], [c_, d]])
    print("\n=== THE RECOGNITION TEST (pooled over modes and families) ===")
    print(f"  named when present : {a}/{a+b} = {100*a/max(a+b,1):.1f}%")
    print(f"  named when absent  : {c_}/{c_+d} = {100*c_/max(c_+d,1):.1f}%")
    print(f"  odds ratio {orr:.2f},  Fisher p = {p:.4g}")

    print(f"\n{'mode':6s} {'recall':>8s} {'false-alarm':>12s}  direction")
    right = 0
    # A mode that is PRESENT in no family of this draw cannot discriminate, so the "at least four of seven
    # modes" half of the criterion is unsatisfiable on a narrow draw and must not be scored as a failure.
    # The pilot (one family, two modes present) printed "RECOGNITION AT CHANCE" off an odds ratio of 2.94 at
    # p = 0.003 for exactly this reason. Count what is evaluable, and say so when it is too few.
    present = [c for c in CODES if per_mode[c][(True, True)] + per_mode[c][(True, False)] > 0]
    for code in CODES:
        pc = per_mode[code]
        rec = pc[(True, True)] / max(pc[(True, True)] + pc[(True, False)], 1)
        fa = pc[(False, True)] / max(pc[(False, True)] + pc[(False, False)], 1)
        ok = (pc[(True, True)] + pc[(True, False)]) > 0 and rec > fa
        right += ok
        note = "named more when present" if ok else ("never present in this draw"
               if pc[(True, True)] + pc[(True, False)] == 0 else "not discriminated")
        print(f"{code:6s} {100*rec:7.1f}% {100*fa:11.1f}%  {note}")

    print()
    need = min(4, len(present))
    # COMPLIANCE GATE, added 2026-09-22 after the first full read. The registration says a draw below 80%
    # compliance yields a LOWER BOUND, not a measurement -- but the verdict below only tested the association
    # clause, so it printed "RECOGNITION EXISTS" on a draw at 65.7% compliance. A precondition that the verdict
    # does not check is not a precondition. It is checked here now.
    lowc = compliant / total < 0.80
    if len(present) < 4:
        print(f"  NOTE: only {len(present)} of 7 modes ({', '.join(present)}) are present in any family of this")
        print(f"  draw, so the registered 'at least four modes' clause is not evaluable here. Requiring {need}.")
    if lowc and len(present) >= 4:
        print(f"  NO VERDICT -- compliance {100*compliant/total:.1f}% is below the registered 80% bar, so these")
        print(f"  numbers are a LOWER BOUND by the criterion's own terms. The evidence in them is"
              f" {'consistent with recognition' if (orr > 1 and p < 0.05 and right >= need) else 'at chance'}")
        print(f"  (odds ratio {orr:.2f}, p = {p:.3g}, {right}/{len(present)} present modes discriminating).")
        print("  Fix the compliance problem or report it as a bound; do not report it as a finding.")
    elif orr > 1 and p < 0.05 and right >= need and len(present) >= 4:
        print("  RECOGNITION EXISTS, by the criterion registered before the data: naming tracks presence, and it")
        print("  does so on at least four modes rather than through one. Read this against the remediation nulls:")
        print("  an agent that can NAME what will go wrong and still not avoid it is a knowing-what-to-do gap.")
    elif len(present) < 4:
        signal = "consistent with recognition" if (orr > 1 and p < 0.05 and right >= need) else "at chance"
        print(f"  NOT A VERDICT -- this draw is too narrow to answer the registered question. The evidence in it is")
        print(f"  {signal} (odds ratio {orr:.2f}, p = {p:.3g}, {right}/{len(present)} present modes discriminating),")
        print("  and the full sweep across eight families is what the criterion was written for.")
    else:
        print("  RECOGNITION AT CHANCE by the registered criterion. The mode layer then organises the ANALYST's")
        print("  view of the benchmark and corresponds to nothing in the model, and the write-up must say so.")


if __name__ == "__main__":
    main()
