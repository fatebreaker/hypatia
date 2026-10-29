#!/usr/bin/env python
"""What does the CONTENTLESS SCAFFOLD cost, per abstract failure mode?

The companion to scripts/content_vs_placebo.py. That one asks what the instruction CONTENT buys over the
scaffold; on DABench the answer is nothing (scripts/dabench_content.py) and on the families it is one confounding
result. This asks the other question: the scaffold is worth +38 points on real data-analysis problems, so what
does it COST, and is the cost organised by mode?

The working account since the mode re-read is "scaffolding buys technical thoroughness and costs structural
judgement": the placebo REDUCES traps about how-to-model-what-you-have (M1-M4) and CAUSES traps about
which-data-identifies-the-answer (M5, M6). Tonight that account lost one leg and gained another -- arr_kinetics
showed the M4 direction REVERSING between 8B and 14B, while causal_iv's M6 harm replicated across two draws. So
this re-derives the whole table under the current rules rather than quoting the old one.

RULES APPLIED, all of them earned the hard way:
  * placebo against its OWN DRAW's control, never a pooled one (draw_compare refuses the alternative);
  * the exhaustion filter -- a non-termination gap >=10pt is CONFOUNDED and dropped, 5-10pt is flagged;
  * ITEM 396's rule -- if the two arms' PASS rates differ by more than 15 points they are not the same
    population and the trap comparison is dropped, not reported. This is new here, and it is what makes the
    arr_kinetics rows disappear rather than contradict each other;
  * the endpoint is the UNION of the family's traps mapped to that mode, among terminated rollouts.

    venv/bin/python scripts/scaffold_liability.py
"""
import sys, os, glob, json, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import fisher_exact
from draw_compare import arm, compare
from trap_modes import MODES, mode_of

PLACEBOS = ("selfver20", "selfver16")
PASS_GAP = 15.0


def fam_modes():
    from sage.registry import families
    F = families()
    out = {}
    for name in sorted(F):
        try:
            f = F[name]; rng = np.random.default_rng(0)
            traps = list(f.traps(f.draw(rng, f.knobs(rng)).data))
        except Exception:
            continue
        d = collections.defaultdict(list)
        for t in traps:
            m = mode_of(t)
            if m:
                d[m].append(t)
        if d:
            out[name] = dict(d)
    return out


def main():
    FM = fam_modes()
    rows, dropped = [], collections.Counter()
    for d in sorted(glob.glob("results/pods/general_agent_*")):
        D = os.path.basename(d)
        for pl in PLACEBOS:
            for p in glob.glob(os.path.join(d, f"{pl}_*.json")):
                fam = os.path.basename(p)[len(pl) + 1:-5]
                if fam not in FM or not os.path.isfile(os.path.join(d, f"control_{fam}.json")):
                    continue
                try:
                    a = arm(D, pl, fam); b = arm(D, "control", fam)
                except Exception:
                    dropped["unreadable arm"] += 1
                    continue
                if abs(a.rate - b.rate) > PASS_GAP:
                    dropped[f"pass-rate gap >{PASS_GAP:.0f}pt (item 396)"] += 1
                    continue
                model = json.load(open(p)).get("agent", "?").split(":")[1]
                for mode, traps in FM[fam].items():
                    ka = sum(a.attrib[t] for t in traps); kb = sum(b.attrib[t] for t in traps)
                    gap = abs(100*a.attrib["none"]/a.n - 100*b.attrib["none"]/b.n)
                    if gap >= 10:
                        dropped["exhaustion-confounded"] += 1
                        continue
                    pa, pb = 100*ka/a.terminated, 100*kb/b.terminated
                    pv = fisher_exact([[ka, a.terminated-ka], [kb, b.terminated-kb]])[1]
                    rows.append((mode, fam, model, D, pa, pb, pv, gap))

    print("WHAT THE CONTENTLESS SCAFFOLD COSTS, BY MODE")
    print("(placebo vs its OWN draw's control, on that mode's trap union among terminated rollouts)\n")
    print(f"dropped before reading: " + ", ".join(f"{v} {k}" for k, v in dropped.most_common()) + "\n")
    for mode in sorted(MODES):
        rs = [r for r in rows if r[0] == mode]
        if not rs:
            continue
        up = [r for r in rs if r[4] > r[5]]
        dn = [r for r in rs if r[4] < r[5]]
        sig_up = [r for r in up if r[6] < 0.05]
        sig_dn = [r for r in dn if r[6] < 0.05]
        fams = sorted({r[1] for r in rs})
        # TWO GUARDS, both added 2026-09-22 after the first run of this script printed verdicts it had not earned.
        # (1) A mode carried by ONE family is a family result wearing a mode's name -- the same error the review
        #     found in M6 ("IV in economics vs Mendelian randomisation" was one family). M5's eleven rows are
        #     ELEVEN DRAWS OF causal_confounded; that is replication, not breadth, and the difference decides
        #     whether "scaffolding costs structural judgement" is a claim about modes or about one dataset.
        # (2) A direction needs a clear majority AND no significant row pointing the other way. The first version
        #     called 2-up/2-down "CAUSES the trap" because one of the ups was significant. It is not a direction.
        share = max(len(up), len(dn)) / len(rs)
        if share < 0.7 or (sig_up and sig_dn) or not (sig_up or sig_dn):
            verdict = "NO CONSISTENT DIRECTION"
        elif len(up) > len(dn):
            verdict = "CAUSES the trap"
        else:
            verdict = "REDUCES the trap"
        if len(fams) < 2:
            verdict += f"  [ONE FAMILY ONLY ({fams[0]}) -- a family result, not a mode result]"
        print(f"=== {mode:18s} {len(rs):2d} clean row(s) over {len(fams)} famil(y/ies) -- {len(up)} up / {len(dn)} down, "
              f"significant {len(sig_up)} up / {len(sig_dn)} down -> {verdict}")
        print(f"      {MODES[mode]}")
        for r in sorted(rs, key=lambda r: r[6]):
            mark = "  <-- significant" if r[6] < 0.05 else ""
            print(f"      {r[1]:18s} {r[2]:12s} {r[3][14:40]:26s} placebo {r[4]:5.1f}% vs control {r[5]:5.1f}% "
                  f"({r[4]-r[5]:+6.1f}pt) p={r[6]:8.3g}{mark}")
        print()

    print("Read the verdicts as a SET, not row by row. The account under test is that the scaffold reduces traps")
    print("about how to model what you have (M1-M4) and causes traps about which data identifies the answer")
    print("(M5, M6). A mode with rows in both directions is evidence against that account, not a mixed result.")


if __name__ == "__main__":
    main()
