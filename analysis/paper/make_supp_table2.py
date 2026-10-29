"""make_supp_table2.py (v05, 10-29) -- Supplementary Table 2: every deviation from the time-stamped analysis plan (docs/RL_PLAN.md),
with when it was decided and its effect on the reported results. The v04 rows (docs/rl4sci/v04/make_supp_table2.py, read without
running it) are kept as written; the rows below are the deviations of the v05 period, each citing the plan entry (UTC) that
records it.
    python3 docs/rl4sci/v05/make_supp_table2.py  ->  docs/rl4sci/v05/supplementary/Supplementary_Table_2_deviations.csv
"""
import ast
import csv
import os

D = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(D, "..", "v04", "make_supp_table2.py")).read()
V04 = next(ast.literal_eval(n.value) for n in ast.parse(src).body
           if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ROWS")
V05 = [
    ("27B reinforcement learning removed from Table 1", "2026-10-21 14:41",
     "The 27B trained with reinforcement learning was a registered Table 1 row. Its in-domain runs were too slow to finish in time, "
     "and on the authors' decision the row was dropped; the RL-trained 27B remained the distillation teacher, and its runs on real "
     "Kepler training stars were stopped.",
     "Table 1 reports the 9B and the 4B (registered at 15:03 the same day); every 27B rule read before the decision stays in "
     "Supplementary Table 1."),
    ("Second frontier comparator added", "2026-10-25 19:39 (registered before its runs); 2026-10-26 01:34 (budget closed)",
     "Claude Opus 5.5 at high effort was added as a frontier comparator on all six benchmarks after the trained models had been "
     "read against GPT-6 Sol. A run of Claude Sonnet 5.5 was stopped by the API budget after three benchmarks.",
     "Claude Opus 5.5 is a Table 1 row; the Sonnet run is not reported. The arms aimed at Claude's level were registered after its "
     "scores were known (Supplementary Note 2)."),
    ("Strongest registered arm per cell", "2026-10-26 14:04",
     "On the authors' instruction, each Table 1 cell uses the strongest registered arm read over three runs, chosen after the reads "
     "of the same test sets.",
     "Cells moved to later arms (MedAgentBench MAB9-T8; QRData QD9-FG, then QD9-FG-L; clinical calculation B-GEN-X3; "
     "health-record questions B-GEN-X2). Every arm tried is in Extended Data Table 4, intervals corrected for the number tried are "
     "in Supplementary Table 8, and confirmations on unused problems and fresh-seed replications are reported beside them."),
    ("Kepler comparison with Claude Opus 5.5 not counted", "2026-10-27 20:23 (adopting a mock review's framing)",
     "On 86 of 256 Kepler problems Claude Opus 5.5 wrote code and its answer in one message, against the harness protocol, so the "
     "code never ran. After this was seen, Kepler was left out of the count of benchmarks on which the 9B is level with Claude.",
     "The Kepler comparison is reported in full, with Claude's accuracy on the 151 light curves where its code ran (Methods); the "
     "count of benchmarks rests on the other five."),
    ("Sandbox outage: answers re-read, trainings restarted", "2026-10-29 01:59",
     "From 23:23 on 28 October to 01:40 on 29 October 2026 (UTC) an agent's package installation broke the shared Python "
     "environment that runs every agent's code. The 515 answers written in that window were set aside and their problems re-run, "
     "and nine trainings that ran during it were restarted from scratch.",
     "No reported result uses an answer or a model from the window; the environment was restored to its exact earlier versions "
     "and made read-only."),
]
os.makedirs(os.path.join(D, "supplementary"), exist_ok=True)
with open(os.path.join(D, "supplementary", "Supplementary_Table_2_deviations.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["deviation", "decided (UTC)", "what changed", "effect on the reported results"])
    w.writerows(list(V04) + V05)
print(f"Supplementary Table 2: {len(V04) + len(V05)} deviations ({len(V04)} from v04, {len(V05)} new)")
