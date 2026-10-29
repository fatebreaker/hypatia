# open-analysts

Code for *Reinforcement learning and distillation close the gap between small open models and frontier models in
scientific data analysis* (manuscript in preparation).

Small open models (Qwen3.5 with 4 and 9 billion parameters) are trained, by reinforcement learning in a notebook
harness and by distillation from a larger model trained the same way, to analyse data in six domains: statistics and
causal inference (QRData), clinical calculation (MedCalc-Bench), health-record questions (EHRSQL), clinical agent tasks
(MedAgentBench), biomedical data science (BioDSBench) and astronomy (Kepler light curves). They are compared, in the
same harness, with GPT-6 Sol and Claude Opus 5.5.

## Layout

| Path | Contents |
|:--|:--|
| `harness/` | The notebook harness: an agent writes one Python cell per turn, a persistent kernel runs it, and the agent submits a JSON answer. Agents for vLLM-served open models, the OpenAI API and the Anthropic API. |
| `rl/` | Training: group relative policy optimization with answer and grounded rewards, the KL anchor, overflow handling, on-policy distillation from a teacher, and maximum-likelihood distillation on teacher notebooks. |
| `generators/`, `sage/` | Generators of synthetic training tasks with known answers (causal-inference designs, survival analysis, light curves and others). |
| `experiments/harness_v3/` | Evaluation drivers (`screen_v3.sh`, `run_v3*.sh`) and readers. |
| `scripts/` | Benchmark fetching and preparation, training-pool builders, teacher-notebook builders, graders, and the cluster job helpers used for the runs. |
| `analysis/paper/` | The scripts behind the paper's tables, figures and statistics: Table 1, the arm-by-arm table, the transfer matrix, multiplicity and margins, the mixed-effects check, false-discovery control, and the table of pre-specified rules. |

## Not in this repository

- **Benchmarks:** they are other people's data. Use the fetch scripts in `scripts/` (`fetch_*.py`) to get them from
  their sources.
- **Model adapters, training pools, teacher notebooks and graded transcripts:** to be released on Hugging Face.
- **The time-stamped analysis plan:** released with the paper as Supplementary Data.

## Running

`requirements.txt` covers the harness and the analysis; `requirements-train.txt` covers training. The scripts were
written for a SLURM cluster: paths in `paths.py` and in the shell scripts point to the authors' storage and need
adapting before use.

## Licence and citation

To be added.
