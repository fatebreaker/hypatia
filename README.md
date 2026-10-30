# Hypatia

Code for *Reinforcement learning and distillation make small open models competitive scientific data analysts*
(manuscript in preparation).

Hypatia is a family of small open models (Qwen3.5 with 4 and 9 billion parameters) trained, by reinforcement learning in a notebook
harness and by distillation from a larger model trained the same way, to analyse data in six domains: statistics and
causal inference (QRData), clinical calculation (MedCalc-Bench), health-record questions (EHRSQL), clinical agent tasks
(MedAgentBench), biomedical data science (BioDSBench) and astronomy (Kepler light curves). They are compared, in the
same harness, with GPT-6 Sol and Claude Opus 5.5.

## Models

Each Hypatia specialist is a LoRA adapter trained for one domain; the paper's Table 1 reports each size's six specialists as
Hypatia-4B and Hypatia-9B. Hypatia-9B-Unified is a single 9B for all six.

| Model | Domain (benchmark) |
|:--|:--|
| Hypatia-9B-Stats, Hypatia-4B-Stats | statistics and causal inference (QRData) |
| Hypatia-9B-MedCalc, Hypatia-4B-MedCalc | clinical calculation (MedCalc-Bench) |
| Hypatia-9B-EHR, Hypatia-4B-EHR | health-record questions (EHRSQL) |
| Hypatia-9B-MedAgent, Hypatia-4B-MedAgent | clinical agent tasks (MedAgentBench) |
| Hypatia-9B-Bio, Hypatia-4B-Bio | biomedical data science (BioDSBench) |
| Hypatia-9B-Astro, Hypatia-4B-Astro | astronomy (Kepler light curves) |
| Hypatia-9B-Unified | all six |
| Hypatia-27B-Teacher | the RL-trained Qwen3.6-27B teachers (clinical, astronomy) |

The adapters will be at `huggingface.co/bio-nlp-umass/<model>` with the paper, on the text-only bases
`bio-nlp-umass/Qwen3.5-9B-text` and `bio-nlp-umass/Qwen3.5-4B-text`.

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
- **Model adapters:** the Hypatia models above, on Hugging Face.
- **Training pools, teacher notebooks and graded transcripts:** to be released on Hugging Face (`bio-nlp-umass/Hypatia-data`).
- **The time-stamped analysis plan:** released with the paper as Supplementary Data.

## Running

`requirements.txt` covers the harness and the analysis; `requirements-train.txt` covers training. The scripts were
written for a SLURM cluster: paths in `paths.py` and in the shell scripts point to the authors' storage and need
adapting before use.

## Licence and citation

The code is released under the Apache License 2.0 (`LICENSE`). The Hypatia adapters on Hugging Face are released under Apache-2.0
as well. The benchmarks keep their own licences and are not redistributed here.

Citation: to be added at publication.
