"""Generate SFT traces for the cis-MVMR family by EXECUTING a reference solution.

WHY
---
Qwen3-8B submits `{"theta_PROTA": 0.0, "theta_PROTB": 0.0}` after running 17
cells. That is a non-answer in a valid schema: reward 0.012 (the proximity
floor), no step attribution, and therefore no GRPO gradient no matter how dense
the reward is. Dense reward only discriminates once the policy lands near a
recognisable analysis. So the policy needs to be taught the pipeline first.

THE ONE RULE THAT MATTERS
-------------------------
Tool outputs in the trace are REAL. Every code cell is executed in the same
Sandbox the evaluation harness uses, against that instance's own data files,
and the captured stdout is what goes into the conversation. Hand-writing
plausible-looking tool output is the classic way to poison an agentic SFT set:
the model learns to hallucinate observations instead of reading them, and it
looks fine until it silently fabricates numbers at eval time.

Every trace is also verified end-to-end: the final answer must land inside the
grader's tolerance of that instance's ground truth, or the trace is discarded.
A trace that teaches the pipeline but ends on a wrong number teaches the wrong
number.

WHY THIS FILE WAS REWRITTEN (FINDINGS C7)
-----------------------------------------
The first version executed ONE fixed script against every instance. It
retained 100% of traces and reached `mean_token_accuracy 1.0` -- and was
worthless: 5 of 6 supervised messages were byte-identical across all 800
traces, so the model memorised a script instead of learning a policy. That
blocks RL from the opposite side of the same wall: if every rollout recites the
same script and passes, within-group advantage is ~0 again.

Diversity has to live in the SUPERVISED span -- the assistant turns. Four
independent sources of it here, none of which is paraphrase noise:

  1. naming scheme     per-trace substitution of every carried variable, so no
                       two traces share token-level surface form
  2. step variants     2-3 genuinely different implementations per step (median
                       rule vs QC column; positional alignment vs merge-on-snp;
                       inv-solve vs Cholesky whitening vs pinv), all of which
                       must reproduce the reference math exactly
  3. recovery traces   one per pipeline step: make the plausible wrong move for
                       real, let the executed diagnostic expose it, then correct.
                       This is the highest-value kind -- it teaches
                       "notice, then act", which is the documented failure mode
                       ("they identify local diagnostic signals but fail to
                       propagate the implication"), and it produces diversity as
                       a by-product of teaching rather than as noise.
  4. varied structure  optional recon cell, adjacent steps merged into one cell,
                       so trace length varies instead of always being five turns

A variant that silently breaks would be invisible -- its traces would just be
dropped and the corpus would quietly narrow back toward one script. So
generation keeps per-variant kept/dropped counts and `main` FAILS if any
variant's drop rate is anomalous. Same rule as everywhere else in this repo:
the eval asserts its own coverage.

Output: JSONL of {"messages": [...]} in the harness's own format -- same system
prompt, same ```python protocol, same final JSON block. Format drift between
SFT and eval would train the model for a protocol it never sees again.

Run:
  venv/bin/python rl/sft_traces.py --split train --limit 20 --out /tmp/sft.jsonl
  venv/bin/python rl/sft_traces.py --audit /tmp/sft.jsonl   # diversity report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from agents import SYSTEM, build_prompt  # noqa: E402
from paths import data_root  # noqa: E402
from rl.reward import StepReward  # noqa: E402
from sandbox import Sandbox  # noqa: E402

IMPORTS = "import numpy as np, pandas as pd, os\n"

# The machine-readable contract between the trace and the verifier. Parsed with
# a regex rather than str.split so a variant may vary the surrounding prose.
FINAL_RE = re.compile(
    r"FINAL\s+theta_PROTA\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)"
    r"[\s,]+theta_PROTB\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)"
)

# --------------------------------------------------------------------------- #
# Naming schemes.
#
# Code below is written with $-placeholders for every variable that CARRIES
# STATE ACROSS CELLS. One scheme is sampled per trace and applied to every cell
# in it, so the trace stays internally consistent while no two traces share
# surface form. Purely local names (loop counters, scratch) are left alone --
# renaming those would be paraphrase noise, not diversity.
# --------------------------------------------------------------------------- #
NAME_POOLS: dict[str, list[str]] = {
    "z":    ["z", "ld_npz", "npz"],
    "meas": ["m", "meas", "conc"],
    "dis":  ["dis", "outcome", "gwas"],
    "sd":   ["sd", "usable_sd", "sd_per_protein"],
    "G":    ["Gamma", "G", "bx"],
    "y":    ["y", "by", "beta_out"],
    "se":   ["se", "se_out", "sy"],
    "R":    ["R", "LD", "Rs"],
    "sel":  ["sel", "iv", "inst"],
    "Sig":  ["Sigma", "V", "Omega"],
    "gls":  ["gls", "fit_gls", "wls"],
    "th0":  ["theta_joint", "th_mv", "theta0"],
}


def sample_names(rng: random.Random) -> dict[str, str]:
    return {k: rng.choice(v) for k, v in NAME_POOLS.items()}


def render(code: str, names: dict[str, str]) -> str:
    return Template(code).substitute(names).strip("\n")


# --------------------------------------------------------------------------- #
# Step variants.
#
# Each step is a list of (tag, code) and a list of prose alternatives. Prose is
# sampled independently of code -- it states the DECISION, which is the same
# whichever idiom implements it, so any pairing is valid and the cross product
# is larger than either pool.
#
# CONTRACT, enforced by every variant of a step (this is what makes the
# variants interchangeable):
#   recon -> np, pd, os imported; $z is the loaded .npz
#   scale -> $sd maps protein_id -> usable within-batch SD
#   inst  -> $G (n x 2, per-SD), $y, $se, $R, $sel
#   fit   -> $gls(G, y, S) callable, $Sig, $th0
#   prune -> prints the FINAL line
#
# Every variant must reproduce generators/cis_mvmr.py::reference_estimate to
# float precision. Differences are idiom, diagnostics and print format -- never
# math. A variant whose math drifts fails the tolerance check and shows up in
# the per-variant drop table rather than quietly shrinking the corpus.
# --------------------------------------------------------------------------- #

RECON_PROSE = [
    "Before modelling anything I need to know what these files actually "
    "contain -- including the keys inside the .npz, which I should not guess.",
    "First a look at what was actually shipped: shapes, column names, and the "
    "contents of the correlation archive. Guessing a key name here would cost "
    "a turn.",
    "I will read the declared metadata before touching the summary statistics "
    "-- the unit each exposure is reported in is a modelling decision, not a "
    "detail.",
]

RECON_CODE = [
    ("recon_glob", """
$z = np.load('data_files/locus_correlation_EUR.npz')
import glob
pd.set_option('display.width', 200)
for f in sorted(glob.glob('data_files/*.tsv.gz')):
    d = pd.read_csv(f, sep='\\t')
    print(os.path.basename(f), d.shape, list(d.columns))
print('npz keys:', list($z.keys()), '| R shape:', $z['R'].shape)
"""),
    ("recon_meta", """
meta = pd.read_csv('data_files/dataset_metadata.tsv.gz', sep='\\t')
print('datasets declared by the study:')
print(meta.to_string(index=False))
$z = np.load('data_files/locus_correlation_EUR.npz')
print('\\nLD reference:', list($z.keys()), $z['R'].shape,
      '| symmetric:', bool(np.allclose($z['R'], $z['R'].T)))
for f in sorted(os.listdir('data_files')):
    if f.endswith('.tsv.gz'):
        print(' ', f, pd.read_csv('data_files/' + f, sep='\\t').shape)
"""),
    ("recon_head", """
print('files:', sorted(os.listdir('data_files')))
$dis = pd.read_csv('data_files/disease_association.tsv.gz', sep='\\t')
print('\\ndisease_association', $dis.shape, list($dis.columns))
print($dis.head(3).to_string(index=False))
print('\\ndeclared units:')
print(pd.read_csv('data_files/dataset_metadata.tsv.gz', sep='\\t')
        [['dataset_id', 'kind', 'phase', 'unit']].to_string(index=False))
$z = np.load('data_files/locus_correlation_EUR.npz')
print('\\nnpz keys:', list($z.keys()), $z['R'].shape)
"""),
]

SCALE_PROSE = [
    "Step 1 -- the protein betas are per log10 concentration unit, but the "
    "estimand is per SD. I need the *usable* SD. The measurements are "
    "batch-structured, so I check per-batch spread and the assay-QC column "
    "before pooling anything.",
    "The exposures are reported per log10 unit and the target is per SD, so I "
    "need a measurement SD. Pooling across batches would fold the between-batch "
    "mean shifts into it, so I look at the batch structure first.",
    "Step 1 is the unit conversion, and the only honest denominator is the "
    "within-batch spread of assay runs that passed QC. I check both the spread "
    "and the control CV before deciding what to keep.",
]

SCALE_CODE = [
    ("scale_median", """
$meas = pd.read_csv('data_files/protein_measurements.tsv.gz', sep='\\t')
sds = $meas.groupby(['protein_id', 'batch']).log10_concentration.std()
cvs = $meas.groupby(['protein_id', 'batch']).control_cv.first()
print(pd.concat([sds.rename('sd'), cvs.rename('control_cv')], axis=1))
print('\\nnaive pooled SD (WRONG -- ignores batch mean shifts):')
print($meas.groupby('protein_id').log10_concentration.std())
cutoff = 1.5 * sds.groupby(level=0).median()
good = sds <= cutoff.reindex(sds.index, level=0)
$sd = sds[good].groupby(level=0).mean()
print('\\nexcluded batches:', list(sds[~good].index))
print('usable within-batch SD:')
print($sd)
"""),
    ("scale_qc", """
$meas = pd.read_csv('data_files/protein_measurements.tsv.gz', sep='\\t')
tab = $meas.groupby(['protein_id', 'batch']).agg(
    sd=('log10_concentration', 'std'),
    mean=('log10_concentration', 'mean'),
    cv=('control_cv', 'first'))
print(tab.round(4).to_string())
cv_cut = (2.0 * tab.groupby(level=0).cv.median()).reindex(tab.index, level=0)
sd_cut = (1.5 * tab.groupby(level=0).sd.median()).reindex(tab.index, level=0)
by_cv, by_sd = tab.cv > cv_cut, tab.sd > sd_cut
print('\\nflagged by control_cv     :', list(tab.index[by_cv]))
print('flagged by 1.5x median SD :', list(tab.index[by_sd]))
print('the two criteria agree    :', bool((by_cv == by_sd).all()))
$sd = tab.sd[~by_sd].groupby(level=0).mean()
print('\\npooled across batches (inflated):',
      $meas.groupby('protein_id').log10_concentration.std().round(4).to_dict())
print('usable within-batch SD          :', $sd.round(4).to_dict())
"""),
    ("scale_pivot", """
$meas = pd.read_csv('data_files/protein_measurements.tsv.gz', sep='\\t')
wide = $meas.pivot_table(index='batch', columns='protein_id',
                         values='log10_concentration', aggfunc=['mean', 'std'])
print(wide.round(4).to_string())
print('\\nthe per-batch means are offset, so a pooled SD absorbs that shift.')
sds = $meas.groupby(['protein_id', 'batch']).log10_concentration.std()
med = sds.groupby(level=0).median()
keep_batch = sds <= (1.5 * med).reindex(sds.index, level=0)
for k, v in sds[~keep_batch].items():
    print('  dropping %s sd=%.4f vs per-protein median %.4f' % (k, v, med[k[0]]))
$sd = sds[keep_batch].groupby(level=0).mean()
print('usable SD per protein:', $sd.round(4).to_dict())
"""),
]

INST_PROSE = [
    "Steps 2 and 3 -- a subset of the holdout file stores effects on the "
    "opposite allele, so I align to the disease effect allele. I also take "
    "instruments from the phase-2 union of genome-wide significant SNPs and use "
    "phase-2 betas, because phase-1 is the discovery screen and its significant "
    "effects are inflated.",
    "Now harmonise and select. Two things to get right: every exposure beta "
    "must refer to the same allele as the outcome, and the betas themselves "
    "must come from the holdout phase -- the discovery phase is where these "
    "SNPs were selected, so its effect sizes are inflated at exactly the SNPs I "
    "am about to use.",
    "Steps 2 and 3 together: sign-align the exposure betas to the disease "
    "effect allele, then build the instrument set from the phase-2 union at "
    "p<5e-8. I quantify the discovery-phase inflation rather than assert it.",
]

INST_CODE = [
    ("inst_positional", """
$dis = pd.read_csv('data_files/disease_association.tsv.gz', sep='\\t')
def aligned(path):
    d = pd.read_csv(path, sep='\\t')
    flip = d.effect_allele.values != $dis.effect_allele.values
    print('  %s: %d/%d SNPs on the opposite allele'
          % (os.path.basename(path), flip.sum(), len(d)))
    return np.where(flip, -d.beta.values, d.beta.values), d.pval.values
ga2, pa2 = aligned('data_files/protein_PROTA_phase2.tsv.gz')
gb2, pb2 = aligned('data_files/protein_PROTB_phase2.tsv.gz')
ga1, _ = aligned('data_files/protein_PROTA_phase1.tsv.gz')
$sel = np.where((pa2 < 5e-8) | (pb2 < 5e-8))[0]
print('instruments selected (phase-2 union p<5e-8):', len($sel))
sig = pa2 < 5e-8
print('phase1 vs phase2 |beta| at significant SNPs -> winner\\'s curse:',
      round(float(np.abs(ga1[sig]).mean() / np.abs(ga2[sig]).mean()), 3), 'x inflated')
$G = np.column_stack([ga2[$sel] / $sd['PROTA'], gb2[$sel] / $sd['PROTB']])
$y = $dis.beta.values[$sel]
$se = $dis.se.values[$sel]
$R = np.load('data_files/locus_correlation_EUR.npz')['R'][np.ix_($sel, $sel)]
print('Gamma', $G.shape, '| y', $y.shape)
"""),
    ("inst_merge", """
$dis = pd.read_csv('data_files/disease_association.tsv.gz', sep='\\t')
def harmonise(tag, phase):
    d = pd.read_csv('data_files/protein_%s_%s.tsv.gz' % (tag, phase), sep='\\t')
    j = $dis[['snp', 'effect_allele']].merge(d, on='snp', suffixes=('_out', ''))
    assert len(j) == len($dis) and (j.snp.values == $dis.snp.values).all()
    flip = j.effect_allele.values != j.effect_allele_out.values
    print('  %s %s: %d betas flipped onto the disease effect allele'
          % (tag, phase, flip.sum()))
    return np.where(flip, -j.beta.values, j.beta.values), j.pval.values
ga2, pa2 = harmonise('PROTA', 'phase2')
gb2, pb2 = harmonise('PROTB', 'phase2')
ga1, _ = harmonise('PROTA', 'phase1')
$sel = np.flatnonzero((pa2 < 5e-8) | (pb2 < 5e-8))
print('selected %d instruments from the phase-2 union at p<5e-8' % len($sel))
sig = pa2 < 5e-8
print('discovery-phase |beta| is %.2fx the holdout at the same SNPs'
      % (np.abs(ga1[sig]).mean() / np.abs(ga2[sig]).mean()))
$G = np.column_stack([ga2, gb2])[$sel] / np.array([$sd['PROTA'], $sd['PROTB']])
$y, $se = $dis.beta.values[$sel], $dis.se.values[$sel]
$R = np.load('data_files/locus_correlation_EUR.npz')['R'][np.ix_($sel, $sel)]
print('exposure matrix', $G.shape, '| outcome', $y.shape)
"""),
    ("inst_dict", """
$dis = pd.read_csv('data_files/disease_association.tsv.gz', sep='\\t')
ref_allele = $dis.effect_allele.values
prot = {}
for tag in ('PROTA', 'PROTB'):
    for ph in ('phase1', 'phase2'):
        d = pd.read_csv('data_files/protein_%s_%s.tsv.gz' % (tag, ph), sep='\\t')
        flip = d.effect_allele.values != ref_allele
        prot[(tag, ph)] = (np.where(flip, -d.beta.values, d.beta.values), d.pval.values)
        print('%s/%s: %d of %d betas sign-flipped' % (tag, ph, flip.sum(), len(d)))
pa2, pb2 = prot[('PROTA', 'phase2')][1], prot[('PROTB', 'phase2')][1]
$sel = np.where((pa2 < 5e-8) | (pb2 < 5e-8))[0]
sig = pa2 < 5e-8
print('\\ninstruments:', len($sel), '| phase-1 inflation at those SNPs: %.2fx'
      % (np.abs(prot[('PROTA', 'phase1')][0][sig]).mean()
         / np.abs(prot[('PROTA', 'phase2')][0][sig]).mean()))
print('-> the exposure matrix uses phase-2 (holdout) betas')
$G = np.column_stack([prot[('PROTA', 'phase2')][0][$sel] / $sd['PROTA'],
                      prot[('PROTB', 'phase2')][0][$sel] / $sd['PROTB']])
$y = $dis.beta.values[$sel]
$se = $dis.se.values[$sel]
$R = np.load('data_files/locus_correlation_EUR.npz')['R'][np.ix_($sel, $sel)]
print('Gamma', $G.shape)
"""),
]

FIT_PROSE = [
    "Step 4 -- the instruments sit in one locus and are correlated, and each "
    "protein partially proxies the other. I fit both proteins jointly with an "
    "LD-aware covariance, and compute the univariable and LD-ignoring versions "
    "alongside so the size of each mistake is visible rather than assumed.",
    "These instruments are one cis locus, so they are correlated with each "
    "other and each protein's instruments partly predict the other protein. "
    "That makes this a multivariable problem with a non-diagonal weight matrix. "
    "I fit it that way and show what the two shortcuts would have given.",
    "Step 4 is the estimator itself: generalised least squares over both "
    "exposures at once, weighted by the LD-implied covariance of the outcome "
    "betas. I print the univariable and diagonal-Sigma alternatives next to it "
    "so the cost of each is measured, not asserted.",
]

FIT_CODE = [
    ("fit_inv", """
def $gls(G, yy, S):
    Si = np.linalg.inv(S)
    return np.linalg.solve(G.T @ Si @ G, G.T @ Si @ yy)
$Sig = np.diag($se) @ $R @ np.diag($se)
$th0 = $gls($G, $y, $Sig)
theta_uni = [$gls($G[:, [k]], $y, $Sig)[0] for k in (0, 1)]
theta_noLD = $gls($G, $y, np.diag($se ** 2))
print('univariable (each protein alone) :', np.round(theta_uni, 4))
print('LD ignored  (diagonal Sigma)     :', np.round(theta_noLD, 4))
print('joint + LD-aware                 :', np.round($th0, 4))
_L = np.linalg.cholesky($Sig)
_r = np.linalg.solve(_L, $y - $G @ $th0)
print()
print('max |standardized residual| %.2f across %d instruments'
      % (np.abs(_r).max(), len(_r)))
print('-> anything above 2.5 is a contaminated instrument, so this fit is')
print('   NOT the final estimate; residual pruning still has to run')
"""),
    ("fit_cholesky", """
from scipy import linalg as sla
$Sig = ($se[:, None] * $R) * $se[None, :]
def $gls(G, yy, S):
    L = np.linalg.cholesky(S)
    Gw = sla.solve_triangular(L, G, lower=True)
    yw = sla.solve_triangular(L, yy, lower=True)
    return np.linalg.lstsq(Gw, yw, rcond=None)[0]
$th0 = $gls($G, $y, $Sig)
print('joint, LD-aware  :', np.round($th0, 4))
print('each alone       :',
      np.round([$gls($G[:, [k]], $y, $Sig)[0] for k in (0, 1)], 4))
print('diagonal Sigma   :', np.round($gls($G, $y, np.diag($se ** 2)), 4))
print('-> dropping either the joint fit or the LD structure moves the estimate')
_L = np.linalg.cholesky($Sig)
_r = np.linalg.solve(_L, $y - $G @ $th0)
print()
print('max |standardized residual| %.2f across %d instruments'
      % (np.abs(_r).max(), len(_r)))
print('-> anything above 2.5 is a contaminated instrument, so this fit is')
print('   NOT the final estimate; residual pruning still has to run')
"""),
    ("fit_pinv", """
def $gls(G, yy, S):
    Si = np.linalg.pinv(S)
    return np.linalg.solve(G.T @ Si @ G, G.T @ Si @ yy)
$Sig = np.diag($se) @ $R @ np.diag($se)
$th0 = $gls($G, $y, $Sig)
for label, est in [('univariable, PROTA alone', $gls($G[:, [0]], $y, $Sig)),
                   ('univariable, PROTB alone', $gls($G[:, [1]], $y, $Sig)),
                   ('joint, Sigma diagonal   ', $gls($G, $y, np.diag($se ** 2))),
                   ('joint, Sigma LD-aware   ', $th0)]:
    print('%s : %s' % (label, np.round(np.atleast_1d(est), 4)))
print('\\ncondition number of the exposure matrix: %.1f' % np.linalg.cond($G))
print('-> collinear enough that dropping one exposure biases the other')
_L = np.linalg.cholesky($Sig)
_r = np.linalg.solve(_L, $y - $G @ $th0)
print()
print('max |standardized residual| %.2f across %d instruments'
      % (np.abs(_r).max(), len(_r)))
print('-> anything above 2.5 is a contaminated instrument, so this fit is')
print('   NOT the final estimate; residual pruning still has to run')
"""),
]

PRUNE_PROSE = [
    "Step 5 -- if the only outcome signal came through these two proteins the "
    "standardized residuals would be exchangeable. I check whether they cluster "
    "by position, which would indicate a further outcome-only pathway "
    "contaminating some instruments, and prune before refitting.",
    "Last check. Under the exclusion restriction the whitened residuals should "
    "look like noise. If a subset of instruments carries outcome signal that "
    "does not pass through either protein, it shows up as residuals too large "
    "to be chance. I drop those iteratively and refit.",
    "Step 5 is the exclusion-restriction diagnostic: whiten the residuals by "
    "the LD covariance and remove instruments the model cannot explain, one at "
    "a time, refitting after each removal until nothing exceeds the threshold.",
]

PRUNE_CODE = [
    ("prune_loop", """
keep = np.ones(len($sel), bool)
theta = $th0.copy()
dropped = []
while True:
    idx = np.where(keep)[0]
    L = np.linalg.cholesky($Sig[np.ix_(idx, idx)])
    r = np.linalg.solve(L, $y[idx] - $G[idx] @ theta)
    w = int(np.argmax(np.abs(r)))
    if abs(r[w]) <= 2.5 or keep.sum() <= 5:
        print('max |standardized residual| now', round(float(abs(r[w])), 3))
        break
    dropped.append(int($sel[idx[w]]))
    keep[idx[w]] = False
    idx = np.where(keep)[0]
    theta = $gls($G[idx], $y[idx], $Sig[np.ix_(idx, idx)])
print('pruned', len(dropped), 'contaminated instruments at SNP indices',
      sorted(dropped)[:12])
print('before pruning :', np.round($th0, 4))
print('after refit    :', np.round(theta, 4))
print('FINAL theta_PROTA = %.6f  theta_PROTB = %.6f' % (theta[0], theta[1]))
"""),
    ("prune_fn", """
def whitened(idx, th):
    L = np.linalg.cholesky($Sig[np.ix_(idx, idx)])
    return np.linalg.solve(L, $y[idx] - $G[idx] @ th)
keep = np.ones(len($sel), bool)
theta = $th0.copy()
removed = []
while keep.sum() > 5:
    idx = np.flatnonzero(keep)
    r = whitened(idx, theta)
    w = int(np.argmax(np.abs(r)))
    if abs(r[w]) <= 2.5:
        break
    removed.append(int($sel[idx[w]]))
    keep[idx[w]] = False
    idx = np.flatnonzero(keep)
    theta = $gls($G[idx], $y[idx], $Sig[np.ix_(idx, idx)])
print('dropped %d of %d instruments; SNP indices %s'
      % (len(removed), len($sel), sorted(removed)[:12]))
print('max |whitened residual| before / after: %.2f / %.2f'
      % (float(np.abs(whitened(np.arange(len($sel)), $th0)).max()),
         float(np.abs(whitened(np.flatnonzero(keep), theta)).max())))
print('theta before pruning: %s' % np.round($th0, 4))
print('FINAL theta_PROTA = %.6f  theta_PROTB = %.6f' % (theta[0], theta[1]))
"""),
]

STEPS: dict[str, tuple[list[str], list[tuple[str, str]]]] = {
    "recon": (RECON_PROSE, RECON_CODE),
    "scale": (SCALE_PROSE, SCALE_CODE),
    "inst": (INST_PROSE, INST_CODE),
    "fit": (FIT_PROSE, FIT_CODE),
    "prune": (PRUNE_PROSE, PRUNE_CODE),
}

# --------------------------------------------------------------------------- #
# Recovery moves -- one per pipeline step, mirroring the five ablations.
#
# The wrong move is EXECUTED, so the diagnostic that exposes it is a real
# number, not a claim. The correction then runs the ordinary variant for that
# step, which reassigns every contract variable -- so a recovery is
# interchangeable with a plain step and composes with everything else.
#
# This is the part that teaches the documented failure mode. A model that has
# only ever seen the correct move has no example of noticing a bad one.
# --------------------------------------------------------------------------- #
RECOVERY: dict[str, dict] = {
    "scale": dict(
        prose="I need a per-SD scale for the exposures. The straightforward "
              "reading is the SD of log10 concentration per protein, so I will "
              "compute that -- and check it against the batch structure before "
              "relying on it.",
        code="""
$meas = pd.read_csv('data_files/protein_measurements.tsv.gz', sep='\\t')
$sd = $meas.groupby('protein_id').log10_concentration.std()
print('SD per protein, pooled over batches:')
print($sd.round(4))
print('\\nsanity check before I use this -- per-batch mean, sd and QC:')
print($meas.groupby(['protein_id', 'batch'])
          .agg(mean=('log10_concentration', 'mean'),
               sd=('log10_concentration', 'std'),
               cv=('control_cv', 'first')).round(4).to_string())
""",
        fixes=[
            "That pooled SD is not the measurement SD. The batch means are "
            "offset from each other, so pooling folds a between-batch shift "
            "into the spread, and one batch has both a much larger SD and a "
            "control CV several times the others -- a degraded assay run. I "
            "recompute within batch and drop the run that fails QC.",
            "I should not use that. Two things in the check contradict it: the "
            "batch means sit at different levels, so a pooled SD is measuring "
            "the offsets as much as the assay noise, and one batch has a "
            "visibly wider spread together with a control CV several times the "
            "other two. That batch is a degraded run, not extra signal. The "
            "denominator has to be a within-batch SD over the runs that passed "
            "QC.",
        ],
    ),
    "inst": dict(
        prose="I have the scale, so now the instruments. The phase-1 files are "
              "the larger scan, so I will start from those betas and select "
              "genome-wide significant SNPs.",
        code="""
$dis = pd.read_csv('data_files/disease_association.tsv.gz', sep='\\t')
def aligned(path):
    d = pd.read_csv(path, sep='\\t')
    flip = d.effect_allele.values != $dis.effect_allele.values
    return np.where(flip, -d.beta.values, d.beta.values), d.pval.values
ga1, pa1 = aligned('data_files/protein_PROTA_phase1.tsv.gz')
gb1, pb1 = aligned('data_files/protein_PROTB_phase1.tsv.gz')
ga2, pa2 = aligned('data_files/protein_PROTA_phase2.tsv.gz')
gb2, pb2 = aligned('data_files/protein_PROTB_phase2.tsv.gz')
$sel = np.where((pa1 < 5e-8) | (pb1 < 5e-8))[0]
$G = np.column_stack([ga1[$sel] / $sd['PROTA'], gb1[$sel] / $sd['PROTB']])
print('phase-1 exposure matrix built:', $G.shape, 'from', len($sel), 'instruments')
sig = pa2 < 5e-8
print('\\nbut compare the two phases at the SAME SNPs:')
print('  phase-1 mean |beta| : %.5f' % np.abs(ga1[sig]).mean())
print('  phase-2 mean |beta| : %.5f' % np.abs(ga2[sig]).mean())
print('  ratio               : %.2fx' % (np.abs(ga1[sig]).mean() / np.abs(ga2[sig]).mean()))
print('  n SNPs compared     : %d' % int(sig.sum()))
""",
        fixes=[
            "The discovery-phase effects are inflated at exactly the SNPs that "
            "were selected for being significant in that scan -- that ratio is "
            "winner's curse, not biology. Using those betas would bias the "
            "exposure-outcome ratio downward. I rebuild the exposure matrix "
            "from the phase-2 holdout betas, keeping the instrument set fixed.",
            "That ratio is the tell: the same SNPs have systematically larger "
            "effects in the phase that selected them than in the phase that "
            "did not. This is winner's curse, so the phase-1 betas overstate "
            "the exposure side and would shrink every ratio estimate. The "
            "instrument set can stay as it is; the betas have to come from the "
            "holdout.",
        ],
    ),
    "fit": dict(
        prose="With aligned per-SD exposures I can estimate each protein's "
              "effect. I will start with the obvious thing -- one protein at a "
              "time -- and look at how much they overlap before trusting it.",
        code="""
def $gls(G, yy, S):
    Si = np.linalg.inv(S)
    return np.linalg.solve(G.T @ Si @ G, G.T @ Si @ yy)
$Sig = np.diag($se) @ $R @ np.diag($se)
uni = [float($gls($G[:, [k]], $y, $Sig)[0]) for k in (0, 1)]
$th0 = np.array(uni)
print('univariable estimates:', np.round(uni, 4))
print('\\nhow separable are the two exposures?')
print('  corr(exposure columns)          : %.3f' % np.corrcoef($G.T)[0, 1])
off = ~np.eye(len($sel), dtype=bool)
print('  mean |LD| between instruments   : %.3f' % np.abs($R[off]).mean())
print('  max  |LD| between instruments   : %.3f' % np.abs($R[off]).max())
""",
        fixes=[
            "Those two estimates cannot both be marginal effects of a single "
            "protein: the exposure columns are correlated, so each protein's "
            "instruments partly predict the other, and a univariable fit "
            "attributes the shared component twice. The instruments are also in "
            "LD with each other, which a diagonal weight matrix would ignore. "
            "Both proteins have to enter one model with an LD-aware covariance.",
            "The correlation between the exposure columns rules that approach "
            "out. Fitting one protein at a time makes the other an omitted "
            "variable that its own instruments partly proxy, so the shared "
            "component is counted into both estimates. The off-diagonal LD is "
            "non-trivial as well, so the residuals are not independent either. "
            "One joint fit, weighted by the LD-implied covariance.",
        ],
    ),
    "prune": dict(
        prose="I have a joint LD-aware estimate. Before submitting it I check "
              "the exclusion restriction: under it, the whitened residuals "
              "should be unstructured.",
        code="""
L = np.linalg.cholesky($Sig)
r = np.linalg.solve(L, $y - $G @ $th0)
print('standardized residuals: max |r| = %.2f, %d of %d exceed 2.5'
      % (np.abs(r).max(), int((np.abs(r) > 2.5).sum()), len(r)))
pos = pd.read_csv('data_files/variant_metadata.tsv.gz', sep='\\t').pos_bp.values[$sel]
bad = np.abs(r) > 2.5
if bad.sum():
    print('outlier positions :', list(pos[bad][:10]))
    print('outliers span %d bp of the %d bp covered by all instruments'
          % (int(pos[bad].max() - pos[bad].min()), int(pos.max() - pos.min())))
print('estimate as it stands:', np.round($th0, 4))
""",
        fixes=[
            "Those residuals are not noise -- too many exceed the threshold and "
            "they sit together in the locus rather than spreading across it. "
            "That is the signature of outcome signal reaching the disease "
            "through something other than these two proteins, on a subset of "
            "instruments. I remove them iteratively and refit.",
            "I cannot submit this. Under the exclusion restriction the whitened "
            "residuals should be unstructured, and these are not: several "
            "exceed the threshold, and they are concentrated in one stretch of "
            "the locus rather than scattered. A further outcome-only pathway is "
            "contaminating that subset of instruments. I drop the worst one at "
            "a time, refitting after each, until none exceed the threshold.",
        ],
    ),
}

# Sentence fragments for the closing summary. Sampled per trace so the final
# supervised message -- the one carrying the answer -- is not a constant either.
SUMMARY_BITS: dict[str, list[str]] = {
    "scale": [
        "the degraded assay batch was excluded before computing the usable "
        "within-batch SD",
        "the per-SD scale came from within-batch spread with the QC-failing run "
        "dropped, not from a pooled SD",
        "exposures were converted to per-SD units using only the assay batches "
        "that passed QC",
    ],
    "inst": [
        "the holdout betas were aligned to the disease effect allele and used in "
        "place of the winner's-cursed screening betas",
        "exposure betas were sign-aligned to the outcome allele and taken from "
        "the phase-2 holdout to avoid discovery-phase inflation",
        "instruments came from the phase-2 union at p<5e-8 with betas harmonised "
        "onto the disease effect allele",
    ],
    "fit": [
        "both proteins were fitted jointly with an LD-aware covariance",
        "the two exposures entered a single GLS weighted by the LD-implied "
        "covariance of the outcome betas",
        "the estimate is multivariable and LD-aware, so neither the shared "
        "component nor the instrument correlation is double-counted",
    ],
    "prune": [
        "the residual cluster from a further outcome-only pathway was pruned "
        "before the final refit",
        "instruments whose whitened residuals violated the exclusion restriction "
        "were removed iteratively and the model refitted",
        "pleiotropic instruments were dropped on the standardized-residual "
        "criterion and the fit repeated on what remained",
    ],
}

SUMMARY_OPEN = [
    "",
    "Summary of the pipeline: ",
    "To recap what the estimate rests on: ",
]

RECOVERY_NOTE = [
    " The first attempt at {step} was wrong and the diagnostic caught it; "
    "the reported numbers come from the corrected analysis.",
    " Note that {step} needed a correction after the initial approach failed "
    "its own check.",
    " {step} was revised once the diagnostic contradicted the first approach.",
]

RECOVERY_LABEL = {
    "scale": "the unit conversion",
    "inst": "the instrument and phase selection",
    "fit": "the estimator",
    "prune": "the exclusion-restriction check",
}


# --------------------------------------------------------------------------- #
# Trace planning
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Family registry.
#
# Everything above this line is the cis-MVMR content; `rl/traces_scrna.py`
# holds family #2's. The machinery below is family-generic -- it was written
# around one family, and RESULTS §15 (cross-family transfer ~0) is the reason a
# second one now has to go through the same path.
# --------------------------------------------------------------------------- #
class _CisFamily:
    name = "cis_mvmr"
    dataset = "cis_mvmr_1k"
    IMPORTS = IMPORTS
    NAME_POOLS = NAME_POOLS
    STEPS = STEPS
    RECOVERY = RECOVERY
    SUMMARY_BITS = SUMMARY_BITS
    RECOVERY_LABEL = RECOVERY_LABEL
    FINAL_RE = FINAL_RE
    ANSWER_KEYS = ("theta_PROTA", "theta_PROTB")
    ORDER = ("recon", "scale", "inst", "fit", "prune")
    RECOVERABLE = ("scale", "inst", "fit", "prune")
    tol = 0.025

    @staticmethod
    def answer(hit):
        return {"theta_PROTA": float(hit.group(1)), "theta_PROTB": float(hit.group(2))}


class _ScrnaFamily:
    name = "scrna_ambient"
    dataset = "scrna_1k"
    tol = 0.05

    @staticmethod
    def answer(hit):
        return {"beta_activated": float(hit.group(1))}


def _load_scrna():
    from rl import traces_scrna as m
    for attr in ("IMPORTS", "NAME_POOLS", "STEPS", "RECOVERY", "SUMMARY_BITS",
                 "RECOVERY_LABEL", "FINAL_RE", "ANSWER_KEYS", "ORDER", "RECOVERABLE"):
        setattr(_ScrnaFamily, attr, getattr(m, attr))
    return _ScrnaFamily


class _BurdenFamily:
    name = "rare_burden"
    dataset = "rare_burden_1k"
    tol = 0.05

    @staticmethod
    def answer(hit):
        return {"beta_burden": float(hit.group(1))}


def _load_burden():
    from rl import traces_burden as m
    for attr in ("IMPORTS", "NAME_POOLS", "STEPS", "RECOVERY", "SUMMARY_BITS",
                 "RECOVERY_LABEL", "FINAL_RE", "ANSWER_KEYS", "ORDER", "RECOVERABLE"):
        setattr(_BurdenFamily, attr, getattr(m, attr))
    return _BurdenFamily


class _GwasFamily:
    name = "gwas_loci"
    dataset = "gwas_loci"
    # Two keys on very different scales (a locus count and a log-odds mean), so
    # the real tolerances live per-key in the grader and rl/reward.py normalises
    # by them. This scalar is only the fallback and is never the binding one.
    tol = 1.0

    @staticmethod
    def answer(hit):
        return {"n_loci": int(hit.group(1)),
                "mean_abs_effect": float(hit.group(2))}


def _load_gwas():
    from rl import traces_gwas as m
    for attr in ("IMPORTS", "NAME_POOLS", "STEPS", "RECOVERY", "SUMMARY_BITS",
                 "RECOVERY_LABEL", "FINAL_RE", "ANSWER_KEYS", "ORDER",
                 "RECOVERABLE", "instance_vars"):
        setattr(_GwasFamily, attr, getattr(m, attr))
    # instance_vars is a plain function; keep it callable as fam.instance_vars(cfg)
    _GwasFamily.instance_vars = staticmethod(m.instance_vars)
    return _GwasFamily


FAMILIES = {"cis_mvmr": lambda: _CisFamily, "scrna_ambient": _load_scrna,
            "rare_burden": _load_burden, "gwas_loci": _load_gwas}


def plan_trace(rng: random.Random, fam, extra: dict | None = None) -> tuple[list[dict], dict]:
    """Assemble one trace's moves. Returns (moves, plan-metadata).

    A move is {prose, code, tag}. Structure varies three ways: the recon cell
    is optional, at most one step is a recovery, and one adjacent pair of plain
    steps may be merged into a single cell.
    """
    names = {k: rng.choice(v) for k, v in fam.NAME_POOLS.items()}
    # Instance-specific substitutions, for families whose task differs per
    # instance rather than only in its numbers. gwas_loci names a real trait,
    # so its code cannot be written without one; families #1-#3 supply nothing
    # and are unaffected.
    if extra:
        names.update(extra)
    # At most one recovery per trace: two would make the trace mostly about
    # error handling, and the point is to teach the pipeline as well.
    recovery_step = rng.choice([None, *fam.RECOVERABLE])
    want_recon = rng.random() < 0.75

    order = list(fam.ORDER) if want_recon else list(fam.ORDER[1:])

    moves: list[dict] = []
    variants: dict[str, str] = {}
    for step in order:
        if step == recovery_step:
            rec = fam.RECOVERY[step]
            moves.append(dict(step=step, tag=f"{step}:recovery",
                              prose=rec["prose"], code=render(rec["code"], names)))
            _, code_pool = fam.STEPS[step]
            tag, code = rng.choice(code_pool)
            # The fix text already states both the diagnosis and what to do
            # next, so it stands alone -- appending the step's ordinary prose
            # on top of it just restates the same reasoning twice.
            moves.append(dict(step=step, tag=f"{step}:{tag}",
                              prose=rng.choice(rec["fixes"]),
                              code=render(code, names), merge_ok=False))
            variants[step] = f"recovery+{tag}"
        else:
            prose_pool, code_pool = fam.STEPS[step]
            tag, code = rng.choice(code_pool)
            moves.append(dict(step=step, tag=f"{step}:{tag}",
                              prose=rng.choice(prose_pool),
                              code=render(code, names), merge_ok=True))
            variants[step] = tag

    # Optionally merge one adjacent pair of plain steps into a single cell, so
    # trace length is not a constant either. Never merge across a recovery --
    # the whole point of a recovery is that the diagnostic lands in its own
    # tool output before the correction is written.
    if rng.random() < 0.35:
        cand = [i for i in range(len(moves) - 1)
                if moves[i].get("merge_ok") and moves[i + 1].get("merge_ok")]
        if cand:
            i = rng.choice(cand)
            a, b = moves[i], moves[i + 1]
            merged = dict(step=f"{a['step']}+{b['step']}",
                          tag=f"{a['tag']}|{b['tag']}",
                          prose=a["prose"] + "\n\n" + b["prose"],
                          code=a["code"] + "\n\n" + b["code"], merge_ok=False)
            moves[i:i + 2] = [merged]

    moves[0]["code"] = fam.IMPORTS + moves[0]["code"]

    plan = dict(names=names, recovery=recovery_step, recon=want_recon,
                variants=variants, n_moves=len(moves))
    return moves, plan


def build_summary(rng: random.Random, recovery: str | None, fam) -> str:
    bits = [rng.choice(fam.SUMMARY_BITS[k]) for k in fam.RECOVERABLE]
    text = rng.choice(SUMMARY_OPEN) + ", ".join(bits[:-1]) + ", and " + bits[-1] + "."
    text = text[0].upper() + text[1:]
    if recovery:
        text += rng.choice(RECOVERY_NOTE).format(step=fam.RECOVERY_LABEL[recovery])
    return text


def _seed_for(instance_id: str) -> int:
    """Stable across processes -- str.__hash__ is salted per interpreter."""
    return int(hashlib.sha256(instance_id.encode()).hexdigest()[:16], 16)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def make_trace(cfg: dict, data_dir: Path, tol: float, fam) -> tuple[dict | None, dict, str]:
    """Execute a planned solution and assemble a verified conversation.

    Returns (trace-or-None, plan, reason). `reason` names the failure so the
    per-variant table can distinguish a broken variant from an instance that
    is genuinely unsolvable.
    """
    rng = random.Random(_seed_for(cfg["id"]))
    extra = fam.instance_vars(cfg) if hasattr(fam, "instance_vars") else None
    moves, plan = plan_trace(rng, fam, extra)

    sb = Sandbox(data_dir=data_dir, timeout=180)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
    try:
        for mv in moves:
            res = sb.run(mv["code"])
            if res.returncode != 0 or res.timed_out:
                return None, plan, f"exec:{mv['tag']}"
            msgs.append({"role": "assistant",
                         "content": f"{mv['prose']}\n\n```python\n{mv['code']}\n```"})
            msgs.append({"role": "user", "content": res.as_feedback()})

        # Parse the executed final line rather than recomputing it in Python --
        # the answer must be what the trace's own tool output actually said.
        hit = fam.FINAL_RE.search(msgs[-1]["content"])
        if not hit:
            return None, plan, "no_final_line"
        answer = fam.answer(hit)

        scored = StepReward(tolerance=tol).score(cfg, answer)
        if not scored.passed:
            return None, plan, "tolerance"      # never ship a wrong-answer trace

        msgs.append({"role": "assistant", "content":
                     "```json\n" + json.dumps(
                         {"answer": answer,
                          "reasoning": build_summary(rng, plan["recovery"], fam)}, indent=2)
                     + "\n```"})
        return ({"id": cfg["id"], "messages": msgs, "reward": scored.reward,
                 "n_cells": sb.attempts, "plan": plan}, plan, "ok")
    finally:
        sb.cleanup()


# --------------------------------------------------------------------------- #
# Diversity audit -- the check that would have caught C7 on day one
# --------------------------------------------------------------------------- #
def audit(path: Path) -> int:
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        print("empty corpus")
        return 1

    sup = [[m["content"] for m in r["messages"] if m["role"] == "assistant"]
           for r in rows]
    flat = [m for t in sup for m in t]
    lens = Counter(len(t) for t in sup)

    print(f"traces: {len(rows)}   supervised messages: {len(flat)}")
    print(f"trace length (assistant turns): "
          f"{', '.join(f'{k}->{v}' for k, v in sorted(lens.items()))}")
    print()

    # By position, which is exactly how C7 was found: 5 of 6 positions had
    # exactly one unique string across 800 traces.
    print("unique assistant messages by position")
    print("  pos      n   unique   most-common-share")
    npos = max(len(t) for t in sup)
    for i in range(npos):
        at = [t[i] for t in sup if len(t) > i]
        c = Counter(at)
        share = c.most_common(1)[0][1] / len(at)
        flag = "  <-- DEGENERATE" if len(c) == 1 and len(at) > 1 else ""
        print(f"  {i:>3} {len(at):>6} {len(c):>8}   {share:>6.1%}{flag}")

    uniq_all = len(set(flat))
    print()
    print(f"unique supervised messages overall : {uniq_all}/{len(flat)} "
          f"({uniq_all / len(flat):.1%})")
    print(f"unique final messages              : "
          f"{len(set(t[-1] for t in sup))}/{len(sup)}")
    print(f"unique first messages              : "
          f"{len(set(t[0] for t in sup))}/{len(sup)}")

    plans = [r.get("plan", {}) for r in rows]
    rec = Counter(p.get("recovery") for p in plans)
    print()
    print("recovery step: " + ", ".join(
        f"{k or 'none'}={v}" for k, v in sorted(rec.items(), key=lambda x: str(x[0]))))
    for step in ("recon", "scale", "inst", "fit", "prune"):
        vs = Counter(p.get("variants", {}).get(step) for p in plans
                     if p.get("variants", {}).get(step))
        if vs:
            print(f"  {step:<6} " + ", ".join(f"{k}={v}" for k, v in sorted(vs.items())))

    # The corpus is usable if no position collapses to one string and the
    # overall unique rate is high. Thresholds are deliberately loose: this is a
    # degeneracy alarm, not a quality score.
    ok = uniq_all / len(flat) > 0.5 and all(
        len(set(t[i] for t in sup if len(t) > i)) > 1
        for i in range(npos) if sum(len(t) > i for t in sup) > 1)
    print()
    print(f"diversity audit: {'PASS' if ok else 'FAIL -- corpus is degenerate'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="cis_mvmr", choices=sorted(FAMILIES))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tol", type=float, default=None,
                    help="override; default is the family's own tolerance")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--audit", metavar="JSONL",
                    help="report supervised-span diversity of an existing corpus")
    args = ap.parse_args()

    if args.audit:
        return audit(Path(args.audit))

    fam = FAMILIES[args.family]()
    tol = args.tol if args.tol is not None else fam.tol
    root = data_root() / "generated" / fam.dataset / args.split
    dirs = sorted(p for p in root.iterdir() if (p / "eval_config.json").exists())
    dirs = dirs[: args.limit]
    out = (Path(args.out) if args.out
           else data_root() / "sft" / f"{fam.name}_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    def job(d: Path):
        cfg = json.loads((d / "eval_config.json").read_text())
        return make_trace(cfg, d / "data_files", tol, fam)

    ok = fail = 0
    seen: Counter = Counter()      # variant -> times planned
    lost: Counter = Counter()      # variant -> times its trace was dropped
    reasons: Counter = Counter()
    with open(out, "w") as fh:
        if args.workers > 1:
            import concurrent.futures as cf
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = ex.map(job, dirs)
        else:
            results = map(job, dirs)
        for i, (t, plan, reason) in enumerate(results, 1):
            tags = list(plan.get("variants", {}).values())
            seen.update(tags)
            if t is None:
                fail += 1
                lost.update(tags)
                reasons[reason] += 1
            else:
                ok += 1
                fh.write(json.dumps(t) + "\n")
            if i % 50 == 0:
                print(f"  ... {i}/{len(dirs)}  kept={ok} dropped={fail}", flush=True)

    print(f"\nkept {ok} verified traces, dropped {fail} "
          f"({fail / max(ok + fail, 1):.1%} failed reference execution or tolerance)")
    if reasons:
        print("drop reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # A variant that is broken rather than merely unlucky would otherwise be
    # invisible: its traces are dropped, the corpus silently narrows back
    # toward one script, and the retention headline still looks healthy. This
    # is the C7 lesson generalised -- measure the thing that can rot.
    print("\nper-variant retention")
    bad = []
    for tag in sorted(seen):
        n, d = seen[tag], lost[tag]
        rate = d / n
        broken = rate > 0.5 and n >= 4
        if broken:
            bad.append(tag)
        mark = "  <-- BROKEN?" if broken else ""
        print(f"  {tag:<28} planned={n:>4}  dropped={d:>4}  ({rate:>5.1%}){mark}")
    if bad:
        print(f"\nFAIL: {len(bad)} variant(s) drop >50% of their traces: "
              f"{', '.join(bad)}")
        print("      A broken variant narrows the corpus silently -- fix it "
              "rather than shipping around it.")
        return 2

    if ok:
        print()
        audit(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
