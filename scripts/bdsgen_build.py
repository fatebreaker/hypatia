"""bdsgen_build.py (10-17, H-GEN27 clean training data) -- BioDSBench-style TRAINING tasks on public cBioPortal studies that
BioDSBench does not use (chosen and downloaded by scripts/bdsgen_fetch.py into generated/bdsgen_raw/<study>/).

Tables. Each study's data_clinical_patient.txt, data_clinical_sample.txt and data_mutations.txt are converted to CSV exactly
as BioDSBench ships its own tables (all columns; cBioPortal's '#' header lines dropped) into generated/bdsgen/data/<study>/,
read-only. NCI-TARGET studies are skipped (bdsgen_fetch.py meant to exclude them; three *_target_gdc studies slipped through
its group filter), and so are studies whose mutation table exceeds MAX_MUT_MB (the largest BioDSBench table is 39 MB; every
rollout copies the tables onto the node's /tmp).

Tasks. TEMPLATES below are the analyses BioDSBench asks for (cohort counts, most-mutated genes, mutation-type indicators,
substitution spectra, mutation burden, clinical distributions, Kaplan-Meier curves, log-rank, Cox, Fisher, multiple-testing
correction), worded in BioDSBench's way but with every definition stated, because an RL reward must not punish a defensible
reading of an ambiguous task. Each applicable template becomes one task: its reference program is RUN on the study's tables,
assert tests are written from the values it produces, and the task is kept only if
  (1) the reference passes the tests under the grader itself (scripts/bdsgen_reference_grader.py, fresh process), and
  (2) a PERTURBED reference (pandas.read_csv patched to drop every third row of every table) FAILS them,
so a solution that computes the right thing on the wrong data is not rewarded. The task text is BioDSBench's (TASK, copied
from scripts/fetch_biodsbench.py) and so is the deliverable (solution.py, graded by hidden asserts in a fresh process).
Output: generated/bdsgen/train/bdst-<study>-<template>/ (eval_config.json + data_files/ hard links) + MANIFEST.json.
    venv/bin/python scripts/bdsgen_build.py [--workers 6] [--studies a,b] [--dry]
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent
G = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated")
RAW, OUT = G / "bdsgen_raw", G / "bdsgen"
DATA = OUT / "data"
SEED, MAX_MUT_MB = 20261017, 40
spec = importlib.util.spec_from_file_location("bdsgen_grader", R / "scripts" / "bdsgen_reference_grader.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
assert grader.DATA == DATA, (grader.DATA, DATA)

# scripts/fetch_biodsbench.py's TASK, verbatim
TASK = """You are given tables from a published biomedical study and a data-science coding task.

Study: {title}
Dataset: {dataset}
Tables (CSV, under data_files/):
  {tables}
{prefix_block}
Task:
{query}
{cot_block}
What to deliver. Explore the data and develop your code in the notebook as usual. Your FINAL deliverable is a
self-contained Python program saved as the file solution.py in your working directory, for example with
    SOLUTION = r'''<your code>'''
    open("solution.py", "w").write(SOLUTION)
It is graded by hidden unit tests (assert statements over the variables the task asks for), run in a FRESH Python
process as:  {order}  -- so solution.py must itself do everything the task needs (imports, loading data from
data_files/, defining every requested variable with exactly the requested name). Nothing else from your notebook
session carries over. Plots are allowed (non-interactive backend) but are not graded unless the task says so.
Available packages include pandas, numpy, scipy, statsmodels, scikit-learn, lifelines, matplotlib and seaborn.
Check it before answering, e.g. exec(open("solution.py").read()) in a fresh namespace.

Return your answer as JSON: {{"answer": {{"{key}": "solution.py"}}}}"""
TABLES = ["data_mutations.csv", "data_clinical_sample.csv", "data_clinical_patient.csv"]
NONSYN = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
          "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
TRUNC = ["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
PERTURB = '''import pandas as _pd
_orig_read_csv = _pd.read_csv
def _read_csv_perturbed(*a, **k):
    _df = _orig_read_csv(*a, **k)
    return _df[[i % 3 != 0 for i in range(len(_df))]]
_pd.read_csv = _read_csv_perturbed
'''

# ---------------------------------------------------------------- shared code (reference programs) and wording
LOAD = '''import pandas as pd
import numpy as np
data_clinical_patient = pd.read_csv("data_files/data_clinical_patient.csv", low_memory=False)
data_clinical_sample = pd.read_csv("data_files/data_clinical_sample.csv", low_memory=False)
data_mutations = pd.read_csv("data_files/data_mutations.csv", low_memory=False)
'''
MAPPED = '''sample_map = data_clinical_sample[["SAMPLE_ID", "PATIENT_ID"]].astype(str).drop_duplicates()
mut = data_mutations.assign(Tumor_Sample_Barcode=data_mutations["Tumor_Sample_Barcode"].astype(str)).merge(
    sample_map, left_on="Tumor_Sample_Barcode", right_on="SAMPLE_ID", how="inner")
profiled = sorted(mut["PATIENT_ID"].unique())
genes = mut[mut["Hugo_Symbol"].notna() & (mut["Hugo_Symbol"] != "Unknown")]
'''
W_MAP = ("Map each mutation row to its patient by matching Tumor_Sample_Barcode in data_mutations.csv to SAMPLE_ID in "
         "data_clinical_sample.csv (drop rows whose sample is not listed there); the profiled patients are the patients with "
         "at least one mapped mutation row. Ignore rows whose Hugo_Symbol is missing or 'Unknown' when counting genes.")
W_MUTATED = "A patient is mutated in a gene if any mapped row of that gene belongs to the patient (any Variant_Classification)."
W_TOP = "Rank genes by the number of mutated patients (descending), ties broken by Hugo_Symbol (ascending)."


def surv_code(prefix, ev, ce, var="os_df"):
    return f'''{var} = data_clinical_patient[["PATIENT_ID", "{prefix}_MONTHS", "{prefix}_STATUS"]].copy()
{var}["PATIENT_ID"] = {var}["PATIENT_ID"].astype(str)
{var}["{prefix}_MONTHS"] = pd.to_numeric({var}["{prefix}_MONTHS"], errors="coerce")
{var} = {var}[{var}["{prefix}_STATUS"].isin({ev!r} + {ce!r}) & {var}["{prefix}_MONTHS"].notna()].copy()
{var}["EVENT"] = {var}["{prefix}_STATUS"].isin({ev!r}).astype(int)
'''


def surv_words(prefix, ev, ce, what):
    q = lambda xs: " or ".join(f"'{x}'" for x in xs)  # noqa: E731
    return (f"Use {prefix}_MONTHS (time) and {prefix}_STATUS from data_clinical_patient.csv: {prefix}_STATUS {q(ev)} is "
            f"{what} (an event) and {q(ce)} is censored; leave out patients with a missing {prefix}_MONTHS or any other "
            f"{prefix}_STATUS value.")


# ---------------------------------------------------------------- tests from probe values
def t_num(expr, v, rel=1e-6):
    if v == "nan" or v is None:
        raise ValueError("nan/None reference value")
    if v in ("inf", "-inf"):
        return f"assert math.isinf(float({expr})) and float({expr}) {'>' if v == 'inf' else '<'} 0"
    return f"assert abs(float({expr}) - {v!r}) <= {rel} * max(1.0, abs({v!r}))"


def t_eq(expr, v):
    return f"assert {expr} == {v!r}"


PROBE_HELPERS = '''import json as _json, math as _math
def _f(x):
    x = float(x)
    return "nan" if x != x else ("inf" if x == _math.inf else ("-inf" if x == -_math.inf else x))
'''


# ---------------------------------------------------------------- templates: ctx -> list of task specs
def T_cohort(c):
    if c["max_samples_per_patient"] < 2:
        return []
    q = ("How many patients and samples are in this study, and how many samples does each patient have? Save:\n"
         "- `n_patients`: the number of unique PATIENT_ID values in data_clinical_patient.csv\n"
         "- `n_samples`: the number of unique SAMPLE_ID values in data_clinical_sample.csv\n"
         "- `samples_per_patient`: a pd.DataFrame with columns `PATIENT_ID` and `N_SAMPLES` (the number of unique samples), "
         "one row per patient that appears in data_clinical_sample.csv")
    code = LOAD + '''n_patients = int(data_clinical_patient["PATIENT_ID"].nunique())
n_samples = int(data_clinical_sample["SAMPLE_ID"].nunique())
samples_per_patient = data_clinical_sample.groupby("PATIENT_ID")["SAMPLE_ID"].nunique().reset_index(name="N_SAMPLES")
'''
    probe = '''print("__PROBE__" + _json.dumps({"np": int(n_patients), "ns": int(n_samples), "rows": len(samples_per_patient),
    "vc": {str(int(k)): int(v) for k, v in samples_per_patient["N_SAMPLES"].value_counts().items()}}))'''

    def tests(p):
        return [t_eq("int(n_patients)", p["np"]), t_eq("int(n_samples)", p["ns"]),
                'assert {"PATIENT_ID", "N_SAMPLES"} <= set(samples_per_patient.columns)',
                t_eq("len(samples_per_patient)", p["rows"]),
                t_eq('{str(int(k)): int(v) for k, v in samples_per_patient["N_SAMPLES"].value_counts().items()}', p["vc"])]
    return [dict(tid="cohort", query=q, code=code, probe=probe, tests=tests)]


def T_topgenes(c):
    q = (f"Find the 10 genes mutated in the most patients. {W_MAP} {W_MUTATED} {W_TOP}\n"
         "Save the result as a pd.DataFrame named `mutated_genes` with 10 rows, in that order, and columns:\n"
         "- `Hugo_Symbol`: the gene\n- `Count`: the number of mutated patients\n"
         "- `Ratio`: Count divided by the number of profiled patients")
    code = LOAD + MAPPED + '''counts = genes.groupby("Hugo_Symbol")["PATIENT_ID"].nunique().reset_index(name="Count")
counts = counts.sort_values(["Count", "Hugo_Symbol"], ascending=[False, True]).head(10).reset_index(drop=True)
counts["Ratio"] = counts["Count"] / len(profiled)
mutated_genes = counts
'''
    probe = '''print("__PROBE__" + _json.dumps({"g": [str(x) for x in mutated_genes["Hugo_Symbol"]],
    "n": [int(x) for x in mutated_genes["Count"]], "r": _f(mutated_genes["Ratio"].sum())}))'''

    def tests(p):
        return [t_eq('[str(x) for x in mutated_genes["Hugo_Symbol"]]', p["g"]),
                t_eq('[int(x) for x in mutated_genes["Count"]]', p["n"]), t_num('mutated_genes["Ratio"].sum()', p["r"])]
    return [dict(tid="topgenes", query=q, code=code, probe=probe, tests=tests)]


def T_vclass(c):
    g = c.get("vclass_gene")
    if not g:
        return []
    q = (f"Get the distribution of mutation types (Variant_Classification) over all rows of data_mutations.csv whose "
         f"Hugo_Symbol is {g}. Save it as a pd.Series named `mutation_types`: the index is the Variant_Classification "
         f"and the value is the number of rows, sorted by the count (descending).")
    code = LOAD + f'''mutation_types = data_mutations.loc[data_mutations["Hugo_Symbol"] == {g!r}, "Variant_Classification"].value_counts()
'''
    probe = '''print("__PROBE__" + _json.dumps({"d": {str(k): int(v) for k, v in mutation_types.items()},
    "top": int(mutation_types.iloc[0])}))'''

    def tests(p):
        return [t_eq("{str(k): int(v) for k, v in mutation_types.items()}", p["d"]), t_eq("int(mutation_types.iloc[0])", p["top"])]
    return [dict(tid="vclass", query=q, code=code, probe=probe, tests=tests)]


def T_protein(c):
    g = c.get("hotspot_gene")
    if not g:
        return []
    q = (f"Look at the protein-level changes (HGVSp_Short) of the {g} mutations in data_mutations.csv (rows with a "
         f"missing HGVSp_Short are ignored). Save `n_changes`, the number of distinct protein changes; `top_change`, the "
         f"change seen in the most rows (ties: the alphabetically first); and `top_count`, its number of rows.")
    code = LOAD + f'''x = data_mutations.loc[data_mutations["Hugo_Symbol"] == {g!r}, "HGVSp_Short"].dropna()
n_changes = int(x.nunique())
vc = x.value_counts()
top_count = int(vc.max())
top_change = sorted(vc[vc == top_count].index)[0]
'''
    probe = '''print("__PROBE__" + _json.dumps({"n": int(n_changes), "c": str(top_change), "k": int(top_count)}))'''

    def tests(p):
        return [t_eq("int(n_changes)", p["n"]), t_eq("str(top_change)", p["c"]), t_eq("int(top_count)", p["k"])]
    return [dict(tid="protein", query=q, code=code, probe=probe, tests=tests)]


def T_spectrum(c):
    if not c["has_snp"]:
        return []
    q = ("For each tumor sample, compute the spectrum of single-nucleotide substitutions. Use the rows of "
         "data_mutations.csv with Variant_Type == 'SNP' whose Reference_Allele and Tumor_Seq_Allele2 are single, different "
         "bases among A, C, G and T. Report every substitution on the pyrimidine strand: when the reference base is G or A, "
         "replace both bases by their complements (G>T becomes C>A, A>G becomes T>C, and so on), which leaves six classes: "
         "C>A, C>G, C>T, T>A, T>C, T>G.\nSave a pd.DataFrame named `spectrum` indexed by Tumor_Sample_Barcode, one row per "
         "sample with at least one such substitution, with the six classes as columns in that order, holding the fraction "
         "of the sample's substitutions in each class (each row sums to 1).")
    code = '''import pandas as pd
data_mutations = pd.read_csv("data_files/data_mutations.csv", low_memory=False)
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
snv = data_mutations[data_mutations["Variant_Type"] == "SNP"].copy()
snv["ref"] = snv["Reference_Allele"].astype(str)
snv["alt"] = snv["Tumor_Seq_Allele2"].astype(str)
snv = snv[snv["ref"].isin(list("ACGT")) & snv["alt"].isin(list("ACGT")) & (snv["ref"] != snv["alt"])]
flip = snv["ref"].isin(["G", "A"])
snv.loc[flip, "ref"] = snv.loc[flip, "ref"].map(comp)
snv.loc[flip, "alt"] = snv.loc[flip, "alt"].map(comp)
snv["cls"] = snv["ref"] + ">" + snv["alt"]
classes = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
spectrum = pd.crosstab(snv["Tumor_Sample_Barcode"], snv["cls"], normalize="index").reindex(columns=classes, fill_value=0.0)
'''
    probe = '''print("__PROBE__" + _json.dumps({"n": int(spectrum.shape[0]), "s": {k: _f(spectrum[k].sum()) for k in spectrum.columns}}))'''

    def tests(p):
        return ([t_eq("list(spectrum.columns)", ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]), t_eq("int(spectrum.shape[0])", p["n"])]
                + [t_num(f'spectrum["{k}"].sum()', v) for k, v in p["s"].items()])
    return [dict(tid="spectrum", query=q, code=code, probe=probe, tests=tests)]


def T_tmb(c):
    q = ("Compute the tumour mutation burden per patient. For each sample in data_mutations.csv (Tumor_Sample_Barcode), "
         f"count its nonsynonymous mutations, i.e. rows whose Variant_Classification is one of {', '.join(NONSYN)}; every "
         "sample that appears in data_mutations.csv is counted, with 0 if it has none of these. Map samples to patients "
         "through SAMPLE_ID in data_clinical_sample.csv (drop samples not listed there) and average the counts over each "
         "patient's samples. Save a pd.DataFrame named `tmb` with columns `PATIENT_ID` and `Mean`, one row per patient.")
    code = LOAD + f'''nonsyn = {NONSYN!r}
dm = data_mutations.assign(Tumor_Sample_Barcode=data_mutations["Tumor_Sample_Barcode"].astype(str))
per_sample = dm.groupby("Tumor_Sample_Barcode")["Variant_Classification"].apply(lambda s: int(s.isin(nonsyn).sum()))
per_sample = per_sample.rename("N").reset_index()
sample_map = data_clinical_sample[["SAMPLE_ID", "PATIENT_ID"]].astype(str).drop_duplicates()
per_sample = per_sample.merge(sample_map, left_on="Tumor_Sample_Barcode", right_on="SAMPLE_ID", how="inner")
tmb = per_sample.groupby("PATIENT_ID")["N"].mean().reset_index(name="Mean")
'''
    probe = '''print("__PROBE__" + _json.dumps({"n": len(tmb), "med": _f(tmb["Mean"].median()), "sum": _f(tmb["Mean"].sum()),
    "max": _f(tmb["Mean"].max())}))'''

    def tests(p):
        return [t_eq("len(tmb)", p["n"]), t_num('tmb["Mean"].median()', p["med"]), t_num('tmb["Mean"].sum()', p["sum"]),
                t_num('tmb["Mean"].max()', p["max"])]
    return [dict(tid="tmb", query=q, code=code, probe=probe, tests=tests)]


def T_onco(c):
    q = (f"Build mutation-type indicators for the 5 genes mutated in the most patients. {W_MAP} {W_MUTATED} {W_TOP} For "
         "each profiled patient and each of the 5 genes, count the patient's mapped rows of that gene in three classes:\n"
         "- `Missense`: Missense_Mutation\n"
         f"- `Truncating`: {', '.join(TRUNC)}\n"
         "- `Inframe`: In_Frame_Del, In_Frame_Ins\n"
         "Save a pd.DataFrame named `onco` with columns `PATIENT_ID`, `Hugo_Symbol`, `Missense`, `Truncating`, `Inframe` "
         "and one row for every (profiled patient, gene) pair, with 0 where the patient has no such mutation.")
    code = LOAD + MAPPED + f'''counts = genes.groupby("Hugo_Symbol")["PATIENT_ID"].nunique().reset_index(name="Count")
top5 = counts.sort_values(["Count", "Hugo_Symbol"], ascending=[False, True]).head(5)["Hugo_Symbol"].tolist()
classes = {{"Missense": ["Missense_Mutation"], "Truncating": {TRUNC!r}, "Inframe": ["In_Frame_Del", "In_Frame_Ins"]}}
onco = pd.MultiIndex.from_product([profiled, top5], names=["PATIENT_ID", "Hugo_Symbol"]).to_frame(index=False)
g = genes[genes["Hugo_Symbol"].isin(top5)]
for name, vcs in classes.items():
    cnt = g[g["Variant_Classification"].isin(vcs)].groupby(["PATIENT_ID", "Hugo_Symbol"]).size().rename(name).reset_index()
    onco = onco.merge(cnt, on=["PATIENT_ID", "Hugo_Symbol"], how="left")
    onco[name] = onco[name].fillna(0).astype(int)
'''
    probe = '''print("__PROBE__" + _json.dumps({"rows": len(onco), "genes": sorted(str(x) for x in onco["Hugo_Symbol"].unique()),
    "by": {k: {str(g): int(v) for g, v in onco.groupby("Hugo_Symbol")[k].sum().items()} for k in ["Missense", "Truncating", "Inframe"]}}))'''

    def tests(p):
        out = [t_eq("len(onco)", p["rows"]), t_eq('sorted(str(x) for x in onco["Hugo_Symbol"].unique())', p["genes"])]
        for k, d in p["by"].items():
            out.append(t_eq(f'{{str(g): int(v) for g, v in onco.groupby("Hugo_Symbol")["{k}"].sum().items()}}', d))
        return out
    return [dict(tid="onco", query=q, code=code, probe=probe, tests=tests)]


def T_dist(c):
    out = []
    for i, (table, attr) in enumerate(c["dist_attrs"]):
        unit = "patients" if table == "patient" else "samples"
        q = (f"Get the distribution of {attr} over the {unit} in data_clinical_{table}.csv, leaving out missing values. "
             f"Save a pd.DataFrame named `dist` with columns:\n- `{attr}`: the category\n- `Count`: the number of {unit}\n"
             f"- `Ratio`: Count as a percentage of the {unit} with a non-missing {attr}\n"
             f"sorted by Count (descending), ties broken by the category (ascending).")
        code = LOAD + f'''x = data_clinical_{table}[{attr!r}].dropna().astype(str)
dist = x.value_counts().rename_axis({attr!r}).reset_index(name="Count")
dist["Ratio"] = 100 * dist["Count"] / dist["Count"].sum()
dist = dist.sort_values(["Count", {attr!r}], ascending=[False, True]).reset_index(drop=True)
'''
        probe = f'''print("__PROBE__" + _json.dumps({{"d": {{str(k): int(v) for k, v in zip(dist[{attr!r}], dist["Count"])}},
    "order": [str(x) for x in dist[{attr!r}]], "r": _f(dist["Ratio"].sum())}}))'''

        def tests(p, attr=attr):
            return [t_eq(f'{{str(k): int(v) for k, v in zip(dist[{attr!r}], dist["Count"])}}', p["d"]),
                    t_eq(f'[str(x) for x in dist[{attr!r}]]', p["order"]), t_num('dist["Ratio"].sum()', p["r"])]
        out.append(dict(tid=f"dist{i + 1}", query=q, code=code, probe=probe, tests=tests))
    return out


def T_age(c):
    a = c.get("age_col")
    if not a:
        return []
    q = (f"Summarise the patients' age ({a} in data_clinical_patient.csv; ignore missing or non-numeric values). Save "
         f"`age_median` (the median), `age_range` (a tuple (min, max)) and `n_age` (the number of patients with an age).")
    code = LOAD + f'''a = pd.to_numeric(data_clinical_patient[{a!r}], errors="coerce").dropna()
age_median = float(a.median())
age_range = (float(a.min()), float(a.max()))
n_age = int(len(a))
'''
    probe = '''print("__PROBE__" + _json.dumps({"m": _f(age_median), "lo": _f(age_range[0]), "hi": _f(age_range[1]), "n": int(n_age)}))'''

    def tests(p):
        return [t_num("age_median", p["m"]), t_num("age_range[0]", p["lo"]), t_num("age_range[1]", p["hi"]), t_eq("int(n_age)", p["n"])]
    return [dict(tid="age", query=q, code=code, probe=probe, tests=tests)]


def _km_all(tid, prefix, ev, ce, what, label):
    q = (f"Estimate {label} for all patients with a Kaplan-Meier fit. {surv_words(prefix, ev, ce, what)} Save the fitted "
         f"lifelines KaplanMeierFitter as `kmf`, the median {label} as `median_time`, and its 95% confidence interval as "
         f"`ci_low` and `ci_high` (lifelines.utils.median_survival_times applied to kmf.confidence_interval_; inf where the "
         f"curve or its bound does not fall to 0.5).")
    code = LOAD + "from lifelines import KaplanMeierFitter\nfrom lifelines.utils import median_survival_times\n" + \
        surv_code(prefix, ev, ce, "d") + f'''kmf = KaplanMeierFitter()
kmf.fit(d["{prefix}_MONTHS"], event_observed=d["EVENT"])
median_time = kmf.median_survival_time_
ci = median_survival_times(kmf.confidence_interval_)
ci_low, ci_high = ci.iloc[0, 0], ci.iloc[0, 1]
'''
    probe = '''print("__PROBE__" + _json.dumps({"n": int(len(kmf.durations)), "e": int(kmf.event_observed.sum()), "m": _f(median_time),
    "lo": _f(ci_low), "hi": _f(ci_high)}))'''

    def tests(p):
        return [t_eq("int(len(kmf.durations))", p["n"]), t_eq("int(kmf.event_observed.sum())", p["e"]),
                t_num("median_time", p["m"]), t_num("ci_low", p["lo"]), t_num("ci_high", p["hi"])]
    return dict(tid=tid, query=q, code=code, probe=probe, tests=tests)


def T_km_all(c):
    out = []
    if c.get("os"):
        out.append(_km_all("km_os", "OS", *c["os"], "a death", "overall survival"))
    for prefix, (ev, ce) in c.get("other_surv", {}).items():
        label = {"PFS": "progression-free survival", "DFS": "disease-free survival"}[prefix]
        out.append(_km_all(f"km_{prefix.lower()}", prefix, ev, ce, "progression or death" if prefix == "PFS" else
                           "recurrence, progression or death", label))
    return out


def _cohort_words():
    return (f"{W_MAP} The cohort is the profiled patients with a usable overall-survival record. {W_MUTATED}")


def T_km_gene(c):
    g = c.get("km_gene")
    if not g or not c.get("os"):
        return []
    ev, ce = c["os"]
    q = (f"Compare overall survival between patients with and without a {g} mutation. {_cohort_words()} "
         f"{surv_words('OS', ev, ce, 'a death')} Fit lifelines KaplanMeierFitter curves for the {g}-mutated and the other "
         f"patients, saved as `kmf_mut` and `kmf_wild`, and compare them with lifelines.statistics.logrank_test; save its "
         f"p-value as `pvalue`.")
    code = LOAD + "from lifelines import KaplanMeierFitter\nfrom lifelines.statistics import logrank_test\n" + MAPPED + \
        surv_code("OS", ev, ce) + f'''os_df = os_df[os_df["PATIENT_ID"].isin(profiled)]
mutated = set(genes.loc[genes["Hugo_Symbol"] == {g!r}, "PATIENT_ID"])
m = os_df["PATIENT_ID"].isin(mutated)
kmf_mut = KaplanMeierFitter().fit(os_df.loc[m, "OS_MONTHS"], os_df.loc[m, "EVENT"], label="mutated")
kmf_wild = KaplanMeierFitter().fit(os_df.loc[~m, "OS_MONTHS"], os_df.loc[~m, "EVENT"], label="wild type")
pvalue = logrank_test(os_df.loc[m, "OS_MONTHS"], os_df.loc[~m, "OS_MONTHS"], os_df.loc[m, "EVENT"], os_df.loc[~m, "EVENT"]).p_value
'''
    probe = '''print("__PROBE__" + _json.dumps({"nm": int(len(kmf_mut.durations)), "em": int(kmf_mut.event_observed.sum()),
    "nw": int(len(kmf_wild.durations)), "ew": int(kmf_wild.event_observed.sum()), "p": _f(pvalue)}))'''

    def tests(p):
        return [t_eq("int(len(kmf_mut.durations))", p["nm"]), t_eq("int(kmf_mut.event_observed.sum())", p["em"]),
                t_eq("int(len(kmf_wild.durations))", p["nw"]), t_eq("int(kmf_wild.event_observed.sum())", p["ew"]),
                t_num("pvalue", p["p"])]
    return [dict(tid="km_gene", query=q, code=code, probe=probe, tests=tests)]


def T_cox_gene(c):
    g = c.get("cox_gene")
    if not g or not c.get("os"):
        return []
    ev, ce = c["os"]
    q = (f"Fit a univariate Cox proportional-hazards model of overall survival on {g} mutation status (1 = mutated, "
         f"0 = not). {_cohort_words()} {surv_words('OS', ev, ce, 'a death')} Use lifelines CoxPHFitter with its default "
         f"settings. Save the coefficient as `coef`, the hazard ratio as `hr`, its 95% confidence interval as `hr_low` and "
         f"`hr_high`, and the p-value as `p_value`.")
    code = LOAD + "from lifelines import CoxPHFitter\n" + MAPPED + surv_code("OS", ev, ce) + f'''os_df = os_df[os_df["PATIENT_ID"].isin(profiled)]
mutated = set(genes.loc[genes["Hugo_Symbol"] == {g!r}, "PATIENT_ID"])
os_df["MUT"] = os_df["PATIENT_ID"].isin(mutated).astype(int)
cph = CoxPHFitter()
cph.fit(os_df[["OS_MONTHS", "EVENT", "MUT"]], duration_col="OS_MONTHS", event_col="EVENT")
s = cph.summary.loc["MUT"]
coef, hr = float(s["coef"]), float(s["exp(coef)"])
hr_low, hr_high = float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])
p_value = float(s["p"])
'''
    probe = '''print("__PROBE__" + _json.dumps({"c": _f(coef), "h": _f(hr), "lo": _f(hr_low), "hi": _f(hr_high), "p": _f(p_value)}))'''

    def tests(p):
        return [t_num("coef", p["c"], 1e-4), t_num("hr", p["h"], 1e-4), t_num("hr_low", p["lo"], 1e-4),
                t_num("hr_high", p["hi"], 1e-4), t_num("p_value", p["p"], 1e-4)]
    return [dict(tid="cox_gene", query=q, code=code, probe=probe, tests=tests)]


def T_fisher_sex(c):
    g, sx = c.get("sex_gene"), c.get("sex")
    if not g or not sx:
        return []
    col, (mv, fv) = sx
    q = (f"Is {g} mutated more often in one sex? {W_MAP} {W_MUTATED} Among the profiled patients whose {col} in "
         f"data_clinical_patient.csv is '{mv}' or '{fv}', build the 2x2 table [[mutated '{mv}', not mutated '{mv}'], "
         f"[mutated '{fv}', not mutated '{fv}']] and run scipy.stats.fisher_exact on it (two-sided). Save `odds_ratio` and "
         f"`pvalue`, and the fractions of mutated patients `frac_male` ('{mv}') and `frac_female` ('{fv}').")
    code = LOAD + "from scipy.stats import fisher_exact\n" + MAPPED + f'''pt = data_clinical_patient[["PATIENT_ID", {col!r}]].copy()
pt["PATIENT_ID"] = pt["PATIENT_ID"].astype(str)
pt = pt[pt["PATIENT_ID"].isin(profiled) & pt[{col!r}].isin([{mv!r}, {fv!r}])]
mutated = set(genes.loc[genes["Hugo_Symbol"] == {g!r}, "PATIENT_ID"])
pt["MUT"] = pt["PATIENT_ID"].isin(mutated)
male, female = pt[pt[{col!r}] == {mv!r}], pt[pt[{col!r}] == {fv!r}]
table = [[int(male["MUT"].sum()), int((~male["MUT"]).sum())], [int(female["MUT"].sum()), int((~female["MUT"]).sum())]]
odds_ratio, pvalue = fisher_exact(table)
frac_male, frac_female = float(male["MUT"].mean()), float(female["MUT"].mean())
'''
    probe = '''print("__PROBE__" + _json.dumps({"o": _f(odds_ratio), "p": _f(pvalue), "m": _f(frac_male), "f": _f(frac_female)}))'''

    def tests(p):
        return [t_num("odds_ratio", p["o"]), t_num("pvalue", p["p"]), t_num("frac_male", p["m"]), t_num("frac_female", p["f"])]
    return [dict(tid="fisher_sex", query=q, code=code, probe=probe, tests=tests)]


def T_cooccur(c):
    pair = c.get("cooccur_pair")
    if not pair:
        return []
    g1, g2 = pair
    q = (f"Do {g1} and {g2} mutations co-occur? {W_MAP} {W_MUTATED} Among the profiled patients, count those mutated in "
         f"both genes (`n_both`), in {g1} only (`n_g1_only`), in {g2} only (`n_g2_only`) and in neither (`n_neither`), "
         f"and run scipy.stats.fisher_exact (two-sided) on the table [[n_both, n_g1_only], [n_g2_only, n_neither]]. Save "
         f"the four counts and the p-value as `pvalue`.")
    code = LOAD + "from scipy.stats import fisher_exact\n" + MAPPED + f'''m1 = set(genes.loc[genes["Hugo_Symbol"] == {g1!r}, "PATIENT_ID"])
m2 = set(genes.loc[genes["Hugo_Symbol"] == {g2!r}, "PATIENT_ID"])
P = set(profiled)
n_both = len(P & m1 & m2)
n_g1_only = len((P & m1) - m2)
n_g2_only = len((P & m2) - m1)
n_neither = len(P - m1 - m2)
_, pvalue = fisher_exact([[n_both, n_g1_only], [n_g2_only, n_neither]])
'''
    probe = '''print("__PROBE__" + _json.dumps({"b": n_both, "a": n_g1_only, "c": n_g2_only, "n": n_neither, "p": _f(pvalue)}))'''

    def tests(p):
        return [t_eq("int(n_both)", p["b"]), t_eq("int(n_g1_only)", p["a"]), t_eq("int(n_g2_only)", p["c"]),
                t_eq("int(n_neither)", p["n"]), t_num("pvalue", p["p"])]
    return [dict(tid="cooccur", query=q, code=code, probe=probe, tests=tests)]


def T_fdr(c):
    if not c.get("fdr_ok") or not c.get("os"):
        return []
    ev, ce = c["os"]
    q = (f"For each of the 5 genes mutated in the most profiled patients, compare overall survival between mutated and "
         f"non-mutated patients. {_cohort_words()} {W_TOP} {surv_words('OS', ev, ce, 'a death')} Test each gene with "
         f"lifelines.statistics.logrank_test, then adjust the five p-values with the Benjamini-Hochberg procedure "
         f"(statsmodels.stats.multitest.multipletests, method='fdr_bh'). Save a pd.DataFrame named `results` with columns "
         f"`Hugo_Symbol`, `N_MUT` (mutated patients in the cohort), `PVALUE` and `FDR`, one row per gene in rank order.")
    code = LOAD + "from lifelines.statistics import logrank_test\nfrom statsmodels.stats.multitest import multipletests\n" + \
        MAPPED + surv_code("OS", ev, ce) + '''os_df = os_df[os_df["PATIENT_ID"].isin(profiled)]
counts = genes.groupby("Hugo_Symbol")["PATIENT_ID"].nunique().reset_index(name="Count")
top5 = counts.sort_values(["Count", "Hugo_Symbol"], ascending=[False, True]).head(5)["Hugo_Symbol"].tolist()
rows = []
for g in top5:
    m = os_df["PATIENT_ID"].isin(set(genes.loc[genes["Hugo_Symbol"] == g, "PATIENT_ID"]))
    p = logrank_test(os_df.loc[m, "OS_MONTHS"], os_df.loc[~m, "OS_MONTHS"], os_df.loc[m, "EVENT"], os_df.loc[~m, "EVENT"]).p_value
    rows.append({"Hugo_Symbol": g, "N_MUT": int(m.sum()), "PVALUE": p})
results = pd.DataFrame(rows)
results["FDR"] = multipletests(results["PVALUE"], method="fdr_bh")[1]
'''
    probe = '''print("__PROBE__" + _json.dumps({"g": [str(x) for x in results["Hugo_Symbol"]], "n": [int(x) for x in results["N_MUT"]],
    "p": [_f(x) for x in results["PVALUE"]], "q": [_f(x) for x in results["FDR"]]}))'''

    def tests(p):
        out = [t_eq('[str(x) for x in results["Hugo_Symbol"]]', p["g"]), t_eq('[int(x) for x in results["N_MUT"]]', p["n"])]
        out += [t_num(f'results["PVALUE"].iloc[{i}]', v) for i, v in enumerate(p["p"])]
        out += [t_num(f'results["FDR"].iloc[{i}]', v) for i, v in enumerate(p["q"])]
        return out
    return [dict(tid="fdr", query=q, code=code, probe=probe, tests=tests)]


def T_mwu_sex(c):
    sx = c.get("sex")
    if not sx or not c.get("mwu_ok"):
        return []
    col, (mv, fv) = sx
    q = (f"Compare the number of nonsynonymous mutations per patient between '{mv}' and '{fv}' patients ({col} in "
         f"data_clinical_patient.csv). {W_MAP} Nonsynonymous rows are those whose Variant_Classification is one of "
         f"{', '.join(NONSYN)}; a patient's count is the number of such mapped rows over all of the patient's samples (0 "
         f"if none). Use the profiled patients with {col} '{mv}' or '{fv}', and scipy.stats.mannwhitneyu (two-sided, "
         f"default settings). Save `pvalue`, and the median counts `median_male` ('{mv}') and `median_female` ('{fv}').")
    code = LOAD + "from scipy.stats import mannwhitneyu\n" + MAPPED + f'''nonsyn = {NONSYN!r}
cnt = mut[mut["Variant_Classification"].isin(nonsyn)].groupby("PATIENT_ID").size()
pt = data_clinical_patient[["PATIENT_ID", {col!r}]].copy()
pt["PATIENT_ID"] = pt["PATIENT_ID"].astype(str)
pt = pt[pt["PATIENT_ID"].isin(profiled) & pt[{col!r}].isin([{mv!r}, {fv!r}])]
pt["N"] = pt["PATIENT_ID"].map(cnt).fillna(0)
a, b = pt.loc[pt[{col!r}] == {mv!r}, "N"], pt.loc[pt[{col!r}] == {fv!r}, "N"]
pvalue = mannwhitneyu(a, b, alternative="two-sided").pvalue
median_male, median_female = float(a.median()), float(b.median())
'''
    probe = '''print("__PROBE__" + _json.dumps({"p": _f(pvalue), "m": _f(median_male), "f": _f(median_female)}))'''

    def tests(p):
        return [t_num("pvalue", p["p"]), t_num("median_male", p["m"]), t_num("median_female", p["f"])]
    return [dict(tid="mwu_sex", query=q, code=code, probe=probe, tests=tests)]


def T_km_sex(c):
    sx = c.get("sex")
    if not sx or not c.get("os") or not c.get("km_sex_ok"):
        return []
    col, (mv, fv) = sx
    ev, ce = c["os"]
    q = (f"Compare overall survival between '{mv}' and '{fv}' patients ({col} in data_clinical_patient.csv), over all "
         f"patients with a usable overall-survival record and {col} '{mv}' or '{fv}'. {surv_words('OS', ev, ce, 'a death')} "
         f"Save lifelines KaplanMeierFitter curves as `kmf_male` ('{mv}') and `kmf_female` ('{fv}'), and the "
         f"lifelines.statistics.logrank_test p-value as `pvalue`.")
    code = LOAD + "from lifelines import KaplanMeierFitter\nfrom lifelines.statistics import logrank_test\n" + \
        surv_code("OS", ev, ce) + f'''sex = data_clinical_patient[["PATIENT_ID", {col!r}]].copy()
sex["PATIENT_ID"] = sex["PATIENT_ID"].astype(str)
os_df = os_df.merge(sex, on="PATIENT_ID")
a, b = os_df[os_df[{col!r}] == {mv!r}], os_df[os_df[{col!r}] == {fv!r}]
kmf_male = KaplanMeierFitter().fit(a["OS_MONTHS"], a["EVENT"], label={mv!r})
kmf_female = KaplanMeierFitter().fit(b["OS_MONTHS"], b["EVENT"], label={fv!r})
pvalue = logrank_test(a["OS_MONTHS"], b["OS_MONTHS"], a["EVENT"], b["EVENT"]).p_value
'''
    probe = '''print("__PROBE__" + _json.dumps({"nm": int(len(kmf_male.durations)), "em": int(kmf_male.event_observed.sum()),
    "nf": int(len(kmf_female.durations)), "ef": int(kmf_female.event_observed.sum()), "p": _f(pvalue)}))'''

    def tests(p):
        return [t_eq("int(len(kmf_male.durations))", p["nm"]), t_eq("int(kmf_male.event_observed.sum())", p["em"]),
                t_eq("int(len(kmf_female.durations))", p["nf"]), t_eq("int(kmf_female.event_observed.sum())", p["ef"]),
                t_num("pvalue", p["p"])]
    return [dict(tid="km_sex", query=q, code=code, probe=probe, tests=tests)]


def T_cox_age(c):
    a = c.get("age_col")
    if not a or not c.get("os"):
        return []
    ev, ce = c["os"]
    q = (f"Is age associated with overall survival? Over all patients with a usable overall-survival record and a numeric "
         f"{a} in data_clinical_patient.csv, fit a Cox proportional-hazards model of overall survival on {a} (in years, "
         f"as a continuous covariate) with lifelines CoxPHFitter and its default settings. "
         f"{surv_words('OS', ev, ce, 'a death')} Save the hazard ratio per year as `hr`, its 95% confidence interval as "
         f"`hr_low` and `hr_high`, the p-value as `p_value`, and the number of patients in the model as `n`.")
    code = LOAD + "from lifelines import CoxPHFitter\n" + surv_code("OS", ev, ce) + f'''age = data_clinical_patient[["PATIENT_ID", {a!r}]].copy()
age["PATIENT_ID"] = age["PATIENT_ID"].astype(str)
age["AGE_YEARS"] = pd.to_numeric(age[{a!r}], errors="coerce")
d = os_df.merge(age[["PATIENT_ID", "AGE_YEARS"]], on="PATIENT_ID").dropna(subset=["AGE_YEARS"])
cph = CoxPHFitter()
cph.fit(d[["OS_MONTHS", "EVENT", "AGE_YEARS"]], duration_col="OS_MONTHS", event_col="EVENT")
s = cph.summary.loc["AGE_YEARS"]
hr, hr_low, hr_high = float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])
p_value, n = float(s["p"]), int(len(d))
'''
    probe = '''print("__PROBE__" + _json.dumps({"h": _f(hr), "lo": _f(hr_low), "hi": _f(hr_high), "p": _f(p_value), "n": n}))'''

    def tests(p):
        return [t_num("hr", p["h"], 1e-4), t_num("hr_low", p["lo"], 1e-4), t_num("hr_high", p["hi"], 1e-4),
                t_num("p_value", p["p"], 1e-4), t_eq("int(n)", p["n"])]
    return [dict(tid="cox_age", query=q, code=code, probe=probe, tests=tests)]


def T_followup(c):
    if not c.get("os") or not c.get("followup_ok"):
        return []
    ev, ce = c["os"]
    q = (f"What is the median follow-up of this cohort? Use the reverse Kaplan-Meier method over all patients with a usable "
         f"overall-survival record: fit a lifelines KaplanMeierFitter on OS_MONTHS with the event indicator reversed (a "
         f"censored patient counts as the event, a death as censored). {surv_words('OS', ev, ce, 'a death')} Save the "
         f"fitted model as `kmf_followup` and its median (median_survival_time_) as `median_followup`.")
    code = LOAD + "from lifelines import KaplanMeierFitter\n" + surv_code("OS", ev, ce) + '''kmf_followup = KaplanMeierFitter()
kmf_followup.fit(os_df["OS_MONTHS"], event_observed=1 - os_df["EVENT"])
median_followup = kmf_followup.median_survival_time_
'''
    probe = '''print("__PROBE__" + _json.dumps({"n": int(len(kmf_followup.durations)), "e": int(kmf_followup.event_observed.sum()),
    "m": _f(median_followup)}))'''

    def tests(p):
        return [t_eq("int(len(kmf_followup.durations))", p["n"]), t_eq("int(kmf_followup.event_observed.sum())", p["e"]),
                t_num("median_followup", p["m"])]
    return [dict(tid="followup", query=q, code=code, probe=probe, tests=tests)]


TEMPLATES = [T_cohort, T_topgenes, T_vclass, T_protein, T_spectrum, T_tmb, T_onco, T_dist, T_age, T_km_all, T_km_gene,
             T_cox_gene, T_fisher_sex, T_cooccur, T_fdr, T_mwu_sex, T_km_sex, T_cox_age, T_followup]
DIST_CANDIDATES = [("patient", "SEX"), ("patient", "RACE"), ("patient", "ETHNICITY"), ("patient", "SMOKING_HISTORY"),
                   ("patient", "STAGE"), ("sample", "SAMPLE_TYPE"), ("sample", "CANCER_TYPE_DETAILED"),
                   ("sample", "ONCOTREE_CODE"), ("sample", "PRIMARY_SITE"), ("sample", "METASTATIC_SITE")]


# ---------------------------------------------------------------- per-study context (what applies, which genes)
def convert(sid):
    """cBioPortal txt -> CSV as BioDSBench ships its tables (pandas round trip, all columns, '#' lines dropped)."""
    src, dst = RAW / sid, DATA / sid
    mfile = next((f for f in ("data_mutations.txt", "data_mutations_extended.txt") if (src / f).exists()), None)
    if mfile is None:
        return None, "no mutation table"
    if (src / mfile).stat().st_size > MAX_MUT_MB * 2 ** 20:
        return None, f"mutation table > {MAX_MUT_MB} MB"
    dst.mkdir(parents=True, exist_ok=True)
    for name, f in (("data_clinical_patient", "data_clinical_patient.txt"), ("data_clinical_sample", "data_clinical_sample.txt"),
                    ("data_mutations", mfile)):
        out = dst / f"{name}.csv"
        if out.exists():
            continue
        with open(src / f, errors="replace") as fh:
            skip = 0
            for line in fh:
                if not line.startswith("#"):
                    break
                skip += 1
        tmp = dst / f"{name}.csv.part"
        pd.read_csv(src / f, sep="\t", skiprows=skip, low_memory=False, quoting=3).to_csv(tmp, index=False)   # plain TSV
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        tmp.rename(out)
    return dst, None


def status_sets(series, kind):
    vals = set(series.dropna().astype(str))
    if kind == "OS":
        ev = sorted(v for v in vals if v in ("1:DECEASED", "DECEASED"))
        ce = sorted(v for v in vals if v in ("0:LIVING", "LIVING"))
    else:
        ev = sorted(v for v in vals if v.startswith("1:"))
        ce = sorted(v for v in vals if v.startswith("0:"))
    return ev, ce


def context(sid, rng):
    d = DATA / sid
    pat = pd.read_csv(d / "data_clinical_patient.csv", low_memory=False)
    smp = pd.read_csv(d / "data_clinical_sample.csv", low_memory=False)
    mut = pd.read_csv(d / "data_mutations.csv", low_memory=False)
    need = {"Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"}
    if not need <= set(mut.columns) or not {"PATIENT_ID"} <= set(pat.columns) or not {"PATIENT_ID", "SAMPLE_ID"} <= set(smp.columns):
        return None, "missing core columns"
    c = {"sid": sid}
    smap = smp[["SAMPLE_ID", "PATIENT_ID"]].astype(str).drop_duplicates()
    m = mut.assign(Tumor_Sample_Barcode=mut["Tumor_Sample_Barcode"].astype(str)).merge(
        smap, left_on="Tumor_Sample_Barcode", right_on="SAMPLE_ID", how="inner")
    c["map_coverage"] = round(len(m) / max(len(mut), 1), 3)
    profiled = set(m["PATIENT_ID"])
    if len(profiled) < 20:
        return None, f"only {len(profiled)} profiled patients"
    genes = m[m["Hugo_Symbol"].notna() & (m["Hugo_Symbol"] != "Unknown")]
    gp = genes.groupby("Hugo_Symbol")["PATIENT_ID"].nunique().reset_index(name="n")
    gp = gp.sort_values(["n", "Hugo_Symbol"], ascending=[False, True])
    ranked = [(str(g), int(n)) for g, n in zip(gp["Hugo_Symbol"], gp["n"])]
    pmut = {g: set(genes.loc[genes["Hugo_Symbol"] == g, "PATIENT_ID"]) for g, _ in ranked[:30]}
    c["max_samples_per_patient"] = int(smp.groupby("PATIENT_ID")["SAMPLE_ID"].nunique().max())
    # variant classes / hotspot
    for g, n in ranked[:10]:
        if mut.loc[mut["Hugo_Symbol"] == g, "Variant_Classification"].nunique() >= 2:
            c["vclass_gene"] = g
            break
    if "HGVSp_Short" in mut.columns:
        best = None
        for g, _ in ranked[:30]:
            vc = mut.loc[mut["Hugo_Symbol"] == g, "HGVSp_Short"].dropna().value_counts()
            if len(vc) and vc.max() >= 3 and (best is None or vc.max() > best[1]):
                best = (g, int(vc.max()))
        c["hotspot_gene"] = best[0] if best else None
    c["has_snp"] = {"Variant_Type", "Reference_Allele", "Tumor_Seq_Allele2"} <= set(mut.columns) and \
        int((mut["Variant_Type"] == "SNP").sum()) >= 50
    # clinical attributes
    cands = []
    for table, attr in DIST_CANDIDATES:
        df = pat if table == "patient" else smp
        if attr in df.columns and df[attr].dtype == object:
            k = df[attr].dropna().nunique()
            if 2 <= k <= 15 and df[attr].notna().sum() >= 20:
                cands.append((table, attr))
    c["dist_attrs"] = sorted(rng.sample(cands, min(2, len(cands))))
    age = next((a for a in ("AGE", "AGE_AT_DIAGNOSIS") if a in pat.columns and
                pd.to_numeric(pat[a], errors="coerce").notna().sum() >= 20), None)
    c["age_col"] = age
    # sex
    if "SEX" in pat.columns:
        vals = set(pat["SEX"].dropna().astype(str))
        mv = next((v for v in vals if v.lower() in ("male", "m")), None)
        fv = next((v for v in vals if v.lower() in ("female", "f")), None)
        c["sex"] = ("SEX", (mv, fv)) if mv and fv else None
    else:
        c["sex"] = None
    sexed = {}
    if c["sex"]:
        mv, fv = c["sex"][1]
        sexed = {str(p): s for p, s in zip(pat["PATIENT_ID"].astype(str), pat["SEX"].astype(str)) if s in (mv, fv)}
    # survival
    c["os"], c["other_surv"] = None, {}
    os_ok = set()
    if {"OS_MONTHS", "OS_STATUS"} <= set(pat.columns):
        ev, ce = status_sets(pat["OS_STATUS"], "OS")
        t = pd.to_numeric(pat["OS_MONTHS"], errors="coerce")
        ok = pat["OS_STATUS"].isin(ev + ce) & t.notna() & (t >= 0)
        if ev and ok.sum() >= 20 and pat.loc[ok, "OS_STATUS"].isin(ev).sum() >= 5 and (t[ok] >= 0).all() and \
                (t[pat["OS_STATUS"].isin(ev + ce) & t.notna()] >= 0).all():
            c["os"] = (ev, ce)
            os_ok = set(pat.loc[ok, "PATIENT_ID"].astype(str))
            c["followup_ok"] = pat.loc[ok, "OS_STATUS"].isin(ce).sum() >= 5
    for pre in ("PFS", "DFS"):
        if {f"{pre}_MONTHS", f"{pre}_STATUS"} <= set(pat.columns):
            ev, ce = status_sets(pat[f"{pre}_STATUS"], pre)
            vals = set(pat[f"{pre}_STATUS"].dropna().astype(str))
            t = pd.to_numeric(pat[f"{pre}_MONTHS"], errors="coerce")
            ok = pat[f"{pre}_STATUS"].isin(ev + ce) & t.notna()
            if ev and ce and vals <= set(ev + ce) and ok.sum() >= 20 and pat.loc[ok, f"{pre}_STATUS"].isin(ev).sum() >= 5 \
                    and (t[ok] >= 0).all():
                c["other_surv"][pre] = (ev, ce)
    cohort = profiled & os_ok
    elig = [g for g, _ in ranked[:30] if g in pmut and len(pmut[g] & cohort) >= 8 and len(cohort - pmut[g]) >= 8]
    c["km_gene"] = elig[0] if elig else None
    c["cox_gene"] = (elig[1] if len(elig) > 1 else elig[0]) if elig else None
    if c["sex"]:
        sx_prof = {p for p in profiled if p in sexed}
        ms = {p for p in sx_prof if sexed[p] == c["sex"][1][0]}
        fs = sx_prof - ms
        c["mwu_ok"] = len(ms) >= 8 and len(fs) >= 8
        sg = [g for g, _ in ranked[:30] if g in pmut and len(pmut[g] & sx_prof) >= 5]
        c["sex_gene"] = sg[0] if (sg and c["mwu_ok"]) else None
        sx_os = {p for p in os_ok if p in sexed}
        c["km_sex_ok"] = sum(sexed[p] == c["sex"][1][0] for p in sx_os) >= 8 and sum(sexed[p] == c["sex"][1][1] for p in sx_os) >= 8
    if len(ranked) >= 2:
        g1, g2 = ranked[0][0], ranked[1][0]
        if len(pmut[g1]) >= 5 and len(pmut[g2]) >= 5:
            c["cooccur_pair"] = (g1, g2)
    top5 = [g for g, _ in ranked[:5]]
    c["fdr_ok"] = bool(c["os"]) and len(top5) == 5 and all(
        len(pmut[g] & cohort) >= 3 and len(cohort - pmut[g]) >= 3 for g in top5)
    return c, None


# ---------------------------------------------------------------- execution
def run_probe(sid, code, probe, timeout=600):
    work = Path(tempfile.mkdtemp(prefix="bdsgen_"))
    try:
        grader._stage_data(sid, work / "data_files")
        (work / "prog.py").write_text(code + "\n" + PROBE_HELPERS + probe + "\n")
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        p = subprocess.run([grader._python(), "prog.py"], cwd=work, capture_output=True, text=True, timeout=timeout, env=env)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("__PROBE__")), None)
        if p.returncode != 0 or line is None:
            return None, (p.stderr or p.stdout)[-800:]
        return json.loads(line[len("__PROBE__"):]), None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_task(sid, t):
    vals, err = run_probe(sid, t["code"], t["probe"])
    if vals is None:
        return None, f"reference failed: {err}"
    try:
        tests = "import math\n\n" + "\n\n".join(t["tests"](vals))
    except ValueError as e:
        return None, f"degenerate value: {e}"
    ok = grader.run_tests("", t["code"], tests, sid)
    if not ok["passed"]:
        return None, f"oracle did not pass: {ok['failed_stage']} {ok['stderr_tail'][-400:]}"
    neg = grader.run_tests("", PERTURB + t["code"], tests, sid)
    if neg["passed"]:
        return None, "insensitive: the perturbed reference passes"
    return {"tests": tests, "probe": vals}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--studies", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="run the checks on N tasks (spread over templates); write nothing")
    a = ap.parse_args()
    if (OUT / "train").exists() and not (a.dry or a.check):
        sys.exit(f"{OUT / 'train'} exists -- not overwriting")
    sel = json.load(open(RAW / "SELECTION.json"))["kept"]
    studies = [s for s in sel if (not a.studies or s in a.studies.split(","))]
    rng = random.Random(SEED)
    skipped, tasks = {}, []
    for sid in studies:
        if "target" in sid.lower():
            skipped[sid] = "NCI-TARGET study"
            continue
        d, why = convert(sid)
        if d is None:
            skipped[sid] = why
            continue
        c, why = context(sid, rng)
        if c is None:
            skipped[sid] = why
            continue
        meta = json.load(open(RAW / sid / "STUDY.json"))
        if a.dry:
            print(f"  {sid}: coverage {c['map_coverage']}, os {bool(c['os'])}, other {list(c['other_surv'])}, km {c['km_gene']}, "
                  f"cox {c['cox_gene']}, sex {c['sex']}, dist {c['dist_attrs']}, age {c['age_col']}", flush=True)
        for T in TEMPLATES:
            for t in T(c):
                tasks.append((sid, meta, c, t))
    by_t = {}
    for sid, _, _, t in tasks:
        by_t[t["tid"].rstrip("0123456789")] = by_t.get(t["tid"].rstrip("0123456789"), 0) + 1
    print(f"{len(studies)} studies; skipped {len(skipped)}: {skipped}")
    print(f"{len(tasks)} candidate tasks from {len({s for s, *_ in tasks})} studies; by template: {by_t}", flush=True)
    if a.dry:
        return
    if a.check:
        seen, pick = {}, []
        for i, (sid, _, _, t) in enumerate(tasks):       # round-robin over templates, different studies
            k = t["tid"].rstrip("0123456789")
            if seen.get(k, 0) < max(1, a.check // len(by_t)) and sid not in {tasks[j][0] for j in pick if tasks[j][3]["tid"] == t["tid"]}:
                seen[k] = seen.get(k, 0) + 1
                pick.append(i)
        with cf.ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda i: make_task(tasks[i][0], tasks[i][3]), pick))
        for i, (r, why) in zip(pick, res):
            print(f"  {'OK  ' if r else 'DROP'} {tasks[i][0]}-{tasks[i][3]['tid']}: " +
                  (r["tests"].replace(chr(10), " ")[:300] if r else why[:300].replace(chr(10), " | ")), flush=True)
        print(f"check: {sum(r is not None for r, _ in res)}/{len(res)} kept")
        return
    results = [None] * len(tasks)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(make_task, sid, t): i for i, (sid, _, _, t) in enumerate(tasks)}
        for k, f in enumerate(cf.as_completed(futs)):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:  # noqa: BLE001
                results[i] = (None, f"build error: {e!r}")
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} done", flush=True)
    (OUT / "train").mkdir(parents=True)
    kept, dropped = [], []
    for (sid, meta, c, t), (res, why) in zip(tasks, results):
        pid = f"bdst-{sid}-{t['tid']}"
        if res is None:
            dropped.append({"id": pid, "why": why[:600]})
            continue
        key = f"{pid}-q1"
        pdir = OUT / "train" / pid
        (pdir / "data_files").mkdir(parents=True)
        for f in TABLES:
            os.link(DATA / sid / f, pdir / "data_files" / f)
        cite = meta.get("citation") or ""
        cfg = {"id": pid,
               "task": TASK.format(title=meta["name"], dataset=f"cBioPortal {sid}" + (f" ({cite})" if cite else ""),
                                   tables="\n  ".join(f"data_files/{x}" for x in TABLES), prefix_block="",
                                   query=t["query"], cot_block="", order="solution.py, then the tests", key=key),
               "data_files": TABLES, "ground_truth": {key: t["code"]},
               "grader": {"type": "bdsgen_tests", "config": {"keys": {key: {"prefix": "", "tests": res["tests"], "study": sid}}}},
               "_provenance": {"benchmark": "BioDSBench-style TRAINING task (H-GEN27)", "study_id": sid, "template": t["tid"],
                               "study_name": meta["name"], "pmid": meta.get("pmid"), "source": "cBioPortal datahub",
                               "not_a_biodsbench_study": True, "reference_values": res["probe"], "built": "2026-10-17"}}
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        kept.append(pid)
    by_t = {}
    for pid in kept:
        k = pid.rsplit("-", 1)[1].rstrip("0123456789")
        by_t[k] = by_t.get(k, 0) + 1
    json.dump({"built": "2026-10-17", "seed": SEED, "n_tasks": len(kept), "by_template": by_t, "skipped_studies": skipped,
               "studies": sorted({p.split("-")[1] for p in kept}), "dropped": dropped, "perturbation": PERTURB},
              open(OUT / "MANIFEST.json", "w"), indent=1)
    print(f"kept {len(kept)} tasks ({by_t}); dropped {len(dropped)}")
    for d in dropped[:40]:
        print("  dropped", d["id"], "--", d["why"][:200].replace("\n", " | "))


if __name__ == "__main__":
    main()
