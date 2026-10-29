"""causal_sem_build.py (10-27, docs/RL_PLAN.md H-SEM) -- generated/causal_sem: causal-direction questions in QRData's format and grader
(qrdata_official, multiple_choice) on binary diagnosis records from LAYERED diagnostic networks with meaningful labels, where the
answer comes from what the labels mean (a cause-level diagnosis causes its syndromes, a syndrome causes its findings, two findings do
not cause each other) more than from the data, which is often weak.

Why (exploratory error analysis, 10-27, on the 9B's QRData-Verified reads): the 20 Neuropathic questions ask for the direction between
a pattern diagnosis (radiculopathy) and a symptom, or between two symptoms; every directional answer is "the radiculopathy causes the
symptom", and the data rarely shows it (true edges with lift <= 1, all-zero columns). Claude Opus 5.5 gets 14 of 20; every 9B read
7-10, mostly answering "no causal relationship". Our discovery pool (causal_disc2) has neutral names (score_J2), so nothing in RL so far
rewarded reading the labels. This pool has NO spinal, nerve-root or neuropathic content and no laterality (L/R) labels.

Twelve hand-written domains (nine clinical, three not: vehicle faults, IT incidents, crop disease), each three levels -- causes ->
syndromes / fault modes -> findings -- with the edges in DOMAINS below; no two syndromes, and no two findings, of a domain where one
plausibly causes the other (so 'no causal relationship' is right for them). Per instance: one domain, a random subset of its syndromes
(at least 3, with their causes and findings), noisy-OR binary data (cause prevalence 0.02-0.35, edge strength 0.05-0.95, leak
0-0.03 for syndromes and 0-0.06 for findings), n = 1,500-8,000 rows; columns in level order (causes, syndromes, findings), shuffled
within a level. ONE question, QRData's wording: 30% two-option pairwise (a direct edge; A or B), 70% three-option (A x causes y,
B y causes x, C no causal relationship) -- 55% a direct edge (syndrome -> finding 80%, cause -> syndrome 20%), 45% no relationship
(two findings 55%, a syndrome and a finding it does not produce 25%, two syndromes 20%). Pairs linked only indirectly (a cause and
a finding) are never asked. The edge's position (A or B) is random.

    venv/bin/python scripts/causal_sem_build.py [--n-train 800] [--n-test 100]
Refuses to overwrite. Writes generated/causal_sem/{train,test}/<id>/{data_files/<Stem>_<k>.csv, eval_config.json} and MANIFEST.json.
"""
import argparse
import json
import os
import sys

import numpy as np

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
OUT = f"{G}/causal_sem"

# domain: stem, units, (cause level, syndrome level, finding level) names, causes {cause: [syndromes]}, syndromes {syndrome: [findings]}
DOMAINS = {   # 10-27 review: no two syndromes, and no two findings, of one domain where one plausibly causes the other
    "cardiac": ("Cardiac", "patients", ("pathophysiological", "syndrome", "symptom"), {
        "coronary atherosclerosis": ["stable angina", "acute myocardial infarction"],
        "rheumatic valve disease": ["mitral stenosis"],
        "calcific valve degeneration": ["aortic stenosis"],
        "conduction system fibrosis": ["complete heart block"],
        "accessory atrioventricular pathway": ["Wolff-Parkinson-White syndrome"]}, {
        "stable angina": ["exertional chest pain", "exertional dyspnea"],
        "acute myocardial infarction": ["crushing chest pain", "troponin elevation", "ST elevation", "diaphoresis"],
        "mitral stenosis": ["diastolic murmur", "exertional dyspnea", "hemoptysis"],
        "aortic stenosis": ["systolic ejection murmur", "exertional syncope", "exertional chest pain"],
        "complete heart block": ["cannon A waves", "wide QRS escape rhythm", "exertional syncope"],
        "Wolff-Parkinson-White syndrome": ["delta wave", "short PR interval", "palpitations"]}),
    "respiratory": ("Respiratory", "patients", ("etiological", "syndrome", "symptom"), {
        "Streptococcus pneumoniae infection": ["lobar pneumonia"],
        "Mycoplasma pneumoniae infection": ["atypical pneumonia"],
        "tobacco smoke exposure": ["chronic obstructive pulmonary disease"],
        "asbestos exposure": ["pleural mesothelioma"],
        "airway hyperresponsiveness": ["asthma exacerbation"],
        "deep venous thrombosis": ["pulmonary embolism"],
        "apical bleb rupture": ["pneumothorax"]}, {
        "lobar pneumonia": ["productive cough", "high fever", "bronchial breath sounds", "pleuritic chest pain"],
        "atypical pneumonia": ["dry cough", "headache", "low-grade fever"],
        "chronic obstructive pulmonary disease": ["chronic sputum production", "barrel chest", "prolonged expiration"],
        "pleural mesothelioma": ["unilateral pleural thickening", "chest wall pain", "weight loss"],
        "asthma exacerbation": ["expiratory wheeze", "nocturnal cough", "chest tightness"],
        "pulmonary embolism": ["sudden dyspnea", "pleuritic chest pain", "elevated D-dimer"],
        "pneumothorax": ["hyperresonance", "absent breath sounds", "sudden dyspnea"]}),
    "hepatic": ("Hepatobiliary", "patients", ("etiological", "disease", "finding"), {
        "chronic hepatitis B infection": ["cirrhosis"],
        "chronic alcohol misuse": ["cirrhosis", "acute pancreatitis"],
        "cholesterol gallstones": ["acute cholecystitis", "choledocholithiasis"],
        "Helicobacter pylori infection": ["duodenal ulcer"]}, {
        "cirrhosis": ["ascites", "spider angiomas", "palmar erythema", "splenomegaly"],
        "acute cholecystitis": ["Murphy sign", "right upper quadrant pain", "fever"],
        "choledocholithiasis": ["jaundice", "pale stools", "dark urine"],
        "acute pancreatitis": ["epigastric pain radiating to the back", "lipase elevation", "vomiting"],
        "duodenal ulcer": ["epigastric pain relieved by food", "melena"]}),
    "renal": ("Renal", "patients", ("etiological", "syndrome", "finding"), {
        "long-standing diabetes mellitus": ["diabetic nephropathy"],
        "group A streptococcal pharyngitis": ["post-streptococcal glomerulonephritis"],
        "calcium oxalate stones": ["ureteric colic"],
        "Escherichia coli bacteriuria": ["cystitis"],
        "podocyte injury": ["minimal change disease"],
        "mucosal IgA immune complexes": ["IgA nephropathy"]}, {
        "diabetic nephropathy": ["albuminuria", "rising creatinine"],
        "post-streptococcal glomerulonephritis": ["cola-colored urine", "periorbital edema", "low complement C3"],
        "ureteric colic": ["colicky flank pain", "microscopic hematuria"],
        "cystitis": ["dysuria", "urinary frequency", "pyuria"],
        "minimal change disease": ["lower-limb edema", "hyperlipidemia", "lipiduria"],
        "IgA nephropathy": ["visible hematuria during infections", "mesangial IgA deposits on biopsy"]}),
    "endocrine": ("Endocrine", "patients", ("pathophysiological", "syndrome", "symptom"), {
        "autoimmune thyroiditis": ["hypothyroidism"],
        "TSH-receptor antibodies": ["Graves hyperthyroidism"],
        "pituitary corticotroph adenoma": ["Cushing syndrome"],
        "pituitary somatotroph adenoma": ["acromegaly"],
        "autoimmune adrenalitis": ["primary adrenal insufficiency"],
        "autoimmune beta-cell destruction": ["type 1 diabetes"]}, {
        "hypothyroidism": ["cold intolerance", "dry skin", "constipation", "weight gain"],
        "Graves hyperthyroidism": ["heat intolerance", "fine tremor", "exophthalmos", "weight loss"],
        "Cushing syndrome": ["moon face", "purple abdominal striae", "proximal muscle weakness", "weight gain"],
        "acromegaly": ["enlarged hands", "jaw protrusion", "increased shoe size"],
        "primary adrenal insufficiency": ["skin hyperpigmentation", "salt craving", "postural hypotension"],
        "type 1 diabetes": ["polyuria", "ketonuria", "weight loss"]}),
    "infection": ("Infection", "patients", ("causal-agent", "syndrome", "sign"), {
        "Plasmodium falciparum": ["intravascular hemolysis", "cerebral malaria"],
        "influenza A virus": ["viral pneumonitis", "influenza-like illness"],
        "Mycobacterium tuberculosis": ["cavitary pulmonary tuberculosis", "tuberculous lymphadenitis"],
        "Salmonella Typhi": ["enteric fever"],
        "dengue virus": ["dengue hemorrhagic fever", "influenza-like illness"]}, {
        "intravascular hemolysis": ["dark urine", "pallor", "scleral icterus"],
        "cerebral malaria": ["impaired consciousness", "malarial retinopathy"],
        "viral pneumonitis": ["dry cough", "hypoxemia"],
        "influenza-like illness": ["myalgia", "high fever", "headache"],
        "cavitary pulmonary tuberculosis": ["chronic cough", "night sweats", "hemoptysis"],
        "tuberculous lymphadenitis": ["painless neck lymph node swelling"],
        "enteric fever": ["stepwise rising fever", "rose spots", "relative bradycardia"],
        "dengue hemorrhagic fever": ["petechiae", "bleeding gums", "hematocrit rise"]}),
    "joint": ("Joint", "patients", ("pathophysiological", "disease", "symptom"), {
        "urate overproduction": ["gout"],
        "autoimmune synovitis": ["rheumatoid arthritis"],
        "articular cartilage degeneration": ["knee osteoarthritis", "hip osteoarthritis"],
        "psoriatic skin disease": ["psoriatic arthritis"],
        "low bone mineral density": ["fragility hip fracture"]}, {
        "gout": ["acute first toe pain", "tophi", "red swollen joint"],
        "rheumatoid arthritis": ["morning stiffness over an hour", "symmetric hand joint swelling", "rheumatoid nodules"],
        "knee osteoarthritis": ["knee crepitus", "knee pain on walking"],
        "hip osteoarthritis": ["groin pain on walking", "reduced hip rotation"],
        "psoriatic arthritis": ["nail pitting", "dactylitis"],
        "fragility hip fracture": ["inability to bear weight", "shortened externally rotated leg", "groin pain on walking"]}),
    "skin": ("Skin", "patients", ("etiological", "disease", "lesion"), {
        "dermatophyte infection": ["tinea corporis"],
        "Malassezia overgrowth": ["pityriasis versicolor"],
        "IgE-mediated atopy": ["atopic dermatitis", "allergic urticaria"],
        "chronic UV exposure": ["actinic keratosis", "basal cell carcinoma"],
        "Staphylococcus aureus colonisation": ["impetigo"]}, {
        "tinea corporis": ["annular scaly plaque", "central clearing"],
        "pityriasis versicolor": ["hypopigmented macules", "fine branny scale"],
        "atopic dermatitis": ["flexural eczema", "intense itching", "dry skin"],
        "allergic urticaria": ["transient wheals", "intense itching"],
        "actinic keratosis": ["rough scaly sun-exposed patch"],
        "basal cell carcinoma": ["pearly papule", "rolled border ulcer"],
        "impetigo": ["honey-colored crusts", "perioral erosions"]}),
    "bowel": ("Bowel", "patients", ("etiological", "disease", "symptom"), {
        "Clostridioides difficile overgrowth": ["pseudomembranous colitis"],
        "gluten-triggered autoimmunity": ["celiac disease"],
        "chronic colonic mucosal inflammation": ["ulcerative colitis"],
        "sigmoid diverticula": ["acute diverticulitis"]}, {
        "pseudomembranous colitis": ["watery diarrhea", "abdominal cramping", "leukocytosis"],
        "celiac disease": ["dermatitis herpetiformis", "bloating", "steatorrhea"],
        "ulcerative colitis": ["bloody diarrhea", "tenesmus", "abdominal cramping"],
        "acute diverticulitis": ["left lower quadrant pain", "fever", "leukocytosis"]}),
    "vehicle": ("Vehicle", "vehicles", ("root-cause", "fault-mode", "symptom"), {
        "worn alternator brushes": ["charging system failure"],
        "clogged fuel filter": ["low fuel pressure"],
        "worn spark plugs": ["ignition misfire"],
        "stuck thermostat": ["engine overheating"],
        "worn brake pads": ["reduced braking performance"]}, {
        "charging system failure": ["battery warning light", "dim headlights", "voltmeter reading below 12 V while driving"],
        "low fuel pressure": ["hesitation on acceleration", "engine stalling", "hard starting"],
        "ignition misfire": ["rough idle", "flashing check-engine light", "poor fuel economy"],
        "engine overheating": ["high temperature gauge", "steam from the hood", "overheat warning chime"],
        "reduced braking performance": ["squealing brakes", "longer stopping distance", "brake pedal vibration"]}),
    "service": ("Service", "incident records", ("root-cause", "failure-mode", "alert"), {
        "disk firmware defect": ["storage latency degradation"],
        "RAID controller cache battery failure": ["storage latency degradation"],
        "memory leak in the application": ["out-of-memory crashes"],
        "expired TLS certificate": ["TLS handshake failure"],
        "faulty network switch": ["packet loss"],
        "misconfigured DNS record": ["name resolution failure"]}, {
        "storage latency degradation": ["high disk IO wait", "disk latency SLO alerts"],
        "out-of-memory crashes": ["rising resident memory graphs", "OOM-killer log entries"],
        "TLS handshake failure": ["HTTPS handshake errors", "client certificate warnings"],
        "packet loss": ["TCP retransmission spikes", "interface error counters"],
        "name resolution failure": ["NXDOMAIN responses", "resolver SERVFAIL counts"]}),
    "crop": ("Crop", "plants", ("causal-agent", "disease", "sign"), {
        "Phytophthora infestans": ["late blight"],
        "Fusarium oxysporum": ["fusarium wilt"],
        "nitrogen deficiency": ["nitrogen chlorosis"],
        "aphid-borne potyvirus": ["mosaic virus disease"],
        "root-knot nematodes": ["root-knot nematode disease"]}, {
        "late blight": ["dark water-soaked leaf lesions", "white sporulation on leaf undersides", "tuber rot"],
        "fusarium wilt": ["one-sided wilting", "vascular browning"],
        "nitrogen chlorosis": ["uniform yellowing of older leaves", "stunted growth"],
        "mosaic virus disease": ["mottled leaves", "leaf distortion"],
        "root-knot nematode disease": ["root galls", "patchy field decline", "stunted growth"]}),
}


def check_domains():
    for k, (_, _, _, causes, syn) in DOMAINS.items():
        assert all(s in syn for v in causes.values() for s in v), (k, "a cause names an unknown syndrome")
        assert all(any(s in v for v in causes.values()) for s in syn), (k, "a syndrome without a cause")
        labels = list(causes) + list(syn) + sorted({f for v in syn.values() for f in v})
        assert len(labels) == len(set(labels)), (k, "a label at two levels")


def build_one(rng):
    dom = list(DOMAINS)[rng.integers(len(DOMAINS))]
    stem, units, lv, causes, syn = DOMAINS[dom]
    S = list(syn)
    keep = [S[i] for i in sorted(rng.choice(len(S), size=int(rng.integers(3, len(S) + 1)), replace=False))]
    C = [c for c, v in causes.items() if any(s in keep for s in v)]
    F = sorted({f for s in keep for f in syn[s]})
    e_cs = [(c, s) for c in C for s in causes[c] if s in keep]
    e_sf = [(s, f) for s in keep for f in syn[s]]
    n = int(rng.integers(1500, 8001))
    val = {}
    for c in C:
        val[c] = rng.random(n) < rng.uniform(0.02, 0.35)
    for lvl, edges, leak_hi in ((keep, e_cs, 0.03), (F, e_sf, 0.06)):
        for v in lvl:
            q = np.full(n, 1.0 - rng.uniform(0.0, leak_hi))     # P(v = 0) = (1 - leak) * prod over active parents of (1 - w)
            for a, b in edges:
                if b == v:
                    q = q * np.where(val[a], 1.0 - rng.uniform(0.05, 0.95), 1.0)
            val[v] = rng.random(n) >= q
    cols = [c for grp in (C, keep, F) for c in rng.permutation(grp).tolist()]
    pairwise = rng.random() < 0.30
    if pairwise or rng.random() < 0.55:
        pool = e_sf if (rng.random() < 0.80 or not e_cs) else e_cs
        x, y = pool[rng.integers(len(pool))]                    # x causes y
        what = "edge"
    else:
        r = rng.random()
        if r < 0.55 and len(F) >= 2:
            i, j = rng.choice(len(F), size=2, replace=False); x, y = F[i], F[j]; what = "none:finding-finding"
        elif r < 0.80:
            cand = [(s, f) for s in keep for f in F if f not in syn[s]]
            if not cand:
                return None
            x, y = cand[rng.integers(len(cand))]; what = "none:syndrome-finding"
        else:
            if len(keep) < 2:
                return None
            i, j = rng.choice(len(keep), size=2, replace=False); x, y = keep[i], keep[j]; what = "none:syndrome-syndrome"
    a, b = (x, y) if rng.random() < 0.5 else (y, x)               # option A is "a causes b"
    if what == "edge":
        ans = "A" if (a, b) == (x, y) else "B"
    else:
        ans = "C"
    X = np.column_stack([val[c].astype(np.int8) for c in cols])
    return dict(dom=dom, stem=stem, units=units, lv=lv, cols=cols, X=X, n=n, a=a, b=b, answer=ans, what=what,
                pairwise=bool(pairwise), causes=C, syndromes=keep, findings=F, edges=e_cs + e_sf)


def write(inst, root, iid, seed, k, rng):
    d = f"{root}/{iid}"
    os.makedirs(f"{d}/data_files")
    fn = f"{inst['stem']}_{k}.csv"
    with open(f"{d}/data_files/{fn}", "w") as fh:
        fh.write(",".join(inst["cols"]) + "\n")
        for row in inst["X"]:
            fh.write(",".join("1" if v else "0" for v in row) + "\n")
    l0, l1, l2 = inst["lv"]; dom = inst["stem"].lower(); units = inst["units"]
    pre = [f"The {dom} diagnosis mainly consists of {l2} diagnosis, {l1} diagnosis, and {l0} diagnosis. The csv file {fn} contains "
           f"{dom} diagnosis records in the form of tables of which the row represents different {units} and the column represents "
           "different diagnostic labels.",
           f"Diagnostic labels in {fn} come at three levels: {l0} diagnoses, {l1} diagnoses and {l2} findings. Each row is one of "
           f"{inst['n']} {units} and each column one binary label (1 = present).",
           f"The csv file {fn} records {inst['n']} {units}; each column is a binary {dom} diagnostic label at the {l0}, {l1} or {l2} level."]
    r = rng.random()
    intro = pre[0] if r < 0.5 else pre[1] if r < 0.75 else pre[2]
    a, b = inst["a"], inst["b"]
    if inst["pairwise"]:
        opts, ask = f"A. {a} causes {b}\nB. {b} causes {a}", "Please answer with A or B."
    else:
        opts, ask = f"A. {a} causes {b}\nB. {b} causes {a}\nC. No causal relationship exists", "Please answer with A, B, or C."
    key = f"{iid}-q1"
    task = ("You are given the data files from a published statistical analysis.\nAnswer the question below by computing the result "
            f"from the data.\n\n{intro}\n\nFiles in data_files/:\n  {fn}\n\nQuestion: Which cause-and-effect relationship is more likely?\n"
            f"{opts}\n{ask}\n\nReturn your answer as JSON: {{\"answer\": {{\"{key}\": <value>}}}}")
    cfg = {"id": iid, "task": task, "data_files": [fn], "ground_truth": {key: inst["answer"]},
           "grader": {"type": "qrdata_official", "config": {"keys": {key: {"question_type": "multiple_choice"}}}},
           "_provenance": {"family": "causal_sem", "seed": seed, "domain": inst["dom"], "n": inst["n"], "p": len(inst["cols"]),
                           "levels": {"cause": inst["causes"], "syndrome": inst["syndromes"], "finding": inst["findings"]},
                           "edges": [list(e) for e in inst["edges"]], "asked": [a, b], "kind": inst["what"],
                           "form": "pairwise" if inst["pairwise"] else "full graph, three options",
                           "keywords": ["Causality", ("Pairwise" if inst["pairwise"] else "Full graph") + " causal discovery",
                                        "Observational data", "Diagnostic labels", "synthetic"]}}
    json.dump(cfg, open(f"{d}/eval_config.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    check_domains()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists -- not overwriting")
    counts = {}
    for split, n0, base in (("train", a.n_train, 0), ("test", a.n_test, 10_000_000)):
        os.makedirs(f"{a.out}/{split}")
        made = k = 0
        tally = {"answers": {}, "kinds": {}, "domains": {}, "pairwise": 0}
        while made < n0:
            seed = a.seed + base + k
            k += 1
            rng = np.random.default_rng(seed)
            inst = build_one(rng)
            if inst is None:
                continue
            iid = f"causal_sem_{split}_{made:05d}"
            write(inst, f"{a.out}/{split}", iid, seed, made + (0 if split == "train" else 5000), rng)
            for f, v in (("answers", inst["answer"]), ("kinds", inst["what"]), ("domains", inst["dom"])):
                tally[f][v] = tally[f].get(v, 0) + 1
            tally["pairwise"] += inst["pairwise"]
            made += 1
        counts[split] = {"n": made, **tally}
        print(split, counts[split], flush=True)
    json.dump({"built": "2026-10-27", "builder": "scripts/causal_sem_build.py", "registration": "docs/RL_PLAN.md H-SEM",
               "seed": a.seed, "splits": counts}, open(f"{a.out}/MANIFEST.json", "w"), indent=1)


if __name__ == "__main__":
    main()
