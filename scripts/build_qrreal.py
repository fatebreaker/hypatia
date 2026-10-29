"""build_qrreal.py (10-26, docs/RL_PLAN.md H-MEET-ALL 14:04) -- the training pool qrreal89 (named for its size): QRData questions that are NOT in QRData-Verified
and share NO data with it. A question is kept only if (a) its task text is not a QRData-Verified task, (b) none of its data file names appears in
any QRData-Verified problem, and (c) none of its data files has the same content (sha256) as any QRData-Verified data file. Their labels are
unverified (the reason QRData-Verified left them out). Pool: generated/qrreal89/train/<id> -> the QRData problem directory (symlinks), so the
trainer reads the same task, data and grader as the benchmark. Refuses to overwrite.   python3 scripts/build_qrreal.py"""
import hashlib, json, os, sys

R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
OUT = f"{G}/qrreal89/train"


def problems(root):
    d = f"{R}/benchmarks/{root}/problems"
    out = {}
    for pid in sorted(os.listdir(d)):
        p = os.path.realpath(f"{d}/{pid}")
        cfg = json.load(open(f"{p}/eval_config.json"))
        df = f"{p}/data_files"
        files = sorted(os.listdir(df)) if os.path.isdir(df) else []
        out[pid] = (p, cfg.get("task", ""), files)
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
allq, ver = problems("qrdata"), problems("qrdata_verified")
v_tasks = {t for _, t, _ in ver.values()}
v_names = {f for _, _, fs in ver.values() for f in fs}
v_hash = {sha(f"{p}/data_files/{f}") for p, _, fs in ver.values() for f in fs}
keep, why = [], {"verified task": 0, "shared file name": 0, "shared file content": 0}
for pid, (p, task, files) in allq.items():
    if task in v_tasks:
        why["verified task"] += 1; continue
    if any(f in v_names for f in files):
        why["shared file name"] += 1; continue
    if any(sha(f"{p}/data_files/{f}") in v_hash for f in files):
        why["shared file content"] += 1; continue
    keep.append((pid, p))
os.makedirs(OUT)
for pid, p in keep:
    os.symlink(p, f"{OUT}/{pid}")
datasets = {tuple(allq[pid][2]) for pid, _ in keep}
print(json.dumps({"qrdata": len(allq), "verified": len(ver), "dropped": why, "kept": len(keep), "datasets": len(datasets), "pool": OUT}))
