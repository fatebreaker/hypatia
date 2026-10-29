"""READINGS of the ASYNC-1 pilot (docs/RL_PLAN.md 2026-09-29 23:35 UTC, registered before any build or run).

    venv/bin/python experiments/harness_v3/read_async.py [--ckpt-root D/ckpt] [--root results/pods/harness_v3] [--out DIR]

Runs: HV3_151 (ASYNC-1; RN_ASYNC) against its synchronous twin HV3_119 (RN_SYNC, same recipe and seed); evals @100:
HV3_152 (V3__<RN_ASYNC>_s100) against HV3_139 (V3__<RN_SYNC>_s100), both the v3 harness, 12x2 per TEST family.

  ASYNC-FASTER      iff mean wall time per step over steps 11-100 (history.jsonl t_step: the step loop from its top to
                    the record, identical definition in both trainers -- rollouts or the wait for them, the gradient step,
                    the publish) <= 0.70 x HV3_119's (496.6 s -> bar 347.6 s).
  ASYNC-EQUIVALENT  iff ALIVE on all three registered criteria -- mean groups_alive_total over steps 51-100 >= 3 (of 8),
                    mean history nll over 51-100 >= 0.05, mean mean_r_answer over 91-100 >= 0.30 (process_history.jsonl;
                    for ASYNC-1 these are over the groups the trainer TRAINED on, as for the synchronous runs) --
                    AND rollout answer pass over steps 91-100 >= 0.323 - 0.05 (0.323 = HV3_119's registered value, which
                    is its mean_r_answer over 91-100 = 0.3235; the history pass_rate, 0.317, is printed beside)
                    AND HO6 @100 (HV3_152) >= HO6 @100 (HV3_139) - 3 points.
  ADOPT-ASYNC       iff both. Every piece that is not logged yet makes the label INCOMPLETE (a provisional value is shown).
  BESIDE (never deciding): steps/hour, the time components (t_wait = the trainer waiting for groups, t_train, t_behav,
  t_publish), the wall clock between consecutive records (t_wall), GPU utilisation / idle fraction and vLLM's running
  requests (sampled by the trainer), the staleness histogram, dropped groups, the importance weights (mean, truncated
  fraction, KL(behaviour || current)), groups already complete when a step started waiting, the paired HO6 contrast.
Last line per step is used (a resumed run logs a step twice). Nothing here reads an answer key: eval cells are pass/fail.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import read_v3 as R  # noqa: E402  (analyze.py helpers: load_config, cell_pass; HO6; never edited)

A, HO6 = R.A, R.HO6
RN_ASYNC = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-async1-lr3e-6-s12"     # HV3_151
RN_SYNC = "rl-czero-causal_rdd-base-echo-sysscience-rc-hv3pr0.3-lr3e-6-s12"             # HV3_119
STEP = 100
FASTER_RATIO = 0.70
TIME_WIN = (11, 100)
ALIVE_GROUPS, ALIVE_NLL, ALIVE_ANSWER = 3.0, 0.05, 0.30
ANSWER_REF, ANSWER_MARGIN = 0.323, 0.05
HO6_MARGIN = 3.0
EPS = 1e-9


def read_jsonl_steps(p: Path) -> dict:
    """{step: the LAST line logged for that step}."""
    out = {}
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and isinstance(x.get("step"), int):
            out[x["step"]] = x
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def window(d: dict, key: str, lo: int, hi: int):
    return _mean(d[s].get(key) for s in range(lo, hi + 1) if s in d)


def has_all(d: dict, lo: int, hi: int) -> bool:
    return all(s in d for s in range(lo, hi + 1))


def run_stats(run_dir: Path) -> dict:
    h, ph = read_jsonl_steps(run_dir / "history.jsonl"), read_jsonl_steps(run_dir / "process_history.jsonl")
    lo, hi = TIME_WIN
    return {"dir": str(run_dir), "h": h, "ph": ph, "last_step": max(h) if h else None,
            "t_step": window(h, "t_step", lo, hi), "t_step_complete": has_all(h, lo, hi),
            "groups_alive": window(ph, "groups_alive_total", 51, 100), "nll": window(h, "nll", 51, 100),
            "r_answer_91": window(ph, "mean_r_answer", 91, 100), "pass_rate_91": window(h, "pass_rate", 91, 100),
            "alive_complete": has_all(ph, 51, 100) and has_all(h, 51, 100)}


def alive_legs(s: dict) -> dict:
    legs = {"groups": (s["groups_alive"], ALIVE_GROUPS), "nll": (s["nll"], ALIVE_NLL),
            "answer": (s["r_answer_91"], ALIVE_ANSWER)}
    return {k: {"value": v, "bar": b, "ok": None if v is None else v >= b - EPS} for k, (v, b) in legs.items()}


def async_diag(h: dict) -> dict:
    """Beside: what ASYNC-1 did, over steps 11-100 (or whatever is logged)."""
    lo, hi = TIME_WIN
    steps = [s for s in range(lo, hi + 1) if s in h]
    xs = [h[s] for s in steps]
    hist, drop = Counter(), 0
    for x in xs:
        hist.update({k: int(v) for k, v in (x.get("async_staleness_hist") or {}).items()})
        drop += int(x.get("async_dropped") or 0)
    walls = [h[s]["t_wall"] - h[s - 1]["t_wall"] for s in steps if s - 1 in h
             and h[s].get("t_wall") is not None and h[s - 1].get("t_wall") is not None]
    tot_is = sum(int(x.get("async_is_tokens") or 0) for x in xs)

    def wmean(key):   # token-weighted over steps
        num = sum(float(x[key]) * int(x.get("async_is_tokens") or 0) for x in xs if x.get(key) is not None)
        return num / tot_is if tot_is else None
    m = lambda k: _mean(x.get(k) for x in xs)   # noqa: E731
    return {"steps": len(steps), "staleness_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "dropped": drop, "dropped_total_last": xs[-1].get("async_dropped_total") if xs else None,
            "t_wait": m("t_wait"), "t_train": m("t_train"), "t_behav": m("t_behav"), "t_publish": m("t_publish"),
            "wall_per_step": _mean(walls), "ready_at_take": m("async_ready_at_take"),
            "gpu_util": m("gpu_util_mean"), "gpu_idle_frac": m("gpu_idle_frac"),
            "vllm_running": m("vllm_running_mean"), "vllm_idle_frac": m("vllm_idle_frac"),
            "is_w_mean": wmean("async_is_w_mean"), "is_trunc_frac": wmean("async_is_trunc_frac"),
            "is_kl": wmean("async_is_kl"), "is_tokens": tot_is,
            "group_s_max": _mean((x.get("group_s") or [None, None])[1] for x in xs),
            "episode_max_s": _mean((x.get("rollout_s") or [None, None, None])[2] for x in xs)}


def ho6_cell(root: Path, name: str, sets) -> dict:
    if not (root / name).is_dir():
        return {"name": name, "present": False, "ho6": None, "complete": False, "res": None}
    res = A.load_config(root, name, sets)
    fams = {f: res["families"].get(("test", f)) for f in HO6}
    complete = all(v is not None and len(v) >= 12 and all(len(r) >= 2 for r in v.values()) for v in fams.values())
    return {"name": name, "present": True, "ho6": R.ho6(res, sets), "complete": complete, "res": res,
            "n_fams": sum(v is not None for v in fams.values())}


def verdict(sa: dict, ss: dict, ea: dict, es: dict) -> dict:
    ref_t = ss["t_step"]
    bar_t = FASTER_RATIO * ref_t if ref_t is not None else None
    faster = None if (sa["t_step"] is None or bar_t is None) else sa["t_step"] <= bar_t + EPS
    faster_final = sa["t_step_complete"] and ss["t_step_complete"]
    legs = alive_legs(sa)
    alive_ok = all(v["ok"] is True for v in legs.values()) if all(v["ok"] is not None for v in legs.values()) else None
    ans_bar = ANSWER_REF - ANSWER_MARGIN
    ans_ok = None if sa["r_answer_91"] is None else sa["r_answer_91"] >= ans_bar - EPS
    ho6_ok = None
    if ea["ho6"] is not None and es["ho6"] is not None:
        ho6_ok = ea["ho6"] >= es["ho6"] - HO6_MARGIN - EPS
    eq_parts = [alive_ok, ans_ok, ho6_ok]
    eq = None if any(p is None for p in eq_parts) else all(eq_parts)
    if eq is None and any(p is False for p in eq_parts):
        eq = False                                    # already failed on a logged piece (still provisional)
    eq_final = sa["alive_complete"] and has_all(sa["ph"], 91, 100) and ea["complete"] and es["complete"]
    final = faster_final and eq_final
    if faster is None or eq is None:
        label = "INCOMPLETE"
    else:
        label = "ADOPT-ASYNC" if (faster and eq) else "NOT ADOPTED (" + ", ".join(
            x for x, ok in (("not FASTER", faster), ("not EQUIVALENT", eq)) if not ok) + ")"
    return {"faster": {"value": sa["t_step"], "ref": ref_t, "bar": bar_t, "ratio": (sa["t_step"] / ref_t)
                       if (sa["t_step"] and ref_t) else None, "ok": faster, "final": faster_final},
            "equivalent": {"alive": {"legs": legs, "ok": alive_ok},
                           "answer": {"value": sa["r_answer_91"], "bar": ans_bar, "ok": ans_ok,
                                      "pass_rate_91": sa["pass_rate_91"], "ref_recomputed": ss["r_answer_91"]},
                           "ho6": {"async": ea["ho6"], "sync": es["ho6"], "bar": (es["ho6"] - HO6_MARGIN)
                                   if es["ho6"] is not None else None, "ok": ho6_ok,
                                   "complete": [ea["complete"], es["complete"]]},
                           "ok": eq, "final": eq_final},
            "label": label, "final": bool(final and label != "INCOMPLETE")}


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(v: dict, sa: dict, ss: dict, ea: dict, es: dict, diag: dict, contrast) -> list:
    f, e = v["faster"], v["equivalent"]
    L = ["# ASYNC-1 pilot readings (docs/RL_PLAN.md 2026-09-29 23:35 UTC; rules: read_async.py docstring)\n",
         f"**{v['label']}**" + ("" if v["final"] else " (provisional: not every registered piece is logged)") + "\n",
         f"runs: ASYNC-1 {Path(sa['dir']).name} (last step {sa['last_step']}) vs synchronous {Path(ss['dir']).name} "
         f"(last step {ss['last_step']})\n",
         "## ASYNC-FASTER\n",
         f"mean t_step over steps {TIME_WIN[0]}-{TIME_WIN[1]}: async {fmt(f['value'], 1)} s vs sync {fmt(f['ref'], 1)} s; "
         f"ratio {fmt(f['ratio'])} (bar <= {FASTER_RATIO}, i.e. <= {fmt(f['bar'], 1)} s) -> "
         f"**{'FASTER' if f['ok'] else ('not faster' if f['ok'] is False else 'n/a')}**"
         + ("" if f["final"] else " (window incomplete)") + "\n",
         "## ASYNC-EQUIVALENT\n", "| criterion | ASYNC-1 | bar | ok |", "|---|---|---|---|"]
    for k, lab in (("groups", "groups_alive_total, steps 51-100"), ("nll", "nll, steps 51-100"),
                   ("answer", "mean_r_answer, steps 91-100 (ALIVE)")):
        x = e["alive"]["legs"][k]
        L.append(f"| {lab} | {fmt(x['value'])} | >= {x['bar']} | {x['ok']} |")
    a = e["answer"]
    L.append(f"| answer vs HV3_119 (mean_r_answer 91-100) | {fmt(a['value'])} | >= {a['bar']:.3f} (0.323 - 0.05; HV3_119 "
             f"recomputed {fmt(a['ref_recomputed'], 4)}) | {a['ok']} |")
    h = e["ho6"]
    L.append(f"| HO6 @100 (HV3_152 vs HV3_139) | {fmt(h['async'], 1)} | >= {fmt(h['bar'], 1)} (HV3_139 {fmt(h['sync'], 1)} "
             f"- 3) | {h['ok']} |")
    L.append(f"\nEQUIVALENT: **{e['ok']}**" + ("" if e["final"] else " (incomplete: " + ", ".join(
        x for x, ok in (("training steps", sa["alive_complete"]), ("HV3_152 eval", ea["complete"]),
                        ("HV3_139 eval", es["complete"])) if not ok) + ")"))
    L.append(f"\nBeside: history pass_rate 91-100 async {fmt(sa['pass_rate_91'])} vs sync {fmt(ss['pass_rate_91'])}; "
             f"sync ALIVE legs: groups {fmt(ss['groups_alive'])}, nll {fmt(ss['nll'])}, answer {fmt(ss['r_answer_91'])}")
    if contrast:
        L.append(f"Paired HO6 contrast (async - sync, by instance): d {fmt(contrast.get('d'), 1)} points, p(>0) "
                 f"{fmt(contrast.get('p_gt'))}, p(<0) {fmt(contrast.get('p_lt'))}, {contrast.get('n_fams')} families")
    d = diag
    sph = 3600.0 / f["value"] if f["value"] else None
    sph0 = 3600.0 / f["ref"] if f["ref"] else None
    L += ["\n## Beside (never deciding): what ASYNC-1 did, steps 11-100\n",
          f"- steps/hour: async {fmt(sph, 2)} vs sync {fmt(sph0, 2)}; wall clock between records (t_wall) "
          f"{fmt(d['wall_per_step'], 1)} s/step",
          f"- time per step: waiting for groups {fmt(d['t_wait'], 1)} s, train {fmt(d['t_train'], 1)} s (of which "
          f"behaviour pass {fmt(d['t_behav'], 1)} s), publish+refill {fmt(d['t_publish'], 1)} s; groups already complete "
          f"when the step began waiting: {fmt(d['ready_at_take'], 2)}",
          f"- slowest episode per step {fmt(d['episode_max_s'], 1)} s; slowest trained group {fmt(d['group_s_max'], 1)} s",
          f"- GPU utilisation {fmt(d['gpu_util'], 1)} % (idle fraction {fmt(d['gpu_idle_frac'])}); vLLM running requests "
          f"{fmt(d['vllm_running'], 2)} (idle fraction {fmt(d['vllm_idle_frac'])}); HV3_119 logged neither",
          f"- staleness histogram (trained groups) {d['staleness_hist']}; dropped groups {d['dropped']} "
          f"(run total {d['dropped_total_last']})",
          f"- importance weights (stale tokens {d['is_tokens']}): mean {fmt(d['is_w_mean'], 4)}, truncated fraction "
          f"{fmt(d['is_trunc_frac'], 5)}, KL(behaviour || current) {fmt(d['is_kl'], 6)}"]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default=None, help="default: paths.data_root()/ckpt")
    ap.add_argument("--root", default=str(ROOT / "results" / "pods" / "harness_v3"))
    ap.add_argument("--rn-async", default=RN_ASYNC)
    ap.add_argument("--rn-sync", default=RN_SYNC)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where ASYNC_READING.md/.json go (default: the HV3_152 eval cell, if it "
                                                "exists; else nothing is written)")
    a = ap.parse_args(argv)
    if a.ckpt_root:
        ck = Path(a.ckpt_root)
    else:
        sys.path.insert(0, str(ROOT))
        from paths import data_root
        ck = data_root() / "ckpt"
    root = Path(a.root)
    sa, ss = run_stats(ck / a.rn_async), run_stats(ck / a.rn_sync)
    sets = A.load_families()
    ea = ho6_cell(root, f"V3__{a.rn_async}_s{STEP}", sets)
    es = ho6_cell(root, f"V3__{a.rn_sync}_s{STEP}", sets)
    contrast = None
    if ea["res"] is not None and es["res"] is not None:
        c = R.contrast(ea["res"], es["res"], sets, a.perms, np.random.default_rng(a.seed))
        contrast = None if c is None else {"d": c[0], "p_gt": c[1], "p_lt": c[2], "n_fams": c[3]}
    v = verdict(sa, ss, ea, es)
    diag = async_diag(sa["h"])
    L = report(v, sa, ss, ea, es, diag, contrast)
    text = "\n".join(L) + "\n"
    print(text)
    out = Path(a.out) if a.out else (root / ea["name"] if ea["present"] else None)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "ASYNC_READING.md").write_text(text)
        (out / "ASYNC_READING.json").write_text(json.dumps({"verdict": v, "diag": diag, "contrast": contrast,
                                                            "async_dir": sa["dir"], "sync_dir": ss["dir"]},
                                                           indent=1, default=str))
    print(f"ASYNC SUMMARY: {v['label']}" + ("" if v["final"] else " (provisional)")
          + (f"; written {out / 'ASYNC_READING.md'}" if out is not None else "; nothing written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
