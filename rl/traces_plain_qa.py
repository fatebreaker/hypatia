"""Short single-turn Q&A traces -- the FORMAT control.

The three-corpus control showed the causal-reasoning loss is the same whether
training data is genomics analysis, causal-discovery tasks, or neutral data
wrangling. One confound survives: all three share the same system prompt, the
same sandbox, and the same multi-turn write-code / read-stdout / write-code
shape. The loss may come from that FORMAT rather than from fine-tuning as such.

This corpus removes the format and keeps nothing else:
  * no system prompt about being a computational biologist
  * no code, no sandbox, no tool use
  * one user turn, one assistant turn -- no multi-turn loop
  * short answers, not long traces

Content is deliberately mundane factual/definitional Q&A, generated from
templates over a fixed vocabulary so nothing here touches statistics,
inference, or data analysis.

If causal reasoning STILL drops by ~0.13, the cause is LoRA fine-tuning at this
rank and scale, full stop. If it does not, the cause is specific to long
agentic traces, which is a materially different and more useful claim.

  venv/bin/python rl/traces_plain_qa.py --n 620
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

# Vocabularies sized so that the template x vocabulary product yields ~620
# DISTINCT questions. The first version drew from 26 combinations and repeated
# each 31 times; training on 26 memorisable facts is not a control for training
# on 620 distinct problems, it is a control for degenerate repetition.
CITY = [("Lisbon","Portugal","Tagus"),("Oslo","Norway","Akerselva"),
        ("Vienna","Austria","Danube"),("Cairo","Egypt","Nile"),
        ("Lima","Peru","Rimac"),("Hanoi","Vietnam","Red River"),
        ("Ottawa","Canada","Rideau"),("Dublin","Ireland","Liffey"),
        ("Prague","Czechia","Vltava"),("Budapest","Hungary","Danube"),
        ("Seoul","South Korea","Han"),("Baghdad","Iraq","Tigris"),
        ("Rome","Italy","Tiber"),("Bern","Switzerland","Aare"),
        ("Warsaw","Poland","Vistula"),("Belgrade","Serbia","Sava"),
        ("Khartoum","Sudan","Nile"),("Kyiv","Ukraine","Dnieper"),
        ("Riga","Latvia","Daugava"),("Bamako","Mali","Niger")]
TOOL = [("a barometer","atmospheric pressure"),("a hygrometer","humidity"),
        ("an odometer","distance travelled"),("a seismograph","ground motion"),
        ("an anemometer","wind speed"),("a caliper","thickness"),
        ("a manometer","fluid pressure"),("a tachometer","rotational speed"),
        ("an altimeter","altitude"),("a pyrometer","high temperature"),
        ("a viscometer","viscosity"),("a refractometer","refractive index"),
        ("a colorimeter","colour intensity"),("a galvanometer","small electric current"),
        ("an audiometer","hearing sensitivity")]
MAT  = [("granite","igneous"),("limestone","sedimentary"),("marble","metamorphic"),
        ("basalt","igneous"),("shale","sedimentary"),("slate","metamorphic"),
        ("obsidian","igneous"),("sandstone","sedimentary"),("gneiss","metamorphic"),
        ("pumice","igneous"),("chalk","sedimentary"),("quartzite","metamorphic"),
        ("gabbro","igneous"),("conglomerate","sedimentary"),("schist","metamorphic")]
UNIT = [("joule","energy"),("pascal","pressure"),("hertz","frequency"),
        ("ampere","electric current"),("lumen","luminous flux"),("newton","force"),
        ("volt","electric potential"),("ohm","electrical resistance"),
        ("tesla","magnetic flux density"),("weber","magnetic flux"),
        ("coulomb","electric charge"),("farad","capacitance"),
        ("henry","inductance"),("sievert","radiation dose"),("katal","catalytic activity"),
        ("candela","luminous intensity"),("becquerel","radioactivity"),("gray","absorbed dose")]
LANG = [("Portuguese","Brazil"),("Swahili","Tanzania"),("Finnish","Finland"),
        ("Amharic","Ethiopia"),("Nepali","Nepal"),("Khmer","Cambodia"),
        ("Icelandic","Iceland"),("Malagasy","Madagascar"),("Quechua","Peru"),
        ("Georgian","Georgia"),("Mongolian","Mongolia"),("Sinhala","Sri Lanka"),
        ("Tagalog","the Philippines"),("Uzbek","Uzbekistan"),("Latvian","Latvia")]
BODY = [("Titan","Saturn"),("Europa","Jupiter"),("Triton","Neptune"),
        ("Phobos","Mars"),("Ganymede","Jupiter"),("Enceladus","Saturn"),
        ("Charon","Pluto"),("Callisto","Jupiter"),("Iapetus","Saturn"),
        ("Miranda","Uranus"),("Io","Jupiter"),("Deimos","Mars")]


def _fact(rng, kind):
    """One (question, answer) fact."""
    if kind == 0:
        c, country, river = rng.choice(CITY)
        return (f"which country {c} is in and the river running through it",
                f"{c} is in {country}, on the {river}")
    if kind == 1:
        tool, meas = rng.choice(TOOL)
        return (f"what {tool} measures", f"{tool} measures {meas}")
    if kind == 2:
        rock, cls = rng.choice(MAT)
        return (f"what class of rock {rock} is", f"{rock} is {cls}")
    if kind == 3:
        unit, qty = rng.choice(UNIT)
        return (f"what the {unit} is a unit of", f"the {unit} is the unit of {qty}")
    if kind == 4:
        lang, place = rng.choice(LANG)
        return (f"where {lang} is chiefly spoken", f"{lang} is chiefly spoken in {place}")
    moon, planet = rng.choice(BODY)
    return (f"which planet {moon} orbits", f"{moon} orbits {planet}")


def build(rng, idx):
    """Two unrelated facts per trace.

    Pairing is what makes the corpus diverse: single facts gave 95 distinct
    questions repeated 14 times each, which is a control for memorising 95
    strings rather than for training on 620 distinct examples. Pairs draw from
    ~95^2 combinations, so repetition effectively vanishes while each trace
    stays a short, single-turn, code-free exchange -- which is the only
    property this control is meant to vary.
    """
    ka, kb = rng.sample(range(6), 2)
    qa, aa = _fact(rng, ka)
    qb, ab = _fact(rng, kb)
    q = f"Two quick questions: {qa}, and {qb}?"
    # .capitalize() lowercases the remainder ("Hanoi is in vietnam"), which
    # would train the model on malformed text -- a confound of its own.
    up = lambda t: t[0].upper() + t[1:]
    a = f"{up(aa)}. {up(ab)}."
    return {"id": f"plainqa_{idx:04d}",
            "messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}],
            "reward": 1.0, "n_cells": 0, "plan": "single-turn factual answer",
            "_family": "plain_qa"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=620)
    a = ap.parse_args()
    rng = random.Random(20260825)
    rows = [build(rng, i) for i in range(a.n)]
    out = data_root() / "sft" / "plain_qa.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lens = [len(r["messages"][1]["content"]) for r in rows]
    print(f"wrote {len(rows)} traces -> {out}")
    print(f"  turns per trace: 2 (vs 7 in the agentic corpora)")
    print(f"  mean answer length: {sum(lens)/len(lens):.0f} chars")
    print(f"  system prompt: none    code blocks: 0")


if __name__ == "__main__":
    main()
