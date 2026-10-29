#!/usr/bin/env python
"""Extract data-generating processes and ablation tables from the public reports.

Each of the 10 public GeneBench-Pro problems ships a 13-page report whose
sections include, verbatim:

    5  Variables and Assumptions
    6  Data-Generating Process     <- the simulator, parameters and all
    8  Estimand
    9  Estimator
    10 Decision-Point and Ablation Walkthrough

Two reasons to pull these out:

1. Section 6 is a generator spec. Ten fully documented DGPs is enough to build
   a task generator, which is the only route to an n large enough to measure
   anything (n=10 gives a ~50pp confidence interval; ~335 needed for 80% power
   at a 10pp gap).

2. Section 10 tabulates every wrong analysis path with its numeric output and
   the step where it breaks. That converts scoring from 1 bit (pass/fail) into
   a step-level diagnosis worth ~2.6 bits, usable as a dense RL reward. It is
   the most valuable artifact in the package and OpenAI's own harness does not
   expose it to solvers.

Writes specs/<problem_id>.json plus specs/_index.json.

Run:  venv/bin/python scripts/extract_dgp.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import GENEBENCH_PUBLIC, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "specs"

# Canonical section names, in order. Reports are consistent enough that
# splitting on "<n> <Title>" at line start is reliable.
SECTIONS = [
    (1, "Overview"),
    (2, "Released Prompt and Files"),
    (3, "Answer Fields and Tolerances"),
    (4, "Structure Diagram"),
    (5, "Variables and Assumptions"),
    (6, "Data-Generating Process"),
    (7, "Analyst Walkthrough"),
    (8, "Estimand"),
    (9, "Estimator"),
    (10, "Decision-Point and Ablation Walkthrough"),
    (11, "References"),
]


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(pg.extract_text() or "" for pg in PdfReader(str(path)).pages)


def split_sections(txt: str) -> dict[str, str]:
    """Return {section_title: body}. Missing sections are simply absent."""
    marks: list[tuple[int, int, str]] = []
    for num, title in SECTIONS:
        # Headings survive extraction as "6 Data-Generating Process", though
        # spacing inside words is sometimes mangled -- match loosely.
        pat = re.compile(rf"\n\s*{num}\s+{re.escape(title[:18])}", re.I)
        m = pat.search(txt)
        if m:
            marks.append((m.start(), num, title))
    marks.sort()
    out: dict[str, str] = {}
    for i, (start, _num, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(txt)
        out[title] = txt[start:end].strip()
    return out


# Row boundaries in the flattened tables. Two complications, both from PDF
# extraction rather than the reports themselves:
#   * failure stages appear as "Step 3" in some reports and "Stage 1 CNV QC" in
#     others, and column order differs between reports
#   * the verdict is frequently glued to the preceding token, e.g. "correctyes",
#     "cnv_no_segmental_dup_qcno" -- so \byes\b never matches
# We therefore anchor on <verdict><failure-point> pairs and treat everything
# between anchors as one row. `raw` stays authoritative.
ROW_ANCHOR = re.compile(
    r"(?P<verdict>yes|no)\s*(?P<point>none|Stage\s*\d+|Step\s*\d+)",
    re.I,
)
# Some reports order the columns method | verdict | numbers | stage, so the
# verdict and the failure point are separated by the output values. Only used
# when the strict anchor finds nothing, since it is easier to fool.
ROW_ANCHOR_LOOSE = re.compile(
    r"(?P<verdict>yes|no)[\s,;]*(?P<nums>(?:-?\d+\.\d+[\s,]*)+)"
    r"(?P<point>Reference|none|Stage\s*\d+[A-Za-z\s]{0,24}|Step\s*\d+)",
    re.I,
)
NUM = re.compile(r"-?\d+\.\d{3,}")


def _squash(section: str) -> str:
    """Join wrapped table rows into one flowing string."""
    body = re.sub(r"^\s*10\s+Decision-Point[^\n]*\n", "", section, flags=re.I)
    return re.sub(r"\s*\n\s*", " ", body).strip()


def _parse_loose(flat: str) -> list[dict]:
    """Fallback for method | verdict | numbers | stage column ordering."""
    rows: list[dict] = []
    matches = list(ROW_ANCHOR_LOOSE.finditer(flat))
    for i, m in enumerate(matches):
        start = matches[i - 1].end() if i else 0
        tail = flat[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(flat)]
        point = m.group("point").strip()
        stage = re.search(r"(?:Stage|Step)\s*(\d+)", point, re.I)
        rows.append(
            {
                "verdict": "pass" if m.group("verdict").lower() == "yes" else "fail",
                "fails_at_stage": int(stage.group(1)) if stage else None,
                "failure_point": None if point.lower() in ("none", "reference") else point,
                "numbers": [float(n) for n in NUM.findall(m.group("nums"))],
                "rationale": " ".join(tail.split())[:300] or None,
                "raw": " ".join(flat[start : m.end()].split())[-300:],
            }
        )
    return rows


def parse_ablations(section: str) -> list[dict]:
    """Pull rows out of the ablation table.

    Heuristic by necessity -- these are PDF tables flattened to text and the
    column order is not consistent across reports. Anchors on verdict+failure
    point, so a row is captured even when the verdict is glued to a method
    name. Always cross-check against `ablation_raw` before relying on a row.
    """
    flat = _squash(section)
    anchors = list(ROW_ANCHOR.finditer(flat))
    if not anchors:
        return _parse_loose(flat)
    rows: list[dict] = []
    for i, m in enumerate(anchors):
        start = anchors[i - 1].end() if i else 0
        chunk = flat[start : m.end()].strip()
        tail = flat[m.end() : anchors[i + 1].start() if i + 1 < len(anchors) else len(flat)]

        point = m.group("point").strip()
        stage = re.search(r"(?:Stage|Step)\s*(\d+)", point, re.I)
        rows.append(
            {
                "verdict": "pass" if m.group("verdict").lower() == "yes" else "fail",
                "fails_at_stage": int(stage.group(1)) if stage else None,
                "failure_point": None if point.lower() == "none" else point,
                "numbers": [float(n) for n in NUM.findall(chunk)],
                "rationale": " ".join(tail.split())[:300] or None,
                "raw": " ".join(chunk.split())[-300:],
            }
        )
    return rows


def main() -> int:
    if not GENEBENCH_PUBLIC.exists():
        print(f"ERROR: {GENEBENCH_PUBLIC} not found", file=sys.stderr)
        return 2
    OUT.mkdir(exist_ok=True)

    index = []
    for pdir in sorted((GENEBENCH_PUBLIC / "problems").iterdir()):
        cfg_path = pdir / "eval_config.json"
        pdf_path = pdir / "report_public.pdf"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        pid = cfg["id"]

        rec: dict = {
            "id": pid,
            "task": cfg.get("task"),
            "ground_truth": cfg.get("ground_truth"),
            "grader": cfg.get("grader"),
            "data_files": cfg.get("data_files"),
        }

        if pdf_path.exists():
            txt = read_pdf(pdf_path)
            secs = split_sections(txt)
            rec["sections_found"] = sorted(secs)
            rec["dgp"] = secs.get("Data-Generating Process")
            rec["variables"] = secs.get("Variables and Assumptions")
            rec["estimand"] = secs.get("Estimand")
            rec["estimator"] = secs.get("Estimator")
            abl = secs.get("Decision-Point and Ablation Walkthrough", "")
            rec["ablation_raw"] = abl
            rec["ablations"] = parse_ablations(abl)
        else:
            rec["sections_found"] = []

        (OUT / f"{pid}.json").write_text(json.dumps(rec, indent=2))
        index.append(
            {
                "id": pid,
                "grader_type": (cfg.get("grader") or {}).get("type"),
                "n_answer_fields": len(cfg.get("ground_truth") or {}),
                "n_data_files": len(cfg.get("data_files") or []),
                "has_dgp": bool(rec.get("dgp")),
                "n_ablations": len(rec.get("ablations") or []),
            }
        )
        print(
            f"  {pid:<48} dgp={'y' if rec.get('dgp') else 'N'} "
            f"ablations={len(rec.get('ablations') or []):>2}"
        )

    (OUT / "_index.json").write_text(json.dumps(index, indent=2))
    have = sum(r["has_dgp"] for r in index)
    tot_abl = sum(r["n_ablations"] for r in index)
    print(f"\n{have}/{len(index)} problems with a DGP section; {tot_abl} ablation rows total")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
