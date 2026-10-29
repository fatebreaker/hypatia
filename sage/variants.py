"""Task-text variants and blind prompts: many verified wordings of one exercise.

Why. A model trained on a corpus with few distinct task texts learns which script answers
which prompt (48 texts for ten genomics families; rewording the prompt dropped accuracy
0.991 -> 0.466). Variants break that key; blind prompts (the method not named) make the
model choose the analysis. Neither touches the data, the gold or the trap table, so the
invariants I1-I4 are unchanged by construction: they depend on the data, never the words.

A variant must keep every fact of the original. That is checked MECHANICALLY (`check`):

  numbers       the set of numeric literals is unchanged (none lost, none added; "70" and
                "70.0" are the same number; digits inside identifiers such as u1 or log10
                are part of the identifier, not numbers)
  file names    every data file name / path survives
  identifiers   every snake_case, mixed letter-digit or ALL-CAPS token, every column named
                in a "Columns:" list or a "name: description" line, every header of the
                instance's data files that the text mentions, and every answer key survives
  answer format the JSON answer template ({"answer": {...}}) survives character for
                character; if the text has none, every answer key survives
  requirements  each sentence that states a requirement (Assume / Report / Treat / Use /
                Do not / ... ) keeps at least 60% of its content words -- a paraphrase
                keeps "circular" and "orbit"; dropping "Assume a circular orbit." fails

Sources of variants (all validated against the original before use):

  plug-in       `Family.task_phrasings(data, blind=False) -> [text]`: the author's own
                templates, rebuilt from the draw (optional hook; default none)
  generic       `reword(text, rng)`: deterministic structural rewording any family gets for
                free -- equivalent lead-ins ("Question:" / "The question:" / ...), persona
                and instruction phrasing, and the question moved first or last. Only
                wording and order change, so every check passes by construction
  llm           rewordings produced OFFLINE by a served model through an OpenAI-compatible
                endpoint (a pod's vLLM server, GBP_BASE_URL, as harness/agents.py calls it):
                `sage reword` writes a cache keyed by the sha1 of the text; emission and
                corpus assembly read the cache, so their output stays deterministic

Blind prompts. A plug-in supports them by accepting `task_text(data, blind=False)`; with
blind=True the text states the question and the data but not the method. `task_text_of`
calls it where supported and reports "unsupported" otherwise (never an error). The corpus
builder has no draw, only the stored text, so blind text is also derived from text:
`blind_from_text` uses the plug-in's `blind_from_text(task)` hook if it has one, else drops
the "Approach:" paragraph -- the convention every plug-in follows. The tests check, on fresh
draws of every family, that this equals `task_text(data, blind=True)` exactly.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# harness/agents.build_prompt: f"{task}\n\nFiles in your working directory:\n{listing}\n{extra}"
PROMPT_SEP = "\n\nFiles in your working directory:\n"

# ----------------------------------------------------------------------------- facts
_NUM = re.compile(r"(?<![A-Za-z_\d])(-?)(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z_\d])")
_FILE = re.compile(r"(?<![\w/.-])((?:[\w-]+/)*[\w.-]+\.(?:csv|tsv|txt|json|jsonl|gz|npz|npy|parquet|h5|h5ad|bed|vcf|"
                   r"fa|fasta|fastq|bam|mtx|xlsx|xls|pkl|tab|dat|zip|feather)(?:\.gz)?)(?![\w])")
_SNAKE = re.compile(r"(?<![\w])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![\w])")
_ALNUM = re.compile(r"(?<![\w])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z\d]+(?![\w])")
_CAPS = re.compile(r"(?<![\w])[A-Z][A-Z0-9]+(?![\w])")
_DEFLINE = re.compile(r"(?m)^[ \t]*(?:-[ \t]*)?([a-z_][A-Za-z0-9_]*)[ \t]*:")
_QUOTED = re.compile(r"`([^`\n]{1,40})`|\"([A-Za-z_][\w.-]{0,39})\"")
_REQ = re.compile(r"(?:^|(?<=[.!?:]\s))((?:Assume|Report|Treat|Use|Do not|Don't|Ignore|Include|Exclude|Round|Express|"
                  r"Give|Consider|Only|Note that|You must|Remember)\b[^.!?\n]*[.!?]?)")
_STOP = frozenset("the a an of to in on for and or is are be by with as at that this these those your you it its from "
                  "each per not no into over than then them they their there which what when where how does do was "
                  "were has have had can may will would should must only also both more most less such same other "
                  "all any".split())
REQ_COVERAGE = 0.6


def _num_key(sign: str, s: str) -> str:
    try:
        v = float(sign + s)
    except ValueError:
        return sign + s
    return repr(v) if math.isfinite(v) else sign + s


def numbers(text: str) -> set[str]:
    return {_num_key(m.group(1), m.group(2)) for m in _NUM.finditer(text)}


def file_names(text: str) -> set[str]:
    return {m.group(1) for m in _FILE.finditer(text)}


def answer_template(text: str) -> str | None:
    """The JSON answer template, verbatim: the brace-balanced object around '"answer"'."""
    i = text.rfind('"answer"')
    if i < 0:
        return None
    start = text.rfind("{", 0, i)
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None


def answer_keys_of(text: str) -> list[str]:
    """Keys of the "answer" object in the JSON template (for texts whose family is unknown)."""
    t = answer_template(text) or ""
    m = re.search(r'"answer"\s*:\s*\{(.*?)\}', t, re.S)
    return re.findall(r'"([^"]+)"\s*:', m.group(1)) if m else []


def _columns_lists(text: str) -> set[str]:
    """Column names listed after "Columns:" (first word of each top-level comma item)."""
    out = set()
    for m in re.finditer(r"Columns?:\s*", text):
        i, depth, item, items = m.end(), 0, "", []
        while i < len(text):
            c = text[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            if depth == 0 and (c == "," or (c == "." and (i + 1 == len(text) or text[i + 1] in " \n")) or c == "\n"):
                items.append(item)
                item = ""
                if c != ",":
                    break
            else:
                item += c
            i += 1
        if item:
            items.append(item)
        for it in items:
            w = re.sub(r"^\s*(?:and|or)\s+", "", it).strip()
            w = re.match(r"[A-Za-z_][A-Za-z0-9_]*", w)
            if w:
                out.add(w.group(0))
    return out


def file_columns(files_dir: Path | None) -> set[str]:
    """Header names of the CSV/TSV data files in a directory (gzip too); empty if none."""
    import gzip
    out: set[str] = set()
    if not files_dir or not Path(files_dir).is_dir():
        return out
    for p in sorted(Path(files_dir).iterdir()):
        n = p.name
        if not re.search(r"\.(csv|tsv|txt)(\.gz)?$", n):
            continue
        try:
            op = gzip.open if n.endswith(".gz") else open
            with op(p, "rt") as fh:
                head = fh.readline()
        except (OSError, UnicodeDecodeError):
            continue
        out |= {c.strip().strip('"') for c in re.split(r"[\t,]", head.strip()) if c.strip()}
    return out


def identifiers(text: str, answer_keys: Iterable[str] = (), protect: Iterable[str] = ()) -> set[str]:
    ids = set(_SNAKE.findall(text)) | set(_ALNUM.findall(text)) | set(_CAPS.findall(text))
    ids |= set(_DEFLINE.findall(text)) | _columns_lists(text)
    for a, b in _QUOTED.findall(text):
        ids.add(a or b)
    ids |= {k for k in answer_keys if k}
    ids |= {p for p in protect if p and re.search(rf"(?<![\w]){re.escape(p)}(?![\w])", text)}
    return {i for i in ids if i.strip()}


def requirements(text: str) -> list[str]:
    """Requirement sentences (outside the answer template), for the coverage check."""
    tpl = answer_template(text)
    body = text.replace(tpl, " ") if tpl else text
    return [m.group(1).strip() for m in _REQ.finditer(body) if len(m.group(1).split()) >= 2]


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in _STOP}


def _has(text: str, tok: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])", text) is not None


def check(original: str, variant: str, answer_keys: Iterable[str] = (), protect: Iterable[str] = ()) -> list[str]:
    """[] if `variant` keeps every fact of `original`; otherwise the reasons it does not."""
    if not isinstance(variant, str) or not variant.strip():
        return ["empty"]
    why = []
    if variant.strip() == original.strip():
        why.append("identical")
    r = len(variant) / max(len(original), 1)
    if not 0.6 <= r <= 1.8:
        why.append(f"length ratio {r:.2f}")
    na, nb = numbers(original), numbers(variant)
    if na - nb:
        why.append(f"numbers lost {sorted(na - nb)[:6]}")
    if nb - na:
        why.append(f"numbers added {sorted(nb - na)[:6]}")
    lost = sorted(f for f in file_names(original) if f not in variant)
    if lost:
        why.append(f"files lost {lost[:6]}")
    keys = list(answer_keys) or answer_keys_of(original)
    lost = sorted(i for i in identifiers(original, keys, protect) if not _has(variant, i))
    if lost:
        why.append(f"identifiers lost {lost[:8]}")
    tpl = answer_template(original)
    if tpl is not None and tpl not in variant:
        why.append("answer format changed")
    elif tpl is None:
        miss = [k for k in keys if not _has(variant, k)]
        if miss:
            why.append(f"answer keys lost {miss}")
    vw = _content_words(variant)
    for s in requirements(original):
        cw = _content_words(s)
        if cw and len(cw & vw) / len(cw) < REQ_COVERAGE:
            why.append(f"requirement dropped: {s[:60]!r}")
    return why


def valid(original: str, variant: str, **kw) -> bool:
    return not check(original, variant, **kw)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


# ----------------------------------------------------------------------------- blind
def blind_supported(fam) -> bool:
    """True if the plug-in's task_text accepts `blind` (the convention: blind: bool = False)."""
    try:
        sig = inspect.signature(fam.task_text)
    except (TypeError, ValueError):
        return False
    return "blind" in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def task_text_of(fam, data, blind: bool = False) -> tuple[str, str]:
    """(text, kind): kind "default", "blind", or "blind-unsupported" (the default text,
    because the plug-in's task_text has no `blind` parameter)."""
    if not blind:
        return fam.task_text(data), "default"
    if not blind_supported(fam):
        return fam.task_text(data), "blind-unsupported"
    return fam.task_text(data, blind=True), "blind"


def drop_paragraphs(text: str, starts: tuple[str, ...] = ("Approach:",)) -> str:
    return "\n\n".join(p for p in text.split("\n\n") if not p.lstrip().startswith(starts))


def blind_from_text(fam, task: str) -> str | None:
    """The blind version of a family's default task text, from the text alone; None if the
    family has no blind support. Exactness is tested against task_text(data, blind=True)."""
    hook = getattr(fam, "blind_from_text", None)
    if callable(hook):
        return hook(task)
    if not blind_supported(fam):
        return None
    return drop_paragraphs(task)


# ----------------------------------------------------------------------------- generic rewording
# Equivalent lead-ins; the first alternative is the original wording. Each is replaced only
# where it starts a paragraph or line (or, for mid-paragraph ones, at its one occurrence).
LEADINS: list[tuple[str, list[str]]] = [
    ("Question: ", ["Question: ", "The question: ", "What to find out: ", "Research question: "]),
    ("Data files:\n", ["Data files:\n", "Files provided:\n", "The data:\n", "Input files:\n"]),
    ("Approach: ", ["Approach: ", "Suggested approach: ", "Method: ", "How to analyse it: "]),
    ("Return your answer as JSON: ", ["Return your answer as JSON: ", "Give the final answer as JSON: ",
                                      "Report the answer in this JSON format: ", "Answer with JSON of the form: "]),
    ("Goal: ", ["Goal: ", "Objective: ", "Aim: ", "Task: "]),
    ("Interpretation: ", ["Interpretation: ", "Definition: ", "How the answer is defined: ", "Meaning of the answer: "]),
    ("The dataset is located at: ", ["The dataset is located at: ", "The data file is: ", "The data are in: ",
                                     "Dataset location: "]),
    ("The dataset has the following description:", ["The dataset has the following description:",
                                                    "Description of the dataset:", "What the data contain:",
                                                    "About the dataset:"]),
    ("The causal question I would like you to answer is:", ["The causal question I would like you to answer is:",
                                                            "The causal question to answer:",
                                                            "Answer this causal question:", "The causal question is:"]),
    ("Estimate the causal effect that answers the question from the data. ",
     ["Estimate the causal effect that answers the question from the data. ",
      "From the data, estimate the causal effect the question asks about. ",
      "Use the data to estimate the causal effect in question. ",
      "Estimate, from the data, the causal effect that answers the question. "]),
    # the GeneBench-Pro boilerplate (genomics families and the legacy corpora)
    ("You are given ", ["You are given ", "You have ", "You are provided with ", "Available to you are "]),
    ("These data came from a real experiment; you will be graded not just on numerical correctness but the quality "
     "of analytical reasoning you exhibit; do not attempt to take any shortcuts.",
     ["These data came from a real experiment; you will be graded not just on numerical correctness but the quality "
      "of analytical reasoning you exhibit; do not attempt to take any shortcuts.",
      "The data come from a real experiment. You will be graded on the quality of the analytical reasoning you "
      "exhibit, not just on numerical correctness; do not attempt to take any shortcuts.",
      "You will be graded not just on numerical correctness but on the quality of the analytical reasoning you "
      "exhibit, and these data came from a real experiment; do not attempt to take any shortcuts.",
      "Do not attempt to take any shortcuts: these data came from a real experiment, and you will be graded on the "
      "quality of analytical reasoning you exhibit as well as on numerical correctness."]),
    ("Return your final answer as exactly one JSON object.",
     ["Return your final answer as exactly one JSON object.", "Your final answer must be exactly one JSON object.",
      "Give your final answer as exactly one JSON object.", "Answer with exactly one JSON object."]),
    ("Return the JSON object in your final answer:",
     ["Return the JSON object in your final answer:", "Use this JSON object in your final answer:",
      "Your final answer is a JSON object of this form:", "The JSON object to return as your final answer:"]),
]
RUN_STARTS = ("Report ", "Do not ")       # runs of such sentences / lines are independent: order is free
_PERSONA = re.compile(r"^You are (an? [^.\n]{3,120}?)\.(?=\s)")
PERSONAS = ["You are {x}.", "Act as {x}.", "Take the role of {x}.", "Work on this as {x}."]
_WILL = re.compile(r"(?<![\w])You will (analyse|analyze|answer|estimate|examine)(?![\w])")
WILLS = ["You will {v}", "Your task is to {v}", "You are asked to {v}", "The job is to {v}"]


def _blocks(text: str) -> tuple[list[str], str]:
    """Paragraphs (or, for a one-paragraph text, lines) and the separator that joins them."""
    paras = text.split("\n\n")
    return (paras, "\n\n") if len(paras) > 1 else (text.split("\n"), "\n")


def _kind(b: str, i: int) -> str:
    s = b.lstrip()
    if '"answer"' in s or s.startswith(("Return your", "Return the JSON")):
        return "answer"                      # checked first: causal's last paragraph starts "Estimate ..."
    if s.startswith(("Approach:", "Here are some example methods", "Method:")):
        return "method"
    if s.startswith(("Question:", "The causal question", "Goal:")):
        return "question"
    if i == 0 and s.startswith("You are "):
        return "persona"
    return "context"


def _shuffle_runs(parts: list[str], rng: np.random.Generator) -> list[str]:
    """Permute each maximal run (length >= 2) of consecutive parts that start with the same
    RUN_STARTS word -- independent "Report X as ..." definitions, "Do not ..." rules."""
    out, i = [], 0
    while i < len(parts):
        st = next((s for s in RUN_STARTS if parts[i].startswith(s)), None)
        j = i + 1
        while st and j < len(parts) and parts[j].startswith(st):
            j += 1
        run = parts[i:j]
        if len(run) >= 2:
            run = [run[x] for x in rng.permutation(len(run))]
        out += run
        i = j
    return out


def reword(text: str, rng: np.random.Generator, order: int | None = None) -> str:
    """One structural rewording: lead-ins, persona and instruction phrasing drawn from rng,
    and the question block kept in place (order 0), moved first -- after the persona -- (1),
    or moved to just before the answer block (2). The answer block (the paragraph holding
    the JSON template, or in a one-paragraph text every line from the first "Return ...")
    always stays last. Content is never touched, so `check` passes by construction."""
    blocks, sep = _blocks(text)
    if sep == "\n":
        for i, b in enumerate(blocks):
            if b.lstrip().startswith("Return") or '"answer"' in b:
                blocks = blocks[:i] + ["\n".join(blocks[i:])]
                break
    kinds = [_kind(b, i) for i, b in enumerate(blocks)]
    order = int(rng.integers(3)) if order is None else order
    if order and "question" in kinds and kinds[-1] == "answer":
        qi = kinds.index("question")
        q = blocks.pop(qi)
        kinds.pop(qi)
        at = (1 if kinds and kinds[0] == "persona" else 0) if order == 1 else len(blocks) - 1
        blocks.insert(at, q)
    s = sep.join(blocks)
    s = "\n".join(_shuffle_runs(s.split("\n"), rng))
    s = "\n".join(" ".join(_shuffle_runs(re.split(r"(?<=\.) (?=[A-Z])", ln), rng)) for ln in s.split("\n"))
    for orig_lead, alts in LEADINS:
        if orig_lead in s:
            s = s.replace(orig_lead, alts[int(rng.integers(len(alts)))], 1)
    m = _PERSONA.match(s)
    if m:
        s = PERSONAS[int(rng.integers(len(PERSONAS)))].format(x=m.group(1)) + s[m.end():]
    m = _WILL.search(s)
    if m:
        s = s[:m.start()] + WILLS[int(rng.integers(len(WILLS)))].format(v=m.group(1)) + s[m.end():]
    return s


def generic_variants(text: str, n: int, seed: int, attempts: int = 60) -> list[str]:
    """Up to n distinct structural rewordings of text (never the text itself), deterministic in seed."""
    rng = np.random.default_rng([seed, 0x5A6E])
    out: list[str] = []
    seen = {text}
    for a in range(attempts):
        if len(out) >= n:
            break
        v = reword(text, rng, order=a % 3 if a < 3 else None)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ----------------------------------------------------------------------------- LLM rewordings (offline)
REWORD_SYSTEM = (
    "You rewrite the task statements of data-analysis exercises so that the same task is worded differently.\n"
    "Rules:\n"
    "1. Keep every number exactly as written (the same digits), every file name and path, every column name and "
    "identifier, every unit, and every requirement or assumption.\n"
    "2. Keep the answer-format instruction, and its JSON exactly character for character, at the end.\n"
    "3. Change the wording and the order of sentences and paragraphs. Do not add information, hints, methods or "
    "examples, and do not remove any.\n"
    "4. Output only the rewritten task statement, between <task> and </task>.")


def _http(method: str, url: str, payload: dict | None, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LLMRewriter:
    """OpenAI-compatible chat client for offline rewording (a pod's vLLM server).

    Endpoint and key come from the arguments or, as in harness/agents.py, GBP_BASE_URL and
    GBP_API_KEY; the model from the argument, GBP_MODEL, or the server's first /models
    entry. `transport(method, url, payload, headers, timeout) -> dict` is injectable, so the
    client is unit-tested without a server."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.9, top_p: float = 0.95, max_tokens: int = 4096, timeout: float = 600,
                 enable_thinking: bool | None = False, transport: Callable | None = None, retries: int = 4):
        self.base_url = (base_url or os.environ.get("GBP_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("no endpoint: pass --base-url or set GBP_BASE_URL to the pod's server "
                               "(http://127.0.0.1:PORT/v1)")
        self.api_key = api_key or os.environ.get("GBP_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        if self.base_url.startswith("https://api.openai.com") and self.api_key == "EMPTY":
            raise RuntimeError("refusing to call api.openai.com without a key (task texts would leave the cluster)")
        self.transport = transport or _http
        self.temperature, self.top_p, self.max_tokens, self.timeout = temperature, top_p, max_tokens, timeout
        self.enable_thinking, self.retries = enable_thinking, retries
        self.model = model or os.environ.get("GBP_MODEL") or self._first_model()

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _first_model(self) -> str:
        res = self.transport("GET", f"{self.base_url}/models", None, self._headers(), self.timeout)
        return res["data"][0]["id"]

    def chat(self, messages: list[dict], seed: int | None = None) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "top_p": self.top_p, "max_tokens": self.max_tokens}
        if seed is not None:
            payload["seed"] = int(seed)
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        last = None
        for attempt in range(self.retries + 1):
            try:
                res = self.transport("POST", f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout)
                return (res["choices"][0]["message"].get("content") or "")
            except urllib.error.HTTPError as e:                  # 429/5xx: retry; 4xx: give up
                last = e
                if e.code not in (429, 500, 502, 503, 529) or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                if attempt == self.retries:
                    raise
            import time
            time.sleep(min(30, 3 * (attempt + 1)))
        raise RuntimeError(f"chat failed: {last}")

    @staticmethod
    def extract(reply: str) -> str:
        reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
        m = re.findall(r"<task>\s*(.*?)\s*</task>", reply, re.S)
        return (m[-1] if m else reply).strip()

    def reword(self, task: str, n: int, seed: int = 0) -> list[str]:
        """n raw (unvalidated) rewordings, one request each, seeded seed..seed+n-1."""
        msgs = [{"role": "system", "content": REWORD_SYSTEM},
                {"role": "user", "content": f"Rewrite this task statement:\n<task>\n{task}\n</task>"}]
        return [self.extract(self.chat(msgs, seed + i)) for i in range(n)]


def llm_variants(rw: LLMRewriter, task: str, k: int, seed: int = 0, attempts: int | None = None,
                 answer_keys=(), protect=()) -> dict:
    """Request rewordings until k pass `check` (or `attempts`, default 2k, are spent)."""
    attempts = attempts or 2 * k
    kept, rejected, i = [], [], 0
    while len(kept) < k and i < attempts:
        batch = rw.reword(task, min(k - len(kept), attempts - i), seed + i)
        i += len(batch)
        for v in batch:
            why = check(task, v, answer_keys=answer_keys, protect=protect)
            if not why and v in kept:
                why = ["duplicate"]
            (rejected.append({"why": why, "text": v}) if why else kept.append(v))
    return {"sha1": sha1(task), "text": task, "variants": kept, "rejected": rejected, "model": rw.model,
            "requests": i}


def load_cache(paths) -> dict[str, list[str]]:
    """sha1(text) -> cached LLM variants (several cache files merge; later ones append)."""
    out: dict[str, list[str]] = {}
    for p in ([paths] if isinstance(paths, (str, Path)) else list(paths or [])):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cur = out.setdefault(r["sha1"], [])
                cur += [v for v in r.get("variants", []) if v not in cur]
    return out


# ----------------------------------------------------------------------------- candidates
def candidates(original: str, k: int, seed: int, *, label: str = "canonical", fam=None, data=None,
               blind: bool = False, cache: dict | None = None, answer_keys=(), protect=(),
               files_dir: Path | None = None) -> list[tuple[str, str]]:
    """[(source, text)]: the original first, then up to k-1 verified rewordings -- the LLM
    cache, the plug-in's phrasings, then the generic rewriter, each validated against the
    original. Deterministic in (original, seed, cache)."""
    out = [(label, original)]
    if k <= 1:
        return out
    prot = set(protect) | file_columns(files_dir)
    seen = {original}

    def add(src, texts):
        for j, t in enumerate(texts):
            if len(out) >= k:
                return
            if t in seen or check(original, t, answer_keys=answer_keys, protect=prot):
                continue
            seen.add(t)
            out.append((f"{src}:{j}", t))
    if cache:
        add("llm", cache.get(sha1(original), []))
    add("plugin", phrasings(fam, data, blind))
    add("generic", generic_variants(original, max(k - len(out), 0) + 3, seed))
    return out


def phrasings(fam, data, blind: bool = False) -> list[str]:
    """The plug-in's own templates (`task_phrasings(data, blind=False)`), [] without the hook.
    A hook without a `blind` parameter is skipped in blind mode: its texts would name the method."""
    hook = getattr(fam, "task_phrasings", None) if fam is not None and data is not None else None
    if not callable(hook):
        return []
    try:
        takes_blind = "blind" in inspect.signature(hook).parameters
    except (TypeError, ValueError):
        takes_blind = False
    if blind and not takes_blind:
        return []
    return list((hook(data, blind=blind) if takes_blind else hook(data)) or [])


def choose(n: int, seed: int) -> int:
    """The variant an instance ships as its task: one index, fixed by the seed."""
    return int(np.random.default_rng([seed, 0xC401CE]).integers(n)) if n > 1 else 0


def stable_int(*parts) -> int:
    """A seed from strings (Python's hash() is salted per process; this is not)."""
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")
