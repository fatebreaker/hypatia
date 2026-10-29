"""Every registered decision rule read since 2026-10-01 (the 9B / 27B programme and its cross-family tests), with its verdict,
from docs/RL_PLAN.md. v2 (10-17, mock review): a verdict is a bold '**RULE MET**' / '**RULE NOT MET**' in a READ / COMPUTED
entry, or 'RULE ...: ... -> MET' in a **READ** entry written that way (the astronomy reads); each rule is linked to the
earliest REGISTERED / AMENDMENT entry that defines it (registration time and wording), and the reading text before the
verdict (its estimate) is kept. v3 (10-29): also a bold block or an arrow ENDING in the verdict (see TAIL). Writes docs/rl4sci/registry_table.md (pipe table) and prints the totals.
    python3 docs/rl4sci/make_registry_table.py"""
import re
R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
lines = open(f"{R}/docs/RL_PLAN.md").read().split("\n")
ENTRY = re.compile(r"\* (2026-\d\d-\d\d) (\d\d:\d\d) UTC -- (.*)")
entries = [m.groups() for m in map(ENTRY.match, lines) if m]
BOLD = re.compile(r"\*\*([A-Z0-9][A-Z0-9\-]+(?:-[A-Z0-9]+)*) (NOT MET|MET)\*\*")
PAREN = r"\(((?:[^()]|\([^()]*\))*)\)"   # a parenthetical, one level of nesting
RULEWORD = re.compile(r"RULE ([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?: \((?:registered|amended)[^)]{0,60}\))? (NOT MET|MET)\b")
ARROW = re.compile(r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?: \([^)]{0,60}\))?:([^;]{0,240}?)-> (NOT MET|MET)\b")
# 10-29 (v3): two more verdict forms that the forms above missed, so Supplementary Table 1 lacked about 60 reads, most of
# them NOT MET: a bold block ENDING in the verdict ('**PR-LONGER-HELPS NOT MET.**', '**B-QR4-9K: QR4-LIFTS NOT MET.**',
# '**BEAT-CLAUDE-MC NOT MET by B-GEN-H.**', '**+1.6 (+26/-22, p = 0.33) -> PR9-BEATS-SCI9 NOT MET**') and an unbolded arrow
# ending in it ('... (CI -8.5 to +0.9) -> BEAT-CLAUDE-MC NOT MET'). The arm is the block's leading 'NAME:', the 'by NAME', or
# the entry's first '**NAME = ' estimate, else the entry's arm; the estimate is the block's text before the rule, else the
# sentence before the block. Rows go through the same de-duplication as the rest.
NM = r"(?<![A-Za-z0-9\-])([A-Z0-9][A-Z0-9]*(?:-[A-Z0-9]+)+)"
TAIL = re.compile(r"\*\*([^*]{0,500}?)" + NM + r" (NOT MET|MET)(?: (?:by|for) ([A-Z][A-Za-z0-9\-]+))?[.!]?\*\*")
RARROW = re.compile(r"-> " + NM + r" (NOT MET|MET)\b")


def clean(t, n):
    t = re.sub(r"\*\*", "", t).replace("|", "/")
    return (t[:n] + "...") if len(t) > n else t


# registration: the earliest REGISTERED / AMENDMENT entry naming the rule as **RULE** (its definition)
reg = {}
for day, hm, body in entries:
    if not re.search(r"REGISTERED|AMENDMENT", body[:120]):
        continue
    for m in re.finditer(r"\*\*([A-Z0-9][A-Z0-9\-]+)\*\*([^*]{0,260})", body):
        name = m.group(1)
        if name not in reg and re.search(r"\biff\b|>=|<=|\bif\b", m.group(2)):
            reg[name] = (f"{day[5:]} {hm}", clean(name + m.group(2), 240))
# fallback (10-20): rules defined on the indented lines under a registration entry ('  NAME (PRIMARY): ...', as H-ASTRO27),
# used only for rules the bold form above did not define; a rule defined inside another's line ('ABOVE means ...' in
# ASTRO27-COMPETITIVE) takes that line
cur = None
for l in lines:
    m = ENTRY.match(l)
    if m:
        cur = m.groups()[:2] if re.search(r"REGISTERED|AMENDMENT|registered before any run", m.group(3)[:120]) else None
        continue
    d = re.match(r"\s+([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?: \(PRIMARY\))?: (.*)", l) if cur else None
    if d and d.group(1) not in reg and re.search(r">=|<=|\biff\b|\bmeans\b|p [<>]", d.group(2)):
        reg[d.group(1)] = (f"{cur[0][5:]} {cur[1]}", clean(d.group(1) + ": " + d.group(2), 240))
        for w in re.findall(r"\b([A-Z]{4,}) means\b", d.group(2)):
            reg.setdefault(d.group(1).rsplit("-", 1)[0] + "-" + w, reg[d.group(1)])
rows = []
for day, hm, body in entries:
    if day < "2026-10-01" or not re.search(r"\*\*READ|READS?\b|COMPUTED|READING", body[:60]):   # 10-24: 'READS (' too
        continue
    arm = re.search(r"\*\*READ ([^*(]+?)(?: \(|\*\*)", body)
    arm = arm.group(1).strip() if arm else ""
    if not arm:
        h = re.search(r"\b(H-[A-Z0-9][A-Z0-9\-]+)", body)
        arm = h.group(1) if h else ""
    found = []
    for v in BOLD.finditer(body):
        if v.group(1) != "NOT":
            found.append((v.group(1), v.group(2), body[max(0, v.start() - 220):v.start()], body[v.end():v.end() + 300]))
    if body.startswith("**READ"):
        names = {f[0] for f in found}
        for v in ARROW.finditer(body):
            if v.group(1) not in names:
                found.append((v.group(1), v.group(3), v.group(2), ""))
    # 10-24: the verdict form of the 10-21+ reads, 'RULE NAME MET' / 'RULE NAME (registered ...) NOT MET' (the 9B and 4B
    # requirement cells), with the estimate before it
    names = {f[0] for f in found}
    for v in RULEWORD.finditer(body):
        if v.group(1) not in names:
            names.add(v.group(1))
            found.append((v.group(1), v.group(2), body[max(0, v.start() - 220):v.start()], body[v.end():v.end() + 300]))
    lead = re.search(r"\*\*([A-Z][A-Za-z0-9\-]+) = ", body)
    tail = []
    if lead and not arm:   # a read with no '**READ NAME' or H-name: its first '**NAME = estimate' names the arm
        arm = lead.group(1)
    for v in TAIL.finditer(body):
        if v.group(2) in names or v.group(2) == "NOT" or any(t[1] == v.group(2) for t in tail):
            continue
        pre = re.match(r"([A-Z][A-Za-z0-9\-]+)(?: \([^)]*\))?:\s*", v.group(1))
        a2 = v.group(4) or (pre.group(1) if pre else (lead.group(1) if lead else arm))
        est = (v.group(1)[pre.end():] if pre else v.group(1)).strip(" :;,-").rstrip(">").strip(" -")
        if not est:
            c2 = body[max(0, v.start() - 220):v.start()]
            est = c2.split(". ")[-1] if ". " in c2 else c2
        tail.append((a2, v.group(2), v.group(3), est))
    for v in RARROW.finditer(body):
        if v.group(1) in names or any(t[1] == v.group(1) for t in tail):
            continue
        c2 = body[max(0, v.start() - 220):v.start()]
        tail.append((lead.group(1) if lead else arm, v.group(1), v.group(2), c2.split(". ")[-1] if ". " in c2 else c2))
    for a2, name, verdict, est in tail:
        names.add(name)
        rows.append((f"{day[5:]} {hm}", clean(a2, 80), name, verdict) + reg.get(name, ("", "")) + (clean(est, 220),))
    for name, verdict, ctx, after in found:
        rt = re.match(r"\s*\(read (\d\d-\d\d \d\d:\d\d); ([^)]*)\)", after)   # 10-24: a 'RECORDED IN STANDARD FORM' entry
        if rt:
            rows.append((rt.group(1), "", name, verdict) + reg.get(name, ("", "")) + (clean(rt.group(2), 220),))
            continue
        reading = ctx.split(". ")[-1] if ". " in ctx else ctx
        if not reading.strip():   # 10-20: the estimate follows the verdict ('**R MET** (estimate)'), or the verdict before it
            m = re.match(r"\s*" + PAREN, after) or re.search(r"\*\*[A-Z0-9][A-Z0-9\-]+ (?:NOT MET|MET)\*\* " + PAREN + r"\.?\s*$", ctx)
            reading = m.group(1) if m else ""
        rows.append((f"{day[5:]} {hm}", clean(arm, 80), name, verdict) + reg.get(name, ("", "")) + (clean(reading, 220),))
# fallback 2 (10-24, mock review: 113 rules had no registration fields): for a rule still unlinked, the earliest entry BEFORE
# its read that names it as a word -- a REGISTERED / AMENDMENT entry first, else any entry -- with the sentence around the name
# as its wording, marked '(by name)' so a reader can tell it from a rule written in the bold 'iff' form
def by_name(name, read_at):
    pat = re.compile(r"(?<![A-Z0-9\-])" + re.escape(name) + r"(?![A-Z0-9\-])")
    best = None
    for day, hm, body in entries:
        if f"{day[5:]} {hm}" >= read_at:
            break
        m = pat.search(body)
        if m and (best is None or (re.search(r"REGISTERED|AMENDMENT", body[:120]) and not best[2])):
            s0 = body.rfind(". ", 0, m.start()) + 2
            s1 = body.find(". ", m.end())
            best = (f"{day[5:]} {hm}", clean(body[max(0, s0):s1 if s1 > 0 else m.end() + 200] + " (by name)", 240),
                    bool(re.search(r"REGISTERED|AMENDMENT", body[:120])))
            if best[2]:
                break
    return best[:2] if best else ("", "")


def by_template(name, read_at):
    """10-24: rules named at read time from a templated definition ('RL4-LIFTS-<bench>', 'SHAM9-LIFTS-<bench>',
    'PROMPT-GROUNDS-<system>') or with a seed-count suffix ('-3S', '-2S', '-S2', '-S3'): try the generalised names in turn"""
    base = re.sub(r"-(3S|2S|S2|S3)$", "", name)
    head = base.rsplit("-", 1)[0]
    for cand in (name, base, head + "-<bench>", head + "-<BENCH>", head + "-<system>", head + "-<arm>", head):
        hit = by_name(cand, read_at)
        if hit[0]:
            return hit
    return ("", "")


def by_hypothesis(arm, read_at):
    """10-24, last resort: a rule named only at read time, under a hypothesis or arm registered as a whole -- the registration
    entry of the hypothesis / arm named in the read (its first H-NAME, else its first upper-case arm token)"""
    toks = re.findall(r"\b(H-[A-Z0-9][A-Z0-9\-]+|[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z]{2,}[0-9][A-Z0-9]*)\b", arm)
    for t in toks:
        hit = by_name(t, read_at)
        if hit[0]:
            return hit[0], clean(hit[1].replace(" (by name)", "") + " (hypothesis entry)", 240)
    return ("", "")


ALIAS = {"ASTRO4-LIFTS": "RL4-LIFTS-KEPLER",   # the same rule, read under two names (10-24 11:15)
         "NEAR-27B": "DISTILL-FULL-RL-NEAR-27B"}   # the 10-11 04:48 completion restates the 00:48 verdict by a short name
rows = [(r[0], r[1], ALIAS.get(r[2], r[2])) + r[3:] for r in rows]
rows = [r if r[4] else r[:4] + by_template(r[2], r[0]) + r[6:] for r in rows]
rows = [r if r[4] else r[:4] + by_hypothesis(r[1], r[0]) + r[6:] for r in rows]
_seen_named = set()
rows = [r for r in rows if not (r[1] == "" and r[2] in _seen_named) and not _seen_named.add(r[2]) or r[1] != ""]
seen = set(); out = []
for r in rows:
    if (r[1], r[2]) in seen:
        continue
    seen.add((r[1], r[2])); out.append(r)
met = sum(r[3] == "MET" for r in out)
hdr = "| Read (UTC) | Arm | Registered rule | Verdict | Registered (UTC) | Rule as registered | Reading |"
md = [hdr, "|---|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in out]
open(f"{R}/docs/rl4sci/registry_table.md", "w").write("\n".join(md) + "\n")
print(f"{len(out)} rule verdicts since 10-01: {met} MET, {len(out) - met} NOT MET; "
      f"{sum(1 for r in out if r[4])} linked to a registration; {sum(1 for r in out if not r[1])} without an arm name")
