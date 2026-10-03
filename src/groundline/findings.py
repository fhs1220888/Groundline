"""Findings and the claim verifier.

A :class:`Finding` is a claim in the report.  It must cite evidence IDs from the
session ledger.  :func:`verify_findings` then checks, mechanically:

1. every cited evidence ID exists;
2. every number written in the title/statement can be found in the cited
   evidence results (allowing rounding and unit scaling s<->ms, fraction<->%);
3. the finding's structured fields (channel, t_start, t_end) are consistent
   with the data and the evidence;
4. each number is used with the meaning of the field it came from: the unit written after it and a
   role word right before it (peak, duration, mean, impulse ...) must fit that field (see
   :mod:`groundline.semantics`);
5. a finding filed under an anomaly category cites evidence in which the matching tool actually
   reported that anomaly, on the finding's channel (a passed check cannot be filed as a fault).

A number may name the exact evidence field it comes from, in brackets right after it (and its unit):
``742.3 K [E4.violations[0].peak_value]``, ``3 spikes [len(E3.issues[0].spikes)]``. Such a number is
checked against that field only, instead of against every field of the cited evidence.

The LLM can phrase things however it likes, but it cannot introduce a number
that no tool produced without the report flagging it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from .semantics import EvField, check_number, evidence_fields, match_scale

CATEGORIES = (
    "combustion_oscillation",
    "redline_violation",
    "sensor_fault",
    "valve_response",
    "performance_deviation",
    "observation",
)
SEVERITIES = ("critical", "warning", "info")


@dataclass
class Finding:
    title: str
    statement: str
    category: str = "observation"
    severity: str = "info"
    channel: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    evidence: list[str] = field(default_factory=list)
    verification: Verification | None = None  # set by verify_findings, never read from input

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        keys = {f for f in cls.__dataclass_fields__} - {"verification"}
        d = {k: v for k, v in d.items() if k in keys}
        d.setdefault("title", "")
        d.setdefault("statement", "")
        ev = d.get("evidence") or []
        d["evidence"] = [ev] if isinstance(ev, str) else list(ev)
        for k in ("t_start", "t_end"):
            if d.get(k) is not None:
                try:
                    d[k] = float(d[k])
                except (TypeError, ValueError):
                    d[k] = None
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(replace(self, verification=None))
        d["verification"] = None if self.verification is None else self.verification.to_dict()
        return d


# numbers not glued to an identifier (skips E3, P2, x1e3 ...)
_NUM_RE = re.compile(r"(?<![A-Za-z_\d.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# a source tag after a number: [E4.peak], [E6.events[0].freq_hz], [len(E3.issues)] (a bare [E4] is just a reference)
_SEG = r"(?:\.[^\s.\[\]()]+|\[\d+\])+"
_CITE_RE = re.compile(r"\[\s*(len\(E\d+" + _SEG + r"\)|E\d+" + _SEG + r")\s*\]")


def citations(text: str) -> list[re.Match]:
    """Source tags in a claim; group(1) is the cited field path."""
    return list(_CITE_RE.finditer(text or ""))


def mask_citations(text: str) -> str:
    """The claim with every source tag blanked out (same length, so positions do not move)."""
    text = text or ""
    for m in citations(text):
        text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    return text


def cited_numbers(text: str) -> dict[int, str]:
    """{start of number: cited field path}. A tag belongs to the nearest number before it in the same clause,
    with only words in between ("742.3 K [E4.peak]", "3 个孤立尖峰 [len(E3.issues)]")."""
    masked = mask_citations(text)
    nums = number_matches(text)
    out: dict[int, str] = {}
    for c in citations(text):
        prev = [m for m in nums if m.end() <= c.start()]
        if not prev:
            continue
        m = prev[-1]
        gap = masked[m.end():c.start()]
        if len(gap) <= 24 and not re.search(r"[\d，,;；。:：\n（()）]", gap):
            path = re.sub(r"\.(\d+)(?=\.|\)|$)", r"[\1]", c.group(1))  # E6.events.0.x -> E6.events[0].x
            # agents see tool output as {"evidence_id": "E3", "result": {...}}: E3.result.x means E3.x
            out[m.start()] = re.sub(r"^(len\()?(E\d+)\.result(?=[.\[])", r"\1\2", path)
    return out


def number_matches(text: str) -> list[re.Match]:
    """Positions of the numbers a claim states (identifiers such as SYN-1013 or run_42, and the source
    tags after numbers, are skipped)."""
    text = mask_citations(text)
    out = []
    for m in _NUM_RE.finditer(text):
        i = m.start()
        if i >= 2 and text[i - 1] in "-_" and text[i - 2].isalpha():
            continue
        out.append(m)
    return out


def _parse_number(tok: str) -> tuple[float, int] | None:
    """Value and number of decimals of a number token ("−4.66" -> (-4.66, 2))."""
    norm = tok.replace("−", "-")
    try:
        val = float(norm)
    except ValueError:
        return None
    mant = norm.lower().split("e")[0]
    return val, len(mant.split(".")[1]) if "." in mant else 0


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """The numbers a claim states, as (token, value, decimals) — exactly what the verifier checks."""
    return [(m.group(0), *p) for m in number_matches(text) if (p := _parse_number(m.group(0))) is not None]


def _flagged_channels(ev, category: str | None = None) -> list[str] | None:
    """Channels on which this evidence entry reports an anomaly, or None if its tool reports none of the
    kind (the list may be empty: the tool ran and found nothing)."""
    r = ev.result
    if ev.tool == "check_redlines":
        return [v.get("channel") for v in r.get("violations") or []]
    if ev.tool == "detect_oscillation":  # pickup present before ignition is an instrumentation problem
        if category == "sensor_fault":
            return [r.get("channel")] if r.get("interference") else []
        return [r.get("channel")] if r.get("detected") else []
    if ev.tool == "check_sensor_health":  # an issue shared by every channel (DAQ dropouts) lists them all
        return [c for i in r.get("issues") or [] for c in (i.get("channels") or [i.get("channel")])]
    if ev.tool == "measure_valve_response":
        return [c for e in r.get("events") or [] if e.get("exceeds_limit") for c in (e.get("response"), e.get("command"))]
    if ev.tool == "check_thrust_pressure_ratio":
        return [] if r.get("consistent", True) else [r.get("thrust"), r.get("pressure")]
    if ev.tool == "compare_reference":
        return [r.get("channel")] if r.get("within_tolerance") is False or r.get("sustained_deviation_intervals") else []
    return None


# the tool whose result must report the anomaly a category claims
_CATEGORY_TOOL = {
    "redline_violation": "check_redlines",
    "combustion_oscillation": "detect_oscillation",
    "sensor_fault": ("check_sensor_health", "detect_oscillation"),
    "valve_response": "measure_valve_response",
    "performance_deviation": ("compare_reference", "check_thrust_pressure_ratio"),
}


def category_problem(f: Finding, cited: list) -> Problem | None:
    """Why the cited evidence does not support the anomaly the finding's category claims, if it does not."""
    tools = _CATEGORY_TOOL.get(f.category)
    if tools is None or not cited:
        return None
    tools = (tools,) if isinstance(tools, str) else tools
    tool = " or ".join(tools)
    flagged = [c for ev in cited if ev.tool in tools for c in (_flagged_channels(ev, f.category) or [])]
    if not any(ev.tool in tools for ev in cited):
        return Problem.of("category_uncited", category=f.category, tool=tool)
    if not flagged:
        return Problem.of("category_no_anomaly", category=f.category, tool=tool)
    if f.channel and f.channel not in flagged:
        return Problem.of("category_other_channel", category=f.category, tool=tool, flagged=sorted(set(flagged)),
                          channel=f.channel)
    return None


# ---------------------------------------------------------------------------- the verifier's interface

class EvidenceEntry(Protocol):
    """What the verifier reads of one ledger entry."""

    id: str
    tool: str
    params: dict
    result: dict


class EvidenceView(Protocol):
    """What the verifier needs of a run: ledger entries by ID and the channel names. A live
    :class:`~groundline.session.Session` is one; a stored ledger is another (:class:`LedgerView`)."""

    @property
    def channels(self) -> list[str]: ...

    def evidence(self, eid: str) -> EvidenceEntry | None: ...


@dataclass(frozen=True)
class StoredEvidence:
    id: str
    tool: str
    params: dict
    result: dict


class LedgerView:
    """A stored ledger ([{"id", "tool", "params", "result"}, ...], as eval results keep it) seen as an
    :class:`EvidenceView`, so saved runs can be verified again without re-running any tool."""

    def __init__(self, ledger: list[dict], channels: list[str]):
        self._ev = {e["id"]: StoredEvidence(e["id"], e["tool"], e["params"], e["result"]) for e in ledger}
        self.channels = list(channels)

    def evidence(self, eid: str) -> StoredEvidence | None:
        return self._ev.get(eid)


# What each problem code says. The message is what agents and report readers see, so it stays prose;
# code is what programs branch on. Problem.from_message reads a code back from these templates, for
# results stored with messages only.
_PROBLEM_TEMPLATES = {
    "unknown_evidence": "unknown evidence id(s): {ids}",
    "unknown_category": "unknown category {category!r}",
    "channel_not_in_data": "channel {channel!r} not in data",
    "tag_outside_evidence": "'{tok}' cites {cite}, but {eid} is not among the finding's evidence",
    "tag_not_numeric": "'{tok}' cites {cite}, which is not a numeric field",
    "tag_mismatch": "'{tok}' cites {cite} = {value:g}, which does not match",
    "time_not_found": "{key}={value} not found in cited evidence",
    "time_not_a_time": "{key}={value} matches only non-time fields ({path})",
    "channel_not_in_evidence": "channel {channel!r} does not appear in the cited evidence",
    "category_uncited": "category {category!r} needs cited {tool} evidence that reports it; none is cited",
    "category_no_anomaly": ("category {category!r}: the cited {tool} evidence reports no anomaly "
                            "(a check that passed is category 'observation')"),
    "category_other_channel": "category {category!r}: the cited {tool} evidence reports it for {flagged}, not {channel!r}",
}
PROBLEM_CODES = tuple(_PROBLEM_TEMPLATES)
# a source tag that names a field which does not hold the number
TAG_PROBLEMS = frozenset({"tag_outside_evidence", "tag_not_numeric", "tag_mismatch"})
# the checks of the verifier before semantic checks: is the claim tied to evidence that exists, on data that exists
GROUNDING_PROBLEMS = frozenset({"unknown_evidence", "unknown_category", "channel_not_in_data", "time_not_found",
                                "channel_not_in_evidence"})


def _template_re(template: str) -> re.Pattern:
    parts = re.split(r"\{[^}]*\}", template)
    return re.compile(".+?".join(re.escape(p) for p in parts))


_PROBLEM_RES = {code: _template_re(t) for code, t in _PROBLEM_TEMPLATES.items()}


@dataclass(frozen=True)
class Problem:
    """Something about a finding as a whole that its evidence does not support."""

    code: str
    message: str

    @classmethod
    def of(cls, code: str, suffix: str = "", **fields) -> "Problem":
        return cls(code, _PROBLEM_TEMPLATES[code].format(**fields) + suffix)

    @classmethod
    def from_message(cls, message: str) -> "Problem":
        """The problem a stored message describes (code "" if no template produces it)."""
        code = next((c for c, rx in _PROBLEM_RES.items() if rx.match(message)), "")
        return cls(code, message)


@dataclass(frozen=True)
class FieldMatch:
    value: float  # the evidence value
    scale: float  # written = value × scale (s->ms, fraction->% ...)
    field: str  # its path, e.g. "E6.events[0].freq_hz"


@dataclass(frozen=True)
class NumberCheck:
    """One number of a title or statement, and what the verifier made of it."""

    text: str
    value: float
    start: int  # position in the title or statement it belongs to
    end: int
    grounded: bool  # its value is in the cited evidence
    cited: str | None  # the field its source tag names, if any
    matched: FieldMatch | None  # the field it is read as
    consistent: bool  # grounded, with a unit and role word that fit that field
    unit: str | None
    role: str | None
    semantic_problem: str | None

    def to_dict(self) -> dict:
        return {"text": self.text, "value": self.value, "grounded": self.grounded, "cited": self.cited,
                "matched": None if self.matched is None else asdict(self.matched),
                "consistent": self.consistent, "unit": self.unit, "role": self.role,
                "semantic_problem": self.semantic_problem}


@dataclass(frozen=True)
class Verification:
    """The verifier's verdict on one finding."""

    status: str  # verified | partial | unsupported
    title_numbers: tuple[NumberCheck, ...]
    statement_numbers: tuple[NumberCheck, ...]
    problems: tuple[Problem, ...]

    @property
    def numbers(self) -> tuple[NumberCheck, ...]:
        return self.title_numbers + self.statement_numbers

    @property
    def ungrounded_numbers(self) -> list[str]:
        return [n.text for n in self.numbers if not n.grounded]

    @property
    def mismatched_numbers(self) -> list[str]:
        return [n.text for n in self.numbers if n.grounded and not n.consistent]

    @property
    def semantic_problems(self) -> list[str]:
        return [n.semantic_problem for n in self.numbers if n.semantic_problem]

    @property
    def problem_messages(self) -> list[str]:
        return [p.message for p in self.problems]

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict:
        """The stored form (report.json, eval results, agent feedback): problems as their messages."""
        nums = self.numbers
        return {
            "status": self.status,
            "n_numbers": len(nums),
            "n_grounded": sum(n.grounded for n in nums),
            "ungrounded_numbers": self.ungrounded_numbers,
            "n_consistent": sum(n.consistent for n in nums),
            "n_cited": sum(n.cited is not None for n in nums),
            "mismatched_numbers": self.mismatched_numbers,
            "semantic_problems": self.semantic_problems,
            "numbers": [n.to_dict() for n in nums],
            "problems": self.problem_messages,
        }


def verify_finding(f: Finding, s: EvidenceView) -> Verification:
    problems: list[Problem] = []
    cited = [s.evidence(e) for e in f.evidence]
    missing = [e for e, ev in zip(f.evidence, cited) if ev is None]
    cited = [ev for ev in cited if ev is not None]
    if missing:
        problems.append(Problem.of("unknown_evidence", ids=", ".join(missing)))
    if f.category not in CATEGORIES:
        problems.append(Problem.of("unknown_category", category=f.category))
    if f.channel and f.channel not in s.channels:
        problems.append(Problem.of("channel_not_in_data", channel=f.channel))

    fields: list[EvField] = []
    text_blob = ""
    for ev in cited:
        fields += evidence_fields(ev.result, ev.id)
        fields += evidence_fields(ev.params, f"{ev.id}.params")
        text_blob += repr(ev.params) + repr(ev.result)

    title_numbers: list[NumberCheck] = []
    statement_numbers: list[NumberCheck] = []
    # title and statement are read as one text, so a unit or role word is found the same way in either
    raw = f"{f.title}\n{f.statement}"
    stmt_at = len(f.title) + 1
    text = mask_citations(raw)  # the semantic checks read units and ranges across a blanked-out tag
    by_path = {fl.path: fl for fl in fields}
    cites = cited_numbers(raw)
    for m in number_matches(raw):
        tok = m.group(0)
        parsed = _parse_number(tok)
        if parsed is None:
            continue
        val, dec = parsed
        cite = cites.get(m.start())
        if cite is not None:  # the writer named the field: check against that one only
            fl = by_path.get(cite)
            sc = None if fl is None else match_scale(val, dec, fl.value, fl.kind)
            cands = [] if sc is None else [(fl, sc)]
            if cands:  # a sibling in the same object holding the very same value shares its meaning
                parent = cite.rsplit(".", 1)[0]  # (stuck_value == channel_max: "its maximum, 5180.25")
                cands += [(g, k) for g in fields if g.path != cite and g.path.rsplit(".", 1)[0] == parent
                          and g.value == fl.value and (k := match_scale(val, dec, g.value, g.kind)) is not None]
            if fl is None or sc is None:
                # point at the field the value does match, so the writer can fix the tag in one round
                alt = sorted(((g, k) for g in fields if (k := match_scale(val, dec, g.value, g.kind)) is not None),
                             key=lambda c: abs(abs(val) - abs(c[0].value * c[1])))
                hint = f" (the value matches {alt[0][0].path})" if alt else ""
                eid = re.search(r"E\d+", cite).group(0)
                if fl is not None:
                    problems.append(Problem.of("tag_mismatch", hint, tok=tok, cite=cite, value=fl.value))
                elif eid not in f.evidence:
                    problems.append(Problem.of("tag_outside_evidence", hint, tok=tok, cite=cite, eid=eid))
                else:
                    problems.append(Problem.of("tag_not_numeric", hint, tok=tok, cite=cite))
        else:
            cands = [(fl, sc) for fl in fields if (sc := match_scale(val, dec, fl.value, fl.kind)) is not None]
            # closest value first, so a message names the field the writer most likely meant
            cands.sort(key=lambda c: abs(abs(val) - abs(c[0].value * c[1])))
        sem = check_number(text, m.start(), m.end(), cands) if cands else None
        best = None
        if cands:
            path = sem["field"] if sem else None
            best = next(((fl, sc) for fl, sc in cands if fl.path == path), cands[0])
        in_title = m.start() < stmt_at
        at = 0 if in_title else stmt_at
        n = NumberCheck(text=tok, value=val, start=m.start() - at, end=m.end() - at, grounded=bool(cands),
                        cited=cite, matched=None if best is None else FieldMatch(best[0].value, best[1], best[0].path),
                        consistent=bool(cands) and sem["ok"], unit=sem["unit"] if sem else None,
                        role=sem["role"] if sem else None, semantic_problem=sem["problem"] if sem else None)
        (title_numbers if in_title else statement_numbers).append(n)
    for key in ("t_start", "t_end"):
        v = getattr(f, key)
        if v is not None:
            cands = [fl for fl in fields if match_scale(v, 2, fl.value, fl.kind) is not None]
            if not cands:
                problems.append(Problem.of("time_not_found", key=key, value=v))
            elif not any(fl.kind == "time" for fl in cands):
                problems.append(Problem.of("time_not_a_time", key=key, value=v, path=cands[0].path))
    if f.channel and cited and f.channel not in text_blob:
        problems.append(Problem.of("channel_not_in_evidence", channel=f.channel))
    if (cp := category_problem(f, cited)) is not None:
        problems.append(cp)

    numbers = title_numbers + statement_numbers
    if not cited:
        status = "unsupported"
    elif problems or any(not n.grounded or not n.consistent for n in numbers):
        status = "partial"
    else:
        status = "verified"
    return Verification(status, tuple(title_numbers), tuple(statement_numbers), tuple(problems))


def verify_findings(findings: list[Finding], s: EvidenceView) -> dict:
    """Verify every finding (each gets its .verification) and count the verdicts."""
    for f in findings:
        f.verification = verify_finding(f, s)
    vs = [f.verification for f in findings]
    counts = {k: 0 for k in ("verified", "partial", "unsupported")}
    for v in vs:
        counts[v.status] += 1
    n_num = sum(len(v.numbers) for v in vs)
    n_ok = sum(n.grounded for v in vs for n in v.numbers)
    n_con = sum(n.consistent for v in vs for n in v.numbers)
    return {**counts, "n_findings": len(findings), "numbers_total": n_num, "numbers_grounded": n_ok,
            "numbers_consistent": n_con, "numbers_cited": sum(n.cited is not None for v in vs for n in v.numbers),
            "grounding_rate": (n_ok / n_num) if n_num else 1.0}
