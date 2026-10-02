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

The LLM can phrase things however it likes, but it cannot introduce a number
that no tool produced without the report flagging it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .semantics import EvField, check_number, evidence_fields
from .session import Session

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
    verification: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        keys = {f for f in cls.__dataclass_fields__}
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
        return asdict(self)


# numbers not glued to an identifier (skips E3, P2, x1e3 ...)
_NUM_RE = re.compile(r"(?<![A-Za-z_\d.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SCALES = (1.0, 1000.0, 0.001, 100.0, 0.01, 60.0)


def number_matches(text: str) -> list[re.Match]:
    """Positions of the numbers a claim states (identifiers such as SYN-1013 or run_42 are skipped)."""
    text = text or ""
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


def _matches(value: float, decimals: int, v: float) -> float | None:
    """The scale (s<->ms, fraction<->% ...) at which evidence value ``v`` rounds to ``value``, if any."""
    a = abs(value)
    for sc in _SCALES:
        x = abs(v * sc)
        if abs(a - x) <= max(0.5 * 10 ** (-decimals), 0.01 * x, 1e-9):
            return sc
    return None


def _flagged_channels(ev) -> list[str] | None:
    """Channels on which this evidence entry reports an anomaly, or None if its tool reports none of the
    kind (the list may be empty: the tool ran and found nothing)."""
    r = ev.result
    if ev.tool == "check_redlines":
        return [v.get("channel") for v in r.get("violations") or []]
    if ev.tool == "detect_oscillation":
        return [r.get("channel")] if r.get("detected") else []
    if ev.tool == "check_sensor_health":
        return [i.get("channel") for i in r.get("issues") or []]
    if ev.tool == "measure_valve_response":
        return [c for e in r.get("events") or [] if e.get("exceeds_limit") for c in (e.get("response"), e.get("command"))]
    if ev.tool == "compare_reference":
        return [r.get("channel")] if r.get("within_tolerance") is False or r.get("sustained_deviation_intervals") else []
    return None


# the tool whose result must report the anomaly a category claims
_CATEGORY_TOOL = {
    "redline_violation": "check_redlines",
    "combustion_oscillation": "detect_oscillation",
    "sensor_fault": "check_sensor_health",
    "valve_response": "measure_valve_response",
    "performance_deviation": "compare_reference",
}


def category_problem(f: Finding, cited: list) -> str | None:
    """Why the cited evidence does not support the anomaly the finding's category claims, if it does not."""
    tool = _CATEGORY_TOOL.get(f.category)
    if tool is None or not cited:
        return None
    flagged = [c for ev in cited if ev.tool == tool for c in (_flagged_channels(ev) or [])]
    if not any(ev.tool == tool for ev in cited):
        return f"category {f.category!r} needs cited {tool} evidence that reports it; none is cited"
    if not flagged:
        return (f"category {f.category!r}: the cited {tool} evidence reports no anomaly "
                "(a check that passed is category 'observation')")
    if f.channel and f.channel not in flagged:
        return f"category {f.category!r}: the cited {tool} evidence reports it for {sorted(set(flagged))}, not {f.channel!r}"
    return None


def verify_finding(f: Finding, s: Session) -> dict:
    problems: list[str] = []
    cited = [s.evidence(e) for e in f.evidence]
    missing = [e for e, ev in zip(f.evidence, cited) if ev is None]
    cited = [ev for ev in cited if ev is not None]
    if missing:
        problems.append(f"unknown evidence id(s): {', '.join(missing)}")
    if f.category not in CATEGORIES:
        problems.append(f"unknown category {f.category!r}")
    if f.channel and f.channel not in s.channels:
        problems.append(f"channel {f.channel!r} not in data")

    fields: list[EvField] = []
    text_blob = ""
    for ev in cited:
        fields += evidence_fields(ev.result, ev.id)
        fields += evidence_fields(ev.params, f"{ev.id}.params")
        text_blob += repr(ev.params) + repr(ev.result)

    numbers = []
    text = f"{f.title}\n{f.statement}"
    for m in number_matches(text):
        tok = m.group(0)
        parsed = _parse_number(tok)
        if parsed is None:
            continue
        val, dec = parsed
        cands = [(fl, sc) for fl in fields if (sc := _matches(val, dec, fl.value)) is not None]
        # closest value first, so a message names the field the writer most likely meant
        cands.sort(key=lambda c: abs(abs(val) - abs(c[0].value * c[1])))
        sem = check_number(text, m.start(), m.end(), cands) if cands else None
        best = None
        if cands:
            path = sem["field"] if sem else None
            best = next(((fl, sc) for fl, sc in cands if fl.path == path), cands[0])
        numbers.append({"text": tok, "value": val, "grounded": bool(cands),
                        "matched": None if best is None else {"value": best[0].value, "scale": best[1],
                                                               "field": best[0].path},
                        "consistent": bool(cands) and sem["ok"],
                        "unit": sem["unit"] if sem else None, "role": sem["role"] if sem else None,
                        "semantic_problem": sem["problem"] if sem else None})
    for key in ("t_start", "t_end"):
        v = getattr(f, key)
        if v is not None:
            cands = [fl for fl in fields if _matches(v, 2, fl.value) is not None]
            if not cands:
                problems.append(f"{key}={v} not found in cited evidence")
            elif not any(fl.kind == "time" for fl in cands):
                problems.append(f"{key}={v} matches only non-time fields ({cands[0].path})")
    if f.channel and cited and f.channel not in text_blob:
        problems.append(f"channel {f.channel!r} does not appear in the cited evidence")
    if (cp := category_problem(f, cited)) is not None:
        problems.append(cp)

    ungrounded = [n["text"] for n in numbers if not n["grounded"]]
    mismatched = [n["text"] for n in numbers if n["grounded"] and not n["consistent"]]
    if not cited:
        status = "unsupported"
    elif ungrounded or mismatched or problems:
        status = "partial"
    else:
        status = "verified"
    return {
        "status": status,
        "n_numbers": len(numbers),
        "n_grounded": sum(n["grounded"] for n in numbers),
        "ungrounded_numbers": ungrounded,
        "n_consistent": sum(n["consistent"] for n in numbers),
        "mismatched_numbers": mismatched,
        "semantic_problems": [n["semantic_problem"] for n in numbers if n["semantic_problem"]],
        "numbers": numbers,
        "problems": problems,
    }


def verify_findings(findings: list[Finding], s: Session) -> dict:
    for f in findings:
        f.verification = verify_finding(f, s)
    counts = {k: 0 for k in ("verified", "partial", "unsupported")}
    for f in findings:
        counts[f.verification["status"]] += 1
    n_num = sum(f.verification["n_numbers"] for f in findings)
    n_ok = sum(f.verification["n_grounded"] for f in findings)
    n_con = sum(f.verification["n_consistent"] for f in findings)
    return {**counts, "n_findings": len(findings), "numbers_total": n_num, "numbers_grounded": n_ok,
            "numbers_consistent": n_con,
            "grounding_rate": (n_ok / n_num) if n_num else 1.0}
