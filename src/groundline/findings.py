"""Findings and the claim verifier.

A :class:`Finding` is a claim in the report.  It must cite evidence IDs from the
session ledger.  :func:`verify_findings` then checks, mechanically:

1. every cited evidence ID exists;
2. every number written in the title/statement can be found in the cited
   evidence results (allowing rounding and unit scaling s<->ms, fraction<->%);
3. the finding's structured fields (channel, t_start, t_end) are consistent
   with the data and the evidence.

The LLM can phrase things however it likes, but it cannot introduce a number
that no tool produced without the report flagging it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

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


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    out = []
    text = text or ""
    for m in _NUM_RE.finditer(text):
        i = m.start()
        if i >= 2 and text[i - 1] in "-_" and text[i - 2].isalpha():
            continue  # part of an identifier such as SYN-1013 or run_42, not a measured value
        tok = m.group(0).replace("−", "-")
        try:
            val = float(tok)
        except ValueError:
            continue
        mant = tok.lower().split("e")[0]
        dec = len(mant.split(".")[1]) if "." in mant else 0
        out.append((m.group(0), val, dec))
    return out


def evidence_numbers(obj) -> list[float]:
    """All numeric leaves of an evidence result, plus list lengths (counts)."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(evidence_numbers(v))
    elif isinstance(obj, (list, tuple)):
        out.append(float(len(obj)))
        for v in obj:
            out.extend(evidence_numbers(v))
    return out


def ground_number(value: float, decimals: int, pool: list[float]) -> tuple[float, float] | None:
    """Return (evidence_value, scale) if ``value`` matches some pool number after rounding/scaling."""
    a = abs(value)
    for v in pool:
        for sc in _SCALES:
            x = abs(v * sc)
            tol = max(0.5 * 10 ** (-decimals), 0.01 * x, 1e-9)
            if abs(a - x) <= tol:
                return v, sc
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

    pool: list[float] = []
    text_blob = ""
    for ev in cited:
        pool.extend(evidence_numbers(ev.result))
        pool.extend(evidence_numbers(ev.params))
        text_blob += repr(ev.params) + repr(ev.result)

    numbers = []
    for tok, val, dec in extract_numbers(f"{f.title}\n{f.statement}"):
        g = ground_number(val, dec, pool)
        numbers.append({"text": tok, "value": val, "grounded": g is not None,
                        "matched": None if g is None else {"value": g[0], "scale": g[1]}})
    for key in ("t_start", "t_end"):
        v = getattr(f, key)
        if v is not None:
            g = ground_number(v, 2, pool)
            if g is None:
                problems.append(f"{key}={v} not found in cited evidence")
    if f.channel and cited and f.channel not in text_blob:
        problems.append(f"channel {f.channel!r} does not appear in the cited evidence")

    ungrounded = [n["text"] for n in numbers if not n["grounded"]]
    if not cited:
        status = "unsupported"
    elif ungrounded or problems:
        status = "partial"
    else:
        status = "verified"
    return {
        "status": status,
        "n_numbers": len(numbers),
        "n_grounded": sum(n["grounded"] for n in numbers),
        "ungrounded_numbers": ungrounded,
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
    return {**counts, "n_findings": len(findings), "numbers_total": n_num, "numbers_grounded": n_ok,
            "grounding_rate": (n_ok / n_num) if n_num else 1.0}
