"""Semantic checks for numbers in a claim: does the evidence field a number came from *mean* what the
sentence says it means?

The grounding check in :mod:`findings` only asks "does this value occur somewhere in the cited evidence".
That passes a real value put in the wrong place (a duration reported as a peak, a time written as a
frequency) and occasionally a made-up value that happens to sit within rounding of an unrelated number.
Here every numeric leaf of the evidence keeps its field name and, where the tool says so, its unit, and
each number in the text is read together with

* the unit written right after it (``s``, ``ms``, ``Hz``, ``%``, ``bar``, ``N·s``, ``个`` ...), and
* the nearest role word before it in the same clause (峰值/peak, 持续/duration, 平均/mean, 积分/总冲/impulse,
  频率/frequency, 偏差/deviation, 延迟/latency).

A number passes when at least one evidence field has the right value *and* a compatible unit *and*, if a
role word is present, a name that fits the role. Everything is plain string and dictionary matching:
deterministic, explainable, and cheap enough to run on every draft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvField:
    value: float
    key: str  # field name the value sits under ("peak", "duration_s", "#len(events)" for list lengths)
    path: str  # e.g. "E6.events[0].freq_hz"
    unit: str | None  # physical unit reported by the tool next to the value, if any
    kind: str  # time | freq | percent | count | physical | plain


_TIME_KEYS = {"t", "time", "t_start", "t_end", "t_peak", "t_cmd", "t_response", "t_center", "t_s"}


def _kind(key: str, unit: str | None) -> str:
    k = key.lower()
    if k.startswith("#len"):
        return "count"
    if k.endswith("_hz") or k in ("hz", "freq", "frequency"):
        return "freq"
    if k.endswith("_pct") or k.endswith("pct") or k == "percent":
        return "percent"
    if k in _TIME_KEYS or k.startswith("t_") or k.endswith("_s") or k.endswith("_ms"):
        return "time"
    if k.startswith("n_") or "count" in k or k.endswith("samples") or k in ("windows", "spikes", "gaps"):
        return "count"
    if unit:
        return "physical"
    return "plain"


def evidence_fields(obj, path: str = "", key: str = "", unit: str | None = None) -> list[EvField]:
    """Every numeric leaf with its field name, path and unit (a sibling ``unit`` / ``<key>_unit`` entry)."""
    out: list[EvField] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(EvField(float(obj), key, path, unit, _kind(key, unit)))
    elif isinstance(obj, dict):
        base = obj.get("unit") if isinstance(obj.get("unit"), str) else None
        for k, v in obj.items():
            ks = str(k)
            u = obj.get(f"{ks}_unit") if isinstance(obj.get(f"{ks}_unit"), str) else base
            # a dict keyed by channel name ({"Pc": {"unit": "bar", "max": ...}}) keeps its own unit
            out.extend(evidence_fields(v, f"{path}.{ks}" if path else ks, ks, u))
    elif isinstance(obj, (list, tuple)):
        out.append(EvField(float(len(obj)), f"#len({key})", f"len({path})", None, "count"))
        for i, v in enumerate(obj):
            out.extend(evidence_fields(v, f"{path}[{i}]", key, unit))
    return out


# ---------------------------------------------------------------------------- reading the text
_BASE_U = r"(?:MPa|kPa|Pa|bar|psi|degC|°C|℃|kg|Hz|hz|ms|sec|s|K|N|g|m|W|J|V|A|rpm)"
_UNIT_RE = re.compile(
    r"\s*(毫秒|秒|赫兹|%|％|个采样点|个|次|段|条|处|samples?|windows?|spikes?|gaps?"
    r"|" + _BASE_U + r"(?:\s?[·*/]\s?" + _BASE_U + r")*)(?![A-Za-z])"
)
_TIME_U = {"毫秒", "秒", "ms", "sec", "s"}
_FREQ_U = {"赫兹", "Hz", "hz"}
_PCT_U = {"%", "％"}
_COUNT_U = {"个采样点", "个", "次", "段", "条", "处", "sample", "samples", "window", "windows", "spike", "spikes",
            "gap", "gaps"}


def _dims(u: str) -> tuple:
    """Units as a cancelled product: "kg/s·s" -> (("kg", 1),), "N·s" -> (("N", 1), ("s", 1))."""
    u = {"°C": "degC", "℃": "degC"}.get(u.strip(), u.strip())
    powers: dict[str, int] = {}
    sign = 1
    for tok in re.findall(r"[·*/]|[^·*/\s]+", u):
        if tok in "·*":
            sign = 1
        elif tok == "/":
            sign = -1
        else:
            powers[tok] = powers.get(tok, 0) + sign
            sign = 1
    return tuple(sorted((k, v) for k, v in powers.items() if v))


def _norm_unit(u: str | None) -> tuple | None:
    return None if u is None else _dims(u)


def unit_after(text: str, end: int) -> str | None:
    m = _UNIT_RE.match(text, end)
    return m.group(1) if m else None


def unit_kind(u: str | None) -> str | None:
    if u is None:
        return None
    if u in _TIME_U:
        return "time"
    if u in _FREQ_U:
        return "freq"
    if u in _PCT_U:
        return "percent"
    if u in _COUNT_U:
        return "count"
    return "physical"


ROLES: dict[str, tuple[tuple[str, ...], str]] = {
    "peak": (("峰值", "最大值", "最大", "最高", "peak", "maximum", "max"), r"peak|max"),
    "duration": (("持续时间", "持续", "时长", "历时", "工作时间", "lasting", "duration", "lasted"),
                 r"duration|action_time|total_s|longest|elapsed|dur"),
    "mean": (("平均值", "平均", "均值", "mean", "average", "averages"), r"mean|average|level|median"),
    "integral": (("总冲量", "总冲", "冲量", "积分", "impulse", "integral", "integrates to"), r"integral|impulse"),
    "freq": (("频率", "frequency"), r"freq|hz"),
    "deviation": (("偏差", "偏离", "deviation", "deviates"), r"dev"),
    "latency": (("延迟", "滞后", "latency", "delay"), r"latenc|lat_|delay"),
}
_GAP_FILLER = re.compile(
    r"\s+|[:：=≈~]|达到|为|约|了|达|近|是|值|"
    r"\b(?:of|about|approximately|approx\.?|around|is|was|at|reached|reaching|reaches|to|by|a|an|the)\b",
    re.IGNORECASE,
)
_CLAUSE_END = set("，,;；。\n（(）)")
_RANGE_SEP = re.compile(r"^\s*(?:[–—~\-]|至|到|to)\s*[-+]?\d")


def role_before(text: str, start: int) -> tuple[str, str] | None:
    """Nearest role word before ``start`` in the same clause, close enough to be about this number."""
    lo = start
    while lo > 0 and text[lo - 1] not in _CLAUSE_END and start - lo < 40:
        lo -= 1
    clause = text[lo:start]
    best: tuple[int, str, str] | None = None
    for role, (words, _) in ROLES.items():
        for w in words:
            i = clause.lower().rfind(w.lower())
            if i < 0:
                continue
            if w.isascii() and w.isalpha():  # whole English words only ("max" not in "maxima_hz")
                a, b = i - 1, i + len(w)
                if (a >= 0 and clause[a].isalpha()) or (b < len(clause) and clause[b].isalpha()):
                    continue
            end = i + len(w)
            if best is None or end > best[0]:
                best = (end, role, w)
    if best is None:
        return None
    # the role word must lead straight into the number ("峰值为 5.2", "peak of about 5.2"); anything else in
    # between ("持续超出红线 700 K" = *continuously* above the 700 K line, "峰值的 10%" = 10 % *of* the peak)
    # means the word is not describing this number
    gap = _GAP_FILLER.sub("", clause[best[0]:])
    if gap:
        return None
    return best[1], best[2]


_NUM_TAIL = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RANGE_END_RE = re.compile(r"\s*(?:[–—~\-]|至|到|to)\s*" + _NUM_TAIL)
_RANGE_START_RE = re.compile(r"(?<![\d.])([-+−]?\d+(?:\.\d+)?)\s*(" + _BASE_U + r"|毫秒|秒)?\s*(?:[–—~\-]|至|到|\bto)\s*$")


def range_end_unit(text: str, end: int) -> str | None:
    """Unit written after the other end of a range that starts here: "3.0–3.7 s" gives 3.0 the unit s."""
    m = _RANGE_END_RE.match(text, end)
    return unit_after(text, m.end()) if m else None


def range_start_value(text: str, start: int) -> float | None:
    """Value of the number that opens a range ending at ``start`` ("2.6–" before "3.7 s"), if any."""
    m = _RANGE_START_RE.search(text[:start])
    return float(m.group(1).replace("−", "-")) if m else None


def _time_scale(tu: str | None, f: EvField) -> float:
    """The scale that turns field ``f`` into the time unit written in the text (s <-> ms)."""
    text_ms = tu in ("ms", "毫秒")
    field_ms = f.key.lower().endswith("_ms") or (f.unit or "").strip() == "ms"
    return 1.0 if text_ms == field_ms else (1000.0 if text_ms else 0.001)


def is_range_endpoint(text: str, start: int, end: int) -> bool:
    after = text[end:]
    u = _UNIT_RE.match(after)
    if u:
        after = after[u.end():]
    if _RANGE_SEP.match(after):
        return True
    before = text[:start].rstrip()
    return bool(re.search(r"(?:[–—~]|至|到|\bto|\d\s*-)$", before))


# ---------------------------------------------------------------------------- the check
def _unit_ok(tk: str | None, tu: str | None, f: EvField, scale: float) -> bool:
    if tk is None:
        return True
    if tk == "time":  # and in the unit written: a field in seconds read as "0.8 ms" is a different number
        return f.kind == "time" and scale == _time_scale(tu, f)
    if tk == "freq":
        return f.kind == "freq"
    if tk == "percent":
        return f.kind == "percent" or (scale == 100.0 and f.kind == "plain")
    if tk == "count":  # "3 个" / "3 spikes" must come from a count or a list length, not any field equal to 3
        return f.kind == "count"
    # physical unit written in the text
    if f.kind == "plain":
        return True
    if f.kind != "physical":
        return False
    return _norm_unit(f.unit) == _norm_unit(tu)


def check_number(text: str, start: int, end: int, candidates: list[tuple[EvField, float]]) -> dict:
    """Semantic verdict for one number, given the evidence fields its value matches.

    Returns {"ok": bool, "unit": ..., "role": ..., "field": path of the accepted/closest field, "problem": str|None}.
    """
    tu = unit_after(text, end) or range_end_unit(text, end)
    tk = unit_kind(tu)
    rb = role_before(text, start)
    role = None if rb is None or is_range_endpoint(text, start, end) else rb
    by_unit = [(f, sc) for f, sc in candidates if _unit_ok(tk, tu, f, sc)]
    tok = text[start:end]
    if tk == "time":  # a time range cannot end before it starts
        lo = range_start_value(text, start)
        if lo is not None and float(tok.replace("−", "-")) < lo:
            f = (by_unit or candidates)[0][0]
            return {"ok": False, "unit": tu, "role": rb and rb[0], "field": f.path,
                    "problem": f"time range ends at {tok} {tu}, before it starts ({lo:g})"}
    if candidates and not by_unit:
        f = candidates[0][0]
        what = f.kind if f.kind != "physical" else f"unit {f.unit}"
        return {"ok": False, "unit": tu, "role": rb and rb[0], "field": f.path,
                "problem": f"'{tok} {tu}' matches only {f.path} ({what})"}
    pool = by_unit or candidates
    if role and pool:
        rx = ROLES[role[0]][1]
        fitting = [(f, sc) for f, sc in pool if re.search(rx, f.key.lower())]
        if not fitting:
            f = pool[0][0]
            return {"ok": False, "unit": tu, "role": role[0], "field": f.path,
                    "problem": f"'{role[1]} {tok}' is read as {role[0]}, but the value comes from {f.path}"}
        pool = fitting
    return {"ok": True, "unit": tu, "role": role and role[0], "field": pool[0][0].path if pool else None,
            "problem": None}
