"""Benchmark the verifier itself: plant known errors in correct findings and count how many it catches.

Correct findings come from the rule agent on synthetic runs (every number verified). Each number in a
statement is then corrupted in one of three ways, one error per mutant:

* ``fabricated``  – the value is changed by -30 % … +50 % (a made-up number of the right size);
* ``swapped``     – the value is replaced by a *different* field of the same cited evidence (a real number
  in the wrong place, e.g. a duration written where the peak should be);
* ``wrong_unit``  – the unit written after the number is replaced by one of another kind (s → Hz, % → s ...).

For every mutant we record whether the grounding-only check (the verifier before semantic checks: is the
value anywhere in the cited evidence?) and the full verifier flag it. Unmodified findings measure false
alarms.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict

from .findings import Finding, number_matches, verify_finding
from .semantics import _UNIT_RE, evidence_fields, unit_kind

_OTHER_UNIT = {"time": "Hz", "freq": "s", "percent": "s", "count": "s", "physical": "s"}


def _grounding_only(v: dict) -> bool:
    """Verdict of the verifier without semantic checks: True = flagged."""
    return bool(v["ungrounded_numbers"]) or any("not found" in p or "unknown" in p or "not in data" in p
                                                  or "does not appear" in p for p in v["problems"]) \
        or v["status"] == "unsupported"


def _mutate(f: Finding, s, rng: random.Random) -> list[tuple[str, Finding, str]]:
    """All single-error mutants of one finding's statement: (kind, mutant, description)."""
    out = []
    text = f.statement
    fields = []
    for eid in f.evidence:
        ev = s.evidence(eid)
        if ev is not None:
            fields += evidence_fields(ev.result, ev.id)
    for m in number_matches(text):
        tok = m.group(0)
        try:
            val = float(tok.replace("−", "-"))
        except ValueError:
            continue
        dec = len(tok.split(".")[1]) if "." in tok else 0
        fmt = lambda x: f"{x:.{dec}f}"  # noqa: E731

        def put(new_tok, a=m.start(), b=m.end()):
            g = copy.copy(f)
            g.statement = text[:a] + new_tok + text[b:]
            g.verification = None
            return g

        # fabricated: same order of magnitude, clearly different value
        if abs(val) >= 10 ** -dec * 5:
            factor = rng.choice([0.7, 0.8, 1.25, 1.5])
            new = fmt(val * factor)
            if new != tok:
                out.append(("fabricated", put(new), f"{tok} -> {new}"))
        # swapped: another field of the cited evidence, clearly different value
        others = [fl for fl in fields if abs(fl.value) > 10 ** -dec and abs(fl.value - val) > 0.05 * max(abs(val), 1e-9)
                  and fmt(fl.value) not in (tok, fmt(0))]
        if others:
            fl = rng.choice(others)
            out.append(("swapped", put(fmt(fl.value)), f"{tok} -> {fmt(fl.value)} ({fl.path})"))
        # wrong unit
        um = _UNIT_RE.match(text, m.end())
        if um:
            k = unit_kind(um.group(1))
            repl = _OTHER_UNIT.get(k or "", None)
            if repl:
                a, b = um.start(1), um.end(1)
                g = copy.copy(f)
                g.statement = text[:a] + repl + text[b:]
                g.verification = None
                out.append(("wrong_unit", g, f"{tok} {um.group(1)} -> {tok} {repl}"))
    return out


def run_verifier_bench(n: int = 50, seed: int = 1000, langs=("zh", "en"), rng_seed: int = 0,
                       examples: int = 3) -> dict:
    from .agent import RuleAgent
    from .session import Session
    from .synth import generate_run

    rng = random.Random(rng_seed)
    stats = defaultdict(lambda: {"n": 0, "caught_grounding": 0, "caught_full": 0})
    samples = defaultdict(list)
    clean = {"findings": 0, "flagged_grounding": 0, "flagged_full": 0}
    for lang in langs:
        for i in range(n):
            run = generate_run(seed + i)
            s = Session(run.data, run.meta, run.reference, run.limits)
            for f in RuleAgent(lang).run(s).findings:
                v = verify_finding(f, s)
                clean["findings"] += 1
                clean["flagged_grounding"] += _grounding_only(v)
                clean["flagged_full"] += v["status"] != "verified"
                for kind, g, desc in _mutate(f, s, rng):
                    vg = verify_finding(g, s)
                    st = stats[kind]
                    st["n"] += 1
                    old, new = _grounding_only(vg), vg["status"] != "verified"
                    st["caught_grounding"] += old
                    st["caught_full"] += new
                    if new and not old and len(samples[kind]) < examples:
                        samples[kind].append({"change": desc, "statement": g.statement,
                                              "why": (vg["semantic_problems"] or vg["problems"] or ["?"])[0]})
                    if not new and len(samples[kind + "_missed"]) < examples:
                        samples[kind + "_missed"].append({"change": desc, "statement": g.statement})
    return {"n_runs": n, "seed": seed, "langs": list(langs), "clean": clean, "mutations": dict(stats),
            "examples": dict(samples)}


def format_bench(res: dict) -> str:
    def pct(a, b):
        return f"{a}/{b}（{100 * a / b:.0f}%）" if b else "–"

    c = res["clean"]
    names = {"fabricated": "编造的数值", "swapped": "真实数值放错位置", "wrong_unit": "单位写错"}
    lines = [
        "| 植入的错误 | 数量 | 只查出处（原校验器） | 出处 + 语义（新校验器） |",
        "|---|---|---|---|",
    ]
    for k in ("fabricated", "swapped", "wrong_unit"):
        st = res["mutations"].get(k)
        if st:
            lines.append(f"| {names[k]} | {st['n']} | {pct(st['caught_grounding'], st['n'])} | "
                         f"{pct(st['caught_full'], st['n'])} |")
    lines.append(f"| 未改动的正确结论（误报） | {c['findings']} | {pct(c['flagged_grounding'], c['findings'])} | "
                 f"{pct(c['flagged_full'], c['findings'])} |")
    return "\n".join(lines)
