"""Benchmark an agent against synthetic runs with known injected anomalies.

For every generated run we compare the agent's findings with the ground truth:

* a truth anomaly is **detected** if some non-observation finding names its channel
  (or a related channel) and overlaps its time window (± ``tol_s``);
* it is **correctly classified** if one of those findings has the expected category;
* a finding that matches no truth anomaly is a **false positive**.

We also report how many claims passed the verifier and how many written numbers
were traceable to evidence.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .findings import Finding
from .session import Session
from .synth import ANOMALY_TYPES, Anomaly, generate_run


def _overlaps(f: Finding, a: Anomaly, tol_s: float) -> bool:
    if f.t_start is None:
        return True
    t1 = f.t_end if f.t_end is not None else f.t_start
    return f.t_start <= a.t_end + tol_s and a.t_start - tol_s <= t1


def match(findings: list[Finding], truth: list[Anomaly], tol_s: float = 0.25) -> dict:
    claims = [f for f in findings if f.category != "observation"]
    used = set()
    per_truth = []
    for a in truth:
        chans = {a.channel, *a.related_channels}
        hits = [i for i, f in enumerate(claims) if f.channel in chans and _overlaps(f, a, tol_s)]
        used.update(hits)
        # loose: a claim that names no channel but has the right category (and a compatible time) —
        # the model found the problem but did not fill in the structured field
        loose = bool(hits) or any(not claims[i].channel and claims[i].category == a.category
                                  and _overlaps(claims[i], a, tol_s) for i in range(len(claims)))
        cat_ok = any(claims[i].category == a.category for i in hits)
        loc_err = None
        primary = [claims[i] for i in hits if claims[i].category == a.category and claims[i].t_start is not None]
        if primary:
            loc_err = min(abs(f.t_start - a.t_start) for f in primary)
        per_truth.append({"type": a.type, "channel": a.channel, "detected": bool(hits), "detected_loose": loose,
                          "category_ok": cat_ok,
                          "t_start_error_s": loc_err})
    fps = [claims[i] for i in range(len(claims)) if i not in used]
    return {"truth": per_truth, "n_claims": len(claims), "false_positives": [
        {"category": f.category, "channel": f.channel, "t_start": f.t_start, "title": f.title} for f in fps]}


def run_benchmark(make_agent, n: int = 30, seed: int = 0, tol_s: float = 0.25, progress=None,
                  keep_going: bool = False) -> dict:
    """Run ``n`` synthetic tests. A failure on the very first run is raised (usually a bad key or URL)
    unless ``keep_going`` is set, in which case every failure is recorded and the benchmark continues."""
    rows = []
    for i in range(n):
        run = generate_run(seed + i)
        s = Session(run.data, run.meta, run.reference, run.limits)
        t0 = time.perf_counter()
        try:
            res = make_agent().run(s)
        except Exception as e:  # one failed run (API error, timeout) should not lose the whole benchmark
            rows.append({"seed": seed + i, "anomalies": [a.type for a in run.truth],
                         "error": f"{type(e).__name__}: {e}"[:500]})
            if progress:
                progress(i + 1, n)
            if i == 0 and not keep_going:
                raise
            continue
        m = match(res.findings, run.truth, tol_s)
        rows.append({
            "seed": seed + i,
            "anomalies": [a.type for a in run.truth],
            "match": m,
            "verification": res.verification,
            "first_submission": res.agent.get("first_submission"),
            "submitted": res.agent.get("submitted", True),
            "fix_rounds_used": res.agent.get("fix_rounds_used", 0),
            "usage": res.agent.get("usage"),
            "elapsed_s": time.perf_counter() - t0,
            "n_evidence": len(s.ledger),
            # enough to audit what the verifier flagged without re-running the model
            "findings": [{"title": f.title, "statement": f.statement, "category": f.category, "channel": f.channel,
                          "evidence": f.evidence, "status": f.verification.get("status"),
                          "ungrounded_numbers": f.verification.get("ungrounded_numbers"),
                          "semantic_problems": f.verification.get("semantic_problems"),
                          "problems": f.verification.get("problems")} for f in res.findings],
            "first_draft_flagged": res.agent.get("first_draft_flagged"),
        })
        if res.agent.get("type") == "llm":
            # keep what is needed to re-verify this run later with an improved verifier (groundline reverify)
            rows[-1]["findings_full"] = [{k: v for k, v in f.to_dict().items() if k != "verification"}
                                         for f in res.findings]
            rows[-1]["first_draft_full"] = res.agent.get("first_draft_findings")
            rows[-1]["ledger"] = [{"id": e.id, "tool": e.tool, "params": e.params, "result": e.result}
                                  for e in s.ledger]
        if progress:
            progress(i + 1, n)
    return {"n_runs": n, "seed": seed, "tol_s": tol_s, "summary": summarize(rows), "runs": rows}


def summarize(rows: list[dict]) -> dict:
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "detected": 0, "category_ok": 0, "loc_err": []})
    n_claims = n_fp = 0
    verified = findings = num_total = num_ok = num_con = 0
    for r in rows:
        for t in r["match"]["truth"]:
            d = by_type[t["type"]]
            d["n"] += 1
            d["detected"] += t["detected"]
            d["category_ok"] += t["category_ok"]
            if t["t_start_error_s"] is not None:
                d["loc_err"].append(t["t_start_error_s"])
        n_claims += r["match"]["n_claims"]
        n_fp += len(r["match"]["false_positives"])
        v = r["verification"]
        verified += v["verified"]
        findings += v["n_findings"]
        num_total += v["numbers_total"]
        num_ok += v["numbers_grounded"]
        num_con += v.get("numbers_consistent", v["numbers_grounded"])
    per_type = {}
    for k in ANOMALY_TYPES:
        d = by_type.get(k)
        if not d or not d["n"]:
            continue
        le = sorted(d["loc_err"])
        per_type[k] = {
            "n": d["n"],
            "recall": d["detected"] / d["n"],
            "category_accuracy": d["category_ok"] / d["n"],
            "median_t_start_error_s": le[len(le) // 2] if le else None,
        }
    firsts = [r["first_submission"] for r in rows if r.get("first_submission")]
    first = None
    if firsts:
        ft = sum(f["numbers_total"] for f in firsts)
        fg = sum(f["numbers_grounded"] for f in firsts)
        first = {"claims_verified": f"{sum(f['verified'] for f in firsts)}/{sum(f['n_findings'] for f in firsts)}",
                 "numbers_grounded": f"{fg}/{ft}", "ungrounded_numbers_caught": ft - fg}
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    for r in rows:
        for k in usage:
            usage[k] += (r.get("usage") or {}).get(k, 0)
    n_truth = sum(d["n"] for d in by_type.values())
    n_det = sum(d["detected"] for d in by_type.values())
    n_nominal = sum(1 for r in rows if not r["anomalies"])
    fp_nominal = sum(len(r["match"]["false_positives"]) for r in rows if not r["anomalies"])
    # a run that crashed or never submitted a report still had anomalies to find
    n_truth_all = n_truth + sum(len(r["anomalies"]) for r in errors)
    n_loose = sum(t.get("detected_loose", t["detected"]) for r in rows for t in r["match"]["truth"])
    submitted = sum(1 for r in rows if r.get("submitted", r["verification"]["n_findings"] > 0))
    fixes = sum(r.get("fix_rounds_used", 0) or 0 for r in rows)
    first_rate = None
    if firsts:
        ft = sum(f["numbers_total"] for f in firsts)
        first_rate = (ft - sum(f["numbers_grounded"] for f in firsts)) / ft if ft else 0.0
        first["semantic_mismatches"] = sum(f["numbers_grounded"] - f.get("numbers_consistent", f["numbers_grounded"])
                                           for f in firsts)
    return {
        "runs_total": len(rows) + len(errors),
        "reports_submitted": submitted,
        "recall_all_runs": n_det / n_truth_all if n_truth_all else None,
        "recall_loose_all_runs": n_loose / n_truth_all if n_truth_all else None,
        "unsupported_claims": findings - verified,
        "ungrounded_numbers": num_total - num_ok,
        "semantic_mismatches": num_ok - num_con,
        "first_draft_ungrounded_rate": first_rate,
        "fix_rounds_used": fixes,
        "recall": n_det / n_truth if n_truth else None,
        "precision": (n_claims - n_fp) / n_claims if n_claims else None,
        "category_accuracy": sum(d["category_ok"] for d in by_type.values()) / n_truth if n_truth else None,
        "false_positives": n_fp,
        "nominal_runs": n_nominal,
        "false_positives_on_nominal_runs": fp_nominal,
        "claims_verified": f"{verified}/{findings}",
        "numbers_grounded": f"{num_ok}/{num_total}",
        "mean_elapsed_s": sum(r["elapsed_s"] for r in rows) / len(rows) if rows else None,
        "first_submission": first,
        "usage": usage if usage["requests"] else None,
        "failed_runs": len(errors),
        "per_type": per_type,
    }


def format_summary(summary: dict) -> str:
    def pct(x):
        return "–" if x is None else f"{100 * x:.0f}%"

    lines = [
        f"recall {pct(summary['recall'])} · precision {pct(summary['precision'])} · "
        f"category accuracy {pct(summary['category_accuracy'])}",
        f"false positives {summary['false_positives']} (on {summary['nominal_runs']} nominal runs: "
        f"{summary['false_positives_on_nominal_runs']})",
        f"claims verified {summary['claims_verified']} · numbers grounded {summary['numbers_grounded']} · "
        f"mean time {summary['mean_elapsed_s']:.2f} s/run",
    ]
    if summary.get("first_submission"):
        f = summary["first_submission"]
        lines.append(f"first draft (before verifier feedback): claims verified {f['claims_verified']} · numbers "
                     f"grounded {f['numbers_grounded']} · invented numbers caught {f['ungrounded_numbers_caught']}")
    if summary.get("usage"):
        u = summary["usage"]
        lines.append(f"LLM usage: {u['requests']} requests · {u['input_tokens']:,} input / "
                     f"{u['output_tokens']:,} output tokens")
    if summary.get("failed_runs"):
        lines.append(f"failed runs: {summary['failed_runs']} (see JSON for errors)")
    lines += [
        "",
        "| anomaly | n | recall | category acc. | median t_start error |",
        "|---|---|---|---|---|",
    ]
    for k, d in summary["per_type"].items():
        le = "–" if d["median_t_start_error_s"] is None else f"{d['median_t_start_error_s'] * 1000:.0f} ms"
        lines.append(f"| {k} | {d['n']} | {pct(d['recall'])} | {pct(d['category_accuracy'])} | {le} |")
    return "\n".join(lines)


def save(result: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return p


class _LedgerView:
    """Just enough of a Session for the verifier: evidence lookup and channel names."""

    def __init__(self, ledger: list[dict], channels: list[str]):
        from types import SimpleNamespace

        self._ev = {e["id"]: SimpleNamespace(**e) for e in ledger}
        self.channels = channels

    def evidence(self, eid):
        return self._ev.get(eid)


def reverify(result: dict) -> dict:
    """Re-run the current verifier on stored LLM runs (findings + ledger) and refresh the summary."""
    from .findings import verify_findings
    from .synth import generate_run

    for r in result["runs"]:
        if "ledger" not in r:
            continue
        chans = [c for c in generate_run(r["seed"]).data.columns if c != "time"]
        view = _LedgerView(r["ledger"], chans)
        fs = [Finding.from_dict(d) for d in r.get("findings_full") or []]
        r["verification"] = verify_findings(fs, view)
        r["findings"] = [{"title": f.title, "statement": f.statement, "category": f.category, "channel": f.channel,
                          "evidence": f.evidence, "status": f.verification["status"],
                          "ungrounded_numbers": f.verification["ungrounded_numbers"],
                          "semantic_problems": f.verification["semantic_problems"],
                          "problems": f.verification["problems"]} for f in fs]
        if r.get("first_draft_full"):
            ff = [Finding.from_dict(d) for d in r["first_draft_full"]]
            r["first_submission"] = verify_findings(ff, view)
    result["summary"] = summarize(result["runs"])
    return result
