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
        cat_ok = any(claims[i].category == a.category for i in hits)
        loc_err = None
        primary = [claims[i] for i in hits if claims[i].category == a.category and claims[i].t_start is not None]
        if primary:
            loc_err = min(abs(f.t_start - a.t_start) for f in primary)
        per_truth.append({"type": a.type, "channel": a.channel, "detected": bool(hits), "category_ok": cat_ok,
                          "t_start_error_s": loc_err})
    fps = [claims[i] for i in range(len(claims)) if i not in used]
    return {"truth": per_truth, "n_claims": len(claims), "false_positives": [
        {"category": f.category, "channel": f.channel, "t_start": f.t_start, "title": f.title} for f in fps]}


def run_benchmark(make_agent, n: int = 30, seed: int = 0, tol_s: float = 0.25, progress=None) -> dict:
    rows = []
    for i in range(n):
        run = generate_run(seed + i)
        s = Session(run.data, run.meta, run.reference, run.limits)
        t0 = time.perf_counter()
        res = make_agent().run(s)
        m = match(res.findings, run.truth, tol_s)
        rows.append({
            "seed": seed + i,
            "anomalies": [a.type for a in run.truth],
            "match": m,
            "verification": res.verification,
            "elapsed_s": time.perf_counter() - t0,
            "n_evidence": len(s.ledger),
        })
        if progress:
            progress(i + 1, n)
    return {"n_runs": n, "seed": seed, "tol_s": tol_s, "summary": summarize(rows), "runs": rows}


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "detected": 0, "category_ok": 0, "loc_err": []})
    n_claims = n_fp = 0
    verified = findings = num_total = num_ok = 0
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
    n_truth = sum(d["n"] for d in by_type.values())
    n_det = sum(d["detected"] for d in by_type.values())
    n_nominal = sum(1 for r in rows if not r["anomalies"])
    fp_nominal = sum(len(r["match"]["false_positives"]) for r in rows if not r["anomalies"])
    return {
        "recall": n_det / n_truth if n_truth else None,
        "precision": (n_claims - n_fp) / n_claims if n_claims else None,
        "category_accuracy": sum(d["category_ok"] for d in by_type.values()) / n_truth if n_truth else None,
        "false_positives": n_fp,
        "nominal_runs": n_nominal,
        "false_positives_on_nominal_runs": fp_nominal,
        "claims_verified": f"{verified}/{findings}",
        "numbers_grounded": f"{num_ok}/{num_total}",
        "mean_elapsed_s": sum(r["elapsed_s"] for r in rows) / len(rows) if rows else None,
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
