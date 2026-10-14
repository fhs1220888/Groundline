"""Hybrid benchmark: known anomalies injected into real test logs.

The synthetic benchmark scores tools against the generator that was written alongside them. Here the
background is a real firing (HANARO, Triton, UVic MULE-1: real noise, dropouts, offsets, start-up
transients and sensor faults), and only the anomaly is synthetic, injected at a known time and size:

* ``oscillation``  a narrow-band oscillation on the chamber pressure, swept over amplitude (% of steady level);
* ``redline``      a chamber-pressure excursion above a redline, swept over duration (against a 10 ms persistence);
* ``deviation``    the chamber pressure departing from a prediction, swept over size (against a 2 % tolerance);
* ``spike``        a one-sample spike on the thrust channel, swept over height (in local noise sigmas);
* ``dropout``      a NaN gap on the thrust channel during the firing, swept over length;
* ``stuck``        the thrust channel frozen during the firing, swept over length.

Real logs already contain genuine problems. The agent (the rule agent unless another is configured) is
therefore first run on the clean background with the same configuration (limits, prediction); a finding in an
injected run counts only if the clean run has no finding of the same category on the same channel at that time.
An injection that lands on such an existing finding is reported as ``masked`` rather than as detected or missed.
Every other new anomaly finding is a false positive.

Each background is analysed clean once per kind of injection (each kind has its own limits and prediction), up
front, and every case is compared with the findings of all of them together. An LLM agent does not give the same
answer twice, so this pools several samples of what it says about the real log. A new finding that the pool
already holds with the same category and check at that time, even on another channel, is the log's own problem
reported differently, not a false positive.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {
    "oscillation": [0.25, 0.5, 1.0, 2.0, 4.0],     # zero-to-peak amplitude, % of steady chamber pressure
    "redline": [0.005, 0.01, 0.02, 0.05, 0.2],     # time above the redline, s (persistence 0.01 s)
    "deviation": [1.0, 2.0, 3.0, 5.0, 8.0],        # % of steady chamber pressure (tolerance 2 %)
    "spike": [6.0, 12.0, 24.0, 48.0],              # spike height, local noise sigmas
    "dropout": [0.01, 0.05, 0.2, 1.0],             # gap length, s
    "stuck": [0.02, 0.05, 0.2, 1.0],               # frozen length, s
}
UNITS = {"oscillation": "% of Pc", "redline": "s above", "deviation": "% of Pc", "spike": "sigma",
         "dropout": "s", "stuck": "s"}
CATEGORY = {"oscillation": "combustion_oscillation", "redline": "redline_violation",
            "deviation": "performance_deviation", "spike": "sensor_fault", "dropout": "sensor_fault",
            "stuck": "sensor_fault"}
PERSISTENCE_S, TOLERANCE_PCT = 0.01, 2.0

ROOT = Path(__file__).resolve().parents[2]
# HANARO is not a default background: its chamber pressure was logged at 10 Hz (a 50 ms injection cannot be
# represented) and its thrust channel has dropouts across the whole record (every thrust injection is masked)
DEFAULT_BACKGROUNDS = {
    "Triton (LOX liquid)": ROOT / "examples" / "triton_lox" / "run.csv",
    "UVic MULE-1 2025-01-18 (hybrid)": ROOT / "examples" / "uvic_mule" / "2025-01-18" / "run.csv",
}


def available_backgrounds() -> dict[str, Path]:
    """Backgrounds present on this machine (Triton and UVic are downloaded by their prepare.py)."""
    return {k: p for k, p in DEFAULT_BACKGROUNDS.items() if p.exists()}


@lru_cache(maxsize=8)
def _background(path: Path):
    from .io import load_run
    from .session import Session

    df, meta = load_run(path)
    s = Session(df, meta)
    # chamber-pressure injections are sized and placed from the chamber pressure's own pulse, even when the
    # default segmentation fell back to thrust; thrust injections use the default segmentation
    seg = s.run("segment_phases").result
    seg_pc = s.run("segment_phases", channel="Pc").result if "Pc" in df.columns else {"fired": False}
    thrust = s.channels_of_kind("force")
    return df, meta, {"default": seg, "Pc": seg_pc}, (thrust[0] if thrust else None)


def _config(kind: str, df: pd.DataFrame, seg: dict, fs: float) -> tuple[dict, pd.DataFrame | None]:
    """Limits and prediction for one kind of injection, shared by the clean and the injected runs."""
    limits = {"redline_persistence_s": PERSISTENCE_S, "reference_tolerance_pct": TOLERANCE_PCT}
    ref = None
    if kind not in ("redline", "deviation"):
        return limits, ref
    pc = df["Pc"].to_numpy(dtype=float)
    if kind == "redline":  # a redline 5 % above anything the clean firing reaches
        limits["redlines"] = {"Pc": {"max": float(np.nanmax(pc)) * 1.05}}
    if kind == "deviation":  # the clean firing smoothed is a perfect prediction; the injection departs from it
        sm = pd.Series(pc).interpolate(limit_direction="both").rolling(max(int(0.2 * fs), 1), center=True,
                                                                       min_periods=1).median()
        step = max(int(fs // 50), 1)
        ref = pd.DataFrame({"time": df["time"].to_numpy()[::step], "Pc": sm.to_numpy()[::step]})
    return limits, ref


def _inject(kind: str, level: float, df: pd.DataFrame, meta: dict, seg: dict, thrust: str | None, fs: float,
            rng: np.random.Generator) -> tuple[pd.DataFrame, dict] | None:
    """A copy of the background with one anomaly, and its truth (channel, t_start, t_end). None if the
    background cannot carry this kind (no thrust channel, chamber pressure sampled too slowly)."""
    t = df["time"].to_numpy()
    seg = seg["Pc"] if kind in ("oscillation", "redline", "deviation") else seg["default"]
    if not seg.get("fired"):  # no clear chamber-pressure pulse (missing, saturated, dead) or no firing at all
        return None
    ms0, ms1, lvl = seg["mainstage_start_s"], seg["mainstage_end_s"], seg["steady_level"]
    base = seg.get("baseline", 0.0)
    out = df.copy()
    edge = lambda x, w: np.clip(x / w, 0, 1)  # noqa: E731
    if kind == "oscillation":
        native = float(meta["channels"].get("Pc", {}).get("native_rate_hz") or fs)
        f_hi = 0.4 * native
        if f_hi < 100:
            return None
        dur = min(1.0, 0.5 * (ms1 - ms0))
        t0 = float(rng.uniform(ms0 + 0.1, ms1 - dur - 0.1))
        f = float(rng.uniform(80, f_hi))
        env = edge(t - t0, 0.05) * edge(t0 + dur - t, 0.05)
        out["Pc"] = out["Pc"] + level / 100 * (lvl - base) * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
        return out, {"channel": "Pc", "t_start": t0, "t_end": t0 + dur, "freq_hz": f}
    if kind == "redline":
        lim = float(np.nanmax(df["Pc"])) * 1.05
        t0 = float(rng.uniform(ms0 + 0.1, ms1 - level - 0.1))
        above = (t >= t0) & (t < t0 + level)
        pc = out["Pc"].to_numpy(dtype=float).copy()
        pc[above] = lim * 1.03 + (pc[above] - lvl) * 0.1  # 3 % over the line, keeping a little of the noise
        out["Pc"] = pc
        return out, {"channel": "Pc", "t_start": t0, "t_end": t0 + level}
    if kind == "deviation":
        dur = 0.4 * (ms1 - ms0)
        t0 = float(rng.uniform(ms0 + 0.05 * (ms1 - ms0), ms1 - dur - 0.05 * (ms1 - ms0)))
        env = edge(t - t0, 0.1) * edge(t0 + dur - t, 0.1)
        out["Pc"] = out["Pc"] - level / 100 * (lvl - base) * env
        return out, {"channel": "Pc", "t_start": t0, "t_end": t0 + dur}
    if thrust is None:
        return None
    x = out[thrust].to_numpy(dtype=float).copy()
    if kind == "spike":
        m = (t >= ms0) & (t <= ms1) & ~np.isnan(x)
        sig = 1.4826 * float(np.nanmedian(np.abs(np.diff(x[m]) - np.nanmedian(np.diff(x[m]))))) / np.sqrt(2)
        idx = np.where(m)[0]
        i = int(rng.choice(idx[len(idx) // 10: -len(idx) // 10]))
        x[i] += level * max(sig, 1e-9)
        out[thrust] = x
        return out, {"channel": thrust, "t_start": float(t[i]), "t_end": float(t[i])}
    t0 = float(rng.uniform(ms0 + 0.1, max(ms0 + 0.1, ms1 - level - 0.1)))
    sel = (t >= t0) & (t < t0 + level)
    if kind == "dropout":
        x[sel] = np.nan
    else:  # stuck: frozen at the reading just before; DAQ dropouts already in the log stay where they are
        i0 = int(np.argmax(sel))
        x[sel & ~np.isnan(x)] = x[i0 - 1] if not np.isnan(x[i0 - 1]) else np.nanmedian(x[sel])
    out[thrust] = x
    return out, {"channel": thrust, "t_start": t0, "t_end": t0 + level}


# tools that describe the run rather than check it: an LLM often cites them first, next to the check itself
_CONTEXT_TOOLS = {"describe_data", "segment_phases", "channel_stats", "plot_window"}


def _findings(df, meta, limits, ref, agent_cfg=None, lang: str = "en") -> dict:
    """The agent's anomaly findings as plain rows, each tagged with the check tool behind its cited evidence,
    plus its covered channels and token usage. An API failure is retried once if the server was unreachable."""
    from .agent import RuleAgent, make_agent
    from .evaluate import is_infra_error
    from .session import Session

    for attempt in range(2):
        s = Session(df, meta, ref, limits)
        try:
            res = make_agent(agent_cfg, lang).run(s) if agent_cfg else RuleAgent(lang).run(s)
            break
        except Exception as e:
            if attempt or not is_infra_error(e):
                raise
            time.sleep(10)
    out = []
    for f in res.findings:
        if f.category == "observation":
            continue
        tools = [ev.tool for ev in (s.evidence(e) for e in f.evidence) if ev is not None]
        out.append({"category": f.category, "channel": f.channel,
                    "tool": next((t for t in tools if t not in _CONTEXT_TOOLS), tools[0] if tools else None),
                    "t_start": f.t_start, "t_end": f.t_end, "title": f.title})
    usage = res.agent.get("usage")
    return {"findings": out, "covered_channels": sorted(res.agent.get("covered_channels", [])),
            **({"usage": usage} if usage else {})}


def _overlap(f: dict, t0: float, t1: float, tol: float = 0.25) -> bool:
    if f["t_start"] is None:
        return True
    return f["t_start"] <= t1 + tol and t0 - tol <= (f["t_end"] if f["t_end"] is not None else f["t_start"])


def _within(a: dict, b: dict, tol: float = 0.25) -> bool:
    """``a``'s time window lies inside ``b``'s: ``a`` says nothing about a time ``b`` does not cover."""
    if a["t_start"] is None or b["t_start"] is None:
        return True
    return b["t_start"] - tol <= a["t_start"] and (a["t_end"] if a["t_end"] is not None else a["t_start"]) <= \
        (b["t_end"] if b["t_end"] is not None else b["t_start"]) + tol


def _same(a: dict, b: dict, any_channel: bool = False, within: bool = False) -> bool:
    """The same finding in the clean and the injected run: same check, category and channel, overlapping (or,
    with ``within``, inside the clean finding's window: an LLM may merge an injection with a real event nearby
    into one wider finding, which then says something new)."""
    if a["category"] != b["category"] or a["tool"] != b["tool"] or not (any_channel or a["channel"] == b["channel"]):
        return False
    return _within(a, b) if within else _overlap(a, b["t_start"] or 0.0, b["t_end"] or b["t_start"] or 1e9)


# the tool whose finding an injection should produce
TOOL = {"oscillation": "detect_oscillation", "redline": "check_redlines", "deviation": "compare_reference",
        "spike": "check_sensor_health", "dropout": "check_sensor_health", "stuck": "check_sensor_health"}


_AGENT: dict = {"cfg": None, "lang": "en"}  # the agent under test, set in each worker by _init_worker


def _init_worker(agent_cfg, lang: str) -> None:
    _AGENT.update(cfg=agent_cfg, lang=lang)


def _failed(e: Exception) -> dict:
    return {"error": f"{type(e).__name__}: {e}"[:500]}


def _clean_case(args) -> dict:
    """The agent on one untouched background, configured for one kind of injection."""
    name, path, kind = args
    try:
        df, meta, seg, _ = _background(Path(path))
        fs = float(meta.get("sample_rate_hz") or 1 / np.median(np.diff(df["time"])))
        limits, ref = _config(kind, df, seg, fs)
        out = _findings(df, meta, limits, ref, _AGENT["cfg"], _AGENT["lang"])
    except Exception as e:  # one failed agent run (API error, timeout) should not lose the whole benchmark
        out = _failed(e)
    return {"background": name, "kind": kind, **out}


def _case(args) -> dict | None:
    """The agent on one injected background (unscored; see _score). None if the injection does not fit."""
    name, path, kind, level, rep, seed = args
    df, meta, seg, thrust = _background(Path(path))
    fs = float(meta.get("sample_rate_hz") or 1 / np.median(np.diff(df["time"])))
    rng = np.random.default_rng([seed, int(level * 1000), rep, sum(map(ord, name + kind))])
    inj = _inject(kind, level, df, meta, seg, thrust, fs, rng)
    if inj is None:
        return None
    dfi, truth = inj
    limits, ref = _config(kind, df, seg, fs)
    try:
        out = _findings(dfi, meta, limits, ref, _AGENT["cfg"], _AGENT["lang"])
    except Exception as e:
        out = _failed(e)
    return {"background": name, "kind": kind, "level": level, "rep": rep, "truth": truth, **out}


def _score(row: dict, pool: list[dict]) -> dict:
    """Score one injected case against the pooled clean runs of its background."""
    if "error" in row:
        return row
    kind, truth = row["kind"], row["truth"]
    clean = [f for c in pool for f in c["findings"]]
    covered = {ch for c in pool for ch in c["covered_channels"]}
    found = [{k: v for k, v in f.items() if k != "new"} for f in row["findings"]]
    new = [not any(_same(f, b, within=True) for b in clean) for f in found]
    hit = [f for f, n in zip(found, new) if n and f["category"] == CATEGORY[kind]
           and f["channel"] in (truth["channel"], None) and _overlap(f, truth["t_start"], truth["t_end"])]
    # masked: the clean run already reports this very thing there, or (for a deviation) already flags the channel
    # as untrustworthy (offset, frozen, dead), in which case its deviations are deliberately not reported
    masked = not hit and (any(b["category"] == CATEGORY[kind] and b["channel"] == truth["channel"]
                              and b["tool"] == TOOL[kind] and _overlap(b, truth["t_start"], truth["t_end"])
                              for b in clean)
                          or (kind == "deviation" and truth["channel"] in covered))
    fps = [f for f, n in zip(found, new) if n and not any(f is h for h in hit)
           and not any(_same(f, b, any_channel=True) for b in clean)]
    return {**row, "detected": bool(hit), "masked": masked, "hit_title": hit[0]["title"] if hit else None,
            "false_positives": [{"category": f["category"], "channel": f["channel"], "title": f["title"]} for f in fps],
            "findings": [{**f, **({"new": True} if n else {})} for f, n in zip(found, new)]}


def rescore_hybrid(res: dict) -> dict:
    """Score every case again from the stored findings (after a change to the scoring)."""
    pools: dict = {}
    for c in res["clean"]:
        if "error" not in c:
            pools.setdefault(c["background"], []).append(c)
    rows = [_score(r, pools.get(r["background"], [])) for r in res["rows"]]
    return {**res, "summary": summarize_hybrid(rows), "rows": rows}


def run_hybrid(backgrounds: dict[str, Path] | None = None, reps: int = 3, seed: int = 0,
               kinds: list[str] | None = None, workers: int | None = None, progress=None,
               agent_cfg=None, lang: str = "en", resume: dict | None = None) -> dict:
    """``agent_cfg``: an :class:`~groundline.config.AgentConfig` for the agent under test (default: the rule
    agent). ``resume``: the result of an earlier call with the same agent and seed; its finished agent runs (clean
    and injected) are kept, only failed or missing ones run again, and everything is scored afresh."""
    backgrounds = backgrounds or available_backgrounds()
    kinds = kinds or list(LEVELS)
    agent = {"type": "rule"} if agent_cfg is None else \
        {"type": agent_cfg.kind, "model": agent_cfg.model, "reasoning_effort": agent_cfg.reasoning_effort}
    kept_clean, kept_rows = {}, {}
    if resume:
        if resume.get("agent") != agent or resume.get("seed") != seed:
            raise ValueError(f"cannot resume: that file is for agent {resume.get('agent')} seed {resume.get('seed')}, "
                             f"this run is {agent} seed {seed}")
        kept_clean = {(c["background"], c["kind"]): c for c in resume.get("clean", []) if "error" not in c}
        kept_rows = {(r["background"], r["kind"], r["level"], r["rep"]): r for r in resume["rows"]
                     if "error" not in r and "findings" in r}
    clean_jobs = [(name, str(path), k) for name, path in backgrounds.items() for k in kinds
                  if (name, k) not in kept_clean]
    jobs = [(name, str(path), k, lvl, r, seed) for name, path in backgrounds.items() for k in kinds
            for lvl in LEVELS[k] for r in range(reps) if (name, k, lvl, r) not in kept_rows]
    total = len(clean_jobs) + len(jobs)
    clean, rows = list(kept_clean.values()), list(kept_rows.values())
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(agent_cfg, lang)) as ex:
        for i, c in enumerate(ex.map(_clean_case, clean_jobs), 1):
            clean.append(c)
            if progress:
                progress(i, total)
        for i, row in enumerate(ex.map(_case, jobs), len(clean_jobs) + 1):
            if row is not None:
                rows.append(row)
            if progress:
                progress(i, total)
    order = {key: i for i, key in enumerate((name, k, lvl, r) for name in backgrounds for k in kinds
                                             for lvl in LEVELS[k] for r in range(reps))}
    rows.sort(key=lambda r: order.get((r["background"], r["kind"], r["level"], r["rep"]), len(order)))
    usage = [x["usage"] for x in clean + rows if x.get("usage")]
    return rescore_hybrid({
        "backgrounds": {k: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                        for k, p in backgrounds.items()},
        "agent": agent, "lang": lang, "reps": reps, "seed": seed, "levels": LEVELS, "units": UNITS,
        "persistence_s": PERSISTENCE_S, "tolerance_pct": TOLERANCE_PCT,
        **({"usage": {key: sum(u.get(key, 0) for u in usage) for key in ("input_tokens", "output_tokens",
                                                                         "requests")}} if usage else {}),
        "clean": clean, "rows": rows})


def summarize_hybrid(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["kind"], {}).setdefault(str(r["level"]), {"n": 0, "detected": 0, "masked": 0,
                                                                         "false_positives": 0, "errors": 0})
        if "error" in r:
            d["errors"] += 1
            continue
        d["n"] += 1
        d["detected"] += r["detected"]
        d["masked"] += r["masked"]
        d["false_positives"] += len(r["false_positives"])
    return out


def format_hybrid(res: dict) -> str:
    errors = any(d.get("errors") for by in res["summary"].values() for d in by.values())
    lines = ["| injected | size | detected | masked | new false positives |" + (" failed runs |" if errors else ""),
             "|---|---|---|---|---|" + ("---|" if errors else "")]
    for kind, by in res["summary"].items():
        for lvl, d in by.items():
            judged = d["n"] - d["masked"]
            rate = f"{d['detected']}/{judged} ({100 * d['detected'] / judged:.0f}%)" if judged else "–"
            lines.append(f"| {kind} | {float(lvl):g} {res['units'][kind]} | {rate} | {d['masked']} | "
                         f"{d['false_positives']} |" + (f" {d.get('errors', 0)} |" if errors else ""))
    failed_clean = [f"{c['background']} / {c['kind']}" for c in res.get("clean", []) if "error" in c]
    if failed_clean:
        lines.append(f"\nclean runs that failed (cases on these backgrounds are scored without them): "
                     f"{', '.join(failed_clean)}")
    u = res.get("usage")
    if u:
        lines.append(f"\ntokens: {u['input_tokens']:,} input / {u['output_tokens']:,} output, "
                     f"{u['requests']:,} requests")
    return "\n".join(lines)
