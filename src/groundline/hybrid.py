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

Real logs already contain genuine problems. The rule agent is therefore first run on the clean background
with the same configuration (limits, prediction); a finding in an injected run counts only if the clean run
has no finding of the same category on the same channel at that time. An injection that lands on such an
existing finding is reported as ``masked`` rather than as detected or missed. Every other new anomaly finding
is a false positive.
"""

from __future__ import annotations

import os
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


def _findings(df, meta, limits, ref):
    """Anomaly findings, each tagged with the tool behind its first cited evidence."""
    from .agent import RuleAgent
    from .session import Session

    s = Session(df, meta, ref, limits)
    res = RuleAgent("en").run(s)
    out = [f for f in res.findings if f.category != "observation"]
    for f in out:
        ev = s.evidence(f.evidence[0]) if f.evidence else None
        f.tool = ev.tool if ev is not None else None
    return out, set(res.agent.get("covered_channels", []))


def _overlap(f, t0: float, t1: float, tol: float = 0.25) -> bool:
    if f.t_start is None:
        return True
    return f.t_start <= t1 + tol and t0 - tol <= (f.t_end if f.t_end is not None else f.t_start)


def _same(a, b) -> bool:
    """The same finding in the clean and the injected run: same check, category and channel, overlapping."""
    return (a.category == b.category and a.channel == b.channel and a.tool == b.tool
            and _overlap(a, b.t_start or 0.0, b.t_end or b.t_start or 1e9))


# the tool whose finding an injection should produce
TOOL = {"oscillation": "detect_oscillation", "redline": "check_redlines", "deviation": "compare_reference",
        "spike": "check_sensor_health", "dropout": "check_sensor_health", "stuck": "check_sensor_health"}


_CLEAN: dict = {}  # (background, kind) -> clean findings, per worker process


def _case(args):
    name, path, kind, level, rep, seed = args
    df, meta, seg, thrust = _background(Path(path))
    fs = float(meta.get("sample_rate_hz") or 1 / np.median(np.diff(df["time"])))
    rng = np.random.default_rng([seed, int(level * 1000), rep, sum(map(ord, name + kind))])
    inj = _inject(kind, level, df, meta, seg, thrust, fs, rng)
    if inj is None:
        return None
    dfi, truth = inj
    limits, ref = _config(kind, df, seg, fs)
    if (path, kind) not in _CLEAN:
        _CLEAN[(path, kind)] = _findings(df, meta, limits, ref)
    clean, covered = _CLEAN[(path, kind)]
    found, _ = _findings(dfi, meta, limits, ref)
    new = [f for f in found if not any(_same(f, b) for b in clean)]
    hit = [f for f in new if f.category == CATEGORY[kind] and f.channel in (truth["channel"], None)
           and _overlap(f, truth["t_start"], truth["t_end"])]
    # masked: the clean run already reports this very thing there, or (for a deviation) already flags the channel
    # as untrustworthy (offset, frozen, dead), in which case its deviations are deliberately not reported
    masked = not hit and (any(b.category == CATEGORY[kind] and b.channel == truth["channel"] and b.tool == TOOL[kind]
                              and _overlap(b, truth["t_start"], truth["t_end"]) for b in clean)
                          or (kind == "deviation" and truth["channel"] in covered))
    fps = [f for f in new if f not in hit]
    return {"background": name, "kind": kind, "level": level, "rep": rep, "truth": truth,
            "detected": bool(hit), "masked": masked, "hit_title": hit[0].title if hit else None,
            "false_positives": [{"category": f.category, "channel": f.channel, "title": f.title} for f in fps]}


def run_hybrid(backgrounds: dict[str, Path] | None = None, reps: int = 3, seed: int = 0,
               kinds: list[str] | None = None, workers: int | None = None, progress=None) -> dict:
    backgrounds = backgrounds or available_backgrounds()
    kinds = kinds or list(LEVELS)
    jobs = [(name, str(path), k, lvl, r, seed) for name, path in backgrounds.items() for k in kinds
            for lvl in LEVELS[k] for r in range(reps)]
    rows = []
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_case, jobs), 1):
            if row is not None:
                rows.append(row)
            if progress:
                progress(i, len(jobs))
    return {"backgrounds": {k: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                            for k, p in backgrounds.items()},
            "reps": reps, "seed": seed, "levels": LEVELS, "units": UNITS,
            "persistence_s": PERSISTENCE_S, "tolerance_pct": TOLERANCE_PCT,
            "summary": summarize_hybrid(rows), "rows": rows}


def summarize_hybrid(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["kind"], {}).setdefault(str(r["level"]), {"n": 0, "detected": 0, "masked": 0,
                                                                         "false_positives": 0})
        d["n"] += 1
        d["detected"] += r["detected"]
        d["masked"] += r["masked"]
        d["false_positives"] += len(r["false_positives"])
    return out


def format_hybrid(res: dict) -> str:
    lines = ["| injected | size | detected | masked | new false positives |", "|---|---|---|---|---|"]
    for kind, by in res["summary"].items():
        for lvl, d in by.items():
            judged = d["n"] - d["masked"]
            rate = f"{d['detected']}/{judged} ({100 * d['detected'] / judged:.0f}%)" if judged else "–"
            lines.append(f"| {kind} | {float(lvl):g} {res['units'][kind]} | {rate} | {d['masked']} | "
                         f"{d['false_positives']} |")
    return "\n".join(lines)
