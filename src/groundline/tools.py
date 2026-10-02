"""Deterministic analysis tools.

Every number that can appear in a report is computed here, by plain
signal-processing code that can be unit-tested and re-run.  The LLM never reads
raw samples; it only chooses which of these tools to call and explains their
outputs.

Each tool has the signature ``fn(session, **params) -> (result: dict, figure_png_b64 | None)``
and is registered in :data:`REGISTRY` together with a JSON schema for its
parameters (used for LLM tool calling and the MCP server).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import signal

from . import plots
from .session import Session, source_hash


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    params: dict
    required: list[str]
    version: str

    def json_schema(self) -> dict:
        return {"type": "object", "properties": self.params, "required": self.required}


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, params: dict | None = None, required: list[str] | None = None):
    def deco(fn):
        REGISTRY[name] = ToolSpec(name, fn, description, params or {}, required or [], source_hash(fn))
        return fn

    return deco


_T = {"type": "number", "description": "window start time in seconds (default: phase start)"}
_T1 = {"type": "number", "description": "window end time in seconds (default: phase end)"}
_CH = {"type": "string", "description": "channel name"}


# ---------------------------------------------------------------------------- helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs of consecutive True values."""
    if mask.size == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


def _filled(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    return pd.Series(x).interpolate(limit_direction="both").to_numpy()


def _default_window(s: Session, t_start, t_end, phase: str | None = "mainstage"):
    if t_start is None or t_end is None:
        p0, p1 = s.phase_window(phase or "mainstage")
        t_start = p0 if t_start is None else t_start
        t_end = p1 if t_end is None else t_end
    return float(t_start), float(t_end)


def _unit(s: Session, ch: str) -> str:
    return s.channel_info(ch).get("unit", "")


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 only has trapz


def _native_rate(s: Session, ch: str) -> float:
    """Rate the channel was actually recorded at (it may have been interpolated onto a faster grid)."""
    r = s.channel_info(ch).get("native_rate_hz")
    return min(float(r), s.fs) if r else s.fs


def _firing_window(s: Session) -> tuple[float, float] | None:
    ph = {p["name"]: p for p in s.phases()}
    if "startup" not in ph or "shutdown" not in ph:
        return None
    return ph["startup"]["t_start"], ph["shutdown"]["t_end"]


def _primary_pressure(s: Session) -> str:
    if "Pc" in s.channels:
        return "Pc"
    p = s.channels_of_kind("pressure")
    if not p:
        raise ValueError("no pressure channel found; pass channel=")
    return max(p, key=lambda c: np.nanmax(s.data[c]))


# ---------------------------------------------------------------------------- describe
@tool(
    "describe_data",
    "Overview of the run: duration, sample rate, and per-channel unit/min/max/mean/NaN count. "
    "Call first to learn the channel names.",
)
def describe_data(s: Session):
    chans = {}
    for c in s.channels:
        x = s.data[c].to_numpy(dtype=float)
        info = s.channel_info(c)
        chans[c] = {
            "unit": info.get("unit", ""),
            "kind": info.get("kind", ""),
            **{k: info[k] for k in ("desc", "native_rate_hz", "note") if k in info},
            "min": np.nanmin(x),
            "max": np.nanmax(x),
            "mean": np.nanmean(x),
            "nan_samples": int(np.isnan(x).sum()),
        }
    t = s.time
    return (
        {
            "test_id": s.meta.get("test_id"),
            **{k: s.meta[k] for k in ("description", "engine_type") if k in s.meta},
            "n_samples": len(t),
            "duration_s": t[-1] - t[0],
            "sample_rate_hz": s.fs,
            "channels": chans,
            "has_reference": s.reference is not None,
            "reference_channels": [c for c in s.reference.columns if c != "time"] if s.reference is not None else [],
            "has_limits": bool(s.limits),
        },
        None,
    )


# ---------------------------------------------------------------------------- phases
def _pulse(s: Session, ch: str) -> dict:
    """Firing pulse on one channel: smoothed trace, pre-test baseline, steady level, and whether the pulse
    stands clearly above the baseline (a saturated or dead sensor shows no clear pulse)."""
    x = _filled(s.require_channel(ch))
    w = max(int(0.02 * s.fs), 1)
    xs = pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
    # steady level: the plateau around the peak of a 0.2 s rolling median (a brief spike does not set the peak,
    # and a sensor that sticks high after shutdown does not get averaged into the level)
    xm = pd.Series(xs).rolling(max(int(0.2 * s.fs), 1), center=True, min_periods=1).median().to_numpy()
    base = float(np.nanpercentile(xm, 5))  # quiet reading before / after the firing, offsets included
    k = int(np.nanargmax(xm))
    peak = float(xm[k])
    above = xm > base + 0.5 * (peak - base)
    lo = k - int(np.argmin(above[k::-1])) + 1 if not above[k::-1].all() else 0
    hi = k + int(np.argmin(above[k:])) if not above[k:].all() else len(xm)
    plateau = xs[lo:hi]
    top = plateau > base + 0.75 * (peak - base)
    level = float(np.nanmedian(plateau[top])) if top.any() else peak
    rise = level - base
    return {"xs": xs, "base": base, "level": level, "rise": rise,
            "clear": bool(rise > 0 and rise >= 0.25 * max(abs(level), abs(base)))}


def compute_phases(s: Session, channel: str | None = None, on_frac: float = 0.1, steady_frac: float = 0.9) -> dict:
    t = s.time
    first = channel or _primary_pressure(s)
    ch, pu, fallback_from = first, _pulse(s, first), None
    if channel is None and not pu["clear"]:
        # the chamber pressure shows no firing pulse (saturated, dead, unplugged): segment on thrust, or failing
        # that on another pressure channel, and say so
        others = s.channels_of_kind("force") + [c for c in s.channels_of_kind("pressure") if c != first]
        for c in others:
            pc = _pulse(s, c)
            if pc["clear"]:
                ch, pu, fallback_from = c, pc, first
                break
    xs, base, level, rise = pu["xs"], pu["base"], pu["level"], pu["rise"]
    on_thr, steady_thr = base + on_frac * rise, base + steady_frac * rise
    steady = np.where(xs > steady_thr)[0]
    if not pu["clear"] or steady.size == 0:
        return {
            "channel": ch,
            "fired": False,
            "phases": [{"name": "no_firing", "t_start": float(t[0]), "t_end": float(t[-1])}],
        }
    i_ms0, i_ms1 = int(steady[0]), int(steady[-1])
    # ignition: where the rise into mainstage leaves the baseline (not the first noise blip above it)
    below = np.where(xs[:i_ms0] < on_thr)[0]
    i_ign = int(below[-1]) + 1 if below.size else 0
    after = np.where(xs[i_ms1:] < on_thr)[0]
    tail_by = "below_on_frac"
    if after.size:
        i_tail = i_ms1 + int(after[0])
    else:
        # the trace never falls back (a sensor that sticks or shifts after shutdown): tail-off ends where it stops
        # falling, i.e. the first 0.2 s window after mainstage whose range is within 2 % of the pulse
        n_set = max(int(0.2 * s.fs), 2)
        rng = (pd.Series(xs[i_ms1:]).rolling(n_set).max() - pd.Series(xs[i_ms1:]).rolling(n_set).min()).to_numpy()
        settled = np.where(rng < 0.02 * rise)[0]
        i_tail, tail_by = ((i_ms1 + int(settled[0]) - n_set + 1, "settled_above_threshold") if settled.size
                           else (len(t) - 1, "end_of_record"))
    # the log stops while the engine is still firing (DAQ cut out, recording stopped early)
    ends_firing = bool(xs[-1] > on_thr and (tail_by == "end_of_record" or i_ms1 >= len(t) - max(int(0.5 * s.fs), 2)))
    bounds = [
        ("pre_test", t[0], t[i_ign]),
        ("startup", t[i_ign], t[i_ms0]),
        ("mainstage", t[i_ms0], t[i_ms1]),
        ("shutdown", t[i_ms1], t[i_tail]),
        ("post_test", t[i_tail], t[-1]),
    ]
    cmd_events = []
    for c in s.channels_of_kind("command"):
        d = np.diff(s.data[c].to_numpy(dtype=float))
        for i in np.where(np.abs(d) > 0.5)[0]:
            cmd_events.append({"command": c, "edge": "open" if d[i] > 0 else "close", "t_s": float(t[i + 1])})
    return {
        "channel": ch,
        **({"fallback_from": fallback_from, "fallback_reason": f"{fallback_from} shows no clear firing pulse"}
           if fallback_from else {}),
        "fired": True,
        "steady_level": level,
        "baseline": base,
        "unit": _unit(s, ch),
        "record_ends_during_firing": ends_firing,
        "record_end_s": float(t[-1]),
        "value_at_record_end": float(xs[-1]),
        "ignition_s": float(t[i_ign]),
        "mainstage_start_s": float(t[i_ms0]),
        "mainstage_end_s": float(t[i_ms1]),
        "tail_off_end_s": float(t[i_tail]),
        "tail_off_end_by": tail_by,
        "mainstage_duration_s": float(t[i_ms1] - t[i_ms0]),
        "phases": [{"name": n, "t_start": float(a), "t_end": float(b)} for n, a, b in bounds],
        "command_events": sorted(cmd_events, key=lambda e: e["t_s"]),
    }


@tool(
    "segment_phases",
    "Split the run into pre_test / startup / mainstage / shutdown / post_test from the chamber-pressure trace "
    "(10 % and 90 % of the way from the pre-test baseline to the steady level). Returns ignition, mainstage "
    "start/end and command edge times. If the chamber pressure shows no clear firing pulse (saturated or dead), "
    "thrust or another pressure channel is used instead (fallback_from). If the trace never falls back below 10 % "
    "(a sensor that sticks after shutdown), tail-off ends where it settles (tail_off_end_by); "
    "record_ends_during_firing says the log stopped while the engine was still firing.",
    {"channel": {**_CH, "description": "pressure channel to segment on (default Pc)"}},
)
def segment_phases(s: Session, channel: str | None = None):
    res = compute_phases(s, channel)
    if channel is None:
        s._cache["phases"] = res
    fig, (ax,) = plots.new_fig(1, height=3.0)
    xlim = None
    if res.get("fired"):
        # a short firing in a long recording would be a sliver: zoom the figure (not the result) onto it
        fire = res["tail_off_end_s"] - res["ignition_s"]
        if fire > 0 and (s.time[-1] - s.time[0]) > 5 * fire:
            pad = max(1.0, 0.5 * fire)
            xlim = (max(float(s.time[0]), res["ignition_s"] - pad), min(float(s.time[-1]), res["tail_off_end_s"] + pad))
    plots.shade_phases(ax, res["phases"], label=True, xlim=xlim)
    ch = res["channel"]
    ax.plot(s.time, s.data[ch], color=plots.SERIES[0], lw=1.0, label=ch)
    for e in res.get("command_events", []):
        ax.axvline(e["t_s"], color=plots.INK_2, lw=0.7, ls=":")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylabel(f"{ch} [{_unit(s, ch)}]")
    ax.set_xlabel("time [s]")
    note = " (dotted: valve commands)" if res.get("command_events") else ""
    ax.set_title(f"Phase segmentation on {ch}{note}", pad=14)
    return res, plots.to_base64(fig)


# ---------------------------------------------------------------------------- redlines
@tool(
    "check_redlines",
    "Check every channel that has a redline in the limits file. A violation must persist for "
    "`persistence_s` (default from limits, 10 ms) — shorter excursions are counted as suppressed transients.",
    {"persistence_s": {"type": "number", "description": "minimum violation duration in seconds"}},
)
def check_redlines(s: Session, persistence_s: float | None = None):
    red = s.limits.get("redlines", {})
    if not red:
        return {"checked": [], "violations": [], "note": "no redlines defined in limits"}, None
    pers = float(persistence_s if persistence_s is not None else s.limits.get("redline_persistence_s", 0.01))
    min_n = max(int(round(pers * s.fs)), 1)
    t = s.time
    violations, suppressed, checked = [], {}, []
    for ch, lim in red.items():
        if ch not in s.channels:
            continue
        x = s.require_channel(ch)
        # decide on a 5 ms moving average so sensor noise does not chop a real excursion into pieces;
        # report peak values from the raw signal
        xs = pd.Series(x).rolling(max(int(0.005 * s.fs), 1), center=True, min_periods=1).mean().to_numpy()
        xs = np.where(np.isnan(x), np.nan, xs)
        checked.append({"channel": ch, **lim, "unit": _unit(s, ch)})
        for kind in ("max", "min"):
            if kind not in lim:
                continue
            with np.errstate(invalid="ignore"):
                mask = xs > lim[kind] if kind == "max" else xs < lim[kind]
            for i0, i1 in _runs(mask):
                if i1 - i0 < min_n:
                    suppressed[ch] = suppressed.get(ch, 0) + 1
                    continue
                seg = x[i0:i1]
                j = int(np.nanargmax(seg) if kind == "max" else np.nanargmin(seg))
                violations.append(
                    {
                        "channel": ch,
                        "kind": kind,
                        "limit": lim[kind],
                        "unit": _unit(s, ch),
                        "t_start": float(t[i0]),
                        "t_end": float(t[i1 - 1]),
                        "duration_s": float(t[i1 - 1] - t[i0]),
                        "peak_value": float(seg[j]),
                        "t_peak": float(t[i0 + j]),
                        "margin_exceeded": float(abs(seg[j] - lim[kind])),
                    }
                )
    res = {
        "persistence_s": pers,
        "smoothing_s": 0.005,
        "checked": checked,
        "violations": violations,
        "n_violations": len(violations),
        "suppressed_transients": suppressed,
    }
    fig = None
    chans = sorted({v["channel"] for v in violations})
    if chans:
        f, axes = plots.new_fig(len(chans), height=2.4)
        for ax, ch in zip(axes, chans):
            plots.shade_phases(ax, s.phases())
            ax.plot(t, s.data[ch], color=plots.SERIES[0], lw=1.0)
            for kind in ("max", "min"):
                if kind in red[ch]:
                    ax.axhline(red[ch][kind], color=plots.CRITICAL, ls="--", lw=1.0)
                    ax.text(t[0], red[ch][kind], f" redline {kind} {red[ch][kind]}", color=plots.CRITICAL,
                            va="bottom", fontsize=7)
            for v in violations:
                if v["channel"] == ch:
                    ax.axvspan(v["t_start"], v["t_end"], color=plots.CRITICAL, alpha=0.15, lw=0)
            ax.set_ylabel(f"{ch} [{_unit(s, ch)}]")
        axes[0].set_title("Redline violations (shaded)")
        axes[-1].set_xlabel("time [s]")
        fig = plots.to_base64(f)
    return res, fig


# ---------------------------------------------------------------------------- oscillation
def _peak_interp(mag: np.ndarray, k: int) -> float:
    if 0 < k < len(mag) - 1:
        a, b, c = np.log(mag[k - 1] + 1e-30), np.log(mag[k] + 1e-30), np.log(mag[k + 1] + 1e-30)
        den = a - 2 * b + c
        if den != 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


@tool(
    "detect_oscillation",
    "Sliding-window FFT search for narrow-band oscillations (e.g. combustion instability) on one channel. "
    "A window is flagged when the dominant peak in [fmin, fmax] exceeds threshold_pct of the channel mean "
    "(zero-mean channels such as accelerometers use only the prominence test). Consecutive flagged windows "
    "with the same frequency are merged into events with frequency and amplitude. An oscillation has to last "
    "min_windows overlapping windows (default 3); shorter narrow-band peaks are listed as short_events: usually "
    "a transient such as a pressure spike, not a sustained oscillation. A second pass with windows five times "
    "longer (at least 0.5 s) catches sustained lines hidden in broadband combustion noise. A line already there before "
    "ignition at half the amplitude or more is listed under interference (electrical or mechanical pickup such as "
    "mains hum), not as an oscillation event.",
    {
        "channel": {**_CH, "description": "channel to analyse (default Pc)"},
        "t_start": _T,
        "t_end": _T1,
        "fmin": {"type": "number", "description": "lower frequency bound in Hz"},
        "fmax": {"type": "number", "description": "upper frequency bound in Hz"},
        "threshold_pct": {"type": "number", "description": "zero-to-peak amplitude threshold in % of mean"},
        "window_s": {"type": "number", "description": "FFT window length in seconds (default 0.1)"},
        "min_windows": {"type": "integer", "description": "flagged windows in a row needed for an event (default 3)"},
    },
)
def detect_oscillation(
    s: Session,
    channel: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    threshold_pct: float | None = None,
    window_s: float = 0.1,
    prominence: float = 8.0,
    min_windows: int | None = None,
    _long_pass: bool = False,
):
    cfg = s.limits.get("oscillation", {})
    min_windows = int(min_windows if min_windows is not None else cfg.get("min_windows", 3))
    channel = channel or _primary_pressure(s)
    fmin = float(fmin if fmin is not None else cfg.get("fmin_hz", 50.0))
    fmax = float(fmax if fmax is not None else cfg.get("fmax_hz", 2000.0))
    thr = float(threshold_pct if threshold_pct is not None else cfg.get("threshold_pct", 0.5))
    t_start, t_end = _default_window(s, t_start, t_end)
    fs = s.fs
    native = _native_rate(s, channel)
    fmax = min(fmax, 0.45 * native)
    if fmax <= fmin:
        return {
            "channel": channel, "t_start": t_start, "t_end": t_end, "fmin_hz": fmin, "fmax_hz": fmax,
            "native_rate_hz": native, "applicable": False, "detected": False, "events": [],
            "note": f"channel recorded at {native:g} Hz: its Nyquist limit is below fmin, so it cannot show "
                    f"oscillations in the requested band",
        }, None
    # at least 16 frequency bins in the band, otherwise the prominence test is meaningless
    window_s = max(window_s, 16.0 / (fmax - fmin))
    # a window longer than the analysed span (a model asked for window_s=400) has nothing to slide over
    window_s = min(window_s, max(t_end - t_start, 16.0 / fs))
    x_all = s.require_channel(channel)
    nan_all = np.isnan(x_all)
    x_all = _filled(x_all)
    t = s.time
    n = max(int(round(window_s * fs)), 16)
    hop = n // 2
    i_lo = int(np.searchsorted(t, t_start))
    i_hi = int(np.searchsorted(t, t_end))
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    band_idx = np.where(band)[0]
    df = fs / n
    rows = []
    for i0 in range(i_lo, max(i_hi - n + 1, i_lo), hop):
        seg = x_all[i0 : i0 + n]
        if nan_all[i0 : i0 + n].mean() > 0.2:
            rows.append({"t_center": float(t[i0 + n // 2]), "skipped": True})
            continue
        tt = np.arange(n)
        mean = float(seg.mean())
        detr = seg - np.polyval(np.polyfit(tt, seg, 1), tt)
        mag = 2 * np.abs(np.fft.rfft(detr * win)) / win.sum()
        mb = mag[band]
        k_rel = int(np.argmax(mb))
        k = band_idx[k_rel]
        delta = _peak_interp(mag, k)
        amp = float(mb[k_rel])
        prom = amp / (float(np.median(mb)) + 1e-30)
        relative = abs(mean) > 3 * float(np.std(seg))
        amp_pct = 100 * amp / abs(mean) if relative else None
        # a peak on the lowest bin of the band is leakage from a slower transient (e.g. the ignition rise)
        # entering the window, not a narrow-band oscillation inside the band
        at_edge = k_rel == 0
        flagged = prom >= prominence and (amp_pct is None or amp_pct >= thr) and not at_edge
        rows.append(
            {
                "t_center": float(t[i0 + n // 2]),
                "freq_hz": float(np.clip((k + delta) * df, fmin, fmax)),
                "amp": amp,
                "amp_pct": amp_pct,
                "prominence": prom,
                "flagged": bool(flagged),
            }
        )
    # merge flagged windows into events
    events: list[dict] = []
    hop_s = hop / fs
    cur = None
    for r in rows:
        if r.get("skipped") or not r["flagged"]:
            if cur:
                events.append(cur)
                cur = None
            continue
        if cur and abs(r["freq_hz"] - cur["_f"][-1]) <= max(2 * df, 0.05 * r["freq_hz"]):
            cur["_rows"].append(r)
            cur["_f"].append(r["freq_hz"])
        else:
            if cur:
                events.append(cur)
            cur = {"_rows": [r], "_f": [r["freq_hz"]]}
    if cur:
        events.append(cur)
    # a sustained oscillation spans several windows; a lone flagged window is a transient (a pressure spike,
    # a step) whose broadband energy happens to peak somewhere in the band. A short analysis span (drilling
    # into one event) cannot hold min_windows windows, so the requirement shrinks to what fits.
    need = max(1, min(min_windows, sum(1 for r in rows if not r.get("skipped"))))
    short = [e for e in events if len(e["_rows"]) < need]
    events = [e for e in events if len(e["_rows"]) >= need]
    out_events = []
    for e in events:
        rs = e["_rows"]
        pk = max(rs, key=lambda r: r["amp"])
        ev = {
            "t_start": rs[0]["t_center"] - hop_s / 2,
            "t_end": rs[-1]["t_center"] + hop_s / 2,
            "duration_s": rs[-1]["t_center"] - rs[0]["t_center"] + hop_s,
            "freq_hz": float(np.median([r["freq_hz"] for r in rs])),
            "peak_amp": pk["amp"],
            "peak_amp_unit": _unit(s, channel),
            "t_peak": pk["t_center"],
            "n_windows": len(rs),
        }
        if pk["amp_pct"] is not None:
            ev["peak_amp_pct"] = pk["amp_pct"]
            ev["mean_amp_pct"] = float(np.mean([r["amp_pct"] for r in rs]))
        out_events.append(ev)
    # a sustained line in rough, broadband combustion noise can hide in short windows: a window five times longer
    # gains about sqrt(5) in signal-to-noise for a steady line. Search again with long windows and keep the
    # events the short-window pass missed (marked with their window).
    long_w = max(0.5, 5 * window_s)
    if not _long_pass and t_end - t_start >= 4 * long_w:
        long_res, _ = detect_oscillation(s, channel, t_start, t_end, fmin, fmax, threshold_pct, long_w, prominence,
                                         min_windows, _long_pass=True)
        for e in long_res["events"]:
            if not any(e["t_start"] < o["t_end"] and o["t_start"] < e["t_end"]
                       and abs(e["freq_hz"] - o["freq_hz"]) <= 0.05 * e["freq_hz"] for o in out_events):
                out_events.append({**e, "window_s": long_res["window_s"]})
        out_events.sort(key=lambda e: e["t_start"])
    if _long_pass:
        return {"events": out_events, "window_s": n / fs}, None
    # a line already there before ignition is pickup (mains hum, a pump, a structural mode), not combustion: the
    # chamber is not burning yet. Compare each event with the same frequency over the last seconds of pre-test.
    interference = []
    ph = {q["name"]: q for q in s.phases()}
    if "pre_test" in ph and "startup" in ph:
        p_hi = ph["startup"]["t_start"] - 0.2
        i_a, i_b = int(np.searchsorted(t, max(ph["pre_test"]["t_start"], p_hi - 2.0))), int(np.searchsorted(t, p_hi))
        if i_b - i_a >= 2 * n:
            for e in list(out_events):
                kf = int(round(e["freq_hz"] / df))
                amps = []
                for j in range(i_a, i_b - n + 1, hop):
                    seg = x_all[j:j + n]
                    if nan_all[j:j + n].mean() > 0.2:
                        continue
                    tt = np.arange(n)
                    m = 2 * np.abs(np.fft.rfft((seg - np.polyval(np.polyfit(tt, seg, 1), tt)) * win)) / win.sum()
                    amps.append(float(m[max(kf - 2, 0):kf + 3].max()))
                if amps and float(np.median(amps)) >= 0.5 * e["peak_amp"]:
                    e["pre_ignition_amp"] = float(np.median(amps))
                    interference.append(e)
                    out_events.remove(e)
    valid = [r for r in rows if not r.get("skipped")]
    worst = max(valid, key=lambda r: r["amp"]) if valid else None
    res = {
        "channel": channel,
        "unit": _unit(s, channel),
        "t_start": t_start,
        "t_end": t_end,
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "native_rate_hz": native,
        "threshold_pct": thr,
        "prominence_min": prominence,
        "window_s": n / fs,
        "freq_resolution_hz": df,
        "n_windows": len(rows),
        "n_skipped_windows": sum(1 for r in rows if r.get("skipped")),
        "min_windows": need,
        "detected": bool(out_events),
        "events": out_events,
        "interference": interference,  # lines present before ignition too: pickup, not combustion
        "short_events": [{"t_start": e["_rows"][0]["t_center"] - hop_s / 2, "t_end": e["_rows"][-1]["t_center"] + hop_s / 2,
                          "freq_hz": float(np.median(e["_f"])), "n_windows": len(e["_rows"]),
                          **({"peak_amp_pct": max(r["amp_pct"] for r in e["_rows"])}
                             if all(r["amp_pct"] is not None for r in e["_rows"]) else {})}
                         for e in short][:10],
        "max_window": None
        if worst is None
        else {k: worst[k] for k in ("t_center", "freq_hz", "amp", "amp_pct", "prominence")},
    }
    # figure: spectrogram + amplitude track
    f, (ax1, ax2) = plots.new_fig(2, height=2.4)
    i_a, i_b = i_lo, max(i_hi, i_lo + n)
    seg = x_all[i_a:i_b] - pd.Series(x_all[i_a:i_b]).rolling(n, center=True, min_periods=1).mean().to_numpy()
    if len(seg) >= n:  # a span shorter than one window (a log cut off at ignition) has no spectrogram to draw
        fr, tt, Sxx = signal.spectrogram(seg, fs=fs, nperseg=n, noverlap=hop, window="hann", mode="magnitude")
        sel = (fr >= fmin) & (fr <= fmax)
        ax1.pcolormesh(tt + t[i_a], fr[sel], 20 * np.log10(Sxx[sel] + 1e-12), shading="auto", cmap="Blues")
    ax1.set_ylabel("freq [Hz]")
    ax1.grid(False)
    ax1.set_title(f"{channel}: spectrogram and dominant-peak amplitude")
    tc = [r["t_center"] for r in valid]
    if any(r["amp_pct"] is not None for r in valid):
        ax2.plot(tc, [r["amp_pct"] if r["amp_pct"] is not None else np.nan for r in valid], color=plots.SERIES[0])
        ax2.axhline(thr, color=plots.CRITICAL, ls="--", lw=1.0)
        ax2.set_ylabel("peak amp [% of mean]")
    else:
        ax2.plot(tc, [r["prominence"] for r in valid], color=plots.SERIES[0])
        ax2.axhline(prominence, color=plots.CRITICAL, ls="--", lw=1.0)
        ax2.set_ylabel("peak prominence [x]")
    for e in out_events:
        for ax in (ax1, ax2):
            ax.axvspan(e["t_start"], e["t_end"], color=plots.CRITICAL, alpha=0.12, lw=0)
        ax2.annotate(f"{e['freq_hz']:.0f} Hz", (e["t_peak"], 1), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8, color=plots.INK)
    ax2.set_xlabel("time [s]")
    return res, plots.to_base64(f)


# ---------------------------------------------------------------------------- sensor health
COINCIDENT_S = 0.02  # spikes on different kinds of sensor this close together belong to one event
# readings no pressure or temperature sensor can truly give: below vacuum (any gauge or absolute pressure is
# above -1 atm) or below absolute zero; a small margin covers noise
_PHYS_MIN = {"psi": -14.7, "bar": -1.013, "kpa": -101.3, "mpa": -0.1013, "pa": -101325.0,
             "degc": -273.15, "c": -273.15, "k": 0.0, "degf": -459.67}


def _impossible_below(unit: str) -> float | None:
    u = (unit or "").strip().lower().replace("°", "deg").replace("℃", "degc")
    lim = _PHYS_MIN.get(u)
    return None if lim is None else lim + 0.01 * max(abs(lim), 1.0)


def _gap_issue(t: np.ndarray, gaps: list[dict], firing) -> dict:
    starts = np.array([g["t_start"] for g in gaps])
    in_fire = [g for g in gaps if firing and g["t_start"] <= firing[1] and firing[0] <= g["t_end"]]
    return {"kind": "recurring_nan_gaps", "t_start": gaps[0]["t_start"], "t_end": gaps[-1]["t_end"],
            "count": len(gaps), "total_s": float(sum(g["duration_s"] for g in gaps)),
            "longest_s": float(max(g["duration_s"] for g in gaps)),
            "median_interval_s": float(np.median(np.diff(starts))),
            "count_during_firing": len(in_fire), "gaps_during_firing": in_fire[:10]}


def _baseline_return(s: Session, frac: float = 0.25, settle_s: float = 1.0) -> dict | None:
    """The segmentation channel (chamber pressure) should fall back to its pre-test reading once the engine is
    off. Returns an issue when, from ``settle_s`` after mainstage end, it stays above baseline by more than
    ``frac`` of the steady level."""
    seg = s._cache.get("phases") or {}
    ch = seg.get("channel")
    if not seg.get("fired") or ch is None:
        return None
    t = s.time
    x = s.require_channel(ch)
    pre = x[(t >= s.phase_window("pre_test")[0]) & (t < seg["ignition_s"])]
    # not before tail-off has ended: a slow tail-off (solid or hybrid motors) is not a reading that failed to return
    t_after = max(seg["mainstage_end_s"] + settle_s, seg.get("tail_off_end_s", 0.0))
    post = x[t >= t_after]
    pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
    if pre.size < 0.5 * s.fs or post.size < 0.5 * s.fs:  # a record that ends soon after shutdown still counts
        return None
    base, after, level = float(np.median(pre)), float(np.median(post)), float(seg["steady_level"])
    if level <= base or after - base <= frac * (level - base):
        return None
    return {"channel": ch, "kind": "no_return_to_baseline", "t_start": float(t_after), "t_end": float(t[-1]),
            "baseline": base, "post_level": after, "steady_level": level, "unit": _unit(s, ch),
            "offset_of_steady_pct": 100 * (after - base) / (level - base)}


@tool(
    "check_sensor_health",
    "Look for instrumentation problems on measurement channels: NaN gaps (more than 3 gaps on one channel are "
    "reported together as recurring gaps; gaps shared by every checked channel are reported once, as dropped DAQ "
    "frames), flatlines (value exactly stuck for >= 50 ms and >= 3 native samples, overlapping the firing — a steady "
    "reading while the engine is off is not a fault; short NaN gaps inside a flatline do not split it, and a value "
    "stuck at the channel's maximum or minimum is marked, as it suggests a saturated sensor), isolated spikes "
    "(>12 local robust sigmas, <= 5 samples wide; spikes that coincide within 20 ms on sensors of different kinds "
    "are reported together as one event, since a single sensor cannot explain them), a chamber pressure that "
    "does not return to its pre-test baseline after shutdown, channels that hold one value for the whole record "
    "(not connected), physically impossible readings (pressure below vacuum, temperature below absolute zero) and "
    "channels that carry exactly the same samples as another (wiring or configuration error).",
    {"channels": {"type": "array", "items": {"type": "string"}, "description": "channels to check (default all measurements)"}},
)
def check_sensor_health(s: Session, channels: list[str] | None = None, spike_sigma: float = 12.0):
    t = s.time
    fs = s.fs
    # commands are not measurements; housekeeping channels (logger temperature, battery, ...) are
    # expected to sit still and are only checked when asked for explicitly
    chans = channels or [c for c in s.channels if s.channel_info(c).get("kind") not in ("command", "housekeeping")]
    issues, summary = [], {}
    firing = _firing_window(s)
    masks = {ch: np.isnan(s.require_channel(ch)) for ch in chans}
    # NaNs present on every checked channel at once come from the DAQ (dropped frames), not from one sensor
    shared = np.logical_and.reduce(list(masks.values())) if len(chans) >= 2 else np.zeros(len(t), bool)
    shared_chans = []
    for ch in chans:
        x = s.require_channel(ch)
        nan = masks[ch]
        ch_issues = []
        gaps = [{"t_start": float(t[i0]), "t_end": float(t[i1 - 1]), "duration_s": float(t[i1 - 1] - t[i0]),
                 "n_samples": i1 - i0} for i0, i1 in _runs(nan)]
        if len(gaps) > 3 and shared.any() and not (nan & ~shared).any():
            shared_chans.append(ch)  # reported once for all channels below
        elif len(gaps) > 3:
            ch_issues.append({"channel": ch, **_gap_issue(t, gaps, firing)})
        else:
            ch_issues += [{"channel": ch, "kind": "nan_gap", **g} for g in gaps]
        min_flat = max(int(0.05 * fs), int(np.ceil(3 * fs / _native_rate(s, ch))), 3)
        # a stuck value interrupted by short dropouts is one flatline, not one per piece between the gaps
        xb = pd.Series(x).interpolate(limit=max(int(0.1 * fs), 1), limit_area="inside").to_numpy()
        same = np.concatenate([[False], np.diff(xb) == 0])
        lo_all, hi_all = (float(np.nanmin(x)), float(np.nanmax(x))) if (~nan).any() else (np.nan, np.nan)
        if (~nan).any() and lo_all == hi_all:
            # one value for the whole record: the sensor is not connected or not working, nothing to judge
            ch_issues.append({"channel": ch, "kind": "dead_channel", "t_start": float(t[0]), "t_end": float(t[-1]),
                              "value": lo_all, "unit": _unit(s, ch),
                              "physically_impossible": bool((lim := _impossible_below(_unit(s, ch))) is not None
                                                            and lo_all < lim)})
            issues.extend(ch_issues)
            summary[ch] = {"nan_samples": int(nan.sum()), "flatline_s": float(t[-1] - t[0]), "spikes": 0, "ok": False}
            continue
        lim = _impossible_below(_unit(s, ch))
        bad = (x < lim) if lim is not None else np.zeros(len(x), bool)
        if bad.sum() >= 0.5 * fs:  # a few noisy samples past the limit are noise; 0.5 s of them is an offset
            ch_issues.append({"channel": ch, "kind": "impossible_value", "t_start": float(t[np.argmax(bad)]),
                              "t_end": float(t[len(bad) - 1 - np.argmax(bad[::-1])]), "min_value": lo_all,
                              "unit": _unit(s, ch), "physical_limit": lim, "share_of_record_pct": 100 * float(bad.mean()),
                              "median_value": float(np.nanmedian(x)), "min_duration_s": 0.5})
        for i0, i1 in _runs(same):
            i0 -= 1
            if firing and min(t[i1 - 1], firing[1]) - max(t[i0], firing[0]) < min_flat / fs:
                continue  # stuck only while the engine was off: indistinguishable from a quiet reading
            if i1 - i0 >= min_flat:
                v = float(xb[i0])
                ch_issues.append({"channel": ch, "kind": "flatline", "t_start": float(t[i0]), "t_end": float(t[i1 - 1]),
                                  "duration_s": float(t[i1 - 1] - t[i0]), "stuck_value": v,
                                  "at_channel_max": v == hi_all, "at_channel_min": v == lo_all,
                                  **({"channel_max": hi_all} if v == hi_all else {}),
                                  **({"channel_min": lo_all} if v == lo_all else {})})
        flats = [i for i in ch_issues if i["kind"] == "flatline"]
        for a, b in zip(flats, flats[1:]):  # a stuck value broken by a blip of a sample or two is one flatline
            if b["stuck_value"] == a["stuck_value"] and b["t_start"] - a["t_end"] < 0.1:
                b["t_start"], b["duration_s"] = a["t_start"], b["t_end"] - a["t_start"]
                ch_issues.remove(a)
        # spikes are judged at the rate the channel was recorded at: on a signal interpolated from a slow
        # logger, one real sample becomes a 10-point triangle and the fast-grid test would flag every peak
        k = max(int(round(fs / _native_rate(s, ch))), 1)
        tk, xk, nk, fk = t[::k], x[::k], nan[::k], fs / k
        xf = _filled(xk)
        med = signal.medfilt(xf, 11)
        res = np.abs(xf - med)
        w = max(int(0.05 * fk), 11)
        local = pd.Series(res).rolling(w, center=True, min_periods=1).median().to_numpy() * 1.4826
        # noise floor from the samples that are off the median: on a quantized, mostly quiet signal the
        # plain median residual is 0 and one-step flicker would otherwise count as a huge spike
        nz = res[res > 0]
        floor = max(float(np.median(nz)) * 1.4826 if nz.size else 0.0, 1e-12)
        sigma = np.maximum(local, floor)
        near_nan = np.convolve(nk.astype(float), np.ones(2 * int(0.01 * fk) + 1), mode="same") > 0
        flag = (res > spike_sigma * sigma) & ~near_nan
        groups = _runs(flag)
        merged: list[list[int]] = []
        gap = int(0.005 * fk)
        for i0, i1 in groups:
            if merged and i0 - merged[-1][1] <= gap:
                merged[-1][1] = i1
            else:
                merged.append([i0, i1])
        spikes = []
        for i0, i1 in merged:
            if i1 - i0 > 5:
                continue
            jj = i0 + int(np.argmax(res[i0:i1]))
            # a spike goes out and comes back: the samples on both sides must lie on the same side of it.
            # A sample on a steep but smooth edge (ignition rise on a slow logger) sits between its neighbours.
            if 0 < i0 and i1 < len(xf):
                lft, rgt = xf[i0 - 1], xf[i1]
                excursion = xf[jj] - 0.5 * (lft + rgt)
                if (xf[jj] - lft) * (xf[jj] - rgt) <= 0 or abs(excursion) < 0.5 * res[jj]:
                    continue
            spikes.append({"t": float(tk[jj]), "value": float(xk[jj]), "local_median": float(med[jj]),
                           "deviation": float(xk[jj] - med[jj]), "sigmas": float(res[jj] / sigma[jj])})
        if spikes:
            ch_issues.append({"channel": ch, "kind": "spike", "t_start": spikes[0]["t"], "t_end": spikes[-1]["t"],
                              "count": len(spikes), "spikes": spikes})  # trimmed to 20 after the coincidence check
        issues.extend(ch_issues)
        summary[ch] = {
            "nan_samples": int(nan.sum()),
            "flatline_s": float(sum(i["duration_s"] for i in ch_issues if i["kind"] == "flatline")),
            "spikes": len(spikes),
            "ok": not ch_issues,
        }
    # spikes on sensors of different kinds within COINCIDENT_S of each other are one event hitting several
    # sensors (a fast physical transient, or interference on the DAQ), not single-sensor glitches
    fam = {ch: (s.channel_info(ch).get("kind") or "other").split("_")[0] for ch in chans}
    allsp = sorted((sp["t"], i["channel"], sp) for i in issues if i["kind"] == "spike" for sp in i["spikes"])
    for tk_, ch, sp in allsp:
        near = {c for t2, c, _ in allsp if abs(t2 - tk_) <= COINCIDENT_S and fam[c] != fam[ch]}
        if near:
            sp["coincident_with"] = sorted(near)
    clusters: list[list] = []
    for tk_, ch, sp in allsp:
        if "coincident_with" in sp:
            if clusters and tk_ - clusters[-1][-1][0] <= COINCIDENT_S:
                clusters[-1].append((tk_, ch, sp))
            else:
                clusters.append([(tk_, ch, sp)])
    for i in [i for i in issues if i["kind"] == "spike"]:
        i["spikes"] = [sp for sp in i["spikes"] if "coincident_with" not in sp]
        if not i["spikes"]:
            issues.remove(i)
        else:
            i.update(t_start=i["spikes"][0]["t"], t_end=i["spikes"][-1]["t"], count=len(i["spikes"]))
            i["spikes"] = i["spikes"][:20]
    for cl in clusters:
        chs = sorted({c for _, c, _ in cl})
        issues.append({"channel": None, "channels": chs, "kind": "coincident_spikes", "t_start": cl[0][0],
                       "t_end": cl[-1][0], "n_channels": len(chs),
                       "spikes": [{"channel": c, "t": t_, "deviation": sp["deviation"]} for t_, c, sp in cl]})
    # two channels carrying exactly the same samples: one is wired or configured to the other's input
    seen: dict[bytes, list[str]] = {}
    for ch in chans:
        x = s.require_channel(ch)
        if np.nanmax(x) > np.nanmin(x) if (~np.isnan(x)).any() else False:
            seen.setdefault(np.nan_to_num(x, nan=np.inf).tobytes(), []).append(ch)
    for group in seen.values():
        if len(group) > 1:
            issues.append({"channel": None, "channels": group, "kind": "duplicate_channels",
                           "t_start": float(t[0]), "t_end": float(t[-1]), "n_channels": len(group)})
            for ch in group:
                summary[ch]["ok"] = False
    if shared_chans:
        gaps = [{"t_start": float(t[i0]), "t_end": float(t[i1 - 1]), "duration_s": float(t[i1 - 1] - t[i0]),
                 "n_samples": i1 - i0} for i0, i1 in _runs(shared)]
        issues.insert(0, {"channel": None, "channels": shared_chans, "shared_by_all_checked": True,
                          **_gap_issue(t, gaps, firing)})
    if (b := _baseline_return(s)) is not None and b["channel"] in chans:
        issues.append(b)
        summary[b["channel"]]["ok"] = False
    res = {"checked": chans, "issues": issues, "n_issues": len(issues), "summary": summary,
           "criteria": {"spike_sigma": spike_sigma, "spike_max_width_samples": 5,
                        "flatline_min_s": max(int(0.05 * fs), 3) / fs, "flatline_min_native_samples": 3,
                        "flatline_only_during_firing": firing is not None, "flatline_bridges_gaps_up_to_s": 0.1,
                        "recurring_gap_threshold": 3, "nan_margin_s": 0.01, "coincident_spike_s": COINCIDENT_S,
                        "baseline_return_frac": 0.25, "baseline_settle_s": 1.0}}
    fig = None
    bad = [c for c in chans if not summary[c]["ok"]][:4]
    if bad:
        f, axes = plots.new_fig(len(bad), height=2.2)
        for ax, ch in zip(axes, bad):
            plots.shade_phases(ax, s.phases())
            ax.plot(t, s.data[ch], color=plots.SERIES[0], lw=0.9)
            for i in issues:
                if i["channel"] != ch:
                    continue
                if i["kind"] == "recurring_nan_gaps":
                    continue  # the NaNs already show as breaks in the trace
                if i["kind"] == "spike":
                    ax.plot([sp["t"] for sp in i["spikes"]], [sp["value"] for sp in i["spikes"]], "o",
                            ms=8, mfc="none", mec=plots.CRITICAL, mew=1.5)
                else:
                    ax.axvspan(i["t_start"], i["t_end"], color=plots.CRITICAL, alpha=0.15, lw=0)
            ax.set_ylabel(f"{ch} [{_unit(s, ch)}]")
        axes[0].set_title("Sensor health issues (shaded: NaN/flatline, circled: spikes)")
        axes[-1].set_xlabel("time [s]")
        fig = plots.to_base64(f)
    return res, fig


# ---------------------------------------------------------------------------- thrust vs chamber pressure
@tool(
    "check_thrust_pressure_ratio",
    "Thrust divided by chamber pressure (both above their pre-test baselines) is proportional to the thrust "
    "coefficient times the nozzle throat area, so it should hold roughly steady while the engine burns. Computed in "
    "short quasi-steady windows (chamber pressure changing by less than 15 % within the window) while thrust is "
    "above half its steady level and chamber pressure above a quarter of its rise; a change larger than "
    "max_change_pct (default 50) points to a throat change (erosion, a broken insert) or to a "
    "drifting or failing sensor.",
    {"thrust": {**_CH, "description": "thrust channel (default the first force channel)"},
     "pressure": {**_CH, "description": "chamber pressure channel (default Pc)"},
     "max_change_pct": {"type": "number", "description": "largest accepted spread of the ratio, % (default 50)"}},
)
def check_thrust_pressure_ratio(s: Session, thrust: str | None = None, pressure: str | None = None,
                                max_change_pct: float = 50.0, window_s: float = 0.25):
    forces = s.channels_of_kind("force")
    thrust = thrust or (forces[0] if forces else None)
    pressure = pressure or (_primary_pressure(s) if s.channels_of_kind("pressure") or "Pc" in s.channels else None)
    if thrust is None or pressure is None:
        return {"applicable": False, "consistent": True,
                "note": "needs a thrust (force) channel and a chamber pressure channel"}, None
    fp, pp = _pulse(s, thrust), _pulse(s, pressure)
    if not (fp["clear"] and pp["clear"]):
        bad = [c for c, q in ((thrust, fp), (pressure, pp)) if not q["clear"]]
        return {"applicable": False, "consistent": True, "thrust": thrust, "pressure": pressure,
                "note": f"{', '.join(bad)} shows no clear firing pulse, so the ratio means nothing"}, None
    t = s.time
    f, p = fp["xs"] - fp["base"], pp["xs"] - pp["base"]
    burning = (f > 0.5 * fp["rise"]) & (p > 0.25 * pp["rise"])
    n = max(int(window_s * s.fs), 2)
    rows = []
    for i0 in range(0, len(t) - n + 1, n):
        sl = slice(i0, i0 + n)
        q = max(n // 4, 1)
        pa, pb, pm = float(np.median(p[i0:i0 + q])), float(np.median(p[i0 + n - q:i0 + n])), float(np.median(p[sl]))
        # quasi-steady windows only: while the chamber pressure is rising or collapsing (start-up, shutdown) the
        # two signals lead and lag each other and the ratio says nothing about the nozzle
        if burning[sl].mean() > 0.9 and abs(pb - pa) <= 0.15 * pm:
            rows.append({"t_center": float(t[i0 + n // 2]), "ratio": float(np.median(f[sl] / p[sl]))})
    if len(rows) < 3:
        return {"applicable": False, "consistent": True, "thrust": thrust, "pressure": pressure,
                "note": "fewer than 3 windows with both thrust and chamber pressure up"}, None
    r = np.array([w["ratio"] for w in rows])
    lo, hi = int(np.argmin(r)), int(np.argmax(r))
    change = 100 * (r[hi] / r[lo] - 1) if r[lo] > 0 else float("inf")
    res = {"applicable": True, "thrust": thrust, "pressure": pressure,
           "ratio_unit": f"{_unit(s, thrust)}/{_unit(s, pressure)}",
           "thrust_baseline": fp["base"], "pressure_baseline": pp["base"],
           "t_start": rows[0]["t_center"] - window_s / 2, "t_end": rows[-1]["t_center"] + window_s / 2,
           "n_windows": len(rows), "ratio_start": float(r[0]), "ratio_end": float(r[-1]),
           "ratio_min": float(r[lo]), "t_ratio_min": rows[lo]["t_center"],
           "ratio_max": float(r[hi]), "t_ratio_max": rows[hi]["t_center"],
           "change_pct": float(change), "max_change_pct": max_change_pct,
           "consistent": bool(change <= max_change_pct)}
    # where it happens matters: a ratio that holds through mainstage and moves only once the chamber pressure has
    # dropped (tail-off) reads differently from one that drifts while the engine is at full power
    ms = [w["ratio"] for w in rows if s.phase_window("mainstage")[0] <= w["t_center"] <= s.phase_window("mainstage")[1]] \
        if any(p_["name"] == "mainstage" for p_ in s.phases()) else []
    if len(ms) >= 3:
        res["mainstage_change_pct"] = float(100 * (max(ms) / min(ms) - 1))
    res["t_ratio_max_after_mainstage"] = bool(any(p_["name"] == "mainstage" for p_ in s.phases())
                                              and rows[hi]["t_center"] > s.phase_window("mainstage")[1])
    fig, ax = plots.new_fig(1, height=2.4)
    ax = ax[0]
    ax.plot([w["t_center"] for w in rows], r, "o-", color=plots.SERIES[0], ms=3)
    ax.set_ylabel(f"{thrust} / {pressure} [{res['ratio_unit']}]")
    ax.set_xlabel("time [s]")
    ax.set_title("Thrust over chamber pressure while burning (should stay roughly steady)")
    return res, plots.to_base64(fig)


# ---------------------------------------------------------------------------- valves
@tool(
    "measure_valve_response",
    "For each valve command edge, measure the latency until the paired response channel (e.g. injector "
    "pressure) departs from its pre-command baseline, and compare with max_latency_ms from the limits file.",
    {"command": {**_CH, "description": "only analyse this command channel (default all configured)"}},
)
def measure_valve_response(s: Session, command: str | None = None):
    cfg = s.limits.get("valve_response", {})
    if not cfg:
        cfg = {c: {"response": None, "max_latency_ms": None} for c in s.channels_of_kind("command")}
    t = s.time
    fs = s.fs
    events = []
    pairs = {k: v for k, v in cfg.items() if command is None or k == command}
    for cmd, c in pairs.items():
        resp = c.get("response")
        if cmd not in s.channels or not resp or resp not in s.channels:
            continue
        u = s.require_channel(cmd)
        y = _filled(s.require_channel(resp))
        span = float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1))
        d = np.diff(u)
        for i in np.where(np.abs(d) > 0.5)[0]:
            i_cmd = i + 1
            rising = d[i] > 0
            b0 = max(i_cmd - int(0.05 * fs), 0)
            base = y[b0:i_cmd]
            baseline = float(np.median(base))
            noise = float(np.std(base)) if base.size > 3 else 0.0
            thr = max(8 * noise, 0.03 * span)
            look = y[i_cmd : i_cmd + int(0.5 * fs)]
            hit = np.where(look > baseline + thr)[0] if rising else np.where(look < baseline - thr)[0]
            lat = float(hit[0] / fs * 1000) if hit.size else None
            lim = c.get("max_latency_ms")
            events.append(
                {
                    "command": cmd,
                    "response": resp,
                    "edge": "open" if rising else "close",
                    "t_cmd": float(t[i_cmd]),
                    "latency_ms": lat,
                    "t_response": float(t[i_cmd]) + lat / 1000 if lat is not None else None,
                    "limit_ms": lim,
                    "exceeds_limit": bool(lim is not None and (lat is None or lat > lim)),
                    "detection_threshold": thr,
                }
            )
    res = {"events": events, "n_exceeding": sum(e["exceeds_limit"] for e in events)}
    fig = None
    if events:
        opens = [e for e in events if e["edge"] == "open"]
        f, axes = plots.new_fig(1, height=3.0)
        ax = axes[0]
        if opens:
            t0 = min(e["t_cmd"] for e in opens) - 0.05
            t1 = max(e["t_response"] or e["t_cmd"] for e in opens) + 0.15
            m = (t >= t0) & (t <= t1)
            for k, e in enumerate(opens):
                col = plots.SERIES[k % len(plots.SERIES)]
                ax.plot(t[m], s.data[e["response"]].to_numpy()[m], color=col, label=e["response"])
                ax.axvline(e["t_cmd"], color=col, ls=":", lw=1.0)
                if e["t_response"]:
                    ax.annotate(f"{e['latency_ms']:.0f} ms", (e["t_response"], 0.9 - 0.12 * k),
                                xycoords=("data", "axes fraction"), color=plots.INK, fontsize=8)
                    ax.axvspan(e["t_cmd"], e["t_response"], color=plots.CRITICAL if e["exceeds_limit"] else col,
                               alpha=0.12, lw=0)
            ax.legend(loc="lower right")
            ax.set_ylabel(f"response [{_unit(s, opens[0]['response'])}]")
            ax.set_title("Valve opening response (dotted: command, shaded: latency)")
            ax.set_xlabel("time [s]")
        fig = plots.to_base64(f)
    return res, fig


# ---------------------------------------------------------------------------- reference comparison
@tool(
    "compare_reference",
    "Compare a measured channel with the simulation prediction (reference) over a phase or time window: "
    "mean deviation %, RMSE, maximum deviation, and intervals where the 0.2 s rolling deviation stays beyond "
    "tolerance_pct for at least 0.5 s.",
    {
        "channel": {**_CH, "description": "channel present in both data and reference (default Pc)"},
        "phase": {"type": "string", "description": "phase name (default mainstage)"},
        "t_start": _T,
        "t_end": _T1,
        "tolerance_pct": {"type": "number", "description": "allowed deviation in % (default from limits, 2 %)"},
    },
)
def compare_reference(
    s: Session,
    channel: str | None = None,
    phase: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    tolerance_pct: float | None = None,
):
    if s.reference is None:
        return {"error": "no reference (simulation prediction) loaded"}, None
    channel = channel or _primary_pressure(s)
    if channel not in s.reference.columns:
        return {"error": f"channel {channel!r} not in reference", "reference_channels": list(s.reference.columns)}, None
    tol = float(tolerance_pct if tolerance_pct is not None else s.limits.get("reference_tolerance_pct", 2.0))
    t_start, t_end = _default_window(s, t_start, t_end, phase or "mainstage")
    m = s.window(t_start, t_end)
    t = s.time[m]
    y = s.require_channel(channel)[m]
    r = np.interp(t, s.reference["time"].to_numpy(), s.reference[channel].to_numpy(dtype=float))
    ok = ~np.isnan(y)
    dev = y - r
    scale = float(np.mean(np.abs(r[ok]))) or 1.0
    dev_pct = 100 * dev / scale
    w = max(int(0.2 * s.fs), 1)
    roll = pd.Series(np.where(ok, dev_pct, np.nan)).rolling(w, center=True, min_periods=w // 2).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        beyond = np.abs(roll) > tol
    intervals = []
    for i0, i1 in _runs(beyond):
        if (t[i1 - 1] - t[i0]) >= 0.5:
            seg = roll[i0:i1]
            intervals.append({
                "t_start": float(t[i0]),
                "t_end": float(t[i1 - 1]),
                "duration_s": float(t[i1 - 1] - t[i0]),
                "mean_dev_pct": float(np.nanmean(seg)),
                "peak_dev_pct": float(seg[np.nanargmax(np.abs(seg))]),
            })
    j = int(np.nanargmax(np.abs(np.where(ok, dev, np.nan))))
    res = {
        "channel": channel,
        "unit": _unit(s, channel),
        "t_start": t_start,
        "t_end": t_end,
        "tolerance_pct": tol,
        "mean_measured": float(np.nanmean(y)),
        "mean_reference": float(np.mean(r[ok])),
        "mean_dev": float(np.nanmean(dev)),
        "mean_dev_pct": float(np.nanmean(dev_pct)),
        "rmse": float(np.sqrt(np.nanmean(dev**2))),
        "max_abs_dev": float(abs(dev[j])),
        "t_max_abs_dev": float(t[j]),
        "sustained_deviation_intervals": intervals,
        "within_tolerance": not intervals and abs(float(np.nanmean(dev_pct))) <= tol,
    }
    f, (ax1, ax2) = plots.new_fig(2, height=2.3)
    tm = s.time
    wide = s.window(t_start - 0.5, t_end + 0.5)
    ax1.plot(tm[wide], s.data[channel].to_numpy()[wide], color=plots.SERIES[0], lw=0.9, label="measured")
    ax1.plot(s.reference["time"], s.reference[channel], color=plots.SERIES[1], lw=1.4, ls="--", label="reference")
    ax1.set_xlim(t_start - 0.5, t_end + 0.5)
    ax1.set_ylabel(f"{channel} [{_unit(s, channel)}]")
    ax1.legend(loc="lower center", ncol=2)
    ax1.set_title(f"{channel}: measured vs simulation prediction")
    ax2.plot(t, roll, color=plots.SERIES[0])
    ax2.axhspan(-tol, tol, color=plots.GRID, alpha=0.6, lw=0)
    for iv in intervals:
        ax2.axvspan(iv["t_start"], iv["t_end"], color=plots.CRITICAL, alpha=0.12, lw=0)
    ax2.set_ylabel("deviation [%] (0.2 s mean)")
    ax2.set_xlabel("time [s]")
    return res, plots.to_base64(f)


# ---------------------------------------------------------------------------- pulse metrics
@tool(
    "pulse_metrics",
    "Metrics of one pulse on a channel (e.g. thrust of a solid motor, a flow transient): peak and its time, action "
    "time from the first crossing of start_pct of the peak to the last crossing of end_pct, the integral over the "
    "action time (total impulse for thrust in N), and the mean over it. The baseline (median of the 1 s before the "
    "start crossing) is subtracted when subtract_baseline is true. NaN samples are linearly interpolated.",
    {
        "channel": {**_CH, "description": "channel to measure (default: first force channel, else Pc)"},
        "start_pct": {"type": "number", "description": "action-time start threshold in % of peak (default 10)"},
        "end_pct": {"type": "number", "description": "action-time end threshold in % of peak (default 10)"},
        "subtract_baseline": {"type": "boolean", "description": "subtract the pre-pulse baseline (default true)"},
    },
)
def pulse_metrics(s: Session, channel: str | None = None, start_pct: float = 10.0, end_pct: float = 10.0,
                  subtract_baseline: bool = True):
    if channel is None:
        force = s.channels_of_kind("force")
        channel = force[0] if force else _primary_pressure(s)
    t = s.time
    raw = s.require_channel(channel)
    x = _filled(raw)
    w = max(int(0.02 * s.fs), 1)
    xs = pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
    # locate the pulse on a 0.1 s rolling median, so a shock or spike of a few milliseconds (a valve slam at
    # shutdown, load-cell ringing) cannot pass for the pulse itself
    xm = pd.Series(xs).rolling(max(int(0.1 * s.fs), 1), center=True, min_periods=1).median().to_numpy()
    i_ref = int(np.nanargmax(xm))
    pre = np.where(xm[:i_ref] < 0.5 * xm[i_ref])[0]
    i_pre = int(pre[-1]) if pre.size else 0
    base_win = (t >= t[i_pre] - 1.5) & (t < t[i_pre] - 0.5)
    baseline = float(np.median(x[base_win])) if subtract_baseline and base_win.any() else 0.0
    y = x - baseline
    ys = xs - baseline
    sustained = float(xm[i_ref] - baseline)

    def window(ref: float, i_c: int) -> tuple[int, int]:
        a = np.where(ys[: i_c + 1] < start_pct / 100 * ref)[0]
        b = np.where(ys[i_c:] < end_pct / 100 * ref)[0]
        return (int(a[-1]) + 1 if a.size else 0), (i_c + int(b[0]) - 1 if b.size else len(t) - 1)

    i0, i1 = window(sustained, i_ref)
    i_pk = i0 + int(np.nanargmax(xs[i0 : i1 + 1]))
    spiky = ys[i_pk] > 1.5 * sustained  # the highest sample is a short spike, not the pulse
    if not spiky:  # thresholds from the peak itself, as before
        i0, i1 = window(float(ys[i_pk]), i_pk)
        i_pk = i0 + int(np.nanargmax(xs[i0 : i1 + 1]))
    peak = float(y[i_pk])
    seg_t, seg_y = t[i0 : i1 + 1], y[i0 : i1 + 1]
    integral = float(_trapz(seg_y, seg_t))
    dur = float(t[i1] - t[i0])
    res = {
        "channel": channel,
        "unit": _unit(s, channel),
        "baseline": baseline,
        "peak": peak,
        "t_peak": float(t[i_pk]),
        "peak_sustained": sustained,
        "peak_is_short_spike": bool(spiky),
        "start_pct": start_pct,
        "end_pct": end_pct,
        "t_start": float(t[i0]),
        "t_end": float(t[i1]),
        "action_time_s": dur,
        "integral": integral,
        "integral_unit": f"{_unit(s, channel)}·s" if _unit(s, channel) else "·s",
        "mean_over_action_time": integral / dur if dur > 0 else None,
        "nan_samples_interpolated": int(np.isnan(raw[i0 : i1 + 1]).sum()),
    }
    f, (ax,) = plots.new_fig(1, height=2.8)
    m = s.window(t[i0] - 0.5, t[i1] + 0.5)
    ax.plot(t[m], y[m], color=plots.SERIES[0], lw=1.0)
    ax.fill_between(seg_t, 0, seg_y, color=plots.SERIES[0], alpha=0.15, lw=0)
    ax.axhline(peak * start_pct / 100, color=plots.INK_2, ls=":", lw=0.8)
    ax.set_ylabel(f"{channel} [{_unit(s, channel)}]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"{channel}: action time {dur:.2f} s, integral {integral:.4g} {res['integral_unit']}")
    return res, plots.to_base64(f)


# ---------------------------------------------------------------------------- drill-down
@tool(
    "channel_stats",
    "Statistics of one channel in a time window or phase: mean, std, min, max, peak-to-peak, NaN count.",
    {"channel": _CH, "t_start": _T, "t_end": _T1, "phase": {"type": "string", "description": "phase name"}},
    required=["channel"],
)
def channel_stats(s: Session, channel: str, t_start: float | None = None, t_end: float | None = None,
                  phase: str | None = None):
    if phase or (t_start is None and t_end is None):
        t_start, t_end = _default_window(s, t_start, t_end, phase or "mainstage")
    x = s.require_channel(channel)[s.window(t_start, t_end)]
    return (
        {
            "channel": channel,
            "unit": _unit(s, channel),
            "t_start": t_start,
            "t_end": t_end,
            "n_samples": int(x.size),
            "nan_samples": int(np.isnan(x).sum()),
            "mean": np.nanmean(x),
            "std": np.nanstd(x),
            "min": np.nanmin(x),
            "max": np.nanmax(x),
            "peak_to_peak": np.nanmax(x) - np.nanmin(x),
        },
        None,
    )


@tool(
    "plot_window",
    "Plot up to 4 channels over a time window to attach as visual evidence; returns per-channel min/max/mean "
    "in that window.",
    {
        "channels": {"type": "array", "items": {"type": "string"}, "description": "channels to plot"},
        "t_start": {"type": "number", "description": "start time in s"},
        "t_end": {"type": "number", "description": "end time in s"},
    },
    required=["channels", "t_start", "t_end"],
)
def plot_window(s: Session, channels: list[str], t_start: float, t_end: float):
    channels = list(channels)[:4]
    m = s.window(t_start, t_end)
    t = s.time[m]
    f, axes = plots.new_fig(len(channels), height=1.9)
    stats = {}
    for k, (ax, ch) in enumerate(zip(axes, channels)):
        x = s.require_channel(ch)[m]
        ax.plot(t, x, color=plots.SERIES[k], lw=0.9)
        ax.set_ylabel(f"{ch} [{_unit(s, ch)}]")
        stats[ch] = {"min": np.nanmin(x), "max": np.nanmax(x), "mean": np.nanmean(x)}
    axes[0].set_title(f"Channels {t_start:.3f}–{t_end:.3f} s")
    axes[-1].set_xlabel("time [s]")
    return {"t_start": t_start, "t_end": t_end, "stats": stats}, plots.to_base64(f)
