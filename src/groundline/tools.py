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
            "min": np.nanmin(x),
            "max": np.nanmax(x),
            "mean": np.nanmean(x),
            "nan_samples": int(np.isnan(x).sum()),
        }
    t = s.time
    return (
        {
            "test_id": s.meta.get("test_id"),
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
def compute_phases(s: Session, channel: str | None = None, on_frac: float = 0.1, steady_frac: float = 0.9) -> dict:
    ch = channel or _primary_pressure(s)
    t = s.time
    x = _filled(s.require_channel(ch))
    w = max(int(0.02 * s.fs), 1)
    xs = pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
    peak = float(np.nanmax(xs))
    level = float(np.nanmedian(xs[xs > 0.5 * peak]))
    on = np.where(xs > on_frac * level)[0]
    steady = np.where(xs > steady_frac * level)[0]
    if on.size == 0 or steady.size == 0 or level <= 0:
        return {
            "channel": ch,
            "fired": False,
            "phases": [{"name": "no_firing", "t_start": float(t[0]), "t_end": float(t[-1])}],
        }
    i_ign, i_ms0, i_ms1 = int(on[0]), int(steady[0]), int(steady[-1])
    after = np.where(xs[i_ms1:] < on_frac * level)[0]
    i_tail = i_ms1 + int(after[0]) if after.size else len(t) - 1
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
        "fired": True,
        "steady_level": level,
        "unit": _unit(s, ch),
        "ignition_s": float(t[i_ign]),
        "mainstage_start_s": float(t[i_ms0]),
        "mainstage_end_s": float(t[i_ms1]),
        "tail_off_end_s": float(t[i_tail]),
        "mainstage_duration_s": float(t[i_ms1] - t[i_ms0]),
        "phases": [{"name": n, "t_start": float(a), "t_end": float(b)} for n, a, b in bounds],
        "command_events": sorted(cmd_events, key=lambda e: e["t_s"]),
    }


@tool(
    "segment_phases",
    "Split the run into pre_test / startup / mainstage / shutdown / post_test from the chamber-pressure trace "
    "(10 % and 90 % of steady level). Returns ignition, mainstage start/end and command edge times.",
    {"channel": {**_CH, "description": "pressure channel to segment on (default Pc)"}},
)
def segment_phases(s: Session, channel: str | None = None):
    res = compute_phases(s, channel)
    if channel is None:
        s._cache["phases"] = res
    fig, (ax,) = plots.new_fig(1, height=3.0)
    plots.shade_phases(ax, res["phases"], label=True)
    ch = res["channel"]
    ax.plot(s.time, s.data[ch], color=plots.SERIES[0], lw=1.0, label=ch)
    for e in res.get("command_events", []):
        ax.axvline(e["t_s"], color=plots.INK_2, lw=0.7, ls=":")
    ax.set_ylabel(f"{ch} [{_unit(s, ch)}]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"Phase segmentation on {ch} (dotted: valve commands)", pad=14)
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
    "with the same frequency are merged into events with frequency and amplitude.",
    {
        "channel": {**_CH, "description": "channel to analyse (default Pc)"},
        "t_start": _T,
        "t_end": _T1,
        "fmin": {"type": "number", "description": "lower frequency bound in Hz"},
        "fmax": {"type": "number", "description": "upper frequency bound in Hz"},
        "threshold_pct": {"type": "number", "description": "zero-to-peak amplitude threshold in % of mean"},
        "window_s": {"type": "number", "description": "FFT window length in seconds (default 0.1)"},
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
):
    cfg = s.limits.get("oscillation", {})
    channel = channel or _primary_pressure(s)
    fmin = float(fmin if fmin is not None else cfg.get("fmin_hz", 50.0))
    fmax = float(fmax if fmax is not None else cfg.get("fmax_hz", 2000.0))
    thr = float(threshold_pct if threshold_pct is not None else cfg.get("threshold_pct", 0.5))
    t_start, t_end = _default_window(s, t_start, t_end)
    fs = s.fs
    fmax = min(fmax, 0.45 * fs)
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
        flagged = prom >= prominence and (amp_pct is None or amp_pct >= thr)
        rows.append(
            {
                "t_center": float(t[i0 + n // 2]),
                "freq_hz": float((k + delta) * df),
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
    valid = [r for r in rows if not r.get("skipped")]
    worst = max(valid, key=lambda r: r["amp"]) if valid else None
    res = {
        "channel": channel,
        "unit": _unit(s, channel),
        "t_start": t_start,
        "t_end": t_end,
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "threshold_pct": thr,
        "prominence_min": prominence,
        "window_s": n / fs,
        "freq_resolution_hz": df,
        "n_windows": len(rows),
        "n_skipped_windows": sum(1 for r in rows if r.get("skipped")),
        "detected": bool(out_events),
        "events": out_events,
        "max_window": None
        if worst is None
        else {k: worst[k] for k in ("t_center", "freq_hz", "amp", "amp_pct", "prominence")},
    }
    # figure: spectrogram + amplitude track
    f, (ax1, ax2) = plots.new_fig(2, height=2.4)
    i_a, i_b = i_lo, max(i_hi, i_lo + n)
    seg = x_all[i_a:i_b] - pd.Series(x_all[i_a:i_b]).rolling(n, center=True, min_periods=1).mean().to_numpy()
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
@tool(
    "check_sensor_health",
    "Look for instrumentation problems on measurement channels: NaN gaps, flatlines (value exactly stuck for "
    ">= 50 ms) and isolated spikes (>12 local robust sigmas, <= 5 samples wide).",
    {"channels": {"type": "array", "items": {"type": "string"}, "description": "channels to check (default all measurements)"}},
)
def check_sensor_health(s: Session, channels: list[str] | None = None, spike_sigma: float = 12.0):
    t = s.time
    fs = s.fs
    chans = channels or [c for c in s.channels if s.channel_info(c).get("kind") != "command"]
    issues, summary = [], {}
    min_flat = max(int(0.05 * fs), 3)
    for ch in chans:
        x = s.require_channel(ch)
        nan = np.isnan(x)
        ch_issues = []
        for i0, i1 in _runs(nan):
            ch_issues.append({"channel": ch, "kind": "nan_gap", "t_start": float(t[i0]),
                              "t_end": float(t[i1 - 1]), "duration_s": float(t[i1 - 1] - t[i0]), "n_samples": i1 - i0})
        same = np.concatenate([[False], np.diff(x) == 0])
        for i0, i1 in _runs(same):
            i0 -= 1
            if i1 - i0 >= min_flat:
                ch_issues.append({"channel": ch, "kind": "flatline", "t_start": float(t[i0]), "t_end": float(t[i1 - 1]),
                                  "duration_s": float(t[i1 - 1] - t[i0]), "stuck_value": float(x[i0])})
        xf = _filled(x)
        med = signal.medfilt(xf, 11)
        res = np.abs(xf - med)
        w = max(int(0.05 * fs), 11)
        local = pd.Series(res).rolling(w, center=True, min_periods=1).median().to_numpy() * 1.4826
        floor = max(float(np.median(res)) * 1.4826, 1e-12)
        sigma = np.maximum(local, floor)
        near_nan = np.convolve(nan.astype(float), np.ones(2 * int(0.01 * fs) + 1), mode="same") > 0
        flag = (res > spike_sigma * sigma) & ~near_nan
        groups = _runs(flag)
        merged: list[list[int]] = []
        gap = int(0.005 * fs)
        for i0, i1 in groups:
            if merged and i0 - merged[-1][1] <= gap:
                merged[-1][1] = i1
            else:
                merged.append([i0, i1])
        spikes = []
        for i0, i1 in merged:
            if i1 - i0 > 5:
                continue
            j = i0 + int(np.argmax(res[i0:i1]))
            spikes.append({"t": float(t[j]), "value": float(x[j]), "local_median": float(med[j]),
                           "deviation": float(x[j] - med[j]), "sigmas": float(res[j] / sigma[j])})
        if spikes:
            ch_issues.append({"channel": ch, "kind": "spike", "t_start": spikes[0]["t"], "t_end": spikes[-1]["t"],
                              "count": len(spikes), "spikes": spikes[:20]})
        issues.extend(ch_issues)
        summary[ch] = {
            "nan_samples": int(nan.sum()),
            "flatline_s": float(sum(i["duration_s"] for i in ch_issues if i["kind"] == "flatline")),
            "spikes": len(spikes),
            "ok": not ch_issues,
        }
    res = {"checked": chans, "issues": issues, "n_issues": len(issues), "summary": summary,
           "criteria": {"spike_sigma": spike_sigma, "spike_max_width_samples": 5,
                        "flatline_min_s": min_flat / fs, "nan_margin_s": 0.01}}
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
