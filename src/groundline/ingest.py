"""Turn a raw test log (CSV) into a Groundline run, from a mapping instead of a hand-written script.

A mapping is a small JSON document::

    {
      "skip_rows": 0,                         # rows above the header to skip (a units row above the names: 1)
      "units_row_below_header": false,        # a units row right under the names (skipped when reading data)
      "time": {"column": "Time (s)", "scale": 1.0},
      "channels": {
        "Pc": {"column": "Chamber Pressure (psi)", "unit": "psi", "kind": "pressure", "desc": "chamber pressure"},
        "F_lc_a": {"column": "Thrust LC A (lbf)", "unit": "lbf", "kind": "force_component"}
      },
      "derived": {"F_thrust": {"sum": ["F_lc_a", "F_lc_b"], "unit": "lbf", "kind": "force"}},
      "window_s": [120, 180],                 # optional: keep this span of the log; output time starts at its start
      "grid_hz": 2000,                        # optional: resample onto a uniform grid ...
      "max_gap_s": 0.002,                     # ... leaving NaN where no logged sample is this close
      "test_id": "...", "description": "..."
    }

Channels may also carry ``scale`` and ``offset`` (value = raw * scale + offset). ``suggest_map`` drafts a
mapping from the header alone: units in parentheses or brackets, or a units row above the names; the time
column; channel kinds from units and keywords; short Groundline names. Read the draft before using it.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_UNIT_IN_NAME = re.compile(r"^(.*?)\s*[(\[]\s*([^()\[\]]*?)\s*[)\]]\s*$")
_PRESSURE_U = {"psi", "psia", "psig", "bar", "bara", "barg", "mbar", "pa", "kpa", "mpa"}
_TEMP_U = {"c", "°c", "degc", "℃", "k", "f", "°f", "degf"}
_FORCE_U = {"n", "kn", "lbf", "kgf"}
_MASS_U = {"kg", "g", "lb", "lbm"}
_TIME_NAMES = ("time", "t", "seconds", "time_s", "t_s", "elapsed", "elapsed_s")


def _split_unit(name: str) -> tuple[str, str]:
    m = _UNIT_IN_NAME.match(name.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (name.strip(), "")


def _norm_unit(u: str) -> str:
    return {"°c": "degC", "℃": "degC", "c": "degC", "degc": "degC", "°f": "degF", "f": "degF"}.get(u.lower(), u)


def _slug(words: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", words)).strip("_").lower()


def _kind(label: str, unit: str) -> str:
    lo, u = label.lower(), unit.lower()
    if lo in ("pc", "p_c", "pcc"):  # the usual short name of chamber pressure, whatever its unit
        return "pressure"
    if any(k in lo for k in ("cjc", "thermistor", "battery", "supply", "voltage")):
        return "housekeeping"
    if any(k in lo for k in ("cmd", "command", "valve state", "solenoid")):
        return "command"
    if "weight" in lo or (u in _MASS_U and "thrust" not in lo):
        return "weight"
    if "thrust" in lo or "load cell" in lo or re.search(r"\blc\b", lo) or u in _FORCE_U:
        return "force"
    if u in _PRESSURE_U or "pressure" in lo:
        return "pressure"
    if u in _TEMP_U or "temp" in lo:
        return "temperature"
    return "other"


def _name(label: str, kind: str, used: set[str]) -> str:
    lo = label.lower()
    label = re.sub(r"^[A-Za-z]_", "", label)  # P_INJECTOR, T_RUN_TANK: the type letter is added back below
    if kind == "pressure" and ("chamber" in lo or "comb" in lo or lo in ("pc", "p_c", "pcc")) and "Pc" not in used:
        base = "Pc"
    else:
        prefix = {"pressure": "P", "temperature": "T", "force": "F", "weight": "W", "command": "cmd"}.get(kind, "")
        words = _slug(re.sub(r"\b(pressure|temperature|temp|weight|thrust)\b", "", label, flags=re.I)) or _slug(label)
        base = f"{prefix}_{words}" if prefix else words
    name, k = base, 2
    while name in used:
        name, k = f"{base}_{k}", k + 1
    used.add(name)
    return name


def _read_head(path: Path, n: int = 3) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for _, row in zip(range(n), csv.reader(f))]


_UNIT_TOKENS = _PRESSURE_U | _TEMP_U | _FORCE_U | _MASS_U | {"s", "sec", "ms", "us", "hz", "khz", "v", "mv", "a", "ma",
                                                              "%", "g", "m/s", "kg/s", "g/s", "lb/s", "rpm", "-", "deg"}


def _unit_like(cells: list[str]) -> bool:
    """A row of units: every non-empty cell a known unit (or bracketed), at least two of them."""
    toks = [c.strip().strip("[]()").lower() for c in cells if c.strip()]
    return len(toks) >= 2 and sum(t in _UNIT_TOKENS for t in toks) >= 0.8 * len(toks)


def _numeric(cells: list[str]) -> float:
    ok = 0
    for c in cells:
        try:
            float(c)
            ok += 1
        except ValueError:
            pass
    return ok / max(len(cells), 1)


def _hint(name: str) -> str | None:
    lo = name.lower()
    if lo in _TIME_NAMES or "time" in lo:
        return "time"
    if lo.startswith(("p_", "pt")) or "pressure" in lo:
        return "pressure"
    if lo.startswith(("t_", "tc")) or "temp" in lo:
        return "temperature"
    if lo.startswith(("l_", "lc", "f_")) or any(k in lo for k in ("thrust", "load", "weight")):
        return "load"
    return None


def _fits(hint: str | None, unit: str) -> bool:
    u = unit.strip().lower()
    return {"time": u in ("s", "sec", "ms"), "pressure": u in _PRESSURE_U, "temperature": u in _TEMP_U,
            "load": u in _FORCE_U | _MASS_U}.get(hint, False)


def _align_units(header: list[str], units: list[str]) -> list[str]:
    """Shift a units row against the names by the offset under which the units fit the names best."""
    best, best_score = units, -1
    for off in range(-3, 4):
        shifted = [units[i + off] if 0 <= i + off < len(units) else "" for i in range(len(header))]
        score = sum(_fits(_hint(h), u) for h, u in zip(header, shifted))
        if score > best_score:
            best, best_score = shifted, score
    return best


def suggest_map(path: str | Path) -> dict:
    """Draft a mapping from the first rows of a CSV log."""
    path = Path(path)
    rows = _read_head(path)
    units: list[str] = []
    skip, below = 0, False
    two_text = len(rows) >= 3 and _numeric(rows[0]) < 0.5 and _numeric(rows[1]) < 0.5 and _numeric(rows[2]) >= 0.5
    if two_text and _unit_like(rows[0]) and not _unit_like(rows[1]):  # units above the names (UVic)
        units, header, skip = rows[0], rows[1], 1
    elif two_text and _unit_like(rows[1]):  # names, then a units row (common DAQ export)
        units, header, below = rows[1], rows[0], True
    else:
        header = rows[0]
    if units and len(units) != len(header):  # misaligned units rows exist (an extra leading or trailing cell)
        units = _align_units(header, units)
    cols: list[tuple[str, str, str]] = []  # (column, label, unit)
    for i, col in enumerate(header):
        label, unit = _split_unit(col)
        if not unit and units and i < len(units):
            unit = units[i].strip()
        cols.append((col, label, _norm_unit(unit)))
    time_col = next((c for c, lab, _ in cols if lab.lower() in _TIME_NAMES), None)
    if time_col is None:
        time_col = next((c for c, lab, _ in cols if "time" in lab.lower()), cols[0][0])
    scale = 1e-3 if re.search(r"\bms\b", next(u for c, _, u in cols if c == time_col) or "") else 1.0
    used: set[str] = set()
    channels: dict[str, dict] = {}
    for col, label, unit in cols:
        if col == time_col or not label or label.lower().startswith("unnamed") or label.lower() in (
                "timestamp", "date", "datetime", "index"):
            continue
        kind = _kind(label, unit)
        channels[_name(label, kind, used)] = {"column": col, "unit": unit, "kind": kind, "desc": label}
    out: dict = {"skip_rows": skip, **({"units_row_below_header": True} if below else {}),
                 "time": {"column": time_col, "scale": scale}, "channels": channels}
    forces = [n for n, c in channels.items() if c["kind"] == "force"]
    if len(forces) == 1 and "F_thrust" not in channels:  # the one thrust channel gets the usual name
        channels = {("F_thrust" if n == forces[0] else n): c for n, c in channels.items()}
        out["channels"] = channels
    elif len(forces) > 1:  # several load cells: thrust is their sum
        for n in forces:
            channels[n]["kind"] = "force_component"
        out["derived"] = {"F_thrust": {"sum": forces, "unit": channels[forces[0]]["unit"], "kind": "force",
                                       "desc": "thrust, sum of " + ", ".join(forces)}}
    out["test_id"] = path.stem
    return out


def ingest(raw: str | Path, mapping: dict, out_dir: str | Path, float_format: str = "%.6g") -> dict[str, Path]:
    """Write ``run.csv`` and ``meta.json`` for ``raw`` according to ``mapping``."""
    raw, out_dir = Path(raw), Path(out_dir)
    skip = int(mapping.get("skip_rows", 0))
    rows = list(range(skip)) + ([skip + 1] if mapping.get("units_row_below_header") else [])
    df = pd.read_csv(raw, skiprows=rows)
    tcfg = mapping["time"]
    t_all = pd.to_numeric(df[tcfg["column"]], errors="coerce").to_numpy(dtype=float) * float(tcfg.get("scale", 1.0))
    cols = {name: pd.to_numeric(df[c["column"]], errors="coerce").to_numpy(dtype=float) * float(c.get("scale", 1.0))
            + float(c.get("offset", 0.0)) for name, c in mapping["channels"].items()}
    # sort, drop rows without a time, and average samples that share a timestamp (some DAQs repeat them),
    # since interpolation needs strictly increasing sample times
    merged = pd.DataFrame({"__t": t_all, **cols}).dropna(subset=["__t"]).groupby("__t", sort=True).mean()
    t = merged.index.to_numpy(dtype=float)
    vals: dict[str, np.ndarray] = {name: merged[name].to_numpy(dtype=float) for name in cols}
    win = mapping.get("window_s")
    t0 = float(win[0]) if win else float(t[0])
    near = (t >= win[0] - 1) & (t <= win[1] + 1) if win else np.ones(len(t), bool)
    native = float(1.0 / np.median(np.diff(t[near])))  # logged rate over the span kept
    if mapping.get("grid_hz"):
        fs = float(mapping["grid_hz"])
        keep = (t >= (win[0] - 1 if win else -np.inf)) & (t <= (win[1] + 1 if win else np.inf))
        tk = t[keep]
        grid = np.arange(t0, float(win[1]) if win else float(tk[-1]), 1.0 / fs)
        j = np.clip(np.searchsorted(tk, grid), 1, len(tk) - 1)
        gap = np.minimum(np.abs(grid - tk[j - 1]), np.abs(tk[j] - grid)) > float(mapping.get("max_gap_s", np.inf))
        out = {"time": np.round(grid - t0, 6)}
        for name, x in vals.items():
            out[name] = np.interp(grid, tk, x[keep])
    else:
        keep = (t >= win[0]) & (t < win[1]) if win else np.ones(len(t), bool)
        gap = np.zeros(int(keep.sum()), bool)
        out = {"time": t[keep] - (t0 if win else 0.0)}
        out.update({name: x[keep] for name, x in vals.items()})
    for name, d in (mapping.get("derived") or {}).items():
        out[name] = np.sum([out[c] for c in d["sum"]], axis=0)
    run = pd.DataFrame(out)
    run.loc[gap, run.columns != "time"] = np.nan
    derived = list((mapping.get("derived") or {}))
    order = ["time"] + [c for c in mapping.get("order", []) if c in run] + \
        [c for c in [*derived, *mapping["channels"]] if c not in mapping.get("order", [])]
    run = run[list(dict.fromkeys(order))]
    chmeta = {}
    for name in run.columns[1:]:
        c = (mapping.get("derived") or {}).get(name) or mapping["channels"][name]
        chmeta[name] = {"unit": c.get("unit", ""), "kind": c.get("kind", "other"),
                        **({"desc": c["desc"]} if c.get("desc") else {}), "native_rate_hz": round(native, 1)}
    meta = {
        "test_id": mapping.get("test_id", raw.stem),
        **({"description": mapping["description"]} if mapping.get("description") else {}),
        **({"engine_type": mapping["engine_type"]} if mapping.get("engine_type") else {}),
        **({"time_base": mapping["time_base"]} if mapping.get("time_base") else {}),
        "sample_rate_hz": float(mapping["grid_hz"]) if mapping.get("grid_hz") else round(native, 3),
        "channels": chmeta,
        "preparation": {k: v for k, v in {"source": mapping.get("source", str(raw)), "window_s": win,
                                          "grid_hz": mapping.get("grid_hz"), "max_gap_s": mapping.get("max_gap_s"),
                                          "grid_points_blanked": int(gap.sum())}.items() if v is not None},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    run.to_csv(out_dir / "run.csv", index=False, float_format=float_format)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return {"run": out_dir / "run.csv", "meta": out_dir / "meta.json"}
