"""Loading and saving test data (CSV, NI TDMS) plus sidecar metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_time_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("time", "t", "time_s", "t_s", "timestamp"):
            return c
    return df.columns[0]


def load_run(path: str | Path, meta_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load a test run into a DataFrame with a ``time`` column (seconds).

    Metadata is read from ``meta_path`` or a ``meta.json`` next to the file, if present.
    """
    path = Path(path)
    if path.suffix.lower() == ".tdms":
        df, meta = read_tdms(path)
    else:
        df = pd.read_csv(path)
        tcol = _find_time_column(df)
        df = df.rename(columns={tcol: "time"})
        meta = {}
    mp = Path(meta_path) if meta_path else path.with_name("meta.json")
    if mp.exists():
        meta = {**meta, **json.loads(mp.read_text())}
    df = df.sort_values("time").reset_index(drop=True)
    if "sample_rate_hz" not in meta and len(df) > 1:
        meta["sample_rate_hz"] = float(1.0 / np.median(np.diff(df["time"].to_numpy())))
    meta.setdefault("channels", {})
    for c in df.columns:
        if c == "time":
            continue
        meta["channels"].setdefault(c, {"unit": "", "kind": _guess_kind(c)})
    meta.setdefault("test_id", path.stem)
    return df, meta


def _guess_kind(name: str) -> str:
    n = name.lower()
    if n.startswith("cmd") or n.endswith("_cmd"):
        return "command"
    if n.startswith(("p", "pc")):
        return "pressure"
    if n.startswith("t"):
        return "temperature"
    if "vib" in n or "acc" in n:
        return "vibration"
    if "mdot" in n or "flow" in n:
        return "flow"
    return "other"


def load_reference(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    return df.rename(columns={_find_time_column(df): "time"}).sort_values("time").reset_index(drop=True)


def load_limits(path: str | Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def read_tdms(path: str | Path) -> tuple[pd.DataFrame, dict]:
    try:
        from nptdms import TdmsFile
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading TDMS needs `pip install veritest[tdms]` (nptdms)") from e
    f = TdmsFile.read(str(path))
    cols: dict[str, np.ndarray] = {}
    units: dict[str, dict] = {}
    time = None
    for group in f.groups():
        for ch in group.channels():
            name = ch.name
            if name.lower() == "time":
                time = ch[:]
                continue
            cols[name] = ch[:]
            units[name] = {"unit": ch.properties.get("unit_string", ""), "kind": _guess_kind(name)}
            if time is None and "wf_increment" in ch.properties:
                n = len(ch)
                time = ch.properties.get("wf_start_offset", 0.0) + np.arange(n) * ch.properties["wf_increment"]
    if time is None:
        raise ValueError("TDMS file has no 'time' channel and no wf_increment property")
    df = pd.DataFrame({"time": time, **cols})
    return df, {"channels": units}


def write_tdms(df: pd.DataFrame, path: str | Path, meta: dict | None = None) -> Path:
    try:
        from nptdms import ChannelObject, TdmsWriter
    except ImportError as e:  # pragma: no cover
        raise ImportError("Writing TDMS needs `pip install veritest[tdms]` (nptdms)") from e
    path = Path(path)
    meta = meta or {}
    dt = float(np.median(np.diff(df["time"].to_numpy())))
    chans = []
    for c in df.columns:
        if c == "time":
            continue
        props = {
            "wf_increment": dt,
            "wf_start_offset": float(df["time"].iloc[0]),
            "unit_string": meta.get("channels", {}).get(c, {}).get("unit", ""),
        }
        chans.append(ChannelObject("hotfire", c, df[c].to_numpy(dtype=float), properties=props))
    with TdmsWriter(str(path)) as w:
        w.write_segment(chans)
    return path
