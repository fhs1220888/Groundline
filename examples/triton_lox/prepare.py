"""Turn a public liquid-engine hot fire (Triton, LOX / fuel, pressure fed) into a Groundline run.

Source: https://github.com/aidenmccollum/Triton-Hotfire-Analysis, file ``Apr_11_TritonHF.csv``
(hot fire of 11 April 2025), pinned to commit ``5a33fe8``. The repository has no license, so the data
is not copied into Groundline: this script downloads it into ``raw/`` (git-ignored) and writes
``run.csv`` / ``meta.json`` next to itself (also git-ignored).

The CSV is one DAQ log: ~1.94 kHz, 25 channels (3 thrust load cells, chamber, manifold, tank and
regulator pressures, tank weights, thermocouples), 0-209 s, the burn at about 148.4-153.3 s (the
team's own analysis script uses 148.35-153.35 s).

What this script does, and every assumption it makes:

1. Keep 120-180 s: pressurisation before, the burn, and the minutes after are all in it; the full
   log is ~400 k rows.
2. Put every channel on a uniform 2 kHz grid by linear interpolation. The log has ~470 timestamp
   gaps longer than 10 ms (up to 68 ms); grid points farther than ``MAX_GAP_S`` from any logged
   sample stay NaN, so the dropouts stay visible instead of being interpolated over.
3. Thrust: ``F_thrust`` is the sum of the three load cells, as in the team's script (which also
   uses the raw lbf values without further calibration). The cells are kept as ``F_lc_a/b/c``.
4. Units are left as logged (psi, lbf, degC). Tank weights are kind ``weight``, not ``force``, so
   pulse-shaped thrust tools do not pick them up.
5. No redlines, valve commands or simulation prediction: none are published with the data.

Run from the repository root::

    python examples/triton_lox/prepare.py
    groundline analyze examples/triton_lox/run.csv
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
URL = ("https://raw.githubusercontent.com/aidenmccollum/Triton-Hotfire-Analysis/"
       "5a33fe8e0da17105c166c59369545e0585db1d44/Apr_11_TritonHF.csv")
FS = 2000.0  # Hz, grid rate (the DAQ logs at ~1.94 kHz)
T0, T1 = 120.0, 180.0  # s, window kept
MAX_GAP_S = 0.002  # leave the grid NaN where the nearest logged sample is farther than this

# logged column -> (Groundline name, unit, kind, description)
CHANNELS = {
    "Chamber Pressure (psi)": ("Pc", "psi", "pressure", "chamber pressure"),
    "LOX Engine Manifold (psi)": ("P_lox_man", "psi", "pressure", "LOX engine manifold (injector inlet)"),
    "Fuel Engine Manifold (psi)": ("P_fu_man", "psi", "pressure", "fuel engine manifold (injector inlet)"),
    "LOX Tank Downstream (psi)": ("P_lox_tank", "psi", "pressure", "LOX tank, downstream"),
    "Fuel Tank Upstream (psi)": ("P_fu_tank_up", "psi", "pressure", "fuel tank, upstream"),
    "Fuel Tank Downstream (psi)": ("P_fu_tank", "psi", "pressure", "fuel tank, downstream"),
    "LOX Reg Upstream (psi)": ("P_lox_reg_up", "psi", "pressure", "LOX pressurant regulator, upstream"),
    "LOX Reg Downstream (psi)": ("P_lox_reg_dn", "psi", "pressure", "LOX pressurant regulator, downstream"),
    "Fuel Reg Upstream (psi)": ("P_fu_reg_up", "psi", "pressure", "fuel pressurant regulator, upstream"),
    "Fuel Reg Downstream (psi)": ("P_fu_reg_dn", "psi", "pressure", "fuel pressurant regulator, downstream"),
    "Muscle Bus (psi)": ("P_muscle", "psi", "pressure", "pneumatic valve actuation bus"),
    "Purge Bus (psi)": ("P_purge", "psi", "pressure", "purge bus"),
    "Fuel Tank Weight (lbf)": ("W_fu_tank", "lbf", "weight", "fuel tank weight"),
    "LOX Tank Weight (lbf)": ("W_lox_tank", "lbf", "weight", "LOX tank weight"),
    "LOX Tank Top (°C)": ("T_lox_tank_top", "degC", "temperature", "LOX tank, top"),
    "LOX Tank Middle (°C)": ("T_lox_tank_mid", "degC", "temperature", "LOX tank, middle"),
    "LOX Tank Bottom (°C)": ("T_lox_tank_bot", "degC", "temperature", "LOX tank, bottom"),
    "Chamber Housing (°C)": ("T_chamber", "degC", "temperature", "chamber housing"),
    "LOX Dragon (°C)": ("T_lox_dragon", "degC", "temperature", "LOX line thermocouple ('LOX Dragon')"),
    "Ignitor (°C)": ("T_ignitor", "degC", "temperature", "igniter"),
    "CJC NTC Thermistor (°C)": ("T_cjc", "degC", "housekeeping", "DAQ cold-junction thermistor"),
}
LOAD_CELLS = {"Thrust LC A (lbf)": "F_lc_a", "Thrust LC B (lbf)": "F_lc_b", "Thrust LC C (lbf)": "F_lc_c"}


def download() -> Path:
    path = RAW / "Apr_11_TritonHF.csv"
    if not path.exists():
        RAW.mkdir(exist_ok=True)
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    return path


def main() -> None:
    df = pd.read_csv(download())
    t = df["Time (s)"].to_numpy(dtype=float)
    keep = (t >= T0 - 1) & (t <= T1 + 1)
    df, t = df[keep], t[keep]
    grid = np.arange(T0, T1, 1.0 / FS)
    # distance from each grid point to the nearest logged sample, to blank out dropouts
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
    gap = near > MAX_GAP_S

    out = {"time": np.round(grid - T0, 6)}
    cells = []
    for col, name in LOAD_CELLS.items():
        out[name] = np.interp(grid, t, df[col].to_numpy(dtype=float))
        cells.append(out[name])
    out["F_thrust"] = np.sum(cells, axis=0)
    for col, (name, *_rest) in CHANNELS.items():
        out[name] = np.interp(grid, t, df[col].to_numpy(dtype=float))
    run = pd.DataFrame(out)
    run.loc[gap, run.columns != "time"] = np.nan
    order = ["time", "Pc", "F_thrust", *LOAD_CELLS.values(), *[v[0] for v in CHANNELS.values() if v[0] != "Pc"]]
    run = run[order]

    rate = float(1.0 / np.median(np.diff(t)))
    channels = {"F_thrust": {"unit": "lbf", "kind": "force", "desc": "thrust, sum of the three load cells",
                             "native_rate_hz": round(rate, 1)}}
    for name in LOAD_CELLS.values():
        channels[name] = {"unit": "lbf", "kind": "force_component", "desc": f"thrust load cell {name[-1].upper()}",
                          "native_rate_hz": round(rate, 1)}
    for name, unit, kind, desc in CHANNELS.values():
        channels[name] = {"unit": unit, "kind": kind, "desc": desc, "native_rate_hz": round(rate, 1)}
    meta = {
        "test_id": "Triton HF 2025-04-11",
        "description": "Pressure-fed LOX/fuel liquid rocket engine hot fire (Triton). Public data from "
                       "github.com/aidenmccollum/Triton-Hotfire-Analysis (no license; not redistributed).",
        "engine_type": "liquid bipropellant, pressure fed (no valve command channels, no simulation prediction)",
        "time_base": f"seconds from {T0:g} s of the original log",
        "sample_rate_hz": FS,
        "channels": {k: channels[k] for k in order if k != "time"},
        "preparation": {"source": URL, "window_s": [T0, T1], "grid_hz": FS, "max_gap_s": MAX_GAP_S,
                        "grid_points_blanked": int(gap.sum())},
    }
    run.to_csv(HERE / "run.csv", index=False, float_format="%.6g")
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {HERE / 'run.csv'} ({len(run)} rows, {int(gap.sum())} grid points blanked) and meta.json")


if __name__ == "__main__":
    main()
