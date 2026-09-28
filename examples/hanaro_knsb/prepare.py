"""Turn SNU HANARO's KNSB static-fire sample data into a Groundline run.

Source: https://github.com/snu-hanaro/static-fire-toolkit (examples/, MIT License,
(c) 2025 SNU Rocket Team HANARO). Files in ``raw/`` are copied unchanged:

* ``KNSB_250220_thrust_raw.csv``   load-cell amplifier output, two rows (time [s], voltage [V]),
  irregular sampling with repeated timestamps
* ``KNSB_250220_pressure_raw.csv`` stand-alone pressure logger, wall-clock timestamps, ~10 Hz, bar
* ``KNSB_250220_thrust.csv`` / ``KNSB_250220_pressure.csv``  HANARO's own processed burn-window
  outputs (100 Hz, filtered, thrust in N); used here only to calibrate volts to newtons and to
  cross-check the analysis
* ``config.xlsx`` motor geometry and test configuration

What this script does, and every assumption it makes:

1. Thrust: average samples sharing a timestamp, then put them on a 100 Hz grid by linear
   interpolation. Grid points farther than ``MAX_GAP_S`` from any raw sample are left NaN, so
   DAQ dropouts stay visible instead of being papered over.
2. Thrust calibration: the load-cell constants are not in the repository, so volts are mapped to
   newtons with a straight line fitted against HANARO's processed thrust (after aligning the two
   by cross-correlation). The fit and its R^2 are written to ``meta.json``.
3. Pressure: the logger clock and the DAQ clock are independent. The offset between them is the
   shift that maximises the correlation of pressure and thrust voltage over the burn. Pressure is
   then linearly interpolated onto the same 100 Hz grid; its native rate (~10 Hz) is recorded in
   ``meta.json`` so tools do not look for content above its Nyquist frequency.
4. No redlines are written: HANARO does not publish a case pressure limit or thrust limit for this
   motor, and inventing one would make the redline check meaningless.

Run from the repository root::

    python examples/hanaro_knsb/prepare.py
    groundline analyze examples/hanaro_knsb/run.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
FS = 100.0  # Hz, same as HANARO's processing
MAX_GAP_S = 0.025  # leave the grid NaN where the nearest raw thrust sample is farther than this


def load_thrust_volts() -> pd.DataFrame:
    rows = (RAW / "KNSB_250220_thrust_raw.csv").read_text().splitlines()
    t, v = (np.array(r.split(","), dtype=float) for r in rows[:2])
    return pd.DataFrame({"t": t, "v": v}).groupby("t", as_index=False)["v"].mean()


def load_pressure() -> pd.DataFrame:
    p = pd.read_csv(RAW / "KNSB_250220_pressure_raw.csv", sep=";")
    ts = pd.to_datetime(p["Datetime"])
    return pd.DataFrame({
        "t": (ts - ts.iloc[0]).dt.total_seconds().to_numpy(),
        "p_bar": p["5600 Pressure (Bar)"].to_numpy(dtype=float),
        "temp_c": p["5600 Temperature (°C)"].to_numpy(dtype=float),
        "wallclock": p["Datetime"],
    })


def best_shift(t_ref, y_ref, t_mov, y_mov, grid, shifts) -> tuple[float, float]:
    """Shift s maximising corr(y_ref(grid), y_mov(grid - s))."""
    a = np.interp(grid, t_ref, y_ref)
    scores = [np.corrcoef(a, np.interp(grid - s, t_mov, y_mov))[0, 1] for s in shifts]
    k = int(np.argmax(scores))
    return float(shifts[k]), float(scores[k])


def on_grid(t_raw, y_raw, grid, max_gap) -> np.ndarray:
    y = np.interp(grid, t_raw, y_raw)
    j = np.clip(np.searchsorted(t_raw, grid), 1, len(t_raw) - 1)
    nearest = np.minimum(np.abs(grid - t_raw[j - 1]), np.abs(t_raw[j] - grid))
    y[nearest > max_gap] = np.nan
    return y


def main() -> None:
    th = load_thrust_volts()
    pr = load_pressure()
    proc_f = pd.read_csv(RAW / "KNSB_250220_thrust.csv")

    # burn window on the DAQ clock, from the thrust voltage itself
    base_v = float(th["v"].median())
    i_pk = int(th["v"].idxmax())
    burn = th[(th["v"] > base_v + 0.1 * (th["v"].iloc[i_pk] - base_v))
              & (abs(th["t"] - th["t"].iloc[i_pk]) < 20)]
    b0, b1 = float(burn["t"].min()), float(burn["t"].max())
    fine = np.arange(b0 - 2, b1 + 2, 0.001)

    # 1) pressure-logger clock -> DAQ clock
    p_pk = float(pr["t"].iloc[pr["p_bar"].idxmax()])
    guess = th["t"].iloc[i_pk] - p_pk
    p_off, p_corr = best_shift(th["t"], th["v"], pr["t"], pr["p_bar"], fine,
                               np.arange(guess - 3, guess + 3, 0.005))

    # 2) volts -> newtons against HANARO's processed thrust (its t=0 is the start of its burn window)
    f_off, f_corr = best_shift(th["t"], th["v"], proc_f["time"], proc_f["thrust"], fine,
                               np.arange(b0 - 2, b0 + 2, 0.002))
    t_proc = proc_f["time"].to_numpy() + f_off
    v_at = np.interp(t_proc, th["t"], th["v"])
    slope, intercept = np.polyfit(v_at, proc_f["thrust"].to_numpy(), 1)
    pred = slope * v_at + intercept
    ss = np.sum((proc_f["thrust"] - pred) ** 2) / np.sum((proc_f["thrust"] - proc_f["thrust"].mean()) ** 2)

    # 3) common 100 Hz grid over the span both instruments recorded
    t0 = np.ceil(max(th["t"].iloc[0], pr["t"].iloc[0] + p_off) * FS) / FS
    t1 = np.floor(min(th["t"].iloc[-1], pr["t"].iloc[-1] + p_off) * FS) / FS
    grid = np.round(np.arange(t0, t1 + 0.5 / FS, 1 / FS), 4)
    thrust = slope * on_grid(th["t"].to_numpy(), th["v"].to_numpy(), grid, MAX_GAP_S) + intercept
    pc = on_grid(pr["t"].to_numpy() + p_off, pr["p_bar"].to_numpy(), grid, 0.3)
    temp = on_grid(pr["t"].to_numpy() + p_off, pr["temp_c"].to_numpy(), grid, 0.3)

    run = pd.DataFrame({"time": grid, "Pc": pc, "F_thrust": thrust, "T_logger": temp})
    run.to_csv(HERE / "run.csv", index=False, float_format="%.6g")

    cfg = pd.read_excel(RAW / "config.xlsx").iloc[0].to_dict()
    p_rate = float(1 / np.median(np.diff(pr["t"])))
    meta = {
        "test_id": "HANARO KNSB_250220",
        "description": "Solid rocket motor static fire (KNSB, 5 grain segments), SNU Rocket Team HANARO. "
                       "Sample data from github.com/snu-hanaro/static-fire-toolkit (MIT License).",
        "engine_type": "solid rocket motor (no valves, no feed system, no simulation prediction)",
        "time_base": "DAQ clock of the thrust channel, seconds",
        "sample_rate_hz": FS,
        "channels": {
            "Pc": {"unit": "bar", "kind": "pressure",
                   "desc": "chamber pressure, stand-alone logger (not zero-corrected)",
                   "native_rate_hz": round(p_rate, 2),
                   "note": "logged at ~10 Hz and linearly interpolated onto the 100 Hz grid"},
            "F_thrust": {"unit": "N", "kind": "force", "desc": "thrust, load cell",
                         "native_rate_hz": round(float(1 / np.median(np.diff(th["t"]))), 1),
                         "note": "irregular DAQ timing; grid points > 25 ms from a raw sample are NaN"},
            "T_logger": {"unit": "degC", "kind": "housekeeping", "desc": "pressure logger internal temperature",
                         "native_rate_hz": round(p_rate, 2)},
        },
        "motor": {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()
                  if k not in ("index",) and not str(k).startswith("비고")},
        "preparation": {
            "script": "examples/hanaro_knsb/prepare.py",
            "pressure_clock_offset_s": round(p_off, 3),
            "pressure_thrust_correlation": round(p_corr, 5),
            "thrust_calibration": {"slope_N_per_V": round(float(slope), 3), "intercept_N": round(float(intercept), 3),
                                   "r2_vs_hanaro_processed": round(float(1 - ss), 5),
                                   "method": "least squares against HANARO processed thrust"},
            "thrust_max_gap_s": MAX_GAP_S,
        },
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (HERE / "limits.json").write_text(json.dumps({"oscillation": {"fmin_hz": 5.0}}, indent=2))

    nan_f = int(np.isnan(thrust).sum())
    print(f"grid {grid[0]:.2f}-{grid[-1]:.2f} s, {len(grid)} samples at {FS:.0f} Hz")
    print(f"pressure clock offset {p_off:+.3f} s (corr {p_corr:.4f}), native {p_rate:.1f} Hz")
    print(f"thrust: {slope:.1f} N/V, {intercept:+.1f} N, R2 {1 - ss:.4f}; NaN {nan_f} samples")


if __name__ == "__main__":
    main()
