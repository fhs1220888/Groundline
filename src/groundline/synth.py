"""Synthetic hot-fire test data with injectable, labelled anomalies.

The generator models a small pressure-fed liquid rocket engine firing:

* two main valves (oxidiser / fuel) with command channels,
* injector pressures that follow valve opening (first-order lag + dead time),
* chamber pressure (Pc) that ignites once both propellants arrive,
* coolant-outlet temperature, mass flows and a vibration accelerometer.

It also writes a noise-free "simulation prediction" (reference) sampled at a
lower rate, a redline/limits file and a ground-truth list of every injected
anomaly.  Because the truth is known, the output doubles as a benchmark for
analysis agents (see ``groundline.evaluate``).

This is an engineering toy model, not a combustion model.  Its job is to
produce data with realistic *structure* (phases, lags, noise, failure
signatures) so the analysis pipeline can be developed without real test data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ANOMALY_TYPES = (
    "oscillation",      # narrow-band pressure oscillation (combustion instability)
    "overtemp",         # coolant outlet temperature exceeds redline
    "sensor_dropout",   # a channel flatlines or returns NaN
    "sensor_spike",     # isolated glitch spikes on a pressure transducer
    "valve_delay",      # fuel valve opens late
    "pc_deficit",       # chamber pressure below prediction at nominal flow (low c* efficiency)
)

# Which finding category an agent is expected to report for each truth type.
TRUTH_TO_CATEGORY = {
    "oscillation": "combustion_oscillation",
    "overtemp": "redline_violation",
    "sensor_dropout": "sensor_fault",
    "sensor_spike": "sensor_fault",
    "valve_delay": "valve_response",
    "pc_deficit": "performance_deviation",
}


@dataclass
class EngineSpec:
    pc_nom: float = 5.0          # MPa
    dp_ox: float = 0.8           # MPa, injector pressure drop at nominal flow
    dp_fu: float = 1.0           # MPa
    mdot_ox: float = 6.1         # kg/s
    mdot_fu: float = 2.4         # kg/s
    t_wall_amb: float = 290.0    # K
    t_wall_ss: float = 620.0     # K, steady coolant outlet temperature
    valve_dead_ms: float = 15.0  # nominal valve dead time
    valve_tau_ms: float = 30.0   # nominal valve/feed time constant
    ign_delay_ms: float = 50.0   # ignition after both propellants arrive
    pc_rise_s: float = 0.35      # startup ramp duration
    pc_tail_tau_s: float = 0.08  # shutdown decay time constant


@dataclass
class Sequence:
    cmd_ox_open: float = 0.80
    cmd_fu_open: float = 0.90
    cmd_close: float = 9.00
    duration: float = 12.0


@dataclass
class Anomaly:
    type: str
    channel: str
    t_start: float
    t_end: float
    params: dict = field(default_factory=dict)
    related_channels: list[str] = field(default_factory=list)

    @property
    def category(self) -> str:
        return TRUTH_TO_CATEGORY[self.type]


@dataclass
class SyntheticRun:
    data: pd.DataFrame
    reference: pd.DataFrame
    truth: list[Anomaly]
    meta: dict
    limits: dict

    def save(self, out_dir: str | Path, fmt: str = "csv") -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if fmt == "tdms":
            from .io import write_tdms

            paths["run"] = write_tdms(self.data, out / "run.tdms", self.meta)
        else:
            paths["run"] = out / "run.csv"
            self.data.to_csv(paths["run"], index=False, float_format="%.6g")
        paths["reference"] = out / "reference.csv"
        self.reference.to_csv(paths["reference"], index=False, float_format="%.6g")
        paths["meta"] = out / "meta.json"
        paths["meta"].write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))
        paths["limits"] = out / "limits.json"
        paths["limits"].write_text(json.dumps(self.limits, indent=2, ensure_ascii=False))
        paths["truth"] = out / "truth.json"
        paths["truth"].write_text(
            json.dumps([asdict(a) | {"category": a.category} for a in self.truth], indent=2)
        )
        return paths


CHANNELS = {
    "Pc": {"unit": "MPa", "kind": "pressure", "desc": "chamber pressure"},
    "P_ox_inj": {"unit": "MPa", "kind": "pressure", "desc": "oxidiser injector pressure"},
    "P_fu_inj": {"unit": "MPa", "kind": "pressure", "desc": "fuel injector pressure"},
    "mdot_ox": {"unit": "kg/s", "kind": "flow", "desc": "oxidiser mass flow"},
    "mdot_fu": {"unit": "kg/s", "kind": "flow", "desc": "fuel mass flow"},
    "T_cool_out": {"unit": "K", "kind": "temperature", "desc": "coolant outlet temperature"},
    "vib_axial": {"unit": "g", "kind": "vibration", "desc": "axial accelerometer"},
    "cmd_ox": {"unit": "-", "kind": "command", "desc": "oxidiser main valve command"},
    "cmd_fu": {"unit": "-", "kind": "command", "desc": "fuel main valve command"},
}


def default_limits(spec: EngineSpec) -> dict:
    return {
        "redlines": {
            "Pc": {"max": round(spec.pc_nom * 1.12, 3)},
            "P_ox_inj": {"max": round((spec.pc_nom + spec.dp_ox) * 1.15, 3)},
            "P_fu_inj": {"max": round((spec.pc_nom + spec.dp_fu) * 1.15, 3)},
            "T_cool_out": {"max": 700.0},
        },
        "redline_persistence_s": 0.010,
        "valve_response": {
            "cmd_ox": {"response": "P_ox_inj", "max_latency_ms": 40.0},
            "cmd_fu": {"response": "P_fu_inj", "max_latency_ms": 40.0},
        },
        "oscillation": {"threshold_pct": 0.5, "fmin_hz": 50.0, "fmax_hz": 2000.0},
        "reference_tolerance_pct": 2.0,
    }


def _valve(t: np.ndarray, t_open: float, t_close: float, dead: float, tau: float) -> np.ndarray:
    """Normalised valve/feed response (0..1) to an open command then a close command."""
    y = np.zeros_like(t)
    t0 = t_open + dead
    rising = t >= t0
    y[rising] = 1.0 - np.exp(-(t[rising] - t0) / tau)
    t1 = t_close + dead
    falling = t >= t1
    y_at_close = 1.0 - np.exp(-(t1 - t0) / tau) if t1 > t0 else 0.0
    y[falling] = y_at_close * np.exp(-(t[falling] - t1) / tau)
    return y


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _envelope(t: np.ndarray, t0: float, t1: float, edge: float = 0.05) -> np.ndarray:
    return _smoothstep((t - t0) / edge) * _smoothstep((t1 - t) / edge)


def _nominal(t: np.ndarray, spec: EngineSpec, seq: Sequence, fu_extra_delay: float = 0.0):
    dead = spec.valve_dead_ms / 1000
    tau = spec.valve_tau_ms / 1000
    v_ox = _valve(t, seq.cmd_ox_open, seq.cmd_close, dead, tau)
    v_fu = _valve(t, seq.cmd_fu_open, seq.cmd_close, dead + fu_extra_delay, tau)
    # both propellants "arrive" when their feed reaches 50 %
    arrive_ox = seq.cmd_ox_open + dead + tau * np.log(2)
    arrive_fu = seq.cmd_fu_open + dead + fu_extra_delay + tau * np.log(2)
    t_ign = max(arrive_ox, arrive_fu) + spec.ign_delay_ms / 1000
    ramp = _smoothstep((t - t_ign) / spec.pc_rise_s)
    t_off = seq.cmd_close + dead
    decay = np.where(t >= t_off, np.exp(-np.clip(t - t_off, 0, None) / spec.pc_tail_tau_s), 1.0)
    pc = spec.pc_nom * ramp * decay
    flow_frac = np.minimum(v_ox, v_fu)
    p_ox = pc + spec.dp_ox * v_ox
    p_fu = pc + spec.dp_fu * v_fu
    mdot_ox = spec.mdot_ox * v_ox
    mdot_fu = spec.mdot_fu * v_fu
    # coolant outlet temperature: first-order heating while firing, slower cooling after
    heat = np.zeros_like(t)
    dt = t[1] - t[0]
    tau_heat, tau_cool = 1.5, 3.0
    target = spec.t_wall_amb + (spec.t_wall_ss - spec.t_wall_amb) * (pc / spec.pc_nom)
    T = spec.t_wall_amb
    for i, tg in enumerate(target):
        tau_i = tau_heat if tg > T else tau_cool
        T += (tg - T) * dt / tau_i
        heat[i] = T
    return {
        "Pc": pc,
        "P_ox_inj": p_ox,
        "P_fu_inj": p_fu,
        "mdot_ox": mdot_ox,
        "mdot_fu": mdot_fu,
        "T_cool_out": heat,
        "_flow_frac": flow_frac,
        "_t_ign": t_ign,
    }


def generate_run(
    seed: int = 0,
    anomalies: list[str] | None = None,
    fs: float = 5000.0,
    spec: EngineSpec | None = None,
    seq: Sequence | None = None,
    test_id: str | None = None,
) -> SyntheticRun:
    """Generate one synthetic hot-fire run.

    ``anomalies``: list of anomaly type names to inject.  ``None`` picks a random
    subset (0-3 anomalies) from the seed; ``[]`` gives a nominal run.
    """
    rng = np.random.default_rng(seed)
    spec = spec or EngineSpec()
    seq = seq or Sequence()
    if anomalies is None:
        k = int(rng.choice([0, 1, 1, 2, 2, 3]))
        anomalies = list(rng.choice(ANOMALY_TYPES, size=k, replace=False))
    for a in anomalies:
        if a not in ANOMALY_TYPES:
            raise ValueError(f"unknown anomaly type {a!r}; choose from {ANOMALY_TYPES}")

    n = int(round(seq.duration * fs))
    t = np.arange(n) / fs
    ms_start_guess, ms_end_guess = 1.8, seq.cmd_close - 0.3  # safe mainstage window for placement

    truth: list[Anomaly] = []
    fu_extra = 0.0
    if "valve_delay" in anomalies:
        fu_extra = float(rng.uniform(0.040, 0.120))
        dead = spec.valve_dead_ms / 1000 + fu_extra
        truth.append(
            Anomaly(
                "valve_delay",
                "P_fu_inj",
                seq.cmd_fu_open,
                seq.cmd_fu_open + dead + 0.01,
                {"extra_delay_ms": round(fu_extra * 1000, 1)},
                related_channels=["cmd_fu", "mdot_fu", "Pc"],
            )
        )

    sig = _nominal(t, spec, seq, fu_extra)
    ref_sig = _nominal(t, spec, seq, 0.0)  # prediction assumes nominal valve timing
    pc_ref = ref_sig["Pc"]

    # ---- deterministic signal modifications (before noise) ----
    pc = sig["Pc"].copy()
    p_ox = sig["P_ox_inj"].copy()
    p_fu = sig["P_fu_inj"].copy()
    temp = sig["T_cool_out"].copy()

    if "pc_deficit" in anomalies:
        frac = float(rng.uniform(0.04, 0.09))
        t0 = float(rng.uniform(ms_start_guess, 4.0))
        drop = frac * _smoothstep((t - t0) / 0.3) * (pc / spec.pc_nom)
        delta = spec.pc_nom * drop
        pc -= delta
        p_ox -= delta
        p_fu -= delta
        truth.append(
            Anomaly(
                "pc_deficit",
                "Pc",
                t0,
                seq.cmd_close,
                {"deficit_pct": round(frac * 100, 2)},
                related_channels=["P_ox_inj", "P_fu_inj"],
            )
        )

    if "oscillation" in anomalies:
        f = float(rng.uniform(150, 1200))
        amp = float(rng.uniform(0.02, 0.06))
        dur = float(rng.uniform(1.0, 2.5))
        t0 = float(rng.uniform(ms_start_guess + 0.2, ms_end_guess - dur))
        env = _envelope(t, t0, t0 + dur)
        phase = rng.uniform(0, 2 * np.pi)
        osc = spec.pc_nom * amp * env * np.sin(2 * np.pi * f * t + phase)
        pc += osc
        p_ox += 0.4 * osc
        p_fu += 0.4 * osc
        truth.append(
            Anomaly(
                "oscillation",
                "Pc",
                t0,
                t0 + dur,
                {"freq_hz": round(f, 1), "amplitude_pct": round(amp * 100, 2)},
                related_channels=["P_ox_inj", "P_fu_inj", "vib_axial"],
            )
        )
    else:
        env = np.zeros_like(t)
        f = None

    if "overtemp" in anomalies:
        excess = float(rng.uniform(110, 170))  # K above nominal steady state
        t0 = float(rng.uniform(3.0, 6.0))
        dur = float(rng.uniform(1.2, 2.5))
        bump = excess * _envelope(t, t0, t0 + dur, edge=0.4)
        temp += bump
        above = np.where(temp > 700.0)[0]
        truth.append(
            Anomaly(
                "overtemp",
                "T_cool_out",
                float(t[above[0]]) if above.size else t0,
                float(t[above[-1]]) if above.size else t0 + dur,
                {"peak_K": round(float(temp.max()), 1)},
            )
        )

    # ---- noise ----
    data = {
        "time": t,
        "Pc": pc + rng.normal(0, 0.003 * spec.pc_nom, n),
        "P_ox_inj": p_ox + rng.normal(0, 0.003 * (spec.pc_nom + spec.dp_ox), n),
        "P_fu_inj": p_fu + rng.normal(0, 0.003 * (spec.pc_nom + spec.dp_fu), n),
        "mdot_ox": sig["mdot_ox"] * (1 + rng.normal(0, 0.005, n)) + rng.normal(0, 0.005, n),
        "mdot_fu": sig["mdot_fu"] * (1 + rng.normal(0, 0.005, n)) + rng.normal(0, 0.003, n),
        "T_cool_out": temp + rng.normal(0, 0.5, n),
        "vib_axial": rng.normal(0, 1, n) * (0.2 + 3.0 * np.clip(pc, 0, None) / spec.pc_nom),
        "cmd_ox": ((t >= seq.cmd_ox_open) & (t < seq.cmd_close)).astype(float),
        "cmd_fu": ((t >= seq.cmd_fu_open) & (t < seq.cmd_close)).astype(float),
    }
    if f is not None:
        data["vib_axial"] += 4.0 * env * np.sin(2 * np.pi * f * t + 0.3)

    # ---- sensor faults (applied after noise: they replace the measurement) ----
    if "sensor_spike" in anomalies:
        ch = "P_ox_inj"
        k = int(rng.integers(1, 4))
        times = np.sort(rng.uniform(ms_start_guess, ms_end_guess, k))
        for ts in times:
            i = int(ts * fs)
            data[ch][i : i + 3] += 0.3 * (spec.pc_nom + spec.dp_ox) * np.array([0.6, 1.0, 0.5])
        truth.append(
            Anomaly(
                "sensor_spike",
                ch,
                float(times[0]),
                float(times[-1]) + 0.001,
                {"count": k, "times_s": [round(float(x), 4) for x in times]},
            )
        )

    if "sensor_dropout" in anomalies:
        candidates = ["mdot_fu", "P_fu_inj", "T_cool_out", "vib_axial"]
        if "overtemp" in anomalies:
            candidates.remove("T_cool_out")
        if "valve_delay" in anomalies or "oscillation" in anomalies:
            candidates.remove("P_fu_inj")
        ch = str(rng.choice(candidates))
        mode = str(rng.choice(["flatline", "nan"]))
        dur = float(rng.uniform(0.4, 1.5))
        t0 = float(rng.uniform(ms_start_guess, ms_end_guess - dur))
        i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
        if mode == "flatline":
            data[ch][i0:i1] = data[ch][i0 - 1]
        else:
            data[ch][i0:i1] = np.nan
        truth.append(Anomaly("sensor_dropout", ch, t0, t0 + dur, {"mode": mode}))

    df = pd.DataFrame(data)

    # ---- reference (simulation prediction), coarser sampling, no noise ----
    step = int(fs // 500)
    ref = pd.DataFrame(
        {
            "time": t[::step],
            "Pc": pc_ref[::step],
            "P_ox_inj": ref_sig["P_ox_inj"][::step],
            "P_fu_inj": ref_sig["P_fu_inj"][::step],
            "mdot_ox": ref_sig["mdot_ox"][::step],
            "mdot_fu": ref_sig["mdot_fu"][::step],
            "T_cool_out": ref_sig["T_cool_out"][::step],
        }
    )

    meta = {
        "test_id": test_id or f"SYN-{seed:04d}",
        "description": "Synthetic hot-fire test (groundline.synth)",
        "sample_rate_hz": fs,
        "channels": CHANNELS,
        "sequence": asdict(seq),
        "engine": asdict(spec),
        "seed": seed,
        "anomalies_injected": sorted(anomalies),
    }
    return SyntheticRun(df, ref, truth, meta, default_limits(spec))
