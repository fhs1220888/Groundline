"""Turn UVic Rocketry's public MULE-1 hybrid-motor test logs into Groundline runs.

Source: https://github.com/UVicRocketry/Propulsion-Test-Data (pinned to commit ``6aa051d``). MULE-1 burns
nitrous oxide (N2O, fed from a run tank pressurised with nitrogen) with paraffin wax: a hybrid motor with
a liquid-style feed system (run tank, flow lines, normally closed valve, injector). The repository has no
license, so nothing is copied into Groundline: this script downloads the logs into ``raw/`` (git-ignored)
and writes ``<test>/run.csv`` and ``<test>/meta.json`` next to itself (also git-ignored).

Every test folder comes with the team's own report of what happened, which makes these logs a check of
what Groundline finds against what the people at the test saw:

* ``2024-12-12`` hot fire: the igniter wiring shorted, the 24 V supply drooped, both valves lost power
  and the DAQ cut out as soon as the engine ignited.
* ``2025-01-18`` hot fire: the team's "first completely nominal hot fire"; thrust ~25 % below
  expectation for an unclear reason; the two chamber thermocouples were not installed / damaged.
* ``2025-02-08`` hot fire: the cast nozzle throat insert broke (throat ~7 mm wider), the phenolic liner
  shattered and the injector snap ring failed; no stable combustion, low thrust.
* ``2025-09-20`` hot fire: no report beyond the date.

What this script does, and every assumption it makes:

1. Each CSV has a units row, then a header row. The units row is misaligned in some files (an index
   column is present in the data but not in the units), so units are taken from the channel name
   instead, as documented in the repository README: pressures psi, run-tank load cell kg, thrust N,
   temperatures degC.
2. Logged at ~450 Hz with jittery timestamps; put on a uniform 500 Hz grid by linear interpolation.
   Grid points farther than ``MAX_GAP_S`` from any logged sample stay NaN.
3. Values are left exactly as logged, offsets and impossible readings included: finding them is the
   point. No redlines or simulation prediction are published, and valve states are not logged.

Run from the repository root::

    python examples/uvic_mule/prepare.py                 # all four hot fires
    groundline analyze examples/uvic_mule/2025-01-18/run.csv
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
BASE = "https://raw.githubusercontent.com/UVicRocketry/Propulsion-Test-Data/6aa051d24a1ff0106c2eef46eafe0034d3d2bee7/"
FS = 500.0  # Hz, grid rate (the DAQ logs at ~450 Hz)
MAX_GAP_S = 0.02

TESTS = {
    "2024-12-12": ("2024-12-12_MULE_1_HOTFIRE/data2.csv",
                   "Hot fire: igniter wiring shorted, supply drooped, valves de-energised, DAQ cut out at ignition "
                   "(team report)."),
    "2025-01-18": ("2025-01-18_MULE_1_HOTFIRE/data1.csv",
                   "Hot fire: first fully nominal one; thrust ~25 % below expectation; chamber thermocouples not "
                   "installed / damaged (team report)."),
    "2025-02-08": ("2025-02-08_MULE_1_HOTFIRE/data0.csv",
                   "Hot fire: nozzle throat insert broke, phenolic liner shattered, snap ring failed; no stable "
                   "combustion, low thrust (team report)."),
    "2025-09-20": ("2025-09-20_MULE_1_HOTFIRE/data0.csv", "Hot fire (no report beyond the date)."),
}

# logged column -> (Groundline name, unit, kind, description)
CHANNELS = {
    "P_COMB_CHMBR": ("Pc", "psi", "pressure", "combustion chamber pressure"),
    "P_INJECTOR": ("P_inj", "psi", "pressure", "injector pressure"),
    "P_N2O_FLOW": ("P_n2o_line", "psi", "pressure", "N2O fill / flow line pressure"),
    "P_N2_FLOW": ("P_n2_line", "psi", "pressure", "nitrogen pressurant line pressure"),
    "P_RUN_TANK": ("P_run_tank", "psi", "pressure", "N2O run tank pressure"),
    "L_RUN_TANK": ("W_run_tank", "kg", "weight", "run tank load cell"),
    "L_THRUST": ("F_thrust", "N", "force", "thrust load cell"),
    "T_RUN_TANK": ("T_run_tank", "degC", "temperature", "run tank"),
    "T_INJECTOR": ("T_injector", "degC", "temperature", "injector"),
    "T_COMB_CHMBR": ("T_chamber", "degC", "temperature", "combustion chamber"),
    "T_POST_COMB": ("T_post_comb", "degC", "temperature", "post-combustion chamber"),
}


def download(path: str) -> Path:
    dst = RAW / path
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BASE + path} ...")
        urllib.request.urlretrieve(BASE + path, dst)
    return dst


def mapping(name: str) -> dict:
    """The conversion as a groundline.ingest mapping (``groundline ingest raw.csv --map map.json`` does the same).
    Units come from the channel names here, not from the units row, which is misaligned in some files."""
    path, _note = TESTS[name]  # the team's account, for people reading this script, not for agents
    return {
        "source": BASE + path,
        "skip_rows": 1,  # row 0: units
        "time": {"column": "seconds"},
        "channels": {ch: {"column": col, "unit": unit, "kind": kind, "desc": desc}
                     for col, (ch, unit, kind, desc) in CHANNELS.items()},
        "grid_hz": FS, "max_gap_s": MAX_GAP_S,
        "test_id": f"UVic MULE-1 {name}",
        # the team's account of the test stays out of the metadata: describe_data hands the description to LLM
        # agents, and the point of these logs is to check what an agent finds against what the team saw
        "description": "N2O / paraffin hybrid motor hot fire, UVic Rocketry MULE-1. Public data from "
                       "github.com/UVicRocketry/Propulsion-Test-Data (no license; not redistributed).",
        "engine_type": "hybrid (N2O / paraffin), pressure-fed oxidiser; no valve states, redlines or prediction",
        "time_base": "seconds from the first logged sample",
    }


def prepare(name: str) -> Path:
    from groundline.ingest import ingest

    paths = ingest(download(TESTS[name][0]), mapping(name), HERE / name)
    print(f"wrote {paths['run']}")
    return paths["run"].parent


if __name__ == "__main__":
    for name in sys.argv[1:] or list(TESTS):
        prepare(name)
