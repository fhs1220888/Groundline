"""Session = one loaded test run + an append-only evidence ledger.

Every analysis step goes through :meth:`Session.run`, which executes a
deterministic tool and records an :class:`Evidence` entry (tool, parameters,
data hash, tool-source hash, result, figure).  Findings in the final report can
only point at these entries, and each entry can be re-executed later with
``groundline reproduce`` to prove the number came from the data.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .io import load_limits, load_reference, load_run, sha256_file


def to_jsonable(obj: Any, sig: int = 6) -> Any:
    """Convert numpy/pandas types to plain JSON, rounding floats to ``sig`` significant digits."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, sig) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        if x == 0:
            return 0.0
        return float(f"{x:.{sig}g}")
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), sig)
    return obj


@dataclass
class Evidence:
    id: str
    tool: str
    params: dict
    result: dict
    data_sha256: str
    tool_version: str
    elapsed_ms: float
    figure_png: str | None = None  # base64
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self, include_figure: bool = True) -> dict:
        d = asdict(self)
        if not include_figure:
            d.pop("figure_png")
        return d

    def reproduce_code(self, run_path: str, reference: str | None, limits: str | None) -> str:
        args = [repr(run_path)]
        if reference:
            args.append(f"reference={reference!r}")
        if limits:
            args.append(f"limits={limits!r}")
        params = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        call = f"s.run({self.tool!r}{', ' if params else ''}{params})"
        return (
            "from groundline import Session\n"
            f"s = Session.open({', '.join(args)})\n"
            f"ev = {call}\n"
            "print(ev.result)"
        )


class Session:
    def __init__(
        self,
        data: pd.DataFrame,
        meta: dict | None = None,
        reference: pd.DataFrame | None = None,
        limits: dict | None = None,
        data_sha256: str = "",
        paths: dict | None = None,
    ):
        self.data = data
        self.meta = meta or {}
        self.reference = reference
        self.limits = limits or {}
        self.data_sha256 = data_sha256 or hashlib.sha256(
            pd.util.hash_pandas_object(data, index=False).values.tobytes()
        ).hexdigest()
        self.paths = paths or {}
        self.ledger: list[Evidence] = []
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def open(
        cls,
        run: str | Path,
        reference: str | Path | None = None,
        limits: str | Path | None = None,
        meta: str | Path | None = None,
    ) -> "Session":
        run = Path(run)
        if reference is None and run.with_name("reference.csv").exists():
            reference = run.with_name("reference.csv")
        if limits is None and run.with_name("limits.json").exists():
            limits = run.with_name("limits.json")
        df, m = load_run(run, meta)
        return cls(
            df,
            m,
            load_reference(reference),
            load_limits(limits),
            data_sha256=sha256_file(run),
            paths={
                "run": str(run),
                "reference": str(reference) if reference else None,
                "limits": str(limits) if limits else None,
            },
        )

    # ------------------------------------------------------------------ helpers
    @property
    def time(self) -> np.ndarray:
        return self.data["time"].to_numpy()

    @property
    def fs(self) -> float:
        return float(self.meta.get("sample_rate_hz") or 1.0 / np.median(np.diff(self.time)))

    @property
    def channels(self) -> list[str]:
        return [c for c in self.data.columns if c != "time"]

    def channel_info(self, ch: str) -> dict:
        return self.meta.get("channels", {}).get(ch, {})

    def channels_of_kind(self, kind: str) -> list[str]:
        return [c for c in self.channels if self.channel_info(c).get("kind") == kind]

    def require_channel(self, ch: str) -> np.ndarray:
        if ch not in self.data.columns:
            raise ValueError(f"unknown channel {ch!r}; available: {self.channels}")
        return self.data[ch].to_numpy(dtype=float)

    def window(self, t_start: float | None, t_end: float | None) -> np.ndarray:
        t = self.time
        lo = -np.inf if t_start is None else t_start
        hi = np.inf if t_end is None else t_end
        return (t >= lo) & (t <= hi)

    def phases(self) -> list[dict]:
        """Deterministic phase segmentation (cached; not itself a ledger entry)."""
        if "phases" not in self._cache:
            from .tools import compute_phases

            self._cache["phases"] = compute_phases(self)
        return self._cache["phases"]["phases"]

    def phase_window(self, name: str) -> tuple[float, float]:
        for p in self.phases():
            if p["name"] == name:
                return p["t_start"], p["t_end"]
        raise ValueError(f"unknown phase {name!r}")

    # ------------------------------------------------------------------ running tools
    def run(self, tool: str, **params) -> Evidence:
        from .tools import REGISTRY

        if tool not in REGISTRY:
            raise ValueError(f"unknown tool {tool!r}; available: {sorted(REGISTRY)}")
        spec = REGISTRY[tool]
        params = {k: v for k, v in params.items() if v is not None}
        unknown = set(params) - set(spec.params)
        if unknown:
            raise ValueError(f"tool {tool!r} got unknown parameter(s) {sorted(unknown)}")
        t0 = time.perf_counter()
        result, fig = spec.fn(self, **params)
        elapsed = (time.perf_counter() - t0) * 1000
        ev = Evidence(
            id=f"E{len(self.ledger) + 1}",
            tool=tool,
            params=to_jsonable(params),
            result=to_jsonable(result),
            data_sha256=self.data_sha256,
            tool_version=spec.version,
            elapsed_ms=round(elapsed, 1),
            figure_png=fig,
        )
        self.ledger.append(ev)
        return ev

    def evidence(self, eid: str) -> Evidence | None:
        for e in self.ledger:
            if e.id == eid:
                return e
        return None

    def ledger_json(self) -> list[dict]:
        return [e.to_dict(include_figure=False) for e in self.ledger]


def source_hash(fn) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover
        src = fn.__name__
    return f"{__version__}+{hashlib.sha256(src.encode()).hexdigest()[:10]}"


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False)
