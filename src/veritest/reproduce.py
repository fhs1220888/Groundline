"""Re-execute the evidence ledger of a saved report and check the results match."""

from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file
from .session import Session


def _resolve(report_path: Path, data: dict, key: str) -> str | None:
    rel = (data.get("paths_relative") or {}).get(key)
    if rel and (report_path.parent / rel).exists():
        return str(report_path.parent / rel)
    p = (data.get("paths") or {}).get(key)
    return p if p and Path(p).exists() else None


def reproduce(report: str | Path, only: list[str] | None = None) -> dict:
    report = Path(report)
    data = json.loads(report.read_text(encoding="utf-8"))
    run = _resolve(report, data, "run")
    if not run:
        raise FileNotFoundError("data file referenced by the report was not found")
    sha = sha256_file(run)
    s = Session.open(run, reference=_resolve(report, data, "reference"), limits=_resolve(report, data, "limits"))
    rows = []
    for ev in data["evidence"]:
        if only and ev["id"] not in only:
            continue
        new = s.run(ev["tool"], **ev["params"])
        same = json.dumps(new.result, sort_keys=True) == json.dumps(ev["result"], sort_keys=True)
        rows.append({
            "id": ev["id"],
            "tool": ev["tool"],
            "reproduced": same,
            "tool_version_changed": new.tool_version != ev["tool_version"],
        })
    return {
        "data_file": run,
        "data_hash_matches": sha == data["data_sha256"],
        "n": len(rows),
        "n_reproduced": sum(r["reproduced"] for r in rows),
        "rows": rows,
    }
