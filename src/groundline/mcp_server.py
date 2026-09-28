"""MCP server: let any MCP client (Claude Desktop, Cursor, your own agent) be the planner.

The client opens a session on a data file, calls analysis tools by name, and
submits findings for verification. Numbers still come only from the tools, so
the verifier works the same way as with the built-in agents.

Run with ``groundline-mcp`` (stdio transport).
"""

from __future__ import annotations

import uuid

from .findings import Finding, verify_findings
from .session import Session


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise ImportError("the MCP server needs `pip install groundline[mcp]`") from e
    from .agent import AnalysisResult
    from .report import write_report
    from .tools import REGISTRY

    mcp = FastMCP("groundline")
    sessions: dict[str, Session] = {}

    def _get(session_id: str) -> Session:
        if session_id not in sessions:
            raise ValueError(f"unknown session {session_id!r}; call open_run first")
        return sessions[session_id]

    @mcp.tool()
    def open_run(run_path: str, reference_path: str | None = None, limits_path: str | None = None) -> dict:
        """Load a test run (CSV or TDMS). reference.csv / limits.json next to the file are picked up automatically.
        Returns a session_id and an overview of the channels."""
        s = Session.open(run_path, reference_path, limits_path)
        sid = uuid.uuid4().hex[:8]
        sessions[sid] = s
        ev = s.run("describe_data")
        return {"session_id": sid, "evidence_id": ev.id, "overview": ev.result}

    @mcp.tool()
    def list_analysis_tools() -> list[dict]:
        """List the deterministic analysis tools and their parameter schemas."""
        return [{"name": t.name, "description": t.description, "parameters": t.json_schema()}
                for t in REGISTRY.values()]

    @mcp.tool()
    def run_analysis(session_id: str, tool: str, params: dict | None = None) -> dict:
        """Run one analysis tool. The call is stored in the evidence ledger; cite the returned evidence_id in findings."""
        ev = _get(session_id).run(tool, **(params or {}))
        return {"evidence_id": ev.id, "result": ev.result, "has_figure": ev.figure_png is not None}

    @mcp.tool()
    def verify(session_id: str, findings: list[dict]) -> dict:
        """Check findings (title, statement, category, severity, channel, t_start, t_end, evidence[]) against the
        ledger. Every number in a statement must be present in the cited evidence."""
        s = _get(session_id)
        fs = [Finding.from_dict(d) for d in findings]
        summary = verify_findings(fs, s)
        return {"summary": summary, "findings": [
            {"title": f.title, **{k: f.verification[k] for k in ("status", "ungrounded_numbers", "problems")}}
            for f in fs]}

    @mcp.tool()
    def write_html_report(session_id: str, findings: list[dict], summary: str, out_path: str,
                          lang: str = "zh") -> dict:
        """Verify the findings and write the HTML report (plus report.json for `groundline reproduce`)."""
        s = _get(session_id)
        fs = [Finding.from_dict(d) for d in findings]
        ver = verify_findings(fs, s)
        res = AnalysisResult(fs, summary, ver, {"type": "llm", "backend": "mcp-client", "model": "external"})
        paths = write_report(s, res, out_path, lang)
        return {"html": str(paths["html"]), "json": str(paths["json"]), "verification": ver}

    return mcp


def main() -> None:
    from .config import load_dotenv

    load_dotenv()
    build_server().run()


if __name__ == "__main__":
    main()
