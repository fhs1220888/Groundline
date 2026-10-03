"""MCP server: let any MCP client (Claude Desktop, Cursor, your own agent) be the planner.

The client opens a session on a data file, calls analysis tools by name, and
submits findings for verification. Numbers still come only from the tools, so
the verifier works the same way as with the built-in agents.

Run with ``groundline mcp`` / ``groundline-mcp`` (stdio transport), or ``uvx groundline mcp``.
"""

from __future__ import annotations

import uuid

from .findings import Finding, verify_findings
from .session import Session


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp 2.x renamed FastMCP to MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        except ImportError as e:  # pragma: no cover
            raise ImportError("the MCP server needs `pip install groundline[mcp]`") from e
    from .agent import AnalysisResult
    from .report import write_report
    from .tools import REGISTRY

    mcp = MCPServer("groundline")
    sessions: dict[str, Session] = {}

    def _get(run_id: str | None) -> Session:
        # `run_id` is optional: some MCP bridges strip argument names they use themselves (a proxy dropped
        # the old name, `session_id`), and with a single open run there is nothing to choose between
        if run_id is None:
            if len(sessions) == 1:
                return next(iter(sessions.values()))
            raise ValueError("run_id is required when more than one run is open; call open_run first"
                             if sessions else "no run is open; call open_run first")
        if run_id not in sessions:
            raise ValueError(f"unknown run_id {run_id!r}; call open_run first")
        return sessions[run_id]

    @mcp.tool()
    def open_run(run_path: str, reference_path: str | None = None, limits_path: str | None = None) -> dict:
        """Load a test run (CSV or TDMS). reference.csv / limits.json next to the file are picked up automatically.
        Returns a run_id and an overview of the channels."""
        s = Session.open(run_path, reference_path, limits_path)
        sid = uuid.uuid4().hex[:8]
        sessions[sid] = s
        ev = s.run("describe_data")
        return {"run_id": sid, "evidence_id": ev.id, "overview": ev.result}

    @mcp.tool()
    def list_analysis_tools() -> list[dict]:
        """List the deterministic analysis tools and their parameter schemas."""
        return [{"name": t.name, "description": t.description, "parameters": t.json_schema()}
                for t in REGISTRY.values()]

    @mcp.tool()
    def run_analysis(tool: str, params: dict | None = None, run_id: str | None = None) -> dict:
        """Run one analysis tool. The call is stored in the evidence ledger; cite the returned evidence_id in findings."""
        ev = _get(run_id).run(tool, **(params or {}))
        return {"evidence_id": ev.id, "result": ev.result, "has_figure": ev.figure_png is not None}

    @mcp.tool()
    def verify(findings: list[dict], run_id: str | None = None) -> dict:
        """Check findings (title, statement, category, severity, channel, t_start, t_end, evidence[]) against the
        ledger. Every number in a statement must be present in the cited evidence, with a unit and role that fit
        the evidence field it came from (e.g. a number after "peak" must come from a peak field). A finding in an
        anomaly category must cite the tool result that reports that anomaly; passed checks are 'observation'.
        Tag a number with its source field to have it checked against exactly that field:
        "742.3 K [E4.violations[0].peak_value]", "3 spikes [len(E3.issues[0].spikes)]"."""
        s = _get(run_id)
        fs = [Finding.from_dict(d) for d in findings]
        summary = verify_findings(fs, s)
        return {"summary": summary, "findings": [
            {"title": f.title, "status": f.verification.status, "ungrounded_numbers": f.verification.ungrounded_numbers,
             "semantic_problems": f.verification.semantic_problems, "problems": f.verification.problem_messages}
            for f in fs]}

    @mcp.tool()
    def write_html_report(findings: list[dict], summary: str, out_path: str, run_id: str | None = None,
                          lang: str = "zh") -> dict:
        """Verify the findings and write the HTML report (plus report.json for `groundline reproduce`)."""
        s = _get(run_id)
        fs = [Finding.from_dict(d) for d in findings]
        ver = verify_findings(fs, s)
        res = AnalysisResult(fs, summary, ver, {"type": "llm", "backend": "mcp-client", "model": "external"})
        paths = write_report(s, res, out_path, lang)
        return {"html": str(paths["html"]), "json": str(paths["json"]), "verification": ver}

    return mcp


def main() -> None:
    from .config import load_dotenv

    load_dotenv()
    try:
        build_server().run()
    except KeyboardInterrupt:  # Ctrl-C in a terminal is a normal way to stop a stdio server
        pass


if __name__ == "__main__":
    main()
