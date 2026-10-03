"""Command line interface: ``groundline demo | synth | analyze | reproduce | eval | leaderboard | tools``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import make_agent


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent", default=os.environ.get("GROUNDLINE_AGENT", "rule"),
                   help="rule (no LLM) | openai (any OpenAI-compatible endpoint: Qwen, DeepSeek, vLLM, Ollama...) "
                        "| anthropic. Default: GROUNDLINE_AGENT from .env, else rule")
    p.add_argument("--model", help="model name (default: GROUNDLINE_LLM_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: GROUNDLINE_LLM_BASE_URL)")
    p.add_argument("--lang", default=os.environ.get("GROUNDLINE_LANG", "zh"), choices=["zh", "en"],
                   help="report language (default: GROUNDLINE_LANG, else zh)")


def cmd_synth(a) -> int:
    from .synth import generate_run

    anomalies = None if a.anomalies is None else [x for x in a.anomalies.split(",") if x]
    run = generate_run(a.seed, anomalies)
    paths = run.save(a.out, fmt=a.format)
    print(f"wrote {paths['run']}")
    print(f"injected: {', '.join(x.type for x in run.truth) or 'none (nominal run)'}  (truth in {paths['truth']})")
    return 0


def _analyze(run_path, reference, limits, a) -> int:
    from . import Session
    from .report import write_report

    s = Session.open(run_path, reference=reference, limits=limits)
    agent = make_agent(a.agent, a.lang, a.model, getattr(a, "base_url", None))
    res = agent.run(s)
    out = a.out or str(Path(run_path).with_name("report.html"))
    paths = write_report(s, res, out, a.lang)
    v = res.verification
    print(res.summary)
    for i, f in enumerate(res.findings, 1):
        mark = "✓" if f.verification.verified else "!"
        print(f"  {i}. [{f.severity}] {mark} {f.title}  ({', '.join(f.evidence)})")
    print(f"claims verified {v['verified']}/{v['n_findings']}, numbers grounded "
          f"{v['numbers_grounded']}/{v['numbers_total']}")
    fs = res.agent.get("first_submission")
    if fs:
        print(f"first draft before verifier feedback: numbers grounded {fs['numbers_grounded']}/{fs['numbers_total']}"
              f" (fix rounds used: {res.agent.get('fix_rounds_used', 0)})")
    u = res.agent.get("usage")
    if u and u.get("requests"):
        print(f"LLM: {res.agent.get('model')} · {u['requests']} requests · {u['input_tokens']:,} input / "
              f"{u['output_tokens']:,} output tokens · {res.elapsed_s:.0f} s")
    print(f"report: {paths['html']}\nledger: {paths['json']}")
    return 0


def cmd_analyze(a) -> int:
    return _analyze(a.run, a.reference, a.limits, a)


def cmd_demo(a) -> int:
    from .synth import generate_run

    anomalies = [x for x in a.anomalies.split(",") if x]
    run = generate_run(a.seed, anomalies)
    paths = run.save(a.out)
    print(f"synthetic run with injected: {', '.join(x.type for x in run.truth)}")
    a.out = str(Path(a.out) / "report.html")
    return _analyze(paths["run"], None, None, a)


def cmd_reproduce(a) -> int:
    from .reproduce import reproduce

    r = reproduce(a.report, a.evidence)
    print(f"data file: {r['data_file']}  hash {'matches' if r['data_hash_matches'] else 'DIFFERS'}")
    for row in r["rows"]:
        flag = "✓" if row["reproduced"] else "✗"
        extra = " (tool code changed since report)" if row["tool_version_changed"] else ""
        print(f"  {flag} {row['id']:>4} {row['tool']}{extra}")
    print(f"{r['n_reproduced']}/{r['n']} evidence entries reproduced exactly")
    return 0 if r["n_reproduced"] == r["n"] and r["data_hash_matches"] else 1


def cmd_eval(a) -> int:
    from .evaluate import format_summary, run_benchmark, save

    def prog(i, n):
        print(f"\r  run {i}/{n}", end="", file=sys.stderr, flush=True)

    res = run_benchmark(lambda: make_agent(a.agent, a.lang, a.model, a.base_url), n=a.n, seed=a.seed, progress=prog,
                        suite=a.suite)
    print(file=sys.stderr)
    print(format_summary(res["summary"]))
    if a.out:
        print(f"details: {save(res, a.out)}")
    return 0


def cmd_leaderboard(a) -> int:
    from .leaderboard import build_table, run_leaderboard

    if a.table_only:
        print(build_table(a.out))
        return 0
    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text())
    if a.n is not None:
        cfg["n"] = a.n
    if a.seed is not None:
        cfg["seed"] = a.seed
    only = [x for x in a.only.split(",") if x] if a.only else None
    md = run_leaderboard(cfg, a.out, force=a.force, only=only, base_dir=Path.cwd())
    print(md.read_text())
    print(f"table: {md}")
    return 0


def cmd_verifier_bench(a) -> int:
    from .evaluate import save
    from .verifier_bench import format_bench, run_verifier_bench

    res = run_verifier_bench(n=a.n, seed=a.seed)
    print(format_bench(res))
    if a.out:
        print(f"details: {save(res, a.out)}")
    return 0


def cmd_ingest(a) -> int:
    from .ingest import ingest, suggest_map

    raw = Path(a.raw)
    out = Path(a.out) if a.out else raw.with_name(raw.stem)
    if a.suggest or not a.map:
        m = suggest_map(raw)
        out.mkdir(parents=True, exist_ok=True)
        (out / "map.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
        print(f"time column: {m['time']['column']!r}" + (f" · skipping {m['skip_rows']} units row" if m["skip_rows"] else ""))
        for name, c in m["channels"].items():
            print(f"  {name:20s} {c['unit'] or '-':6s} {c['kind']:15s} <- {c['column']}")
        for name, d in (m.get("derived") or {}).items():
            print(f"  {name:20s} {d['unit'] or '-':6s} {d['kind']:15s} = {' + '.join(d['sum'])}")
        print(f"draft mapping: {out / 'map.json'}\nread and edit it (names, kinds, window_s, grid_hz, max_gap_s), "
              f"then: groundline ingest {raw} --map {out / 'map.json'}")
        return 0
    paths = ingest(raw, json.loads(Path(a.map).read_text()), out)
    print(f"wrote {paths['run']} and {paths['meta']}\nnext: groundline analyze {paths['run']}")
    return 0


def cmd_hybrid_bench(a) -> int:
    from pathlib import Path as _P

    from .evaluate import save
    from .hybrid import available_backgrounds, format_hybrid, run_hybrid

    bgs = {_P(p).parent.name if _P(p).name == "run.csv" else _P(p).stem: _P(p) for p in a.backgrounds} \
        if a.backgrounds else available_backgrounds()
    if not bgs:
        print("no backgrounds found: run examples/triton_lox/prepare.py and examples/uvic_mule/prepare.py, "
              "or pass run.csv paths")
        return 1

    def prog(i, n):
        print(f"\r  case {i}/{n}", end="", file=sys.stderr, flush=True)

    res = run_hybrid(bgs, reps=a.reps, seed=a.seed, progress=prog)
    print(file=sys.stderr)
    print("backgrounds: " + ", ".join(bgs))
    print(format_hybrid(res))
    if a.out:
        print(f"details: {save(res, a.out)}")
    return 0


def cmd_reverify(a) -> int:
    from .evaluate import format_summary, reverify, save

    for p in a.results:
        res = reverify(json.loads(Path(p).read_text()))
        save(res, p)
        print(f"== {p}\n{format_summary(res['summary'])}\n")
    return 0


def cmd_mcp(a) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_tools(a) -> int:
    from .tools import REGISTRY

    for t in REGISTRY.values():
        params = ", ".join(t.params) or "—"
        print(f"{t.name}({params})\n    {t.description}\n")
    return 0


def cmd_config(a) -> int:
    from .config import describe

    for k, v in describe().items():
        print(f"{k:>22}: {v if v is not None else '-'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .config import load_dotenv

    load_dotenv()  # .env in the current directory or a parent; shell variables win
    from . import __version__

    p = argparse.ArgumentParser(prog="groundline", description="Verifiable test-data analysis agent")
    p.add_argument("--version", action="version", version=f"groundline {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="generate a synthetic hot-fire run and analyse it")
    d.add_argument("--out", default="groundline_demo")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--anomalies", default="oscillation,overtemp,valve_delay,sensor_spike,pc_deficit")
    _add_agent_args(d)
    d.set_defaults(fn=cmd_demo)

    s = sub.add_parser("synth", help="generate synthetic test data with labelled anomalies")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--anomalies", help="comma separated; omit for random, '' for nominal")
    s.add_argument("--format", default="csv", choices=["csv", "tdms"])
    s.set_defaults(fn=cmd_synth)

    an = sub.add_parser("analyze", help="analyse a run (CSV or TDMS) and write an HTML report")
    an.add_argument("run")
    an.add_argument("--reference", help="simulation prediction CSV (default: reference.csv next to run)")
    an.add_argument("--limits", help="limits JSON (default: limits.json next to run)")
    an.add_argument("--out", help="report path (default: report.html next to run)")
    _add_agent_args(an)
    an.set_defaults(fn=cmd_analyze)

    r = sub.add_parser("reproduce", help="re-run every evidence entry of a report and compare")
    r.add_argument("report", help="report .json written next to the HTML")
    r.add_argument("--evidence", nargs="*", help="only these IDs, e.g. E3 E5")
    r.set_defaults(fn=cmd_reproduce)

    e = sub.add_parser("eval", help="benchmark an agent on synthetic runs with known anomalies")
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--out", help="write full results JSON here")
    e.add_argument("--suite", default="classic", choices=["classic", "realistic"],
                   help="classic: the original six anomaly types; realistic: adds faults and nuisances seen in real "
                        "test logs (DAQ dropouts, saturation, offsets, duplicated / dead channels, mains hum, ...)")
    _add_agent_args(e)
    e.set_defaults(fn=cmd_eval)

    mp = sub.add_parser("mcp", help="run the MCP server on stdio (same as groundline-mcp)")
    mp.set_defaults(fn=cmd_mcp)

    vb = sub.add_parser("verifier-bench", help="plant known errors in correct findings and count what the verifier catches")
    vb.add_argument("--n", type=int, default=50)
    vb.add_argument("--seed", type=int, default=1000)
    vb.add_argument("--out", help="write details JSON here")
    vb.set_defaults(fn=cmd_verifier_bench)

    ig = sub.add_parser("ingest", help="turn a raw CSV log into a run (run.csv + meta.json) from a mapping")
    ig.add_argument("raw", help="raw CSV log")
    ig.add_argument("--map", help="mapping JSON; without it a draft mapping is written for you to edit")
    ig.add_argument("--suggest", action="store_true", help="only write a draft mapping (map.json)")
    ig.add_argument("--out", help="output directory (default: a folder named after the raw file)")
    ig.set_defaults(fn=cmd_ingest)

    hb = sub.add_parser("hybrid-bench", help="inject known anomalies into real test logs and count what is caught")
    hb.add_argument("backgrounds", nargs="*", help="run.csv files (default: the real-data examples present locally)")
    hb.add_argument("--reps", type=int, default=5, help="injections per size and background (default 5)")
    hb.add_argument("--seed", type=int, default=0)
    hb.add_argument("--out", help="write details JSON here")
    hb.set_defaults(fn=cmd_hybrid_bench)

    rv = sub.add_parser("reverify", help="re-run the current verifier on saved LLM benchmark results")
    rv.add_argument("results", nargs="+", help="eval / leaderboard JSON files (updated in place)")
    rv.set_defaults(fn=cmd_reverify)

    lb = sub.add_parser("leaderboard", help="benchmark several agents/models on the same runs and compare them")
    lb.add_argument("config", nargs="?", default="examples/leaderboard.json", help="models JSON")
    lb.add_argument("--out", default="leaderboard", help="directory for per-model results and LEADERBOARD.md")
    lb.add_argument("--n", type=int, help="override n from the config")
    lb.add_argument("--seed", type=int, help="override seed from the config")
    lb.add_argument("--only", help="comma separated model names to (re)run")
    lb.add_argument("--force", action="store_true", help="re-run models that already have results")
    lb.add_argument("--table-only", action="store_true", help="only rebuild the table from --out")
    lb.set_defaults(fn=cmd_leaderboard)

    c = sub.add_parser("config", help="show the configuration the CLI will use (.env, agent, model; keys masked)")
    c.set_defaults(fn=cmd_config)

    t = sub.add_parser("tools", help="list analysis tools")
    t.set_defaults(fn=cmd_tools)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
