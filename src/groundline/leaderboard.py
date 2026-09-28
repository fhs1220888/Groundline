"""Run the synthetic benchmark for several agents/models and compare them in one table.

Config file (JSON)::

    {
      "n": 14, "seed": 1000, "lang": "zh",
      "models": [
        {"name": "rule", "agent": "rule"},
        {"name": "gpt-5.6-sol", "agent": "openai", "model": "gpt-5.6-sol", "api": "responses",
         "from": "eval_sol_v2.json"},
        {"name": "qwen2.5-7b", "agent": "openai", "model": "qwen2.5:7b-16k",
         "base_url": "http://localhost:11434/v1", "api_key": "ollama"}
      ]
    }

Per entry: ``agent`` (rule | openai | anthropic), ``model``, ``base_url``, ``api`` (responses | chat, default:
responses for api.openai.com, chat otherwise), ``api_key`` or ``api_key_env`` (name of the variable holding the
key), ``reasoning_effort``, and ``from`` to import an existing ``groundline eval`` JSON instead of re-running it
(only if it used the same n and seed).

Every model runs the same seeds. Results go to ``<out>/<name>.json`` and are reused on the next call, so an
interrupted leaderboard resumes where it stopped; ``<out>/LEADERBOARD.md`` is rewritten each time.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .evaluate import run_benchmark, save, summarize


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "model"


def make_entry_agent(e: dict, lang: str):
    from . import agent as A

    kind = e.get("agent", "openai")
    if kind == "rule":
        return A.RuleAgent(lang)
    key = e.get("api_key")
    if key is None and e.get("api_key_env"):
        import os

        key = os.environ.get(e["api_key_env"], "")
    if kind == "anthropic":
        return A.LLMAgent(A.AnthropicBackend(e.get("model"), key), lang)
    url = e.get("base_url") or "https://api.openai.com/v1"
    api = e.get("api") or ("responses" if "api.openai.com" in url else "chat")
    if api == "responses":
        be = A.OpenAIResponses(e.get("model"), url, key, reasoning_effort=e.get("reasoning_effort"))
    else:
        be = A.OpenAICompatible(e.get("model"), url, key)
        if e.get("reasoning_effort"):
            be.reasoning_effort = e["reasoning_effort"]
        if e.get("temperature") is not None:
            be.temperature = float(e["temperature"])
    return A.LLMAgent(be, lang)


def run_leaderboard(config: dict, out: str | Path, force: bool = False, only: list[str] | None = None,
                    base_dir: Path | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n, seed, lang = int(config.get("n", 14)), int(config.get("seed", 1000)), config.get("lang", "zh")
    for e in config["models"]:
        name = e["name"]
        if only and name not in only:
            continue
        path = out / f"{_slug(name)}.json"
        if path.exists() and not force:
            prev = json.loads(path.read_text())
            if prev.get("n_runs") == n and prev.get("seed") == seed:
                print(f"[{name}] already done ({path}), skipping", file=sys.stderr)
                continue
        if e.get("from"):
            src = Path(e["from"])
            if not src.is_absolute() and base_dir:
                src = base_dir / src
            res = json.loads(src.read_text())
            if res.get("n_runs") != n or res.get("seed") != seed:
                raise SystemExit(f"[{name}] {src} used n={res.get('n_runs')} seed={res.get('seed')}, "
                                 f"leaderboard needs n={n} seed={seed}")
            print(f"[{name}] imported {src}", file=sys.stderr)
        else:
            print(f"[{name}] running {n} tests...", file=sys.stderr)

            def prog(i, total, name=name):
                print(f"\r  [{name}] run {i}/{total}", end="", file=sys.stderr, flush=True)

            res = run_benchmark(lambda e=e: make_entry_agent(e, lang), n=n, seed=seed, progress=prog,
                                keep_going=True)
            print(file=sys.stderr)
            if res["summary"].get("failed_runs") == n:
                err = next((r["error"] for r in res["runs"] if "error" in r), "")
                print(f"[{name}] every run failed, e.g.: {err}", file=sys.stderr)
        res["entry"] = {k: v for k, v in e.items() if k not in ("api_key",)}
        save(res, path)
    table = build_table(out, [e["name"] for e in config["models"]])
    md = out / "LEADERBOARD.md"
    md.write_text(table + "\n")
    return md


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def build_table(out: str | Path, order: list[str] | None = None) -> str:
    out = Path(out)
    rows = []
    for p in sorted(out.glob("*.json")):
        res = json.loads(p.read_text())
        if "runs" not in res:
            continue
        name = (res.get("entry") or {}).get("name") or p.stem
        s = summarize(res["runs"])  # recomputed, so files from older versions get the same columns
        rows.append((name, res, s))
    if order:
        rank = {n: i for i, n in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r[0], len(rank)))
    head = [
        "| 模型 | 交出报告 | 检出率 | 精确率 | 正常试车误报 | 初稿中无出处的数字 | 校验后仍无出处 | 用到修正轮 | token/次 | 秒/次 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines = []
    for name, res, s in rows:
        total = s["runs_total"]
        f = s.get("first_submission")
        is_llm = f is not None or (s.get("usage") is not None)
        if f:
            ft = int(f["numbers_grounded"].split("/")[1])
            first = f"{f['ungrounded_numbers_caught']} / {ft}（{_pct(s['first_draft_ungrounded_rate'])}）"
        else:
            first = "–" if not is_llm else "未交报告"
        u = s.get("usage") or {}
        ok_runs = total - s["failed_runs"]
        tok = f"{(u.get('input_tokens', 0) + u.get('output_tokens', 0)) / ok_runs / 1000:.1f}k" \
            if u and ok_runs else "–"
        if s["mean_elapsed_s"] is None:  # every run crashed
            lines.append(f"| {name} | 0/{total} | 0% | – | – | – | – | – | – | – |")
            continue
        fixes = s["fix_rounds_used"] if is_llm else "–"
        lines.append(
            f"| {name} | {s['reports_submitted']}/{total} | {_pct(s['recall_all_runs'])} | {_pct(s['precision'])} | "
            f"{s['false_positives_on_nominal_runs']}（{s['nominal_runs']} 次） | {first} | "
            f"{s['ungrounded_numbers']} 个数字、{s['unsupported_claims']} 条结论 | {fixes} | {tok} | "
            f"{s['mean_elapsed_s']:.1f} |"
        )
    notes = [
        "",
        f"同一组 {rows[0][1].get('n_runs', '?')} 次合成试车（种子从 {rows[0][1].get('seed', '?')} 开始）。"
        if rows else "",
        "检出率按全部试车计算：崩溃或没有交报告的试车，其中的异常都算漏检。",
        "“初稿中无出处的数字”是校验器第一次拦下的数字（在所引用的证据里找不到），也就是没有校验器时会进入报告的编造数字。",
    ]
    return "\n".join(head + lines + notes)
