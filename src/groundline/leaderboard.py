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

from .evaluate import BenchmarkInterrupted, is_infra_error, run_benchmark, save, summarize


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
        be = A.OpenAICompatible(e.get("model"), url, key, timeout=float(e.get("timeout", 600)),
                                max_tokens=int(e.get("max_tokens", 2048)))
        # the entry is the whole configuration: settings meant for the default model in .env
        # (e.g. GROUNDLINE_LLM_REASONING_EFFORT for gpt-5.6-sol) must not leak into a local model
        be.reasoning_effort = e.get("reasoning_effort")
        be.temperature = float(e["temperature"]) if e.get("temperature") is not None else None
    return A.LLMAgent(be, lang, max_seconds=float(e.get("run_timeout_s", 1200)))


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
        prev = json.loads(path.read_text()) if path.exists() and not force else None
        if prev and (prev.get("n_runs") != n or prev.get("seed") != seed):
            prev = None
        if prev and not prev.get("incomplete") and not any(is_infra_error(r.get("error", ""))
                                                           for r in prev.get("runs", [])) \
                and any("error" not in r for r in prev.get("runs", [])):
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
            resume = [r for r in prev["runs"]] if prev else None
            kept = sum(1 for r in resume or [] if not is_infra_error(r.get("error", "")))
            print(f"[{name}] running {n} tests" + (f" (resuming, {kept} already done)" if kept else "") + "...",
                  file=sys.stderr)

            def prog(i, total, name=name):
                print(f"\r  [{name}] run {i}/{total}", end="", file=sys.stderr, flush=True)

            entry = {k: v for k, v in e.items() if k not in ("api_key",)}

            def checkpoint(rows, path=path, entry=entry):
                # written after every run, so Ctrl-C or a crash loses at most the run in progress
                save({"n_runs": n, "seed": seed, "incomplete": True, "runs": rows, "summary": summarize(rows),
                      "entry": entry}, path)

            try:
                # a non-connection failure on the first run is almost always configuration (URL, key, model
                # name, unsupported option): stop this model instead of failing the same way n times
                res = run_benchmark(lambda e=e: make_entry_agent(e, lang), n=n, seed=seed, progress=prog,
                                    resume_rows=resume, infra_retries=1, stop_on_infra=True,
                                    on_row=checkpoint)
            except BenchmarkInterrupted as bi:
                part = {"n_runs": n, "seed": seed, "incomplete": True, "runs": bi.rows,
                        "summary": summarize(bi.rows) if bi.rows else {},
                        "entry": {k: v for k, v in e.items() if k not in ("api_key",)}}
                save(part, path)
                print(f"\n[{name}] lost the model server after {len(bi.rows)} of {n} runs ({bi}).\n"
                      f"  Saved what is done. Check that the server (e.g. the Ollama app) is running, then run the "
                      f"same command again: it continues from run {len(bi.rows) + 1}.", file=sys.stderr)
                continue
            except KeyboardInterrupt:
                print(f"\n[{name}] stopped. Finished runs are saved in {path}; run the same command to continue.",
                      file=sys.stderr)
                raise SystemExit(130) from None
            except Exception as err:
                print(f"\n[{name}] first run failed, skipping this model: {type(err).__name__}: {err}"[:900],
                      file=sys.stderr)
                continue
            print(file=sys.stderr)
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
        if res.get("incomplete"):
            name += f"（未跑完，{len(res['runs'])}/{res.get('n_runs')}）"
        s = summarize(res["runs"])  # recomputed, so files from older versions get the same columns
        rows.append((name, res, s))
    if order:
        rank = {n: i for i, n in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r[0].split("（未跑完")[0], len(rank)))
    head = [
        "| 模型 | 交出报告 | 检出率（严格 / 宽松） | 精确率 | 正常试车误报 | 初稿中无出处的数字 | 校验后仍有问题 | 用到修正轮 | token/次 | 秒/次 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines = []
    for name, _res, s in rows:
        total = s["runs_total"]
        f = s.get("first_submission")
        is_llm = f is not None or (s.get("usage") is not None)
        if f:
            ft = int(f["numbers_grounded"].split("/")[1])
            first = f"{f['ungrounded_numbers_caught']} / {ft}（{_pct(s['first_draft_ungrounded_rate'])}）"
            if f.get("semantic_mismatches"):
                first += f"，另有 {f['semantic_mismatches']} 个含义不符"
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
            f"| {name} | {s['reports_submitted']}/{total} | {_pct(s['recall_all_runs'])} / "
            f"{_pct(s['recall_loose_all_runs'])} | {_pct(s['precision'])} | "
            f"{s['false_positives_on_nominal_runs']}（{s['nominal_runs']} 次） | {first} | "
            f"{s['ungrounded_numbers'] + s.get('semantic_mismatches', 0)} 个数字、{s['unsupported_claims']} 条结论 | "
            f"{fixes} | {tok} | "
            f"{s['mean_elapsed_s']:.1f} |"
        )
    notes = [
        "",
        f"同一组 {rows[0][1].get('n_runs', '?')} 次合成试车（种子从 {rows[0][1].get('seed', '?')} 开始）。"
        if rows else "",
        "检出率按全部试车计算：崩溃或没有交报告的试车，其中的异常都算漏检。严格：结论必须写明通道且时间对得上；"
        "宽松：结论没填通道时，只要类别对、时间不冲突也算检出（模型发现了问题，但没填好结构化字段）。",
        "“初稿中无出处的数字”是校验器第一次拦下的数字（在所引用的证据里找不到），也就是没有校验器时会进入报告的编造数字。",
    ]
    return "\n".join(head + lines + notes)
