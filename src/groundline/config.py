"""Project configuration from a ``.env`` file.

Groundline looks for ``.env`` in the current directory and its parents (up to
the filesystem root) and loads ``KEY=value`` lines into the environment.
Variables already set in the shell win over the file.  No extra dependency.

Recognised keys (see ``.env.example``)::

    OPENAI_API_KEY / ANTHROPIC_API_KEY / GROUNDLINE_LLM_API_KEY
    GROUNDLINE_AGENT        rule | openai | anthropic   (default agent for the CLI)
    GROUNDLINE_LLM_MODEL    model name
    GROUNDLINE_LLM_BASE_URL OpenAI-compatible endpoint (Qwen, DeepSeek, vLLM, Ollama ...)
    GROUNDLINE_LLM_TEMPERATURE
    GROUNDLINE_LLM_REASONING_EFFORT  none | low | medium | high (reasoning models; Anthropic also xhigh | max)
    GROUNDLINE_LANG         zh | en
    GROUNDLINE_CITE_NUMBERS 1 (default) asks LLM agents to tag numbers with their source field, 0 does not
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def find_dotenv(start: str | Path | None = None) -> Path | None:
    d = Path(start or Path.cwd()).resolve()
    for p in (d, *d.parents):
        f = p / ".env"
        if f.is_file():
            return f
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:  # inline comment on an unquoted value
            val = val.split(" #", 1)[0].rstrip()
        if key:
            out[key] = val
    return out


def load_dotenv(path: str | Path | None = None, override: bool = False) -> Path | None:
    f = Path(path) if path else find_dotenv()
    if not f or not f.is_file():
        return None
    for k, v in parse_dotenv(f.read_text(encoding="utf-8")).items():
        if override or k not in os.environ:
            os.environ[k] = v
    return f


def describe() -> dict:
    """What the CLI will use, with secrets masked (for `groundline config`)."""
    def mask(v: str | None) -> str | None:
        if not v:
            return None
        return v[:3] + "…" + v[-4:] if len(v) > 10 else "***"

    return {
        "dotenv": str(find_dotenv() or "(none found)"),
        "agent": os.environ.get("GROUNDLINE_AGENT", "rule"),
        "model": os.environ.get("GROUNDLINE_LLM_MODEL"),
        "base_url": os.environ.get("GROUNDLINE_LLM_BASE_URL", "https://api.openai.com/v1 (default)"),
        "reasoning_effort": os.environ.get("GROUNDLINE_LLM_REASONING_EFFORT", "(model default)"),
        "lang": os.environ.get("GROUNDLINE_LANG", "zh"),
        "cite_numbers": os.environ.get("GROUNDLINE_CITE_NUMBERS", "1"),
        "OPENAI_API_KEY": mask(os.environ.get("OPENAI_API_KEY")),
        "GROUNDLINE_LLM_API_KEY": mask(os.environ.get("GROUNDLINE_LLM_API_KEY")),
        "ANTHROPIC_API_KEY": mask(os.environ.get("ANTHROPIC_API_KEY")),
    }


# agent kinds served by an OpenAI-compatible endpoint (the aliases only name the server)
OPENAI_KINDS = ("openai", "openai-compatible", "qwen", "deepseek", "ollama", "vllm")
OPENAI_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class AgentConfig:
    """Which agent runs and how, resolved once. Backends read nothing from the environment: what is not set
    here takes the backend's own default."""

    kind: str = "rule"  # rule | anthropic | one of OPENAI_KINDS
    model: str | None = None
    base_url: str | None = None  # OpenAI-compatible endpoint (default: OPENAI_URL)
    api_key: str | None = None
    api: str | None = None  # responses | chat, for OPENAI_KINDS
    temperature: float | None = None  # None: not sent (reasoning models reject one)
    reasoning_effort: str | None = None
    timeout: float | None = None  # seconds per request
    max_tokens: int | None = None  # cap on one reply
    max_seconds: float | None = None  # wall-clock budget for one analysis
    cite_numbers: bool = True  # ask the model to tag each number with its source field

    @classmethod
    def from_env(cls, kind: str | None = None, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, env: Mapping[str, str] = os.environ) -> "AgentConfig":
        """The CLI's configuration: arguments, then GROUNDLINE_* variables (shell or .env)."""
        kind = kind or env.get("GROUNDLINE_AGENT", "rule")
        base_url = base_url or env.get("GROUNDLINE_LLM_BASE_URL")
        t = env.get("GROUNDLINE_LLM_TEMPERATURE")
        return cls(kind, model=model or env.get("GROUNDLINE_LLM_MODEL"), base_url=base_url,
                   api_key=api_key or _key_from_env(kind, env),
                   api=_resolve_api(kind, base_url, env.get("GROUNDLINE_OPENAI_API")),
                   temperature=float(t) if t else None,
                   reasoning_effort=env.get("GROUNDLINE_LLM_REASONING_EFFORT") or None,
                   cite_numbers=env.get("GROUNDLINE_CITE_NUMBERS", "1") != "0")

    @classmethod
    def from_entry(cls, e: Mapping, env: Mapping[str, str] = os.environ) -> "AgentConfig":
        """A leaderboard entry. The entry is the whole configuration: settings meant for the default model in
        .env (e.g. GROUNDLINE_LLM_REASONING_EFFORT for gpt-5.6-sol) must not leak into a local model. Only the
        API key may come from the environment (``api_key_env``, else the usual key variable)."""
        kind = e.get("agent", "openai")
        api = _resolve_api(kind, e.get("base_url"), e.get("api"))
        chat = api == "chat"  # local models: longer requests, and a reply cap so one stuck in a loop stops
        key = e.get("api_key") or (env.get(e["api_key_env"]) if e.get("api_key_env") else None)
        return cls(kind, model=e.get("model"), base_url=e.get("base_url"), api_key=key or _key_from_env(kind, env),
                   api=api, temperature=float(e["temperature"]) if e.get("temperature") is not None else None,
                   reasoning_effort=e.get("reasoning_effort"),
                   timeout=float(e["timeout"]) if "timeout" in e else (600.0 if chat else None),
                   max_tokens=int(e["max_tokens"]) if "max_tokens" in e else (2048 if chat else None),
                   max_seconds=float(e.get("run_timeout_s", 1200)), cite_numbers=bool(e.get("cite_numbers", True)))


def _resolve_api(kind: str, base_url: str | None, api: str | None) -> str | None:
    """Official OpenAI -> Responses API (tools + reasoning); other endpoints -> Chat Completions."""
    if kind not in OPENAI_KINDS:
        return None
    return api or ("responses" if kind == "openai" and "api.openai.com" in (base_url or OPENAI_URL) else "chat")


def _key_from_env(kind: str, env: Mapping[str, str]) -> str | None:
    if kind == "anthropic":
        return env.get("ANTHROPIC_API_KEY") or None
    if kind in OPENAI_KINDS:
        return env.get("GROUNDLINE_LLM_API_KEY") or env.get("OPENAI_API_KEY") or None
    return None
