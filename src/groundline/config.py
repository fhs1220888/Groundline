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
    GROUNDLINE_LANG         zh | en
"""

from __future__ import annotations

import os
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
        "lang": os.environ.get("GROUNDLINE_LANG", "zh"),
        "OPENAI_API_KEY": mask(os.environ.get("OPENAI_API_KEY")),
        "GROUNDLINE_LLM_API_KEY": mask(os.environ.get("GROUNDLINE_LLM_API_KEY")),
        "ANTHROPIC_API_KEY": mask(os.environ.get("ANTHROPIC_API_KEY")),
    }
