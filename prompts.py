from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
    _RAW_PROMPTS = yaml.safe_load(f)


def load_prompt(key: str) -> ChatPromptTemplate:
    """Loads a named prompt's text from prompts.yaml and wraps it exactly
    like an inline ChatPromptTemplate.from_template(...) call would.
    {placeholder} syntax works identically to before — nothing about how a
    pattern file INVOKES a prompt changes, only where the text lives."""
    if key not in _RAW_PROMPTS:
        raise KeyError(f"No prompt named '{key}' found in prompts.yaml")
    return ChatPromptTemplate.from_template(_RAW_PROMPTS[key])


def load_raw(key: str) -> str:
    """Returns the raw prompt text without wrapping — needed for
    agentic_rag.py's SystemMessage, which isn't a ChatPromptTemplate."""
    if key not in _RAW_PROMPTS:
        raise KeyError(f"No prompt named '{key}' found in prompts.yaml")
    return _RAW_PROMPTS[key]