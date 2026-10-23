"""Helpers for structured (JSON-schema constrained) LLM output."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Type

from pydantic import BaseModel

# Keys that some providers reject inside a response schema.
_STRIP_KEYS = {"title", "default", "examples", "$schema"}


def inline_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Pydantic JSON schema with every $ref inlined and decorative keys removed.

    Gemini, Ollama and Anthropic tool schemas each support a different JSON-schema
    subset; a self-contained schema without $defs works on all of them.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                merged = {**copy.deepcopy(defs[name]), **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            out = {}
            for k, v in node.items():
                if k in _STRIP_KEYS:
                    continue
                if k == "properties" and isinstance(v, dict):
                    out[k] = {name: resolve(prop) for name, prop in v.items()}  # property names are data, keep all
                else:
                    out[k] = resolve(v)
            # Optional[X] renders as anyOf[X, null]; collapse to X + nullable for picky providers.
            any_of = out.get("anyOf")
            if isinstance(any_of, list) and len(any_of) == 2 and {"type": "null"} in any_of:
                other = next(s for s in any_of if s != {"type": "null"})
                out.pop("anyOf")
                out = {**other, **out, "nullable": True}
            return out
        if isinstance(node, list):
            return [resolve(n) for n in node]
        return node

    return resolve(raw)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_text(text: str) -> str:
    """Best-effort extraction of a JSON object from model output (fences, chatter around it)."""
    text = (text or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def schema_instruction(schema: Dict[str, Any]) -> str:
    """Prompt text for providers without native schema support."""
    return (
        "Respond with a single JSON object only (no prose, no code fences) that conforms to this JSON Schema:\n"
        + json.dumps(schema, separators=(",", ":"))
    )
