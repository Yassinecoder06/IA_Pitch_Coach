"""Markdown skill loading for LLM coaching prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@lru_cache(maxsize=64)
def load_skill(name: str) -> str:
    """Load a markdown skill by filename stem."""
    safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-"))
    path = SKILLS_DIR / f"{safe_name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def compose_skills(primary_skill: str, helper_skills: Iterable[str]) -> str:
    """Compose the primary mode skill with shared helper skills."""
    blocks: List[str] = []
    for skill_name in [primary_skill, *helper_skills]:
        skill = load_skill(skill_name)
        if skill:
            blocks.append(skill)

    return "\n\n---\n\n".join(blocks)
