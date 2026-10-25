"""
Filler word counting (compatibility wrapper around backend.analysis.fillers).
"""

from typing import Dict

from backend.analysis.fillers import count_fillers


def count_filler_words(text: str) -> Dict[str, int]:
    """Count filler words in text. Returns {filler: count}."""
    return count_fillers(text or "")


def get_total_filler_count(text: str) -> int:
    return sum(count_filler_words(text).values())
