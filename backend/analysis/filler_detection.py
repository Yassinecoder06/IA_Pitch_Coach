"""
Filler Word Detection
=====================
Lightweight detection of filler words in speech transcripts.
"""

import re
from typing import Dict, List
from dataclasses import dataclass


# Default filler words to detect
FILLER_WORDS = [
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "literally",
    "so",
    "well",
    "right",
    "i mean",
    "kind of",
    "sort of",
    "okay so"
]


@dataclass
class FillerAnalysis:
    """Result of filler word analysis."""
    total_count: int
    word_count: int
    filler_ratio: float  # Ratio of fillers to total words
    details: Dict[str, int]  # Count per filler word


class FillerDetector:
    """
    Detects and counts filler words in text.
    Configurable list of filler words.
    """

    def __init__(self, filler_words: List[str] = None):
        """
        Initialize the detector.

        Args:
            filler_words: List of filler words/phrases to detect.
                         Uses default list if not provided.
        """
        self.filler_words = filler_words or FILLER_WORDS.copy()

    def analyze(self, text: str) -> FillerAnalysis:
        """
        Analyze text for filler words.

        Args:
            text: Text to analyze

        Returns:
            FillerAnalysis with counts and details
        """
        if not text.strip():
            return FillerAnalysis(
                total_count=0,
                word_count=0,
                filler_ratio=0.0,
                details={}
            )

        text_lower = text.lower()
        word_count = len(text.split())
        filler_counts = {}

        for filler in self.filler_words:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(filler) + r"\b"
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                filler_counts[filler] = count

        total_fillers = sum(filler_counts.values())
        filler_ratio = total_fillers / word_count if word_count > 0 else 0.0

        return FillerAnalysis(
            total_count=total_fillers,
            word_count=word_count,
            filler_ratio=filler_ratio,
            details=filler_counts
        )

    def add_filler_word(self, word: str):
        """Add a filler word to detect."""
        if word.lower() not in self.filler_words:
            self.filler_words.append(word.lower())

    def remove_filler_word(self, word: str):
        """Remove a filler word from detection."""
        word_lower = word.lower()
        if word_lower in self.filler_words:
            self.filler_words.remove(word_lower)


# Global instance and convenience functions
_detector: FillerDetector = None


def get_detector() -> FillerDetector:
    """Get the global filler detector instance."""
    global _detector
    if _detector is None:
        _detector = FillerDetector()
    return _detector


def count_filler_words(text: str) -> Dict[str, int]:
    """
    Count filler words in text.

    Args:
        text: Text to analyze

    Returns:
        Dict mapping filler words to their counts
    """
    analysis = get_detector().analyze(text)
    return analysis.details


def get_total_filler_count(text: str) -> int:
    """
    Get total count of all filler words.

    Args:
        text: Text to analyze

    Returns:
        Total filler word count
    """
    analysis = get_detector().analyze(text)
    return analysis.total_count
