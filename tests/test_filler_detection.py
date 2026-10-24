import json
from pathlib import Path

import pytest

from backend.analysis.filler_detection import count_filler_words, get_total_filler_count
from backend.analysis.fillers import find_fillers_in_words

CASES = json.loads((Path(__file__).parents[1] / "shared" / "filler_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:40] for c in CASES])
def test_shared_filler_cases(case):
    """Same cases as web/src/lib/__tests__/fillers.test.ts, so both detectors agree."""
    assert count_filler_words(case["text"]) == case["expected"]


def test_total_count():
    assert get_total_filler_count("Um, so basically, our product is like super fast, you know?") == 5
    assert get_total_filler_count("") == 0


def test_fillers_get_word_timestamps():
    words = [
        {"w": "So,", "start": 0.0, "end": 0.3},
        {"w": "um,", "start": 0.4, "end": 0.6},
        {"w": "you", "start": 0.7, "end": 0.8},
        {"w": "know,", "start": 0.8, "end": 1.0},
        {"w": "we", "start": 1.1, "end": 1.2},
        {"w": "grew.", "start": 1.2, "end": 1.5},
    ]
    found = find_fillers_in_words(words)
    assert [(f["filler"], f["start"], f["end"]) for f in found] == [
        ("so", 0.0, 0.3), ("um", 0.4, 0.6), ("you know", 0.7, 1.0)
    ]
