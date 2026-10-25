"""
Clarity flags from recognition confidence, plus key deck terms that were never said clearly.

This is a proxy, not phoneme-level scoring: Whisper's per-word probability drops on
mumbled, rushed or mispronounced words.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

LOW_PROB = 0.45
MIN_LEN = 4


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


COMMON = {
    "because", "therefore", "everything", "something", "customers", "different", "important", "business",
    "today", "first", "second", "third", "problem", "solution", "market", "traction", "about", "their",
    "there", "these", "those", "which", "where", "while", "would", "could", "should", "every", "other",
}


def key_terms(texts: Sequence[str], limit: int = 25) -> List[str]:
    """Distinctive terms from slide titles and bullets: names (capitalised mid-sentence) and
    long words. Sentence-initial capitals are ignored. Figures are left out on purpose:
    "$1.5M" on a slide is transcribed as "1.5 million", so it would always look missing."""
    terms: Dict[str, int] = {}
    for text in texts:
        for sentence in re.split(r"(?<=[.!?:])\s+|\n+", text or ""):
            tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
            for i, tok in enumerate(tokens):
                low = tok.lower()
                is_name = i > 0 and tok[0].isupper() and len(tok) >= 4
                is_long = len(tok) >= 9 and tok.isalpha()
                if (is_name or is_long) and low not in COMMON:
                    terms[low] = terms.get(low, 0) + 1
    return sorted(terms, key=lambda t: -terms[t])[:limit]


def clarity(words: Sequence[dict], deck_texts: Sequence[str] = ()) -> Dict[str, Any]:
    unclear = [
        {"word_index": i, "text": w["w"].strip(), "t": w["start"], "prob": w.get("prob", 1.0)}
        for i, w in enumerate(words)
        if w.get("prob", 1.0) < LOW_PROB and len(_norm(w["w"])) >= MIN_LEN
    ]
    probs = [w.get("prob", 1.0) for w in words]
    spoken: Dict[str, List[float]] = {}
    for i, w in enumerate(words):
        spoken.setdefault(_norm(w["w"]), []).append(w.get("prob", 1.0))
        if i + 1 < len(words):  # "Freight Pilot" said for "FreightPilot"
            pair = _norm(w["w"]) + _norm(words[i + 1]["w"])
            spoken.setdefault(pair, []).append(min(w.get("prob", 1.0), words[i + 1].get("prob", 1.0)))
    missing, mumbled = [], []
    for term in key_terms(deck_texts):
        if term not in spoken:
            missing.append(term)
        elif max(spoken[term]) < LOW_PROB:
            mumbled.append(term)
    return {
        "avg_confidence": round(sum(probs) / len(probs), 3) if probs else 0.0,
        "unclear_count": len(unclear),
        "unclear": unclear[:40],
        "key_terms_missing": missing,
        "key_terms_unclear": mumbled,
    }
