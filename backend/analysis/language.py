"""
Word-choice coaching: hedging, vague words, apologies, jargon, passive voice,
weak verbs and repeated words. Lexicon in shared/language.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

LEXICON = json.loads((Path(__file__).resolve().parents[2] / "shared" / "language.json").read_text(encoding="utf-8"))
RISK = sorted(LEXICON["risk"], key=lambda e: -len(e["tokens"]))  # longest patterns first
PLAIN = LEXICON["plain_english"]
WEAK = LEXICON["weak_verbs"]
BE = set(LEXICON["be_verbs"])
STOP = set(LEXICON["stopwords"])
PARTICIPLE_EXCEPTIONS = {"need", "speed", "seed", "feed", "red", "bed", "shed", "hundred", "based", "used", "excited", "interested"}


def _norm(word: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", word.lower().replace("’", "'"))


def analyse_language(words: Sequence[dict]) -> Dict[str, Any]:
    tokens = [_norm(w["w"]) for w in words]
    items: List[Dict[str, Any]] = []
    taken = set()

    for entry in RISK:
        pat = entry["tokens"]
        n = len(pat)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == pat and not taken.intersection(range(i, i + n)):
                taken.update(range(i, i + n))
                items.append({
                    "kind": entry["category"], "text": " ".join(pat), "word_index": i,
                    "t": words[i]["start"], "tip": entry["tip"],
                })

    for i, tok in enumerate(tokens):
        if tok in PLAIN:
            items.append({"kind": "jargon", "text": tok, "word_index": i, "t": words[i]["start"],
                          "tip": f'Say "{PLAIN[tok]}" instead.'})
        if tok in WEAK and i + 1 < len(tokens):
            items.append({"kind": "weak_verb", "text": tok, "word_index": i, "t": words[i]["start"],
                          "tip": f"Use a stronger verb: {WEAK[tok]}."})
        # Passive voice: be-verb (+ optional adverb) + past participle, e.g. "was created", "are being used".
        if tok in BE:
            j = i + 1
            if j < len(tokens) and tokens[j] in ("being", "been"):
                j += 1
            if j < len(tokens) and tokens[j].endswith("ly"):
                j += 1
            if j < len(tokens) and len(tokens[j]) > 3 and tokens[j] not in PARTICIPLE_EXCEPTIONS \
                    and (tokens[j].endswith("ed") or tokens[j] in ("built", "made", "done", "given", "taken", "shown", "seen")):
                items.append({"kind": "passive", "text": " ".join(tokens[i:j + 1]), "word_index": i,
                              "t": words[i]["start"], "tip": "Say who did it: active voice is clearer and stronger."})

    # Repetition: a content word used 4+ times within any 60-word stretch.
    repeated: Dict[str, int] = {}
    for start in range(0, max(1, len(tokens) - 59), 20):
        window = Counter(t for t in tokens[start:start + 60] if t and t not in STOP and len(t) > 3)
        for word, c in window.items():
            if c >= 4:
                repeated[word] = max(repeated.get(word, 0), c)
    for word, c in repeated.items():
        first = tokens.index(word)
        items.append({"kind": "repetition", "text": word, "word_index": first, "t": words[first]["start"],
                      "tip": f'"{word}" comes up {c} times in a short stretch: vary it or cut repeats.'})

    items.sort(key=lambda x: x["t"])
    counts = Counter(i["kind"] for i in items)
    minutes = max((words[-1]["end"] - words[0]["start"]) / 60, 1 / 60) if words else 1
    return {
        "counts": dict(counts),
        "hedges_per_minute": round((counts["hedge"] + counts["apology"]) / minutes, 1),
        "items": items,
    }
