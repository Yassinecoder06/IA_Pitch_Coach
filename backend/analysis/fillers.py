"""
Context-aware filler detection.

Rules live in shared/fillers.json and are mirrored by web/src/lib/fillers.ts;
both implementations are tested against shared/filler_cases.json.

Unlike a plain word list, ambiguous words only count in filler position:
"like" in "it was, like, huge" but not "I'd like to"; "so" in "So, uh," but
not "So we raised"; "right" as a tag question ("..., right?") only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RULES_PATH = Path(__file__).resolve().parents[2] / "shared" / "fillers.json"
RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))

ALWAYS = set(RULES["always"])
DISCOURSE = set(RULES["discourse"])
PHRASES: List[List[str]] = RULES["phrases"]
YOU_KNOW_NOT_BEFORE = set(RULES["you_know_not_before"])
LIKE_FILLER_AFTER = set(RULES["like_filler_after"])
LIKE_NOT_AFTER = set(RULES["like_not_after"])
SENTENCE_MARKERS = set(RULES["sentence_markers"])

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


@dataclass
class Token:
    text: str  # lowercase, curly apostrophes normalised
    start: int  # char offsets in the source text
    end: int
    comma_before: bool
    punct_after: str  # "," "." "?" "!" or ""
    sentence_start: bool


@dataclass
class FillerMatch:
    filler: str
    start: int  # char offsets of the matched span
    end: int


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(text):
        between = text[prev_end:m.start()]
        after = text[m.end():m.end() + 1]
        sentence_start = not tokens or bool(re.search(r"[.!?]", between))
        tokens.append(Token(
            text=m.group(0).lower().replace("’", "'"),
            start=m.start(),
            end=m.end(),
            comma_before="," in between or ";" in between,
            punct_after=after if after in ",.?!;" else "",
            sentence_start=sentence_start,
        ))
        prev_end = m.end()
    return tokens


def _is_plain_filler(tok: Optional[Token]) -> bool:
    return tok is not None and (tok.text in ALWAYS or tok.text in DISCOURSE)


def find_fillers(text: str) -> List[FillerMatch]:
    tokens = tokenize(text)
    matches: List[FillerMatch] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        prev = tokens[i - 1] if i > 0 else None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        phrase = _match_phrase(tokens, i)
        if phrase:
            n = len(phrase)
            following = tokens[i + n] if i + n < len(tokens) else None
            # "you know what users want" is a real question, not a filler.
            if phrase != ["you", "know"] or following is None or following.text not in YOU_KNOW_NOT_BEFORE \
                    or tokens[i + n - 1].punct_after:
                matches.append(FillerMatch(" ".join(phrase), tok.start, tokens[i + n - 1].end))
                i += n
                continue

        w = tok.text
        if w in ALWAYS or w in DISCOURSE:
            matches.append(FillerMatch(w, tok.start, tok.end))
        elif w == "like" and _like_is_filler(tok, prev):
            matches.append(FillerMatch(w, tok.start, tok.end))
        elif w == "right" and _right_is_filler(tok, nxt):
            matches.append(FillerMatch(w, tok.start, tok.end))
        elif w in SENTENCE_MARKERS and w != "right" and _marker_is_filler(tok, prev, nxt):
            matches.append(FillerMatch(w, tok.start, tok.end))
        i += 1
    return matches


def _match_phrase(tokens: Sequence[Token], i: int) -> Optional[List[str]]:
    for phrase in PHRASES:
        n = len(phrase)
        if i + n <= len(tokens) and [t.text for t in tokens[i:i + n]] == phrase:
            # Punctuation inside the phrase breaks it ("you. Know") except after the last word.
            if all(not t.punct_after for t in tokens[i:i + n - 1]):
                return phrase
    return None


def _like_is_filler(tok: Token, prev: Optional[Token]) -> bool:
    if prev is not None and prev.text in LIKE_NOT_AFTER:
        return False
    if tok.comma_before or tok.punct_after == ",":
        return True
    return prev is not None and prev.text in LIKE_FILLER_AFTER


def _right_is_filler(tok: Token, nxt: Optional[Token]) -> bool:
    tag_question = tok.punct_after == "?" and (nxt is None or nxt.sentence_start)
    return tag_question or (tok.comma_before and tok.punct_after == ",")


def _marker_is_filler(tok: Token, prev: Optional[Token], nxt: Optional[Token]) -> bool:
    at_clause_start = tok.sentence_start or _is_plain_filler(prev)
    return at_clause_start and (tok.punct_after == "," or _is_plain_filler(nxt))


def count_fillers(text: str) -> Dict[str, int]:
    return dict(Counter(m.filler for m in find_fillers(text)))


def find_fillers_in_words(words: Sequence[dict]) -> List[dict]:
    """Fillers with timestamps, for Whisper word lists [{"w", "start", "end"}, ...]."""
    pieces, owner, offset = [], [], 0
    for idx, word in enumerate(words):
        text = word["w"]
        pieces.append(text)
        owner.extend([idx] * (len(text) + 1))  # +1 for the joining space
        offset += len(text) + 1
    joined = " ".join(pieces)
    out = []
    for m in find_fillers(joined):
        first, last = owner[m.start], owner[max(m.start, m.end - 1)]
        out.append({
            "filler": m.filler,
            "start": words[first]["start"],
            "end": words[last]["end"],
            "word_index": first,
        })
    return out
