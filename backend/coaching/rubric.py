"""
6-Dimensional Pitch Rubric and resilient Scorecard parser.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    score: float = Field(ge=0, le=10, description="0-10")
    feedback: str = Field(default="", description="One specific sentence that cites the speaker's words.")


class StructureChecklist(BaseModel):
    hook: bool = False
    problem: bool = False
    solution: bool = False
    market: bool = False
    traction: bool = False
    ask: bool = False


class Scorecard(BaseModel):
    """6-dimension pitch rubric. Produced by the judge model as structured output."""

    overall: float = Field(ge=0, le=100, description="0-100 holistic score")
    clarity: DimensionScore
    structure: DimensionScore
    persuasion: DimensionScore
    delivery: DimensionScore
    confidence: DimensionScore
    relevance: DimensionScore
    checklist: StructureChecklist = Field(default_factory=StructureChecklist)
    strengths: List[str] = Field(default_factory=list)
    top_fixes: List[str] = Field(default_factory=list)
    rewrite: Optional[str] = Field(default=None, description="Punchier rewrite of the weakest sentence.")


def parse_scorecard_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Legacy fallback: extract scorecard JSON or parse "Clarity: 8/10" style scores from free text.
    New code asks the judge model for a `Scorecard` via structured output instead.
    """
    if not text:
        return None

    # Try matching explicit JSON block
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "clarity" in data or "overall" in data:
                return _normalize_scorecard_dict(data)
        except Exception:
            pass

    # Try finding naked JSON object with overall or clarity
    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            data = json.loads(candidate)
            if "clarity" in data or "overall" in data:
                return _normalize_scorecard_dict(data)
    except Exception:
        pass

    # Fallback regex extraction for legacy SCORES: Clarity: X/10 format
    clarity_m = re.search(r"Clarity:\s*(\d+)(?:/10)?", text, re.IGNORECASE)
    language_m = re.search(r"(?:Language|Persuasion):\s*(\d+)(?:/10)?", text, re.IGNORECASE)
    confidence_m = re.search(r"Confidence:\s*(\d+)(?:/10)?", text, re.IGNORECASE)
    relevance_m = re.search(r"(?:Topic Relevance|Relevance):\s*(\d+)(?:/10)?", text, re.IGNORECASE)
    structure_m = re.search(r"Structure:\s*(\d+)(?:/10)?", text, re.IGNORECASE)
    delivery_m = re.search(r"Delivery:\s*(\d+)(?:/10)?", text, re.IGNORECASE)

    c = int(clarity_m.group(1)) if clarity_m else 7
    lang = int(language_m.group(1)) if language_m else 7
    conf = int(confidence_m.group(1)) if confidence_m else 7
    r = int(relevance_m.group(1)) if relevance_m else 7
    s = int(structure_m.group(1)) if structure_m else 7
    d = int(delivery_m.group(1)) if delivery_m else 7

    overall = int(round((c + lang + conf + r + s + d) / 6.0 * 10))

    # Infer checklist from transcript/response
    lower = text.lower()
    checklist = {
        "hook": any(w in lower for w in ["hook", "grab", "attention"]),
        "problem": any(w in lower for w in ["problem", "pain", "need", "frustration"]),
        "solution": any(w in lower for w in ["solution", "product", "platform", "app"]),
        "market": any(w in lower for w in ["market", "billion", "tam", "customers", "users"]),
        "traction": any(w in lower for w in ["traction", "growth", "revenue", "waitlist", "pilot"]),
        "ask": any(w in lower for w in ["ask", "raising", "seeking", "looking for"]),
    }

    return {
        "overall": overall,
        "clarity": {"score": min(max(c, 0), 10), "feedback": ""},
        "structure": {"score": min(max(s, 0), 10), "feedback": ""},
        "persuasion": {"score": min(max(lang, 0), 10), "feedback": ""},
        "delivery": {"score": min(max(d, 0), 10), "feedback": ""},
        "confidence": {"score": min(max(conf, 0), 10), "feedback": ""},
        "relevance": {"score": min(max(r, 0), 10), "feedback": ""},
        "checklist": checklist,
        "strengths": [],
        "top_fixes": [],
        "rewrite": None
    }


def _normalize_scorecard_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    def norm_dim(val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            s = int(val.get("score", 7))
            return {"score": min(max(s, 0), 10), "feedback": str(val.get("feedback", ""))}
        try:
            s = int(val)
            return {"score": min(max(s, 0), 10), "feedback": ""}
        except Exception:
            return {"score": 7, "feedback": ""}

    clarity = norm_dim(data.get("clarity", 7))
    structure = norm_dim(data.get("structure", 7))
    persuasion = norm_dim(data.get("persuasion", data.get("language", 7)))
    delivery = norm_dim(data.get("delivery", 7))
    confidence = norm_dim(data.get("confidence", 7))
    relevance = norm_dim(data.get("relevance", data.get("topic_relevance", 7)))

    overall = data.get("overall")
    if overall is None:
        overall = int(round((clarity["score"] + structure["score"] + persuasion["score"] + delivery["score"] + confidence["score"] + relevance["score"]) / 6.0 * 10))
    else:
        overall = min(max(int(overall), 0), 100)

    raw_checklist = data.get("checklist") or {}
    checklist = {
        "hook": bool(raw_checklist.get("hook", False)),
        "problem": bool(raw_checklist.get("problem", False)),
        "solution": bool(raw_checklist.get("solution", False)),
        "market": bool(raw_checklist.get("market", False)),
        "traction": bool(raw_checklist.get("traction", False)),
        "ask": bool(raw_checklist.get("ask", False)),
    }

    return {
        "overall": overall,
        "clarity": clarity,
        "structure": structure,
        "persuasion": persuasion,
        "delivery": delivery,
        "confidence": confidence,
        "relevance": relevance,
        "checklist": checklist,
        "strengths": data.get("strengths", []),
        "top_fixes": data.get("top_fixes", []),
        "rewrite": data.get("rewrite")
    }
