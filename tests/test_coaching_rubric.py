from backend.coaching.rubric import parse_scorecard_from_text


def test_parse_json_scorecard():
    sample_response = """
Here is my feedback on your pitch. Your delivery was energetic and clear.

```json
{
  "overall": 88,
  "clarity": {"score": 9, "feedback": "Value proposition is crystal clear."},
  "structure": {"score": 8, "feedback": "Good progression from problem to solution."},
  "persuasion": {"score": 9, "feedback": "Compelling urgency."},
  "delivery": {"score": 8, "feedback": "Good cadence."},
  "confidence": {"score": 9, "feedback": "Authoritative tone."},
  "relevance": {"score": 10, "feedback": "Directly targets enterprise pain."},
  "checklist": {"hook": true, "problem": true, "solution": true, "market": false, "traction": true, "ask": false},
  "strengths": ["Clear hook", "Strong metric on traction"],
  "top_fixes": ["Clarify the size of the total addressable market", "State your funding ask clearly"],
  "rewrite": "We save enterprise teams 10 hours a week by automating document workflows."
}
```
    """
    card = parse_scorecard_from_text(sample_response)
    assert card is not None
    assert card["overall"] == 88
    assert card["clarity"]["score"] == 9
    assert card["checklist"]["hook"] is True
    assert card["checklist"]["market"] is False
    assert len(card["strengths"]) == 2
    assert "save enterprise teams" in card["rewrite"]


def test_fallback_regex_scorecard():
    legacy_response = """
SCORES:
Clarity: 8/10
Language: 7/10
Confidence: 9/10
Topic Relevance: 8/10
Structure: 8/10
Delivery: 7/10

ANALYSIS:
Strong pitch with clear focus on the customer problem.

ADVICE:
- Add a specific customer metric.
- Mention competitor differentiation.
    """
    card = parse_scorecard_from_text(legacy_response)
    assert card is not None
    assert card["clarity"]["score"] == 8
    assert card["persuasion"]["score"] == 7
    assert card["confidence"]["score"] == 9
    assert card["overall"] > 0
