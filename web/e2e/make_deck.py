"""Writes sample.pptx and sample.pdf (via LibreOffice) for the deck-import E2E tests.

Usage: python -I make_deck.py OUT_DIR
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.decks.pptx_tools import _convert_sync, make_sample_pptx  # noqa: E402

SLIDES = [
    {"title": "Shipping eats retail margins", "bullets": ["Small stores overpay 30%"], "notes": "Open with a store owner's story."},
    {"title": "FreightPilot pools the volume", "bullets": ["Big-brand carrier rates", "Automatic booking"], "notes": "Show the saving per parcel."},
    {"title": "Raising 1.5M seed", "bullets": ["40 paying stores", "500 by next year"], "notes": "Make the ask clearly."},
]

out = Path(sys.argv[1])
pptx = make_sample_pptx(SLIDES)
(out / "sample.pptx").write_bytes(pptx)
(out / "sample.pdf").write_bytes(_convert_sync(pptx))
print("sample deck written")
