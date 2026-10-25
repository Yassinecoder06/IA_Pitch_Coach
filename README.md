# AI Pitch Coach

A private, local-first rehearsal studio for founders and speakers. Rehearse a deck slide by slide while your
camera and microphone are analysed **on your own machine**, then get a full report: pace, fillers, pauses,
vocal variety, upspeak, eye contact, posture, word choice, per-slide timing and AI coaching from the model of
your choice. Track progress across attempts, compare takes, set goals, export a PDF, and drill specific
habits in the training arcade.

## What it does

| Area | Features |
|---|---|
| **Coach** | Voice or text pitch turns; 8 coaching modes (pitch analysis, investor Q&A, objection handling...); streamed feedback; structured 6-dimension scorecard |
| **Live practice** | Live captions (local Whisper), eye contact, posture, fidgeting, sway, pitch range, energy, pace and fillers on a heads-up display |
| **Decks** | Upload PDF or PPTX (speaker notes kept), write an outline, or generate a pitch deck with AI; slide critique; export to PPTX |
| **Rehearsal Studio** | Presenter view with current/next slide, teleprompter notes, per-slide and total timers, keyboard navigation |
| **Report** | Overall, delivery and content scores; 11 delivery metrics; timeline synced to the recording; clickable transcript; slide-by-slide table; evidence-backed coaching notes; word-choice tips; rewrite of the weakest passage; one-click investor Q&A on your weak points |
| **History** | Trends across attempts, best take, side-by-side comparison (deltas plus resolved/persisting/new issues), goals, printable PDF report |
| **Arcade** | Filler Ninja, Pitch Surfer, Eye Lock, Stage Presence, Pace Racer, Pause Power; difficulties, personal bests, daily streak, a daily queue that targets your weakest metrics |
| **Accounts** | One owner passphrase per install, plus local speaker profiles (optional 4-digit PIN), each with its own data |
| **Bring your own AI** | OpenAI, Anthropic, Gemini, Ollama (local or cloud) and any OpenAI-compatible API; model per task (chat, scoring, deck generation, Q&A) with automatic fallback |

## Privacy model

- **Camera frames never leave the browser.** MediaPipe face and pose models run in a Web Worker from files
  served by this app (no CDN). Only numbers (eye-contact share, posture score...) are stored.
- **Audio is transcribed locally** with faster-whisper. Recordings can be kept, auto-deleted after N days,
  or not kept at all (Settings › Privacy).
- **Only the transcript and measurements** go to an AI provider, and only to the one you configure. With
  local Ollama nothing leaves the machine.
- API keys are encrypted at rest (AES-GCM) with a random key generated on first run in `data/secret.key`.
- The session is an HttpOnly, SameSite=Strict cookie; the WebSocket checks it and the page origin.

## Quick start

### Docker

```bash
docker compose up --build
```

Open http://localhost:8000. The first run prints a **setup code** in the logs; enter it with a passphrase to
claim the install. The default image includes LibreOffice for PPTX import (`APP_TARGET=runtime` builds a
smaller image without it).

### Local

Requirements: Python 3.10+, Node 22, ffmpeg. Optional: LibreOffice (PPTX import), an NVIDIA GPU (Whisper on CUDA).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd web && npm ci && npm run build && cd ..     # also downloads the MediaPipe models into web/public/mediapipe
uvicorn backend.main:app --port 8000
```

For development, `make dev` runs the API with reload and the Vite dev server on http://localhost:5173.

### Configure AI

Settings › AI providers: add a provider and key, pick a primary model, optional fallbacks, and optionally a
different model per task. Without any provider, the app uses a local Ollama at `OLLAMA_ENDPOINT` if one is
running. Every measured part of the report works without AI; only the written feedback needs a model.

### Useful settings (`.env`)

| Variable | Default | |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/pitch_coach.db` | or `postgresql://user:pw@host/db` |
| `SECRET_KEY` | generated into `data/secret.key` | set explicitly to share between replicas |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` | `tiny.en` / `cpu` | e.g. `medium.en` / `cuda` |
| `SECURE_COOKIES` | `false` | `true` when served over HTTPS |
| `CORS_ORIGINS` | Vite dev origins | |

Forgot the passphrase? `python -m backend.cli reset-owner`, then restart for a new setup code (data and API
keys are kept).

## Testing

```bash
make test        # backend unit/integration tests (SQLite; set TEST_POSTGRES_URL to run them on PostgreSQL)
make test-web    # typecheck, lint, vitest
make test-live   # real provider calls; needs GEMINI_API_KEY and/or OLLAMA_API_KEY
make e2e         # Playwright + Chrome against the real backend, with fake camera/microphone input
```

The end-to-end suite feeds Chrome a Piper-voiced pitch as the microphone and MediaPipe's own test images as
the camera, so speech recognition, eye contact and posture are checked on real models. `E2E_GPU=1` runs
browser vision on an NVIDIA GPU (PRIME offload on hybrid laptops).

## Architecture

```
Browser                                              FastAPI (local)
  AudioWorklet → 16 kHz PCM16 ── binary WebSocket ─▶ chunked Whisper (word timestamps) → live captions
  YIN pitch / energy / pauses                        analysis: pace, pauses, fillers, pYIN prosody,
  MediaPipe worker → gaze, posture, motion           upspeak, word choice, clarity, per-slide segments,
  pdf.js renders uploaded decks                      cues → LLM judge (structured output, cited evidence)
  React 18 · router · TanStack Query · zustand ◀REST  SQLAlchemy 2 + Alembic (SQLite or PostgreSQL)
                                                     LLM router: per-task model, fallback, cooldown
```

- `backend/analysis/` delivery analysis (pure functions), `backend/rehearsals/` lifecycle, compare, goals,
  privacy, `backend/decks/` decks, `backend/llm/` providers and router, `backend/db/` models and migrations
- `web/src/lib/sensors/` browser sensors, `web/src/features/` one folder per screen
- `shared/` data used by both sides (filler rules, word-choice lexicon), with parity tests

## License

MIT. Third-party notices, including code and data adapted from iamspeaker (MIT), are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
