# AI Pitch Coach

AI Pitch Coach is a real-time, voice-first coaching application for startup pitch practice and public speaking improvement. It combines faster-whisper for speech-to-text, Piper for text-to-speech, multiple LLM providers, and optional Supabase-backed session persistence.

The platform captures spoken input from the browser, transcribes speech, analyzes content with a selectable LLM provider, and returns both written and spoken coaching feedback.

## Features

- **Real-time Voice Pipeline:** WebSocket-based streaming audio capture and response.
- **Speech-to-Text (STT):** Local transcription using faster-whisper.
- **Dynamic Provider Registry:** Multi-provider LLM orchestration supporting Ollama, OpenAI, Anthropic, Google, Azure, DeepSeek, and more.
- **Text-to-Speech (TTS):** Real-time synthesized feedback with Piper.
- **Browser-based UI:** Plain HTML/CSS/JS frontend featuring live transcripts, streaming analysis, scorecards, and audio playback.
- **Coaching Modes:** Pitch Analysis, Interactive Coaching, Investor Q&A, and more.
- **Filler-word Detection:** Real-time computation and reporting.
- **Session Persistence:** Optional Supabase integration for storing sessions, messages, and speech metrics.
- **MCP Tool Server:** Optional standalone server for web search and URL fetching.

## Coaching Modes & AI Skills

The application uses specific AI skill prompts (located in `backend/skills/*.md`) to drive the LLM behavior:

- **Pitch Analysis:** Evaluates clarity, language, confidence, and topic relevance with structured scoring and actionable advice.
- **Interactive Coaching:** Provides back-and-forth conversational feedback, offering one specific piece of advice per turn.
- **Investor Q&A:** Simulates a seasoned investor probing problem/solution fit, market size, competition, and traction.
- **Conversation Practice:** Supportive partner for general speaking practice and communication coaching.
- **Delivery Coaching:** Focuses on pace, pauses, filler words, and vocal confidence.
- **Objection Handling:** Helps speakers identify hidden concerns and formulate calm, direct answers to common investor pushback.
- **Pitch Rewrite:** Suggests clearer, more concise wording that leads with customer pain and specific outcomes.
- **Web Research:** Integrates live market facts, competitor context, and investor information into coaching responses.

## System Architecture

1. Browser records microphone audio and streams chunks over WebSocket.
2. Backend combines chunks to WAV.
3. faster-whisper transcribes audio and returns transcript + confidence.
4. Filler-word analyzer computes count + details.
5. Pitch analyzer streams LLM response in chunks.
6. Backend streams text feedback to client in real time.
7. Backend sends scores (for pitch-analysis mode).
8. Backend synthesizes TTS sentence-by-sentence and streams WAV audio chunks back.

**Main Components:**
- Backend API and WebSocket server: FastAPI
- STT engine: faster-whisper
- LLM layer: Local and cloud providers via unified interfaces
- TTS engine: Piper
- Frontend: HTML/CSS/JavaScript UI

---

## Quick Start & Setup

### 1. Requirements

- Python 3.10 or newer
- 8 GB RAM minimum
- A browser with microphone access

### 2. Environment Setup

Create a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install Python requirements:
```bash
pip install -r requirements.txt
```

Create your environment file:
```bash
cp .env.example .env.local
```
*(On Windows, use `copy .env.example .env.local`)*

### 3. Install Speech-to-Text (faster-whisper)

The app uses `faster-whisper`. You can explicitly pre-cache the `tiny` English model using Python:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8', download_root='models/whisper')"
```

Set your preferred model size and settings in `.env.local`:
```env
WHISPER_MODEL_SIZE=tiny.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
WHISPER_MODELS_DIR=models/whisper
```
*(Available sizes: `tiny`, `tiny.en`, `base`, `small`, `medium`, `large-v3`)*

### 4. Install Text-to-Speech (Piper)

Install the Piper TTS library:

```bash
pip install piper-tts
```

Download the `en_US-lessac-medium` voice model:

```bash
# Linux / macOS / Windows (Git Bash or PowerShell with curl)
mkdir -p models/piper
cd models/piper
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ../..
```

*(Optional) The backend can also auto-install Piper voices at runtime if you set `AUTO_INSTALL_TTS=true` in your `.env.local`.*

### 5. Local LLM (Ollama)

Ollama is optional but recommended for local processing.
1. Download from [Ollama.com](https://ollama.com/download)
2. Run in a terminal:
```bash
ollama pull qwen3:0.6b
ollama serve
```

Configure your `.env.local`:
```env
DEFAULT_LLM=ollama
OLLAMA_ENDPOINT=http://localhost:11434
```
You can also configure cloud providers (OpenAI, Anthropic, Google, etc.) by adding their respective API keys.

---

## Extended Setup

### Supabase Integration (Sessions & Metrics)

Supabase provides persistent session history, messages, and speech metrics.

1. **Create a Project:** Go to [Supabase](https://supabase.com) and create a project.
2. **Environment Variables:** Add to `.env.local`:
```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SESSION_CONTEXT_WINDOW=8
```
*(Note: Use `SUPABASE_SERVICE_ROLE_KEY` only on the backend.)*

3. **Database Schema:** Run the following SQL in the Supabase SQL Editor to create required tables and policies:

<details>
<summary>Click to view full SQL schema</summary>

```sql
create extension if not exists pgcrypto;

create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  title text not null default 'Untitled Session',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  current_mode text not null default 'pitch_analysis',
  context_md text not null default ''
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  transcript text,
  created_at timestamptz not null default now(),
  audio_url text
);

create table if not exists public.speech_metrics (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  words_per_minute double precision not null default 0,
  pause_frequency double precision not null default 0,
  pause_duration double precision not null default 0,
  energy_variation double precision not null default 0,
  rhythm_score double precision not null default 0,
  created_at timestamptz not null default now()
);

alter table public.sessions alter column user_id set default auth.uid();
alter table public.messages alter column user_id set default auth.uid();
alter table public.speech_metrics alter column user_id set default auth.uid();

create index if not exists idx_messages_session_created_at on public.messages(session_id, created_at);
create index if not exists idx_speech_metrics_session_created_at on public.speech_metrics(session_id, created_at desc);
create index if not exists idx_sessions_updated_at on public.sessions(updated_at desc);
create index if not exists idx_sessions_user_updated_at on public.sessions(user_id, updated_at desc);
create index if not exists idx_messages_user_session_created_at on public.messages(user_id, session_id, created_at);
create index if not exists idx_speech_metrics_user_session_created_at on public.speech_metrics(user_id, session_id, created_at desc);

create or replace function public.set_updated_at() returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_sessions_set_updated_at on public.sessions;
create trigger trg_sessions_set_updated_at before update on public.sessions for each row execute function public.set_updated_at();

alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.speech_metrics enable row level security;

create policy sessions_select_own on public.sessions for select using (auth.uid() = user_id);
create policy sessions_insert_own on public.sessions for insert with check (auth.uid() = user_id);
create policy sessions_update_own on public.sessions for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy sessions_delete_own on public.sessions for delete using (auth.uid() = user_id);

create policy messages_select_own on public.messages for select using (auth.uid() = user_id);
create policy messages_insert_own on public.messages for insert with check (auth.uid() = user_id);
create policy messages_update_own on public.messages for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy messages_delete_own on public.messages for delete using (auth.uid() = user_id);

create policy metrics_select_own on public.speech_metrics for select using (auth.uid() = user_id);
create policy metrics_insert_own on public.speech_metrics for insert with check (auth.uid() = user_id);
create policy metrics_update_own on public.speech_metrics for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy metrics_delete_own on public.speech_metrics for delete using (auth.uid() = user_id);
```
</details>

*(If you only need shared session storage and not user login, the backend will operate correctly with just the `SUPABASE_SERVICE_ROLE_KEY`.)*

### MCP Tool Server (Web Research)

The application includes a standalone MCP (Model Context Protocol) server for web tools like `web_search` and `fetch_url`.

1. **Install Dependencies:**
```bash
cd services/mcp_server
pip install -r requirements.txt
```
2. **Run Independently:**
```bash
python server.py
```
3. **Enable in Backend:** Add to `.env.local`:
```env
MCP_ENABLED=true
MCP_AUTO_SEARCH=true
MCP_SERVER_CMD=python
MCP_SERVER_ARGS=services/mcp_server/server.py
```

---

## Running the Application

### Local Workflow

1. Start the FastAPI backend:
```bash
cd backend
python main.py
```
2. Open the web UI at `http://localhost:8000`

### Docker Deployment

The repository includes a multi-stage Dockerfile and a Compose stack.
```bash
cd docker
docker-compose up --build
```
This sets up the backend alongside Ollama, with volumes mapped for persistent models.

---

## API Reference

### REST Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/health` | GET | Health check |
| `/api/status` | GET | Component status |
| `/api/providers` | GET | Available LLM providers |
| `/api/models/{provider}` | GET | Models for a provider |
| `/api/settings` | GET | Current settings |
| `/api/sessions` | GET, POST | List or create Supabase sessions |
| `/api/sessions/{id}` | GET | Fetch session and recent messages |
| `/api/sessions/{id}/mode` | PATCH | Update coaching mode |
| `/api/sessions/{id}/summary`| POST | Generate context summary |

### WebSocket Protocol
- **Client Messages:** `config`, `start`, `audio`, `stop`, `reset`, `ping`
- **Server Messages:** `config_ack`, `status`, `transcript`, `filler_words`, `analysis`, `scores`, `audio`, `complete`, `error`, `pong`

---

## Known Gaps & Roadmap

1. **Automated Tests:** Unit tests for analyzer parsing, integration tests for WebSocket events.
2. **Observability:** Structured logging, latency metrics.
3. **Resilience:** Retry/backoff and clear provider failover behavior.
4. **Security:** Rate limiting and abuse protections.
5. **UX Polish:** Richer conversation timeline, better transcript distinctions, session summaries.

## License
MIT License
