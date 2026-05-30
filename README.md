# AI Pitch Coach

AI Pitch Coach is a modular voice coaching app for practicing startup pitches and public speaking. It combines faster-whisper for speech-to-text, Piper for text-to-speech, multiple LLM providers, and optional Supabase-backed session persistence.

## What You Need

- Python 3.10 or newer
- 8 GB RAM minimum
- A browser with microphone access
- Optional local LLM runtime: Ollama
- Optional Supabase project for session storage and authentication

## What This Project Uses

- `requirements.txt` for Python dependencies
- faster-whisper for transcription
- Piper for voice output
- Supabase for sessions, messages, and speech metrics
- `backend/config/settings.py` loads `.env.local` first, then `.env`

## Quick Start

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Python Requirements

```bash
pip install -r requirements.txt
```

This installs the backend runtime plus the Supabase Python client.

### 3. Install Whisper Support

The app uses faster-whisper. The model downloads automatically on first run into `models/whisper`, but you can choose the size in your environment file.

Recommended settings for a CPU machine:

```env
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
WHISPER_MODELS_DIR=models/whisper
```

If you want to pre-download manually, just start the backend once after setting the model size. faster-whisper will fetch the model into the configured folder.

### 4. Install Piper TTS

You can install Piper manually:

```bash
pip install piper-tts
```

The backend can also auto-install Piper and download the default voice model on startup when `AUTO_INSTALL_TTS=true`.

```env
AUTO_INSTALL_TTS=true
PIPER_VOICE=en_US-lessac-medium
PIPER_MODELS_DIR=models/piper
```

If you want to download the voice files yourself, place these in `models/piper`:

```bash
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
```

### 5. Install Ollama for a Local LLM

Ollama is optional, but it is the easiest local provider to use.

Windows:

1. Download Ollama from https://ollama.com/download
2. Open a terminal and run:

```bash
ollama pull qwen3:0.6b
ollama serve
```

macOS / Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:0.6b
ollama serve
```

### 6. Set Up Supabase

Supabase is used for session persistence, message history, and speech metrics. You can use either a cloud Supabase project or a local Supabase instance.

#### Cloud Supabase

1. Create a project at https://supabase.com.
2. From Project Settings > API, copy:
   - Project URL
   - `anon` public key
   - `service_role` key
3. Create the required tables.
4. Add the Supabase values to your `.env.local` or `.env` file.

#### Local Supabase

1. Install the Supabase CLI.
2. Start local services with `supabase start`.
3. Use the local API URL and keys from the CLI output.
4. Create the required tables in the local database.

#### Required Supabase Tables

The backend expects these tables:

- `sessions`
- `messages`
- `speech_metrics`

The full SQL schema is documented in [docs/supabase_setup.md](docs/supabase_setup.md).

#### Supabase Environment Variables

Add these to `.env.local` for local development or `.env` for shared defaults:

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SESSION_CONTEXT_WINDOW=8
```

Use `SUPABASE_SERVICE_ROLE_KEY` only on the backend. Do not expose it in frontend code.

### 7. Add Authentication with Supabase

Supabase Auth is the right place to add sign-in for the app. The backend already supports Supabase-backed persistence, but if you want user login, configure it in Supabase first.

Recommended setup:

1. In Supabase Dashboard, open Authentication.
2. Enable Email/Password auth or your preferred OAuth provider.
3. Set your site URL and redirect URLs for local development and production.
4. Keep using the `anon` key in browser-facing code.
5. Keep using the `service_role` key only in backend code.
6. If you later add user-aware session filtering, scope rows by authenticated user ID.

If you only want shared session storage and do not need login yet, Supabase still works with the backend using the keys above.

### 8. Create Your Environment File

Copy the example file and edit it:

```bash
copy .env.example .env.local
```

or on macOS / Linux:

```bash
cp .env.example .env.local
```

Minimum useful configuration:

```env
DEFAULT_LLM=ollama
OLLAMA_ENDPOINT=http://localhost:11434

WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu

PIPER_VOICE=en_US-lessac-medium

SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_ANON_KEY=YOUR_ANON_KEY
```

Add cloud LLM keys only if you want those providers enabled:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
DEEPSEEK_API_KEY=
MISTRAL_API_KEY=
GROK_API_KEY=
```

### 9. Run the Backend

```bash
cd backend
python main.py
```

Open the web UI at http://localhost:8000

## Environment Reference

### Core Settings

```env
DEFAULT_LLM=ollama
DEFAULT_MODEL=
HOST=0.0.0.0
PORT=8000
DEBUG=false
PRELOAD_STT_MODEL=true
SESSION_CONTEXT_WINDOW=8
```

### Speech-to-Text Settings

```env
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
WHISPER_MODELS_DIR=models/whisper
```

### Text-to-Speech Settings

```env
PIPER_VOICE=en_US-lessac-medium
PIPER_MODELS_DIR=models/piper
AUTO_INSTALL_TTS=true
```

### Supabase Settings

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_ANON_KEY=YOUR_ANON_KEY
```

## Coaching Modes

1. Pitch Analysis Mode - structured feedback with scores and suggestions
2. Interactive Coaching Mode - refine your pitch through conversation
3. Investor Q&A Mode - practice answering investor questions

## Docker

If you prefer Docker, the repository includes files in `docker/`.

```bash
cd docker
docker-compose up --build
```

## Troubleshooting

### Ollama does not connect

1. Make sure Ollama is running with `ollama serve`
2. Check your model is installed with `ollama list`
3. Pull the default model again with `ollama pull qwen3:0.6b`

### Piper does not start

1. Run `pip install piper-tts`
2. Confirm `models/piper/en_US-lessac-medium.onnx` exists
3. Leave `AUTO_INSTALL_TTS=true` if you want the backend to bootstrap it automatically

### Whisper does not load

1. Confirm faster-whisper is installed from `requirements.txt`
2. Check the `WHISPER_MODEL_SIZE` and `WHISPER_MODELS_DIR` values
3. Delete the cached model folder and restart if the download was interrupted

### Supabase is disabled

1. Confirm `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set
2. Make sure the `sessions`, `messages`, and `speech_metrics` tables exist
3. Verify the Supabase keys are copied from the correct project

### Authentication is not working yet

1. Enable Auth in the Supabase dashboard
2. Set the correct redirect URLs
3. Keep `SUPABASE_ANON_KEY` in browser-side code and `SUPABASE_SERVICE_ROLE_KEY` only on the backend

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
| `/api/sessions/{session_id}` | GET | Fetch a session and its recent messages |
| `/api/sessions/{session_id}/mode` | PATCH | Update the stored coaching mode |
| `/api/sessions/{session_id}/summary` | POST | Generate a compact context summary |

### WebSocket Messages

Client messages include `config`, `start`, `audio`, `stop`, `reset`, and `ping`.

Server messages include `config_ack`, `status`, `transcript`, `filler_words`, `analysis`, `scores`, `audio`, `complete`, and `error`.

## License

MIT License

## Credits

- faster-whisper - speech recognition
- Ollama - local LLM runtime
- Piper - text-to-speech
- FastAPI - backend web framework
