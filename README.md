# AI Pitch Coach

A **modular, extensible AI voice coaching platform** for improving startup pitches and public speaking. Supports both local and cloud language models with real-time voice interaction.

## Features

### Core Features
- **Real-time Speech-to-Text** using faster-whisper (4x faster than OpenAI Whisper)
- **Multi-Provider LLM Support** - OpenAI, Anthropic, Google, Ollama, and more
- **Text-to-Speech Feedback** using Piper TTS
- **WebSocket Communication** for low-latency streaming
- **Structured Scoring** for Clarity, Language, Confidence, and Topic Relevance
- **Filler Word Detection** (um, uh, like, you know, basically)

### Coaching Modes
1. **Pitch Analysis Mode** - Get structured feedback with scores and suggestions
2. **Interactive Coaching Mode** - Refine your pitch through conversation
3. **Investor Q&A Mode** - Practice answering investor questions

### Supported LLM Providers
| Provider | Type | Models |
|----------|------|--------|
| Ollama | Local | qwen3, mistral, llama3, etc. |
| OpenAI | Cloud | GPT-4o, GPT-4 Turbo, GPT-3.5 |
| Anthropic | Cloud | Claude 3.5, Claude 3 |
| Google | Cloud | Gemini 2.0, Gemini 1.5 |
| DeepSeek | Cloud | DeepSeek Chat, Coder |
| Mistral | Cloud | Mistral Large, Medium, Small |
| Azure OpenAI | Cloud | Deployed GPT models |
| Grok | Cloud | Grok 2, Grok 2 Mini |

## System Requirements

- **RAM**: 8GB minimum
- **CPU**: Any modern processor (GPU optional)
- **OS**: Windows, macOS, or Linux
- **Browser**: Chrome, Firefox, Edge (with microphone access)

## Architecture

```
Browser                           Backend (FastAPI)
   |                                    |
   | ---- WebSocket Connection -----> |
   |                                    |
   | -- Config (provider/mode) -----> |
   | -- Audio Chunks (250ms) -------> |
   |                                    |
   |                              faster-whisper (STT)
   |                                    |
   |                              LLM Provider (configurable)
   |                              - Ollama (local)
   |                              - OpenAI
   |                              - Anthropic
   |                              - Google
   |                              - ...
   |                                    |
   |                              Piper (TTS)
   |                                    |
   | <-- Transcript + Scores --------- |
   | <-- Streaming AI Feedback ------- |
   | <-- Audio Response -------------- |
```

## Project Structure

```
ai_pitch_coach/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI server & WebSocket handler
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Environment-based configuration
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider_interface.py  # Abstract LLM provider
│   │   ├── registry.py         # Provider registration
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── ollama_provider.py
│   │   ├── google_provider.py
│   │   ├── azure_provider.py
│   │   ├── deepseek_provider.py
│   │   ├── mistral_provider.py
│   │   └── grok_provider.py
│   │
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py              # Speech-to-Text (faster-whisper)
│   │   ├── tts.py              # Text-to-Speech (Piper)
│   │   └── voice_loop.py       # Continuous conversation loop
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── pitch_analysis.py   # Pitch analysis & coaching prompts
│       └── filler_detection.py # Filler word detection
│
├── frontend/
│   ├── index.html              # Main UI
│   ├── style.css               # Open WebUI inspired styles
│   └── script.js               # WebSocket & audio handling
│
├── models/
│   ├── whisper/                # Whisper models (auto-downloaded)
│   └── piper/                  # Piper voice models
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .env.example                # Environment configuration template
└── requirements.txt
```

## Quick Start

### 1. Install Prerequisites

#### Install Ollama (for local LLM)

**Windows:**
Download from https://ollama.com/download

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Pull the LLM Model

```bash
ollama pull qwen3:0.6b
```

#### Start Ollama Server

```bash
ollama serve
```

### 2. Install Piper TTS

```bash
pip install piper-tts
```

Note:
The backend now supports automatic TTS bootstrap on startup. If `piper-tts` or the
default voice model (`en_US-lessac-medium`) is missing, it will try to install/download
them automatically.

To disable auto-bootstrap:

```env
AUTO_INSTALL_TTS=false
```

#### Download Voice Model

```bash
mkdir -p models/piper
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

```bash
# Copy example config
cp .env.example .env

# Edit .env to add cloud provider API keys (optional)
```

### 5. Run the Server

```bash
cd backend
python main.py
```

### 6. Open the Web UI

Navigate to: http://localhost:8000

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Default LLM Provider (ollama, openai, anthropic, etc.)
DEFAULT_LLM=ollama

# Cloud Provider API Keys (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# STT Configuration
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu

# TTS Configuration
PIPER_VOICE=en_US-lessac-medium
```

### Whisper Model Sizes

| Model | Size | RAM Usage | Speed | Accuracy |
|-------|------|-----------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Good |
| base | 74M | ~1GB | Fast | Better |
| small | 244M | ~2GB | Medium | Best |

## Usage

### Basic Usage
1. Select your **AI Provider** and **Model** from the dropdowns
2. Choose a **Coaching Mode**
3. Click **Start Recording** and speak your pitch
4. Click **Stop Recording** when finished
5. Review the AI feedback and scores

### Coaching Modes

#### Pitch Analysis Mode
- Get structured feedback with scores (0-10)
- Categories: Clarity, Language, Confidence, Topic Relevance
- Actionable improvement suggestions

#### Interactive Coaching Mode
- Have a conversation to refine your pitch
- Get iterative feedback on improvements
- Build on previous responses

#### Investor Q&A Mode
- Practice answering investor questions
- Questions cover problem, market, competition, team
- Get evaluation of your answers

## Docker Deployment

```bash
cd docker
docker-compose up --build
```

This will:
1. Build the pitch coach container
2. Start Ollama container
3. Auto-pull the qwen3:0.6b model
4. Start the web server on port 8000

To add cloud provider support, uncomment and set API keys in docker-compose.yml.

## API Reference

### WebSocket Protocol (`/ws`)

**Client Messages:**
```json
{"type": "config", "provider": "openai", "model": "gpt-4o", "mode": "pitch_analysis"}
{"type": "start"}
{"type": "audio", "data": "base64"}
{"type": "stop"}
{"type": "reset"}
{"type": "ping"}
```

**Server Messages:**
```json
{"type": "config_ack", "config": {...}}
{"type": "status", "message": "..."}
{"type": "transcript", "text": "...", "final": true}
{"type": "filler_words", "count": 5, "details": {...}}
{"type": "analysis", "text": "...", "streaming": true}
{"type": "scores", "data": {...}}
{"type": "audio", "data": "base64", "format": "wav"}
{"type": "complete"}
{"type": "error", "message": "..."}
```

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/health` | GET | Health check |
| `/api/status` | GET | Component status |
| `/api/providers` | GET | Available LLM providers |
| `/api/models/{provider}` | GET | Models for a provider |
| `/api/settings` | GET | Current settings |

## Troubleshooting

### "Cannot connect to Ollama"
1. Ensure Ollama is running: `ollama serve`
2. Check if model is installed: `ollama list`
3. Pull model if missing: `ollama pull qwen3:0.6b`

### "Provider not available"
1. Check API key is set in `.env`
2. Verify endpoint URL is correct
3. Check network connectivity

### "Microphone not working"
1. Allow microphone access in browser
2. Check browser console for errors
3. Try a different browser (Chrome recommended)

### "No speech detected"
1. Speak closer to the microphone
2. Reduce background noise
3. Try recording a longer segment (5+ seconds)

## Scaling Suggestions

1. **Horizontal Scaling**: Deploy multiple backend instances behind a load balancer
2. **Caching**: Add Redis for caching provider responses
3. **Queue System**: Use Celery/RabbitMQ for async processing
4. **GPU Acceleration**: Enable CUDA for faster Whisper transcription
5. **CDN**: Serve frontend assets via CDN for better performance

## License

MIT License - feel free to use and modify for your projects.

## Credits

Built with:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [Ollama](https://ollama.com) - Local LLM runtime
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [FastAPI](https://fastapi.tiangolo.com) - Web framework
- Architecture inspired by [Open WebUI](https://github.com/open-webui/open-webui)
