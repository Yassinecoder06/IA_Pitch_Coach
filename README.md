# AI Pitch Coach

A **local-first, real-time AI Pitch & Speaking Coach** that analyzes your speech and provides instant feedback. Everything runs locally on your machine - no cloud APIs required.

## Features

- **Real-time Speech-to-Text** using faster-whisper (4x faster than OpenAI Whisper)
- **AI Pitch Analysis** via Ollama with qwen3:0.6b (optimized for 8GB RAM)
- **Text-to-Speech Feedback** using Piper TTS
- **WebSocket Communication** for low-latency streaming
- **Structured Scoring** for Clarity, Language, Confidence, and Topic Relevance
- **Filler Word Detection** (um, uh, like, you know, basically)

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
   | -- Audio Chunks (every 250ms) --> |
   |                                    |
   |                              faster-whisper (STT)
   |                                    |
   |                              Ollama (LLM Analysis)
   |                                    |
   |                              Piper (TTS)
   |                                    |
   | <-- Transcript + Scores --------- |
   | <-- Streaming AI Feedback ------- |
   | <-- Audio Response -------------- |
```

## Quick Start

### 1. Install Prerequisites

#### Install Ollama

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

**Option A: Install via pip**
```bash
pip install piper-tts
```

**Option B: Download binary**
1. Download from https://github.com/rhasspy/piper/releases
2. Extract to a folder
3. Add to PATH or set `PIPER_PATH` environment variable

#### Download Voice Model

```bash
# Create models directory
mkdir -p ai_pitch_coach/models/piper

# Download voice (en_US-lessac-medium)
cd ai_pitch_coach/models/piper
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
cd ai_pitch_coach
pip install -r requirements.txt
```

### 4. Run the Server

```bash
cd ai_pitch_coach/backend
python main.py
```

### 5. Open the Web UI

Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
ai_pitch_coach/
├── backend/
│   ├── __init__.py         # Package init
│   ├── main.py             # FastAPI server & WebSocket handler
│   ├── stt.py              # Speech-to-Text (faster-whisper)
│   ├── llm.py              # LLM integration (Ollama)
│   └── tts.py              # Text-to-Speech (Piper)
│
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Styles
│   └── script.js           # WebSocket & audio handling
│
├── models/
│   ├── whisper/            # Whisper models (auto-downloaded)
│   └── piper/              # Piper voice models
│
├── docker/
│   ├── Dockerfile          # Container build
│   └── docker-compose.yml  # Full stack orchestration
│
└── requirements.txt        # Python dependencies
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `tiny` | Whisper model: tiny, base, small |
| `WHISPER_DEVICE` | `cpu` | Device: cpu, cuda |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type: int8, float16, float32 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `qwen3:0.6b` | LLM model name |
| `PIPER_PATH` | `piper` | Path to Piper executable |
| `PIPER_VOICE` | `en_US-lessac-medium` | Voice model name |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Whisper Model Sizes

| Model | Size | RAM Usage | Speed | Accuracy |
|-------|------|-----------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Good |
| base | 74M | ~1GB | Fast | Better |
| small | 244M | ~2GB | Medium | Best |

## Docker Deployment

### Build and Run

```bash
cd ai_pitch_coach/docker
docker-compose up --build
```

This will:
1. Build the pitch coach container
2. Start Ollama container
3. Auto-pull the qwen3:0.6b model
4. Start the web server on port 8000

### Stop

```bash
docker-compose down
```

## Usage

1. **Click "Start Recording"** and speak your pitch clearly
2. **Click "Stop Recording"** when finished
3. **Wait for analysis** - typically 5-15 seconds depending on length
4. **Review feedback**:
   - Live transcript of your speech
   - Scores for Clarity, Language, Confidence, Topic Relevance
   - Filler word count with breakdown
   - AI-generated improvement suggestions
5. **Listen to audio feedback** (if TTS is configured)

## Pitch Scoring Criteria

| Score | Range | Description |
|-------|-------|-------------|
| Clarity | 0-10 | How clear and understandable is the message |
| Language | 0-10 | Grammar, vocabulary, and professional tone |
| Confidence | 0-10 | Perceived confidence from speech patterns |
| Topic Relevance | 0-10 | How well the content fits a pitch format |

## Troubleshooting

### "Cannot connect to Ollama"

1. Ensure Ollama is running: `ollama serve`
2. Check if model is installed: `ollama list`
3. Pull model if missing: `ollama pull qwen3:0.6b`

### "Microphone not working"

1. Allow microphone access in browser
2. Check browser console for errors
3. Try a different browser (Chrome recommended)

### "No speech detected"

1. Speak closer to the microphone
2. Reduce background noise
3. Try recording a longer segment (5+ seconds)

### "TTS not working"

1. Verify Piper is installed: `piper --help`
2. Check voice model exists in `models/piper/`
3. TTS is optional - text feedback still works

### High RAM Usage

1. Use `WHISPER_MODEL_SIZE=tiny`
2. Reduce `num_ctx` in llm.py options
3. Close other applications

## API Reference

### WebSocket Protocol (`/ws`)

**Client Messages:**
```json
{"type": "start"}                    // Start recording
{"type": "audio", "data": "base64"}  // Audio chunk
{"type": "stop"}                     // Stop and process
{"type": "ping"}                     // Keep-alive
```

**Server Messages:**
```json
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

## License

MIT License - feel free to use and modify for your projects.

## Credits

Built with:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [Ollama](https://ollama.com) - Local LLM runtime
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [FastAPI](https://fastapi.tiangolo.com) - Web framework
