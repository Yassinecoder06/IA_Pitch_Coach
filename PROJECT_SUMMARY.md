# AI Pitch Coach - Project Summary

Last updated: March 31, 2026

## 1) Project Purpose

AI Pitch Coach is a real-time, voice-first coaching application for startup pitch practice and public speaking improvement.

The platform captures spoken input from the browser, transcribes speech, analyzes content with a selectable LLM provider, and returns both written and spoken coaching feedback.

## 2) What We Built

### Core Product Capabilities

- Real-time voice pipeline over WebSocket
- Speech-to-text transcription using faster-whisper
- Multi-provider LLM orchestration with runtime provider/model selection
- Coaching analysis across multiple speaking modes
- Text-to-speech feedback with Piper
- Filler-word detection and reporting
- Structured pitch scoring in pitch-analysis mode
- Browser-based UI with live transcript, streamed analysis, scores, and audio playback

### Coaching Modes Implemented

1. Pitch Analysis
- Structured output sections: SCORES, ANALYSIS, ADVICE
- Scores for Clarity, Language, Confidence, Topic Relevance

2. Interactive Coaching
- Conversational back-and-forth with context retention
- Response style tuned for concise, actionable coaching

3. Investor Q&A
- Investor-style question flow with focused probing and brief evaluation

## 3) System Architecture

### Runtime Flow

1. Browser records microphone audio and streams chunks over WebSocket.
2. Backend combines chunks to WAV.
3. faster-whisper transcribes audio and returns transcript + confidence.
4. Filler-word analyzer computes count + details.
5. Pitch analyzer streams LLM response in chunks.
6. Backend streams text feedback to client in real time.
7. Backend sends scores (for pitch-analysis mode).
8. Backend synthesizes TTS sentence-by-sentence and streams WAV audio chunks back.

### Main Components

- Backend API and WebSocket server: FastAPI
- STT engine: faster-whisper
- LLM layer: local and cloud providers via unified provider interfaces
- TTS engine: Piper
- Frontend: plain HTML/CSS/JavaScript UI with streaming updates

## 4) Provider and Model Layer

### Provider Strategy

The project uses a dynamic provider registry that enables providers based on environment configuration:

- Always available locally: Ollama
- Enabled when key/config exists: OpenAI, Anthropic, Google, Azure OpenAI, DeepSeek, Mistral, Grok, Alibaba, ModelScope, Moonshot, SiliconFlow

### Provider Features

- Unified streaming interface
- Provider availability checks
- Dynamic model listing
- Default-provider fallback behavior

## 5) Config and Environment

### Centralized Settings

- Environment-first configuration with dotenv loading
- Config groups: provider settings, STT, TTS, and server
- Safe settings export endpoint for frontend status/config

### Notable Environment Controls

- DEFAULT_LLM, DEFAULT_MODEL
- Provider keys and endpoints
- Whisper model/device/compute settings
- Piper voice/models directory
- Host and port

## 6) Frontend UX Delivered

- Provider dropdown with model auto-loading
- Coaching mode selection
- Record/stop controls
- Live audio visualizer and recording timer
- Real-time transcript updates
- Streaming feedback rendering
- Score cards and filler-word details
- Audio response queue playback
- Conversation history support for interactive modes

## 7) Deployment and Operations

### Local Run

- Python virtual environment workflow
- FastAPI launched via backend/main.py
- Browser client served from backend

### Containerization

- Multi-stage Dockerfile (builder + runtime)
- docker-compose stack includes backend + Ollama
- Named volumes for model persistence (whisper, piper, ollama)
- Health check endpoint support

## 8) Key Improvements and Fixes Completed

### A) Spoken Feedback Completeness (Previous improvement)

- Removed forced truncation behavior that clipped coaching responses.
- Ensured TTS can speak full generated feedback instead of an artificially short subset.
- Relaxed strict token cap behavior in analysis/provider flow where appropriate.

Result:
- Spoken coaching feedback is now more complete and less likely to end abruptly.

### B) Asterisk Handling for TTS (Latest improvement)

Problem:
- Some model responses included markdown asterisks, and TTS pronounced them (for example, saying "asterisk").

Fixes implemented:
- Added backend output sanitization to strip * characters from streaming LLM chunks before they are stored/sent.
- Updated coaching system prompts to explicitly forbid markdown and * usage.
- Added fallback sanitization in TTS sentence splitting to remove * before speech synthesis.

Result:
- Markdown emphasis/list markers no longer leak into spoken output.

## 9) API and Communication Surface

### REST Endpoints

- / -> serves frontend entry page
- /health -> component health and provider availability
- /api/status -> high-level runtime status
- /api/providers -> available providers and defaults
- /api/models/{provider} -> models for selected provider
- /api/settings -> safe settings snapshot

### WebSocket Protocol

Client message types:
- config
- start
- audio
- stop
- reset
- ping

Server message types:
- status
- config_ack
- transcript
- filler_words
- analysis (streaming + completion marker)
- scores
- audio
- complete
- error
- pong

## 10) Dependencies and Technology Stack

### Python Runtime

- Python 3.11 (container baseline)
- FastAPI + Uvicorn
- websockets
- httpx
- faster-whisper
- numpy
- soundfile
- python-dotenv
- python-multipart
- piper-tts

### Infra and Tooling

- Docker + Docker Compose
- Ollama for local model serving

## 11) Current Strengths

- End-to-end real-time voice coaching pipeline is implemented.
- Provider layer is extensible and supports local-first and cloud-first setups.
- Coaching modes are distinct and practical.
- UX supports streaming text and streamed audio responses.
- Deployment path exists for local and containerized workflows.

## 12) Known Gaps / Suggested Next Iterations

1. Add automated tests
- Unit tests for analyzer parsing/sanitization
- Integration tests for WebSocket event flow

2. Add observability
- Structured logging
- Latency metrics for STT, LLM, TTS phases

3. Improve resilience
- Retry/backoff handling and clearer provider failover behavior

4. Security and hardening
- Authn/authz for API access in non-local deployments
- Rate limiting and abuse protections

5. UX polish
- Better partial/final transcript distinction
- Richer conversation timeline and downloadable session summaries

## 13) Summary

This project has evolved into a modular, practical AI voice coaching platform that supports many LLM backends, delivers actionable pitch feedback, and speaks results back to the user in real time.

Recent work focused on response quality and spoken-output quality:
- preserving complete feedback,
- preventing markdown symbols from being spoken,
- and keeping the coaching experience natural and professional.
