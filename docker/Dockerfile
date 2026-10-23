# ==============================================================================
# AI Pitch Coach - Production Multi-Stage Dockerfile
# ==============================================================================

# --- Stage 1: Build React Frontend ---
FROM node:22-alpine AS web-builder
WORKDIR /app/web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# --- Stage 2: Build Python Dependencies ---
FROM python:3.12-slim AS py-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 3: Runtime Image ---
FROM python:3.12-slim AS runtime
WORKDIR /app

# System audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from py-builder
COPY --from=py-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create persistent model directories
RUN mkdir -p /app/models/whisper /app/models/piper /app/data

# Copy backend application and skills
COPY backend /app/backend
COPY services /app/services

# Copy compiled frontend from web-builder
COPY --from=web-builder /app/web/dist /app/web/dist

# Default runtime configuration
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATABASE_URL=sqlite:////app/data/pitch_coach.db
ENV WHISPER_MODEL_SIZE=tiny.en
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
ENV OLLAMA_ENDPOINT=http://host.docker.internal:11434

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
