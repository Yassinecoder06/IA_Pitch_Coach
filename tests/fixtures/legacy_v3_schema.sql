CREATE TABLE IF NOT EXISTS app_owner (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    passphrase_hash TEXT NOT NULL,
    session_salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_providers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT,
    api_key_encrypted TEXT,
    key_hint TEXT,
    models_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1,
    last_fetched_at TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_preferences (
    user_id TEXT PRIMARY KEY,
    chat_models_json TEXT NOT NULL DEFAULT '[]',
    fallback_models_json TEXT NOT NULL DEFAULT '[]',
    tts_ref TEXT,
    tts_voice TEXT,
    stt_ref TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'Untitled Session',
    current_mode TEXT NOT NULL DEFAULT 'pitch_analysis',
    context_md TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    transcript TEXT,
    scores_json TEXT,
    metrics_json TEXT,
    audio_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS speech_metrics (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    words_per_minute REAL NOT NULL DEFAULT 0.0,
    pause_frequency REAL NOT NULL DEFAULT 0.0,
    pause_duration REAL NOT NULL DEFAULT 0.0,
    energy_variation REAL NOT NULL DEFAULT 0.0,
    rhythm_score REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
