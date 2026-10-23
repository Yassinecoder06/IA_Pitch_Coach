import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.ws import coach
from backend.storage.database import db
from backend.auth.owner import owner_service


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db.sqlite_file = tmp_path / "test_ws_audio.db"
    db.is_postgres = False
    db.init_schema()
    owner_service._setup_code = None


def _pcm16_tone(seconds: float, rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes()


def test_combine_audio_chunks_wraps_pcm_as_wav():
    pcm = _pcm16_tone(1.0)
    wav_bytes = coach.combine_audio_chunks([pcm[:16000], pcm[16000:]])
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 16000
        assert wf.readframes(16000) == pcm


def test_combine_audio_chunks_rejects_tiny_input():
    assert coach.combine_audio_chunks([]) is None
    assert coach.combine_audio_chunks([b"\x00" * 100]) is None


def test_binary_pcm_frames_reach_stt_as_wav(monkeypatch):
    captured = {}

    async def fake_transcribe(audio: bytes, sample_rate: int = 16000):
        captured["wav"] = audio
        return "", 0.0  # empty transcript ends the turn before the LLM is called

    monkeypatch.setattr(coach, "async_transcribe_audio", fake_transcribe)

    client = TestClient(app)
    code = owner_service.get_or_create_setup_code()
    client.post("/api/auth/owner/setup", json={"passphrase": "CorrectHorse1", "setupCode": code})

    pcm = _pcm16_tone(2.0)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        assert ws.receive_json()["type"] == "status"
        for i in range(0, len(pcm), 3200):  # 100 ms frames, like the AudioWorklet
            ws.send_bytes(pcm[i:i + 3200])
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["message"] == "Transcribing speech..."
        assert ws.receive_json()["type"] == "error"  # "No speech detected"

    with wave.open(io.BytesIO(captured["wav"])) as wf:
        assert wf.getframerate() == 16000
        assert wf.readframes(wf.getnframes()) == pcm


def test_binary_frames_ignored_when_not_recording(monkeypatch):
    called = False

    async def fake_transcribe(audio: bytes, sample_rate: int = 16000):
        nonlocal called
        called = True
        return "", 0.0

    monkeypatch.setattr(coach, "async_transcribe_audio", fake_transcribe)
    client = TestClient(app)
    code = owner_service.get_or_create_setup_code()
    client.post("/api/auth/owner/setup", json={"passphrase": "CorrectHorse1", "setupCode": code})

    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(_pcm16_tone(1.0))
        ws.send_json({"type": "stop"})
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
    assert not called
