"""Render text with the local Piper voice into a 48 kHz mono WAV (Chrome's fake mic input).

Usage: python -I make_wav.py OUT.wav "text"
"""

import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.voice.tts import synthesize_speech  # noqa: E402


def main(out_path: str, text: str) -> None:
    import librosa

    wav = synthesize_speech(text)
    with wave.open(io.BytesIO(wav)) as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), "<i2").astype(np.float32) / 32768
    samples = librosa.resample(samples, orig_sr=rate, target_sr=48000)
    lead = np.zeros(int(0.5 * 48000), dtype=np.float32)  # half a second of silence before speaking
    samples = np.concatenate([lead, samples, lead])
    with wave.open(out_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
