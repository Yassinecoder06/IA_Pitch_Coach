"""
Chunked live transcription.

Incoming 16 kHz PCM16 is split at natural pauses (or every MAX_CHUNK_S at the
quietest point) and each chunk is transcribed exactly once. Word timestamps are
shifted to the start of the take, so the chunk results concatenate into the
final transcript without a second full-length pass.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import numpy as np

from backend.voice.stt import TranscriptionResult, async_transcribe_detailed

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 50  # 20 ms
MIN_CHUNK_S = 2.0
MAX_CHUNK_S = 12.0
CUT_SILENCE_S = 0.5
MIN_SPEECH_S = 0.3  # chunks with less speech are skipped (avoids hallucinated text on silence)

Transcriber = Callable[[bytes, str], Awaitable[TranscriptionResult]]


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def default_transcriber(wav: bytes, context: str) -> TranscriptionResult:
    return await async_transcribe_detailed(wav, verbatim=True, context=context)


@dataclass
class ChunkResult:
    start: float  # seconds from the start of the take
    end: float
    text: str
    words: List[dict] = field(default_factory=list)


class StreamingTranscriber:
    def __init__(
        self,
        on_chunk: Optional[Callable[[ChunkResult], Awaitable[None]]] = None,
        transcriber: Optional[Transcriber] = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate
        self.frame = sample_rate // 50
        self.on_chunk = on_chunk
        self._transcriber = transcriber
        self.samples = np.zeros(0, dtype=np.int16)  # whole take, kept for the final analysis
        self.levels: List[float] = []  # dBFS per 20 ms frame
        self.chunk_start = 0  # frame index
        self.chunks: List[ChunkResult] = []
        self._pending = b""
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._context = ""

    # ---- input ----
    def feed(self, pcm: bytes) -> None:
        data = self._pending + pcm
        usable = len(data) - (len(data) % 2)
        self._pending = data[usable:]
        new = np.frombuffer(data[:usable], dtype="<i2")
        start_frame = len(self.samples) // self.frame
        self.samples = np.concatenate([self.samples, new])
        for i in range(start_frame, len(self.samples) // self.frame):
            seg = self.samples[i * self.frame:(i + 1) * self.frame].astype(np.float32) / 32768
            self.levels.append(float(20 * np.log10(max(np.sqrt(np.mean(seg * seg)), 1e-6))))
        self._maybe_cut()

    def _threshold(self) -> float:
        """Speech/silence boundary: 10 dB over the noise floor, clamped to [-50, -35] dBFS.

        The upper clamp matters before any silence has been heard: then the "floor" is
        the speech level itself and would classify continuous speech as silence.
        """
        recent = self.levels[-1500:]  # last 30 s
        floor = float(np.percentile(recent, 10)) if recent else -60.0
        return min(-35.0, max(-50.0, floor + 10.0))

    def _maybe_cut(self) -> None:
        n = len(self.levels)
        chunk_frames = n - self.chunk_start
        if chunk_frames < MIN_CHUNK_S * 50:
            return
        thr = self._threshold()
        silence_frames = int(CUT_SILENCE_S * 50)
        tail = self.levels[-silence_frames:]
        if all(level < thr for level in tail):
            self._cut(n - silence_frames // 2)
        elif chunk_frames >= MAX_CHUNK_S * 50:
            window = self.levels[n - 100:n]  # quietest point in the last 2 s
            self._cut(n - 100 + int(np.argmin(window)))

    def _cut(self, end_frame: int) -> None:
        start, self.chunk_start = self.chunk_start, end_frame
        self._enqueue(start, end_frame)

    def _enqueue(self, start_frame: int, end_frame: int) -> None:
        if end_frame <= start_frame:
            return
        thr = self._threshold()
        speech = sum(1 for level in self.levels[start_frame:end_frame] if level >= thr)
        if speech < MIN_SPEECH_S * 50:
            return
        pcm = self.samples[start_frame * self.frame:end_frame * self.frame].tobytes()
        self._queue.put_nowait((start_frame / 50, end_frame / 50, pcm))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())

    # ---- transcription ----
    async def _drain(self) -> None:
        while not self._queue.empty():
            start_s, end_s, pcm = await self._queue.get()
            try:
                transcribe = self._transcriber or default_transcriber
                result = await transcribe(pcm_to_wav(pcm, self.sample_rate), self._context)
            except Exception as e:  # one bad chunk must not end the take
                logger.warning(f"[stream] chunk {start_s:.1f}-{end_s:.1f}s failed: {e}")
                continue
            words = [
                {**w, "start": round(w["start"] + start_s, 3), "end": round(w["end"] + start_s, 3)}
                for w in result.words
            ]
            chunk = ChunkResult(start=start_s, end=end_s, text=result.text.strip(), words=words)
            if chunk.text:
                self.chunks.append(chunk)
                self._context = (self._context + " " + chunk.text)[-400:]
                if self.on_chunk:
                    await self.on_chunk(chunk)

    async def finish(self) -> ChunkResult:
        """Flushes the last chunk and returns the whole take as one result."""
        self._enqueue(self.chunk_start, len(self.levels))
        self.chunk_start = len(self.levels)
        if self._worker:
            await self._worker
        await self._drain()
        words = [w for c in self.chunks for w in c.words]
        return ChunkResult(
            start=0.0,
            end=len(self.samples) / self.sample_rate,
            text=" ".join(c.text for c in self.chunks),
            words=words,
        )

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def wav(self) -> bytes:
        return pcm_to_wav(self.samples.tobytes(), self.sample_rate)
