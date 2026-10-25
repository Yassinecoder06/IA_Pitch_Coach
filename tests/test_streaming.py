import asyncio

import numpy as np

from backend.voice.streaming import StreamingTranscriber
from backend.voice.stt import TranscriptionResult

SR = 16000


def tone(seconds, hz=180.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype("<i2").tobytes()


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype="<i2").tobytes()


def make_fake():
    calls = []

    async def fake(wav: bytes, context: str) -> TranscriptionResult:
        n = len(calls)
        calls.append((len(wav), context))
        return TranscriptionResult(
            text=f"chunk{n}", confidence=1, language="en", segments=[],
            words=[{"w": f"chunk{n}", "start": 0.1, "end": 0.5, "prob": 0.9}],
        )
    return fake, calls


def feed_in_frames(st, pcm, frame_bytes=3200):
    for i in range(0, len(pcm), frame_bytes):
        st.feed(pcm[i:i + frame_bytes])


def test_cuts_at_pauses_and_offsets_word_times():
    async def scenario():
        fake, calls = make_fake()
        seen = []

        async def on_chunk(c):
            seen.append(c)
        st = StreamingTranscriber(on_chunk=on_chunk, transcriber=fake)
        feed_in_frames(st, silence(0.5) + tone(2.5) + silence(0.8))
        await asyncio.sleep(0)  # let the worker run
        assert len(seen) == 1  # first sentence emitted before the take ends
        feed_in_frames(st, tone(3.0) + silence(0.2))
        final = await st.finish()
        return seen, calls, final, st

    seen, calls, final, st = asyncio.run(scenario())
    assert [c.text for c in seen] == ["chunk0", "chunk1"]
    assert seen[1].start > 3.0  # second chunk starts after the pause
    assert final.words[1]["start"] == round(seen[1].start + 0.1, 3)
    assert final.text == "chunk0 chunk1"
    assert calls[1][1].strip().endswith("chunk0")  # previous text passed as context
    assert abs(st.duration - 7.0) < 0.01


def test_long_monologue_is_cut_at_max_length():
    async def scenario():
        fake, calls = make_fake()
        st = StreamingTranscriber(transcriber=fake)
        feed_in_frames(st, tone(26.0))
        final = await st.finish()
        return calls, final
    calls, final = asyncio.run(scenario())
    assert len(calls) == 3  # ~12 s + ~12 s + remainder
    assert final.end == 26.0


def test_silence_only_produces_no_transcription():
    async def scenario():
        fake, calls = make_fake()
        st = StreamingTranscriber(transcriber=fake)
        feed_in_frames(st, silence(5.0))
        await st.finish()
        return calls
    assert asyncio.run(scenario()) == []


def test_odd_byte_boundaries_are_handled():
    async def scenario():
        fake, _ = make_fake()
        st = StreamingTranscriber(transcriber=fake)
        pcm = tone(1.0)
        st.feed(pcm[:1001])
        st.feed(pcm[1001:])
        return st
    st = asyncio.run(scenario())
    assert len(st.samples) == SR


def test_failed_chunk_does_not_stop_the_take():
    async def scenario():
        n = 0

        async def flaky(wav, context):
            nonlocal n
            n += 1
            if n == 1:
                raise RuntimeError("GPU hiccup")
            return TranscriptionResult(text="ok", confidence=1, language="en", segments=[], words=[])
        st = StreamingTranscriber(transcriber=flaky)
        feed_in_frames(st, tone(2.5) + silence(0.8) + tone(2.5))
        return await st.finish()
    assert asyncio.run(scenario()).text == "ok"
