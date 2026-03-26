"""
Voice Loop Module
=================
Implements continuous voice conversation loop similar to ChatGPT Voice Mode.
Supports voice activity detection, interruptions, and streaming responses.
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, AsyncGenerator
import time


class VoiceLoopState(str, Enum):
    """States for the voice conversation loop."""
    IDLE = "idle"           # Waiting for user to start
    LISTENING = "listening"  # Recording user speech
    PROCESSING = "processing"  # Transcribing and analyzing
    SPEAKING = "speaking"    # AI is speaking response
    INTERRUPTED = "interrupted"  # User interrupted AI


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)
    audio_duration: Optional[float] = None
    scores: Optional[dict] = None


@dataclass
class VoiceLoopConfig:
    """Configuration for the voice loop."""
    # Silence detection
    silence_threshold_ms: int = 1500  # How long to wait after user stops speaking
    min_speech_duration_ms: int = 500  # Minimum speech to process

    # Interruption handling
    allow_interruption: bool = True
    interruption_threshold_ms: int = 300  # How quickly to detect interruption

    # Processing
    max_response_sentences: int = 3  # Limit TTS sentences for speed
    stream_tts: bool = True  # Start TTS before full response

    # Context
    max_conversation_turns: int = 10  # Keep recent turns for context


class VoiceLoop:
    """
    Continuous voice conversation loop.

    Implements the flow:
    1. User speaks
    2. Speech is transcribed
    3. LLM generates response (streaming)
    4. Response is spoken (can start before complete)
    5. System listens again when AI finishes speaking
    """

    def __init__(
        self,
        config: Optional[VoiceLoopConfig] = None,
        on_state_change: Optional[Callable[[VoiceLoopState], Any]] = None,
        on_transcript: Optional[Callable[[str], Any]] = None,
        on_response_chunk: Optional[Callable[[str], Any]] = None,
        on_audio_ready: Optional[Callable[[bytes], Any]] = None
    ):
        self.config = config or VoiceLoopConfig()
        self.state = VoiceLoopState.IDLE
        self.conversation_history: List[ConversationTurn] = []

        # Callbacks
        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_response_chunk = on_response_chunk
        self.on_audio_ready = on_audio_ready

        # Internal state
        self._running = False
        self._audio_buffer: List[bytes] = []
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._current_response: str = ""
        self._interrupt_flag = False

    def _set_state(self, new_state: VoiceLoopState):
        """Update state and notify callback."""
        if self.state != new_state:
            self.state = new_state
            if self.on_state_change:
                self.on_state_change(new_state)

    def start(self):
        """Start the voice loop (begin listening)."""
        self._running = True
        self._set_state(VoiceLoopState.LISTENING)

    def stop(self):
        """Stop the voice loop."""
        self._running = False
        self._set_state(VoiceLoopState.IDLE)

    def interrupt(self):
        """Signal that user is interrupting."""
        if self.state == VoiceLoopState.SPEAKING:
            self._interrupt_flag = True
            self._set_state(VoiceLoopState.INTERRUPTED)

    def add_audio_chunk(self, chunk: bytes, has_voice: bool = True):
        """
        Add an audio chunk from the microphone.

        Args:
            chunk: Audio data bytes
            has_voice: Whether voice activity was detected
        """
        if not self._running:
            return

        current_time = time.time() * 1000  # Convert to ms

        if has_voice:
            self._audio_buffer.append(chunk)
            self._silence_start = None

            if self._speech_start is None:
                self._speech_start = current_time

            # Check for interruption
            if self.state == VoiceLoopState.SPEAKING and self.config.allow_interruption:
                speech_duration = current_time - self._speech_start
                if speech_duration >= self.config.interruption_threshold_ms:
                    self.interrupt()

        else:
            # Silence detected
            if self._silence_start is None:
                self._silence_start = current_time
            else:
                silence_duration = current_time - self._silence_start

                # Check if we should process
                if (
                    self.state == VoiceLoopState.LISTENING
                    and self._audio_buffer
                    and silence_duration >= self.config.silence_threshold_ms
                ):
                    speech_duration = (self._silence_start - self._speech_start) if self._speech_start else 0

                    if speech_duration >= self.config.min_speech_duration_ms:
                        # Ready to process
                        self._set_state(VoiceLoopState.PROCESSING)

    def get_audio_buffer(self) -> bytes:
        """Get and clear the current audio buffer."""
        if not self._audio_buffer:
            return b""
        combined = b"".join(self._audio_buffer)
        self._audio_buffer = []
        self._speech_start = None
        self._silence_start = None
        return combined

    def add_conversation_turn(
        self,
        role: str,
        text: str,
        scores: Optional[dict] = None
    ):
        """Add a turn to conversation history."""
        turn = ConversationTurn(
            role=role,
            text=text,
            scores=scores
        )
        self.conversation_history.append(turn)

        # Trim history if too long
        if len(self.conversation_history) > self.config.max_conversation_turns:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_turns:]

    def get_conversation_context(self) -> List[dict]:
        """Get conversation history as message list for LLM."""
        return [
            {"role": turn.role, "content": turn.text}
            for turn in self.conversation_history
        ]

    def begin_response(self):
        """Signal that AI is starting to respond."""
        self._current_response = ""
        self._interrupt_flag = False
        self._set_state(VoiceLoopState.SPEAKING)

    def add_response_chunk(self, chunk: str):
        """Add a chunk of the AI response."""
        if self._interrupt_flag:
            return False  # Stop adding if interrupted

        self._current_response += chunk

        if self.on_response_chunk:
            self.on_response_chunk(chunk)

        return True

    def finish_response(self):
        """Signal that AI has finished responding."""
        if not self._interrupt_flag:
            # Add to conversation history
            if self._current_response:
                self.add_conversation_turn("assistant", self._current_response)

        self._current_response = ""
        self._interrupt_flag = False

        if self._running:
            self._set_state(VoiceLoopState.LISTENING)
        else:
            self._set_state(VoiceLoopState.IDLE)

    async def process_audio(
        self,
        audio_data: bytes,
        transcribe_fn: Callable[[bytes], tuple[str, float]],
        analyze_fn: Callable[[str, List[dict]], AsyncGenerator[str, None]],
        synthesize_fn: Callable[[str], Optional[bytes]]
    ):
        """
        Process recorded audio through the full pipeline.

        Args:
            audio_data: Combined audio bytes
            transcribe_fn: Function to transcribe audio
            analyze_fn: Async generator function for LLM analysis
            synthesize_fn: Function to synthesize speech
        """
        self._set_state(VoiceLoopState.PROCESSING)

        try:
            # Step 1: Transcribe
            transcript, confidence = transcribe_fn(audio_data)

            if not transcript.strip():
                self.finish_response()
                return

            if self.on_transcript:
                self.on_transcript(transcript)

            # Add user turn to history
            self.add_conversation_turn("user", transcript)

            # Step 2: Get LLM response (streaming)
            self.begin_response()

            sentences_buffer = ""
            sentences_synthesized = 0

            async for chunk in analyze_fn(transcript, self.get_conversation_context()):
                if self._interrupt_flag:
                    break

                if not self.add_response_chunk(chunk):
                    break

                # Progressive TTS
                if self.config.stream_tts:
                    sentences_buffer += chunk

                    # Check for complete sentences
                    while '.' in sentences_buffer or '!' in sentences_buffer or '?' in sentences_buffer:
                        # Find first sentence end
                        for i, char in enumerate(sentences_buffer):
                            if char in '.!?' and i < len(sentences_buffer) - 1:
                                sentence = sentences_buffer[:i+1].strip()
                                sentences_buffer = sentences_buffer[i+1:].strip()

                                if sentence and sentences_synthesized < self.config.max_response_sentences:
                                    audio = synthesize_fn(sentence)
                                    if audio and self.on_audio_ready:
                                        self.on_audio_ready(audio)
                                    sentences_synthesized += 1
                                break
                        else:
                            break

            # Synthesize any remaining text
            if sentences_buffer.strip() and sentences_synthesized < self.config.max_response_sentences:
                if not self._interrupt_flag:
                    audio = synthesize_fn(sentences_buffer.strip())
                    if audio and self.on_audio_ready:
                        self.on_audio_ready(audio)

            self.finish_response()

        except Exception as e:
            print(f"[VoiceLoop] Error in process_audio: {e}")
            import traceback
            traceback.print_exc()
            self.finish_response()

    def reset(self):
        """Reset the voice loop state."""
        self._audio_buffer = []
        self._silence_start = None
        self._speech_start = None
        self._current_response = ""
        self._interrupt_flag = False
        self.conversation_history = []
        self._set_state(VoiceLoopState.IDLE)
