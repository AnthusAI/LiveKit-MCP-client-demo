"""
Local streaming STT using faster-whisper for frequent interim transcripts.

This provides true streaming transcription with partial results emitted every 1-2 seconds,
which is critical for responsive turn detection.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, aio

logger = logging.getLogger(__name__)


@dataclass
class _STTOptions:
    model_size: str
    language: str
    device: str
    compute_type: str


class LocalWhisperSTT(stt.STT):
    """
    Local streaming STT using faster-whisper.

    Provides frequent interim transcripts (every 1-2 seconds) for responsive turn detection.
    """

    def __init__(
        self,
        *,
        model_size: str = "base",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Create a new instance of Local Whisper STT.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Language code (e.g., "en" for English)
            device: Device to run on ("cpu", "cuda", or "auto")
            compute_type: Computation type ("int8", "int16", "float16", "float32")
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False
            )
        )

        self._opts = _STTOptions(
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
        )

        logger.info(f"Initializing faster-whisper model: {model_size} ({device}/{compute_type})")
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper model loaded successfully")

    @property
    def model(self) -> str:
        return f"faster-whisper-{self._opts.model_size}"

    @property
    def provider(self) -> str:
        return "faster-whisper-local"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Batch transcription (non-streaming)"""
        # Combine all audio frames into one
        audio_frame = rtc.combine_audio_frames(buffer)

        # Convert to numpy array (float32, normalized to [-1, 1])
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        lang = language if language != NOT_GIVEN else self._opts.language

        # Transcribe
        segments, info = self._model.transcribe(
            audio_data,
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Combine all segments
        full_text = " ".join([segment.text for segment in segments]).strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=full_text,
                    language=lang,
                )
            ]
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> "WhisperRecognizeStream":
        lang = language if language != NOT_GIVEN else self._opts.language
        return WhisperRecognizeStream(
            stt=self,
            model=self._model,
            language=lang,
            conn_options=conn_options,
        )


class WhisperRecognizeStream(stt.RecognizeStream):
    """
    Streaming recognition using faster-whisper with sliding window.

    Emits interim transcripts every 1-2 seconds for responsive turn detection.
    """

    def __init__(
        self,
        *,
        stt: LocalWhisperSTT,
        model: WhisperModel,
        language: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=16000,  # Whisper expects 16kHz
        )

        self._model = model
        self._language = language
        self._audio_buffer: list[bytes] = []
        self._last_transcript_time = 0.0
        self._transcript_interval = 1.5  # Emit interim every 1.5 seconds
        self._window_duration = 10.0  # Process last 10 seconds of audio
        self._sample_rate = 16000
        self._bytes_per_second = self._sample_rate * 2  # 16-bit audio = 2 bytes per sample

    async def _run(self) -> None:
        """Main streaming loop"""
        logger.info("Starting LocalWhisper streaming recognition")

        # Start background task to process audio periodically
        process_task = asyncio.create_task(self._process_audio_periodically())

        # Track if we've detected speech
        speech_detected = False

        try:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    # Add audio to buffer
                    audio_bytes = frame.data.tobytes()
                    self._audio_buffer.append(audio_bytes)

                    # Emit START_OF_SPEECH on first audio
                    if not speech_detected and len(self._audio_buffer) > 0:
                        speech_detected = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        )
                        logger.info("🎤 START_OF_SPEECH emitted")

                    # Trim buffer to window duration to prevent memory growth
                    max_bytes = int(self._window_duration * self._bytes_per_second)
                    total_bytes = sum(len(chunk) for chunk in self._audio_buffer)

                    while total_bytes > max_bytes and len(self._audio_buffer) > 1:
                        removed = self._audio_buffer.pop(0)
                        total_bytes -= len(removed)

                elif isinstance(frame, self._FlushSentinel):
                    # User stopped speaking - emit END_OF_SPEECH then final transcript
                    if speech_detected:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                        logger.info("🔇 END_OF_SPEECH emitted")
                        speech_detected = False

                    await self._process_and_emit(final=True)
                    self._audio_buffer.clear()

        finally:
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass

            logger.info("LocalWhisper streaming recognition stopped")

    async def _process_audio_periodically(self):
        """Background task that processes audio every transcript_interval"""
        while True:
            try:
                await asyncio.sleep(self._transcript_interval)

                # Only process if enough time has passed and we have audio
                if self._audio_buffer and time.time() - self._last_transcript_time >= self._transcript_interval:
                    await self._process_and_emit(final=False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic audio processing: {e}", exc_info=True)

    async def _process_and_emit(self, final: bool):
        """Process current audio buffer and emit transcript"""
        if not self._audio_buffer:
            return

        try:
            # Combine audio chunks
            combined_audio = b"".join(self._audio_buffer)

            # Convert to numpy array
            audio_array = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip if too short (less than 0.5 seconds)
            if len(audio_array) < self._sample_rate * 0.5:
                return

            # Transcribe in executor to avoid blocking
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_array,
                    language=self._language,
                    beam_size=5,
                    vad_filter=False,  # CRITICAL: Disable faster-whisper's internal VAD
                    without_timestamps=True,  # Faster processing
                    condition_on_previous_text=False,  # Each transcript is independent
                    no_speech_threshold=1.0,  # NEVER skip due to silence detection
                )
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()

            if text:
                # IMPORTANT: Always emit FINAL_TRANSCRIPT (not INTERIM)
                # Without VAD, only FINAL_TRANSCRIPT events trigger turn detection
                # Each chunk is "final" for its time window - this allows turn detection
                # to run frequently (every 1.5s) based on what you're actually saying
                event_type = stt.SpeechEventType.FINAL_TRANSCRIPT

                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        alternatives=[
                            stt.SpeechData(
                                text=text,
                                language=self._language,
                            )
                        ]
                    )
                )

                self._last_transcript_time = time.time()

                logger.info(f"📝 Emitted FINAL transcript: {text[:100]}...")

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
