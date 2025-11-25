"""
Local streaming STT using Vosk for ultra-low latency and frequent updates.

Vosk provides true streaming with partial results every ~100ms, ideal for responsive turn detection.
"""

import asyncio
import json
import logging
import os
from typing import Union

import numpy as np
from vosk import Model, KaldiRecognizer
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)


class VoskSTT(stt.STT):
    """
    Local streaming STT using Vosk.

    Provides ultra-low latency with partial results every ~100ms for responsive turn detection.
    """

    def __init__(
        self,
        *,
        model_path: str,
        language: str = "en",
    ):
        """
        Create a new instance of Vosk STT.

        Args:
            model_path: Path to the Vosk model directory
            language: Language code (e.g., "en" for English)
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False
            )
        )

        self._model_path = model_path
        self._language = language

        logger.info(f"Initializing Vosk model from: {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Vosk model not found at: {model_path}")

        self._model = Model(model_path)
        logger.info("Vosk model loaded successfully")

    @property
    def model(self) -> str:
        return os.path.basename(self._model_path)

    @property
    def provider(self) -> str:
        return "vosk-local"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Batch transcription (non-streaming)"""
        # Combine all audio frames
        audio_frame = rtc.combine_audio_frames(buffer)

        # Convert to numpy array
        audio_data = audio_frame.data.tobytes()

        # Create recognizer (Vosk expects 16kHz)
        rec = KaldiRecognizer(self._model, 16000)

        # Process in chunks
        chunk_size = 4000
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            rec.AcceptWaveform(chunk)

        # Get final result
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=self._language,
                )
            ]
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> "VoskRecognizeStream":
        lang = language if language != NOT_GIVEN else self._language
        return VoskRecognizeStream(
            stt=self,
            model=self._model,
            language=lang,
            conn_options=conn_options,
        )


class VoskRecognizeStream(stt.RecognizeStream):
    """
    Streaming recognition using Vosk with ultra-low latency.

    Emits partial results every ~100ms for responsive turn detection.
    """

    def __init__(
        self,
        *,
        stt: VoskSTT,
        model: Model,
        language: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=16000,  # Vosk expects 16kHz
        )

        self._model = model
        self._language = language
        self._recognizer = KaldiRecognizer(model, 16000)
        self._recognizer.SetWords(True)
        self._last_partial_text = ""
        self._speech_started = False

    async def _run(self) -> None:
        """Main streaming loop - processes audio in real-time"""
        logger.info("Starting Vosk streaming recognition")

        try:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    # Convert audio to bytes
                    audio_bytes = frame.data.tobytes()

                    # Emit START_OF_SPEECH on first audio
                    if not self._speech_started:
                        self._speech_started = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        )
                        logger.info("🎤 START_OF_SPEECH emitted")

                    # Process audio with Vosk
                    if self._recognizer.AcceptWaveform(audio_bytes):
                        # Complete result available
                        result = json.loads(self._recognizer.Result())
                        text = result.get("text", "").strip()

                        if text:
                            # Emit FINAL transcript for this segment
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=text,
                                            language=self._language,
                                        )
                                    ]
                                )
                            )
                            logger.info(f"📝 FINAL: {text}")
                            self._last_partial_text = ""
                    else:
                        # Partial result
                        partial = json.loads(self._recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()

                        # Only emit if text changed
                        if partial_text and partial_text != self._last_partial_text:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=partial_text,
                                            language=self._language,
                                        )
                                    ]
                                )
                            )
                            logger.debug(f"💬 PARTIAL: {partial_text}")
                            self._last_partial_text = partial_text

                elif isinstance(frame, self._FlushSentinel):
                    # Flush and get final result
                    result = json.loads(self._recognizer.FinalResult())
                    text = result.get("text", "").strip()

                    if self._speech_started:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                        logger.info("🔇 END_OF_SPEECH emitted")
                        self._speech_started = False

                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=self._language,
                                    )
                                ]
                            )
                        )
                        logger.info(f"📝 FINAL (flush): {text}")

                    # Reset recognizer for next turn
                    self._recognizer = KaldiRecognizer(self._model, 16000)
                    self._recognizer.SetWords(True)
                    self._last_partial_text = ""

        except Exception as e:
            logger.error(f"Error in Vosk streaming: {e}", exc_info=True)
        finally:
            logger.info("Vosk streaming recognition stopped")
