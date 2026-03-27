# app/realtime/local_stt_client.py
from __future__ import annotations

import asyncio
import audioop
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from app.realtime.deepgram_stt_client import DeepgramSTTCallbacks


@dataclass(slots=True)
class LocalWhisperSTTConfig:
    model_name: str = "medium.en"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    sample_rate: int = 16000

    input_encoding: str = "mulaw"
    input_sample_rate: int = 8000
    input_channels: int = 1
    sample_width_bytes: int = 2

    endpointing_ms: int = 300
    utterance_end_ms: int = 1000
    vad_enabled: bool = True
    rms_threshold: int = 220
    min_utterance_ms: int = 250
    speech_start_min_ms: int = 180


class LocalWhisperSTTEngine:
    """
    Shared STT engine loaded once for the whole application lifecycle.
    """

    def __init__(self, config: LocalWhisperSTTConfig) -> None:
        self.config = config
        self.model: WhisperModel | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        print(
            "[LOCAL STT] loading shared model...",
            {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
            },
        )

        self.model = WhisperModel(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        print(
            "[LOCAL STT] shared model loaded",
            {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
            },
        )

    def unload(self) -> None:
        self.model = None

    def create_session(
        self,
        *,
        callbacks: DeepgramSTTCallbacks,
    ) -> "LocalWhisperSTTSession":
        if self.model is None:
            raise RuntimeError("LocalWhisperSTTEngine is not loaded.")
        return LocalWhisperSTTSession(engine=self, callbacks=callbacks)


class LocalWhisperSTTSession:
    """
    Per-call STT session that reuses the shared Whisper model.
    """

    def __init__(
        self,
        *,
        engine: LocalWhisperSTTEngine,
        callbacks: DeepgramSTTCallbacks,
    ) -> None:
        self.engine = engine
        self.callbacks = callbacks
        self.config = engine.config

        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self._speech_active = False
        self._speech_started_at_monotonic: float | None = None
        self._last_voice_at_monotonic: float | None = None

        self._speech_candidate_started_at_monotonic: float | None = None
        self._utterance_pcm8k = bytearray()

    async def start(self) -> None:
        if self._running:
            return

        if self.engine.model is None:
            raise RuntimeError("Shared Whisper model is not loaded.")

        self._running = True
        self._monitor_task = asyncio.create_task(self._endpoint_monitor_loop())

    async def send_audio(self, audio_bytes: bytes) -> None:
        if not self._running or not audio_bytes:
            return

        try:
            pcm8k = self._decode_mulaw_to_pcm16(audio_bytes)
            is_voiced = self._is_voiced_chunk(pcm8k)
            now = time.perf_counter()

            async with self._lock:
                if is_voiced:
                    if not self._speech_active:
                        if self._speech_candidate_started_at_monotonic is None:
                            self._speech_candidate_started_at_monotonic = now

                        candidate_ms = (
                            now - self._speech_candidate_started_at_monotonic
                        ) * 1000.0

                        if candidate_ms >= self.config.speech_start_min_ms:
                            self._speech_active = True
                            self._speech_started_at_monotonic = (
                                self._speech_candidate_started_at_monotonic
                            )
                            await self._emit_event(
                                "message:SpeechStarted",
                                {
                                    "provider": "local",
                                    "timestamp_monotonic": self._speech_started_at_monotonic,
                                },
                            )

                    if self._speech_active:
                        self._last_voice_at_monotonic = now
                        self._utterance_pcm8k.extend(pcm8k)

                else:
                    self._speech_candidate_started_at_monotonic = None

                    if self._speech_active:
                        self._utterance_pcm8k.extend(pcm8k)

        except Exception as exc:
            await self._emit_error(
                "local_stt:send_audio_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    async def finalize(self) -> None:
        if not self._running:
            return
        await self._flush_current_utterance(force=True)

    async def close(self) -> None:
        self._running = False

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._monitor_task = None

    async def _endpoint_monitor_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(0.05)

                async with self._lock:
                    if not self._speech_active or self._last_voice_at_monotonic is None:
                        continue

                    silence_ms = (time.perf_counter() - self._last_voice_at_monotonic) * 1000.0
                    if silence_ms < self.config.utterance_end_ms:
                        continue

                await self._flush_current_utterance(force=False)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit_error(
                "local_stt:endpoint_monitor_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    async def _flush_current_utterance(self, *, force: bool) -> None:
        async with self._lock:
            if not self._utterance_pcm8k:
                self._reset_current_utterance()
                return

            pcm8k = bytes(self._utterance_pcm8k)
            speech_started_at = self._speech_started_at_monotonic
            self._reset_current_utterance()

        utterance_ms = self._pcm_duration_ms(
            pcm8k,
            sample_rate=self.config.input_sample_rate,
            sample_width_bytes=self.config.sample_width_bytes,
        )

        if utterance_ms < self.config.min_utterance_ms and not force:
            return

        try:
            transcript = await self._transcribe_pcm8k_bytes(pcm8k)
        except Exception as exc:
            await self._emit_error(
                "local_stt:transcription_failed",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "utterance_ms": round(utterance_ms, 3),
                },
            )
            return

        cleaned = " ".join((transcript or "").split()).strip()
        if not cleaned:
            await self._emit_event(
                "message:UtteranceEnd",
                {
                    "provider": "local",
                    "speech_started_at_monotonic": speech_started_at,
                    "utterance_ms": round(utterance_ms, 3),
                    "text": "",
                },
            )
            return

        await self._emit_transcript(
            cleaned,
            True,
            {
                "provider": "local",
                "is_final": True,
                "speech_final": True,
                "utterance_ms": round(utterance_ms, 3),
                "speech_started_at_monotonic": speech_started_at,
            },
        )

        await self._emit_event(
            "message:UtteranceEnd",
            {
                "provider": "local",
                "speech_started_at_monotonic": speech_started_at,
                "utterance_ms": round(utterance_ms, 3),
                "text": cleaned,
            },
        )

    async def _transcribe_pcm8k_bytes(self, pcm8k: bytes) -> str:
        if self.engine.model is None:
            raise RuntimeError("Shared Whisper model is not initialized.")

        audio_float32 = self._pcm8k_to_resampled_float32(
            pcm8k=pcm8k,
            src_rate=self.config.input_sample_rate,
            dst_rate=self.config.sample_rate,
        )

        segments, _info = self.engine.model.transcribe(
            audio_float32,
            language=self.config.language,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", "") or ""
            text = text.strip()
            if text:
                parts.append(text)

        return " ".join(parts).strip()

    def _decode_mulaw_to_pcm16(self, audio_bytes: bytes) -> bytes:
        if self.config.input_encoding != "mulaw":
            raise ValueError(f"Unsupported local STT input encoding: {self.config.input_encoding}")
        return audioop.ulaw2lin(audio_bytes, self.config.sample_width_bytes)

    def _is_voiced_chunk(self, pcm16_bytes: bytes) -> bool:
        if not self.config.vad_enabled:
            return True
        if not pcm16_bytes:
            return False
        rms = audioop.rms(pcm16_bytes, self.config.sample_width_bytes)
        return rms >= self.config.rms_threshold

    def _pcm8k_to_resampled_float32(
        self,
        *,
        pcm8k: bytes,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        audio_int16 = np.frombuffer(pcm8k, dtype=np.int16)
        if audio_int16.size == 0:
            return np.zeros((0,), dtype=np.float32)

        if src_rate != dst_rate:
            audio_float = resample_poly(audio_int16.astype(np.float32), dst_rate, src_rate)
        else:
            audio_float = audio_int16.astype(np.float32)

        audio_float32 = audio_float / 32768.0
        return np.asarray(audio_float32, dtype=np.float32)

    def _pcm_duration_ms(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        sample_width_bytes: int,
    ) -> float:
        if not pcm_bytes:
            return 0.0
        samples = len(pcm_bytes) / sample_width_bytes
        return (samples / sample_rate) * 1000.0

    def _reset_current_utterance(self) -> None:
        self._speech_active = False
        self._speech_started_at_monotonic = None
        self._last_voice_at_monotonic = None
        self._speech_candidate_started_at_monotonic = None
        self._utterance_pcm8k.clear()

    async def _emit_transcript(self, transcript: str, is_final: bool, payload: dict[str, Any]) -> None:
        callback = getattr(self.callbacks, "on_transcript", None)
        if callback is not None:
            await callback(transcript, is_final, payload)

    async def _emit_event(self, name: str, payload: dict[str, Any]) -> None:
        callback = getattr(self.callbacks, "on_event", None)
        if callback is not None:
            await callback(name, payload)

    async def _emit_error(self, name: str, payload: dict[str, Any]) -> None:
        callback = getattr(self.callbacks, "on_error", None)
        if callback is not None:
            await callback(name, payload)