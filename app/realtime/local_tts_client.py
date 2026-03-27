# app/realtime/local_tts_client.py
from __future__ import annotations

import asyncio
import audioop
import os
import tempfile
from pathlib import Path


class LocalPiperTTSClient:
    """
    Local TTS client backed by Piper.

    Flow:
    - call piper.exe with the configured ONNX model
    - generate a temporary WAV file
    - read PCM frames from the WAV
    - resample to 8 kHz mono if needed
    - convert PCM16 -> μ-law
    - yield Twilio-compatible 160-byte μ-law chunks
    """

    def __init__(
        self,
        *,
        model_path: str,
        config_path: str = "",
        speaker_id: int = 0,
        target_sample_rate: int = 8000,
    ) -> None:
        self.model_path = model_path.strip()
        self.config_path = config_path.strip()
        self.speaker_id = speaker_id
        self.target_sample_rate = target_sample_rate

        self.piper_binary = os.getenv("COMPASS_PIPER_BINARY", "").strip()
        self.chunk_size_bytes = int(
            os.getenv("COMPASS_LOCAL_TTS_PCM_READ_CHUNK_BYTES", "4096")
        )

        if not self.piper_binary:
            raise ValueError(
                "COMPASS_PIPER_BINARY is required for LocalPiperTTSClient."
            )

        if not self.model_path:
            raise ValueError(
                "COMPASS_LOCAL_TTS_MODEL_PATH is required for LocalPiperTTSClient."
            )

        if not Path(self.piper_binary).exists():
            raise FileNotFoundError(
                f"Piper binary not found: {self.piper_binary}"
            )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Piper model not found: {self.model_path}"
            )

        if self.config_path and not Path(self.config_path).exists():
            raise FileNotFoundError(
                f"Piper config not found: {self.config_path}"
            )

    async def stream_mulaw_8k(self, text: str):
        """
        Synthesize text to μ-law 8 kHz and yield Twilio-compatible chunks.
        """
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return

        with tempfile.TemporaryDirectory(prefix="compass_piper_") as tmp_dir:
            wav_path = Path(tmp_dir) / "tts.wav"
            await self._run_piper(cleaned, wav_path)

            async for chunk in self._wav_to_mulaw_chunks(wav_path):
                yield chunk

    async def _run_piper(self, text: str, wav_path: Path) -> None:
        cmd = [
            self.piper_binary,
            "--model",
            self.model_path,
            "--output_file",
            str(wav_path),
            "--speaker",
            str(self.speaker_id),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate(input=text.encode("utf-8"))

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "Piper synthesis failed. "
                f"returncode={process.returncode}; "
                f"stdout={stdout_text}; stderr={stderr_text}"
            )

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise RuntimeError(
                f"Piper did not produce a valid WAV file at {wav_path}"
            )

    async def _wav_to_mulaw_chunks(self, wav_path: Path):
        import wave

        loop = asyncio.get_running_loop()

        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            input_rate = wav_file.getframerate()

            if sample_width != 2:
                raise ValueError(
                    f"Unsupported WAV sample width: {sample_width}. Expected PCM16."
                )

            pcm16 = await loop.run_in_executor(
                None,
                self._read_all_frames,
                wav_file,
                self.chunk_size_bytes,
            )

        pcm16 = self._normalize_channels(pcm16, channels=channels, sample_width=sample_width)
        pcm16 = self._resample_pcm16(
            pcm16,
            src_rate=input_rate,
            dst_rate=self.target_sample_rate,
            sample_width=sample_width,
        )

        mulaw_bytes = audioop.lin2ulaw(pcm16, 2)

        twilio_frame_bytes = 160
        for offset in range(0, len(mulaw_bytes), twilio_frame_bytes):
            chunk = mulaw_bytes[offset : offset + twilio_frame_bytes]
            if not chunk:
                continue

            if len(chunk) < twilio_frame_bytes:
                chunk += b"\xFF" * (twilio_frame_bytes - len(chunk))

            yield chunk
            await asyncio.sleep(0)

    def _read_all_frames(
        self,
        wav_file,
        read_chunk_bytes: int,
    ) -> bytes:
        pcm_parts: list[bytes] = []

        # read_chunk_bytes here refers to raw PCM bytes.
        # For wave.readframes, we need frame count.
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        bytes_per_frame = channels * sample_width
        frames_per_read = max(1, read_chunk_bytes // bytes_per_frame)

        while True:
            frames = wav_file.readframes(frames_per_read)
            if not frames:
                break
            pcm_parts.append(frames)

        return b"".join(pcm_parts)

    def _normalize_channels(
        self,
        pcm16: bytes,
        *,
        channels: int,
        sample_width: int,
    ) -> bytes:
        if channels == 1:
            return pcm16

        if channels != 2:
            raise ValueError(
                f"Unsupported channel count: {channels}. Expected mono or stereo."
            )

        # Convert stereo PCM16 to mono PCM16.
        return audioop.tomono(pcm16, sample_width, 0.5, 0.5)

    def _resample_pcm16(
        self,
        pcm16: bytes,
        *,
        src_rate: int,
        dst_rate: int,
        sample_width: int,
    ) -> bytes:
        if src_rate == dst_rate:
            return pcm16

        resampled, _ = audioop.ratecv(
            pcm16,
            sample_width,
            1,
            src_rate,
            dst_rate,
            None,
        )
        return resampled