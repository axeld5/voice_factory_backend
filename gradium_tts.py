# gradium_tts_local.py
from __future__ import annotations

import asyncio
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gradium


@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    request_id: str
    output_path: Path

    def as_frontend_payload(self) -> Dict[str, Any]:
        """
        JSON-friendly payload you can later return from an API endpoint.
        (You can also skip base64 and stream bytes directly in a real backend.)
        """
        mime, _ = mimetypes.guess_type(self.output_path.name)
        if not mime:
            mime = "audio/wav"
        return {
            "filename": self.output_path.name,
            "mime_type": mime,
            "sample_rate": self.sample_rate,
            "request_id": self.request_id,
            "audio_base64": base64.b64encode(self.audio_bytes).decode("ascii"),
        }


async def tts_from_text(
    text: str,
    *,
    output_path: str | Path = "output.wav",
    model_name: str = "default",
    voice_id: str = "YTpq7expH9539ERJ",
    output_format: str = "wav",
    client: Optional["gradium.client.GradiumClient"] = None,
) -> TTSResult:
    """
    Generate TTS locally (writes a wav file) and returns bytes + metadata.
    """
    output_path = Path(output_path)

    _client = client or gradium.client.GradiumClient()
    result = await _client.tts(
        setup={
            "model_name": model_name,
            "voice_id": voice_id,
            "output_format": output_format,
        },
        text=text,
    )

    audio_bytes = result.raw_data
    output_path.write_bytes(audio_bytes)

    return TTSResult(
        audio_bytes=audio_bytes,
        sample_rate=int(result.sample_rate),
        request_id=str(result.request_id),
        output_path=output_path,
    )


async def tts_from_file(
    text_file_path: str | Path,
    *,
    output_path: str | Path = "output.wav",
    encoding: str = "utf-8",
    model_name: str = "default",
    voice_id: str = "YTpq7expH9539ERJ",
    output_format: str = "wav",
    client: Optional["gradium.client.GradiumClient"] = None,
) -> TTSResult:
    """
    Given a text file, read it and generate TTS to a local wav file.
    Returns a TTSResult, which can be turned into a frontend payload.
    """
    text_file_path = Path(text_file_path)
    if not text_file_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_file_path}")

    text = text_file_path.read_text(encoding=encoding).strip()
    if not text:
        raise ValueError(f"Text file is empty: {text_file_path}")

    return await tts_from_text(
        text,
        output_path=output_path,
        model_name=model_name,
        voice_id=voice_id,
        output_format=output_format,
        client=client,
    )


def tts_from_file_sync(*args, **kwargs) -> TTSResult:
    """
    Convenience sync wrapper if you don't want to deal with asyncio yet.
    """
    return asyncio.run(tts_from_file(*args, **kwargs))


def tts_from_text_sync(*args, **kwargs) -> TTSResult:
    """
    Convenience sync wrapper if you don't want to deal with asyncio yet.
    """
    return asyncio.run(tts_from_text(*args, **kwargs))

