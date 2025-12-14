# gradium_tts_streaming_local.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union, Callable

import gradium

PCM_SAMPLE_RATE_HZ = 48_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2  # 16-bit signed int little-endian


@dataclass(frozen=True)
class StreamingTTSMeta:
    request_id: str
    output_format: str  # "pcm" typically for streaming
    sample_rate: int = PCM_SAMPLE_RATE_HZ
    channels: int = PCM_CHANNELS
    sample_width_bytes: int = PCM_SAMPLE_WIDTH_BYTES


async def tts_stream_bytes(
    text: Union[str, AsyncIterator[str]],
    *,
    client: Optional["gradium.client.GradiumClient"] = None,
    model_name: str = "default",
    voice_id: str = "YTpq7expH9539ERJ",
    output_format: str = "pcm",
    json_config: Optional[Dict[str, Any]] = None,
) -> tuple[StreamingTTSMeta, AsyncIterator[bytes]]:
    """
    Start a Gradium streaming TTS request and return:
      (meta, async_iterator_of_audio_bytes)

    Notes:
    - For output_format="pcm", chunks are raw PCM (48kHz, 16-bit signed LE, mono).
    - This is perfect to forward to a frontend as a stream later.
    """
    _client = client or gradium.client.GradiumClient()

    setup: Dict[str, Any] = {
        "model_name": model_name,
        "voice_id": voice_id,
        "output_format": output_format,
    }
    if json_config:
        setup["json_config"] = json_config

    stream = await _client.tts_stream(setup=setup, text=text)

    # Some SDKs expose request_id on the stream; if not, keep empty string safely.
    request_id = getattr(stream, "request_id", "") or ""

    meta = StreamingTTSMeta(
        request_id=request_id,
        output_format=output_format,
        # For now we assume Gradium's documented PCM settings.
        sample_rate=PCM_SAMPLE_RATE_HZ if output_format == "pcm" else 0,
        channels=PCM_CHANNELS if output_format == "pcm" else 0,
        sample_width_bytes=PCM_SAMPLE_WIDTH_BYTES if output_format == "pcm" else 0,
    )

    async def iterator() -> AsyncIterator[bytes]:
        async for chunk in stream.iter_bytes():
            yield chunk

    return meta, iterator()


async def write_stream_to_file(
    chunks: AsyncIterator[bytes],
    output_path: str | Path,
) -> Path:
    """
    Consume an async byte-stream and write it to disk (binary).
    For PCM output, this writes raw .pcm bytes (not a WAV container).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        async for chunk in chunks:
            f.write(chunk)

    return output_path


async def tts_stream_to_file(
    text: Union[str, AsyncIterator[str]],
    *,
    output_path: str | Path = "output.pcm",
    client: Optional["gradium.client.GradiumClient"] = None,
    model_name: str = "default",
    voice_id: str = "YTpq7expH9539ERJ",
    output_format: str = "pcm",
    json_config: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable[[bytes], None]] = None,
) -> tuple[StreamingTTSMeta, Path]:
    """
    Convenience: start streaming, optionally call on_chunk(chunk) for each chunk,
    and write all chunks to a local file.
    """
    meta, chunks = await tts_stream_bytes(
        text,
        client=client,
        model_name=model_name,
        voice_id=voice_id,
        output_format=output_format,
        json_config=json_config,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        async for chunk in chunks:
            if on_chunk:
                on_chunk(chunk)
            f.write(chunk)

    return meta, output_path


def tts_stream_to_file_sync(*args, **kwargs):
    """Sync wrapper if you want to use it outside async code."""
    return asyncio.run(tts_stream_to_file(*args, **kwargs))


def pcm_stream_frontend_info(meta: StreamingTTSMeta) -> Dict[str, Any]:
    """
    A small JSON-serializable blob you can send alongside streaming PCM to help
    the frontend decode/play it correctly.
    """
    if meta.output_format != "pcm":
        return {"output_format": meta.output_format, "request_id": meta.request_id}

    return {
        "output_format": "pcm",
        "request_id": meta.request_id,
        "sample_rate_hz": meta.sample_rate,        # 48000
        "channels": meta.channels,                 # 1
        "sample_width_bytes": meta.sample_width_bytes,  # 2 (16-bit)
        "endianness": "little",
        "encoding": "signed-int",
    }
