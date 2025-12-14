from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Literal, Optional, Union

import requests

from dotenv import load_dotenv
load_dotenv()

PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"

class PyannoteAIError(RuntimeError):
    pass


def _format_mmss(seconds: Optional[float]) -> str:
    """Format seconds as m:ss (or ??:?? if missing)."""
    if seconds is None:
        return "??:??"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "??:??"
    m, s = divmod(max(total, 0), 60)
    return f"{m}:{s:02d}"


def format_turn_level_transcript(turns: Any) -> str:
    """
    Pretty-print pyannote's turn-level transcript.

    Accepts a list of dicts (typical fields: speaker, start, end, text). This is
    intentionally tolerant to schema variations across pyannote versions.

    Output format (one line per turn):
      SPEAKER (m:ss): text
    """
    if not turns:
        return ""
    if not isinstance(turns, list):
        # best-effort: avoid crashing callers
        return str(turns)

    lines: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            lines.append(str(t))
            continue

        speaker = (
            t.get("speaker")
            or t.get("label")
            or t.get("speakerLabel")
            or "SPEAKER"
        )

        # pyannote sometimes nests timestamps under "segment"
        seg = t.get("segment") if isinstance(t.get("segment"), dict) else {}
        start = t.get("start", seg.get("start"))
        text = (t.get("text") or "").strip()

        lines.append(f"{speaker} ({_format_mmss(start)}): {text}".rstrip())

    return "\n".join(lines)


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _json_headers(api_key: str) -> Dict[str, str]:
    h = _auth_headers(api_key)
    h["Content-Type"] = "application/json"
    return h


def create_media_upload_url(
    *,
    object_key: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Step 1: Declare a media://object-key and get a pre-signed PUT URL.
    POST https://api.pyannote.ai/v1/media/input  {"url": "media://object-key"}
    """
    api_key = api_key or os.getenv("PYANNOTE_API_KEY")
    if not api_key:
        raise PyannoteAIError("Missing API key. Set PYANNOTE_API_KEY or pass api_key=...")

    declare_url = f"{PYANNOTE_API_BASE}/media/input"
    body = {"url": f"media://{object_key}"}

    r = requests.post(declare_url, headers=_json_headers(api_key), json=body, timeout=60)
    if r.status_code not in (200, 201):
        raise PyannoteAIError(f"Failed to get pre-signed upload URL: {r.status_code} - {r.text}")

    data = r.json()
    presigned_put_url = data.get("url")
    if not presigned_put_url:
        raise PyannoteAIError(f"Unexpected response from media/input: {data}")

    return presigned_put_url


def upload_audio_stream_to_presigned_url(
    presigned_put_url: str,
    audio_stream: BinaryIO,
) -> None:
    """
    Step 2: PUT local audio bytes to the pre-signed URL.

    IMPORTANT:
    - Do NOT stream via a generator: requests will use Transfer-Encoding: chunked.
    - Many S3 presigned PUT URLs reject chunked uploads and require Content-Length.
    """
    # Compute remaining bytes in the stream (works for normal file objects)
    try:
        cur = audio_stream.tell()
        audio_stream.seek(0, 2)  # end
        end = audio_stream.tell()
        audio_stream.seek(cur, 0)
        content_length = end - cur
    except Exception:
        # Fallback: if stream isn't seekable, read into memory (last resort)
        data = audio_stream.read()
        content_length = len(data)
        audio_stream = None  # so we use `data=` below

    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(content_length),
    }

    if audio_stream is not None:
        r = requests.put(
            presigned_put_url,
            data=audio_stream,   # file-like object, not a generator
            headers=headers,
            timeout=10 * 60,
        )
    else:
        r = requests.put(
            presigned_put_url,
            data=data,           # bytes fallback
            headers=headers,
            timeout=10 * 60,
        )

    if r.status_code not in (200, 201, 204):
        raise PyannoteAIError(f"Upload PUT failed: {r.status_code} - {r.text}")


def upload_local_audio(
    audio: Union[str, Path, BinaryIO, bytes],
    *,
    api_key: Optional[str] = None,
    object_key: Optional[str] = None,
) -> str:
    """
    Upload local audio and return a media:// URL you can use in /diarize.

    audio can be:
      - path (str/Path)
      - file-like (BinaryIO)
      - raw bytes
    """
    api_key = api_key or os.getenv("PYANNOTE_API_KEY")
    if not api_key:
        raise PyannoteAIError("Missing API key. Set PYANNOTE_API_KEY or pass api_key=...")

    if object_key is None:
        # keep it unique + readable
        object_key = f"uploads/{uuid.uuid4().hex}"

    presigned_put_url = create_media_upload_url(object_key=object_key, api_key=api_key)

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        with path.open("rb") as f:
            upload_audio_stream_to_presigned_url(presigned_put_url, f)
    elif isinstance(audio, (bytes, bytearray)):
        # upload bytes (smallish payloads); for big files use a stream
        r = requests.put(
            presigned_put_url,
            data=audio,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10 * 60,
        )
        if r.status_code not in (200, 201, 204):
            raise PyannoteAIError(f"Upload PUT failed: {r.status_code} - {r.text}")
    else:
        # BinaryIO
        upload_audio_stream_to_presigned_url(presigned_put_url, audio)

    return f"media://{object_key}"


def perform_stt(
    audio_url_or_media_url: str,
    *,
    api_key: Optional[str] = None,
    poll_interval_s: int = 10,
    timeout_s: int = 15 * 60,
    transcript_level: Literal["turn", "word", "both"] = "turn",
) -> Dict[str, Any]:
    """
    Perform speaker-attributed transcription using /diarize with transcription=true.
    audio_url_or_media_url can be:
      - https://... (public)
      - media://... (uploaded via Media API)
    """
    api_key = api_key or os.getenv("PYANNOTE_API_KEY")
    if not api_key:
        raise PyannoteAIError("Missing API key. Set PYANNOTE_API_KEY or pass api_key=...")

    # Create diarization+transcription job
    create_url = f"{PYANNOTE_API_BASE}/diarize"
    payload = {"url": audio_url_or_media_url, "transcription": True}

    r = requests.post(create_url, headers=_json_headers(api_key), json=payload, timeout=60)
    if r.status_code != 200:
        raise PyannoteAIError(f"Create job failed: {r.status_code} - {r.text}")

    job = r.json()
    job_id = job.get("jobId")
    if not job_id:
        raise PyannoteAIError(f"Unexpected create-job response (no jobId): {job}")

    # Poll job
    get_url = f"{PYANNOTE_API_BASE}/jobs/{job_id}"
    deadline = time.time() + timeout_s

    while True:
        if time.time() > deadline:
            raise PyannoteAIError(f"Timed out after {timeout_s}s waiting for job {job_id}.")

        rr = requests.get(get_url, headers=_auth_headers(api_key), timeout=60)
        if rr.status_code != 200:
            raise PyannoteAIError(f"Get job failed: {rr.status_code} - {rr.text}")

        data = rr.json()
        status = data.get("status")

        if status in {"succeeded", "failed", "canceled"}:
            if status != "succeeded":
                raise PyannoteAIError(f"Job {job_id} ended with status={status}: {data}")

            output = data.get("output") or {}
            turn = output.get("turnLevelTranscription")
            word = output.get("wordLevelTranscription")

            if transcript_level == "turn":
                transcript = {"turnLevelTranscription": turn}
            elif transcript_level == "word":
                transcript = {"wordLevelTranscription": word}
            else:
                transcript = {
                    "turnLevelTranscription": turn,
                    "wordLevelTranscription": word,
                }

            return {
                "jobId": job_id,
                "status": status,
                "transcript": transcript,
                "raw_output": output,
            }

        time.sleep(poll_interval_s)


def perform_stt_from_local_audio(
    audio: Union[str, Path, BinaryIO, bytes],
    *,
    api_key: Optional[str] = None,
    object_key: Optional[str] = None,
    poll_interval_s: int = 10,
    timeout_s: int = 15 * 60,
    transcript_level: Literal["turn", "word", "both"] = "turn",
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      local audio -> upload to media:// -> run perform_stt(media://...)
    """
    api_key = api_key or os.getenv("PYANNOTE_API_KEY")
    if not api_key:
        raise PyannoteAIError("Missing API key. Set PYANNOTE_API_KEY or pass api_key=...")

    media_url = upload_local_audio(audio, api_key=api_key, object_key=object_key)
    return perform_stt(
        media_url,
        api_key=api_key,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
        transcript_level=transcript_level,
    )
