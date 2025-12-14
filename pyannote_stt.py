"""
pyannoteAI STT (speaker-attributed transcription) via diarization + transcription.

Based on pyannoteAI docs:
- POST https://api.pyannote.ai/v1/diarize with {"url": ..., "transcription": true}
- Poll   https://api.pyannote.ai/v1/jobs/{jobId} until status is succeeded/failed/canceled
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Literal, Optional

import requests

from dotenv import load_dotenv

load_dotenv()


PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"


class PyannoteAIError(RuntimeError):
    pass


def perform_stt(
    audio_url: str,
    *,
    api_key: Optional[str] = None,
    poll_interval_s: int = 10,
    timeout_s: int = 15 * 60,
    transcript_level: Literal["turn", "word", "both"] = "turn",
) -> Dict[str, Any]:
    """
    Perform STT with speaker attribution using pyannoteAI's diarize endpoint
    with transcription enabled.

    Args:
        audio_url: Publicly accessible direct URL to an audio file (.wav, .mp3, etc.)
        api_key: pyannoteAI API key (defaults to env var PYANNOTE_API_KEY)
        poll_interval_s: Seconds between polling job status
        timeout_s: Max seconds to wait for completion
        transcript_level: "turn" for turnLevelTranscription, "word" for wordLevelTranscription,
                         "both" for both fields.

    Returns:
        Dict containing:
          - jobId
          - status
          - transcript (turn/word/both)
          - raw_output (full job output payload)

    Raises:
        PyannoteAIError on HTTP errors, timeouts, or failed/canceled jobs.
    """
    api_key = api_key or os.getenv("PYANNOTE_API_KEY")
    if not api_key:
        raise PyannoteAIError(
            "Missing API key. Pass api_key=... or set PYANNOTE_API_KEY in your environment."
        )

    # 1) Create diarization job with transcription enabled
    create_url = f"{PYANNOTE_API_BASE}/diarize"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"url": audio_url, "transcription": True}

    r = requests.post(create_url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise PyannoteAIError(f"Create job failed: {r.status_code} - {r.text}")

    job = r.json()
    job_id = job.get("jobId")
    if not job_id:
        raise PyannoteAIError(f"Unexpected create-job response (no jobId): {job}")

    # 2) Poll job until completion
    get_url = f"{PYANNOTE_API_BASE}/jobs/{job_id}"
    get_headers = {"Authorization": f"Bearer {api_key}"}

    deadline = time.time() + timeout_s
    while True:
        if time.time() > deadline:
            raise PyannoteAIError(
                f"Timed out after {timeout_s}s waiting for job {job_id}."
            )

        rr = requests.get(get_url, headers=get_headers, timeout=60)
        if rr.status_code != 200:
            raise PyannoteAIError(f"Get job failed: {rr.status_code} - {rr.text}")

        data = rr.json()
        status = data.get("status")

        if status in {"succeeded", "failed", "canceled"}:
            if status != "succeeded":
                raise PyannoteAIError(f"Job {job_id} ended with status={status}: {data}")
            output = data.get("output") or {}

            # 3) Extract transcript fields
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


def format_turn_level_transcript(turn_level_transcription: list[dict]) -> str:
    """Pretty-print turn-level transcript as SPEAKER (m:ss): text"""
    lines: list[str] = []
    for turn in turn_level_transcription or []:
        start = float(turn.get("start", 0.0))
        speaker = turn.get("speaker", "SPEAKER_??")
        text = (turn.get("text") or "").strip()
        timestamp = f"{int(start // 60)}:{int(start % 60):02d}"
        lines.append(f"{speaker} ({timestamp}): {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage (set PYANNOTE_API_KEY in your env, or pass api_key=...)
    result = perform_stt(
        "https://files.pyannote.ai/marklex1min.wav",
        transcript_level="turn",
        poll_interval_s=10,
        timeout_s=10 * 60,
    )

    turns = result["transcript"]["turnLevelTranscription"]
    print(format_turn_level_transcript(turns))
