# speech_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from gradium_tts import tts_from_text_sync

# IMPORTANT:
# Update this import to match *your* pyannote file name.
# Example: if your file is named "pyannote_stt.py", keep as-is.
from pyannote_stt import perform_stt, format_turn_level_transcript


def stt_then_tts(
    audio_url: str,
    *,
    tts_output_path: str | Path = "transcript.wav",
    transcript_level: str = "turn",
    voice_id: str = "YTpq7expH9539ERJ",
) -> Dict[str, Any]:
    """
    1) Run pyannote STT on an audio URL (speaker-attributed)
    2) Convert resulting transcript into a single text string
    3) Run Gradium TTS locally and write a wav
    4) Return both: STT raw + TTS frontend-friendly payload
    """
    stt_result = perform_stt(audio_url, transcript_level=transcript_level)

    # Prefer turn-level for readable text
    transcript_text = ""
    if transcript_level in ("turn", "both"):
        turns = stt_result["transcript"].get("turnLevelTranscription") or []
        transcript_text = format_turn_level_transcript(turns)
    elif transcript_level == "word":
        # Fallback: try to join word-level transcription if present
        words = stt_result["transcript"].get("wordLevelTranscription") or []
        transcript_text = " ".join((w.get("text") or "").strip() for w in words).strip()

    if not transcript_text:
        transcript_text = "[No transcript text produced]"

    tts_result = tts_from_text_sync(
        transcript_text,
        output_path=tts_output_path,
        voice_id=voice_id,
        output_format="wav",
        model_name="default",
    )

    return {
        "stt": stt_result,
        "tts": {
            "output_path": str(tts_result.output_path),
            "sample_rate": tts_result.sample_rate,
            "request_id": tts_result.request_id,
            "frontend_payload": tts_result.as_frontend_payload(),
        },
    }


if __name__ == "__main__":
    # Example
    audio_url = "https://files.pyannote.ai/marklex1min.wav"
    bundle = stt_then_tts(audio_url, tts_output_path="transcript.wav")
    print("Wrote:", bundle["tts"]["output_path"])
    print("TTS sample rate:", bundle["tts"]["sample_rate"])
    print("TTS request ID:", bundle["tts"]["request_id"])
