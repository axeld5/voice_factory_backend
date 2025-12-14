from __future__ import annotations

import base64
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from gradium_tts import tts_from_text
from main import build_transcript_text, ensure_env
from pyannote_stt import perform_stt_from_local_audio
from text2sql import (
    execute_sql,
    generate_sql,
    generate_tts_answer,
    get_data_path,
    get_prompt_path,
)

load_dotenv()

TranscriptLevel = Literal["turn", "word", "both"]


@dataclass(frozen=True)
class VoiceFactoryInput:
    """
    Internal representation of the frontend input.

    Frontend can provide either:
      - text (natural language query)
      - audio_bytes (stream / upload) which will be transcribed to text
    """

    text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_filename: Optional[str] = None
    transcript_level: TranscriptLevel = "turn"


@dataclass(frozen=True)
class VoiceFactoryOutput:
    """
    Internal representation of the pipeline output.

    - answer_text: the final text answer (what you call "answer text")
    - audio_bytes: TTS audio bytes of answer_text
    """

    answer_text: str
    audio_bytes: Optional[bytes] = None
    audio_filename: Optional[str] = None
    audio_mime_type: Optional[str] = None

    def to_api_payload(self) -> dict:
        payload: dict = {"answer_text": self.answer_text}
        if self.audio_bytes is None:
            return payload

        payload["audio"] = {
            "filename": self.audio_filename or "answer.wav",
            "mime_type": self.audio_mime_type or "audio/wav",
            "audio_base64": base64.b64encode(self.audio_bytes).decode("ascii"),
        }
        return payload


def _default_paths() -> dict:
    outputs_dir = (Path(__file__).parent / "outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "text2sql_prompt": str(get_prompt_path("text2sql_prompt.txt")),
        "output2answer_prompt": str(get_prompt_path("output2answer_prompt.txt")),
        "machine_csv": str(get_data_path("Machine_Data.csv")),
        "sensor_csv": str(get_data_path("Sensor_Data.csv")),
        "telemetry_csv": str(get_data_path("Telemetry_Data.csv")),
        "outputs_dir": outputs_dir,
    }


async def run_voice_factory_pipeline(
    vfi: VoiceFactoryInput,
    *,
    include_audio: bool = True,
) -> VoiceFactoryOutput:
    # Keys needed for the pipeline
    ensure_env("OPENAI_API_KEY")
    if include_audio:
        ensure_env("GRADIUM_API_KEY")

    paths = _default_paths()

    # 1) Determine user_query (from text or STT(audio))
    if vfi.text and vfi.text.strip():
        user_query = vfi.text.strip()
    else:
        if not vfi.audio_bytes:
            raise ValueError("Missing input: provide either text or audio.")
        ensure_env("PYANNOTE_API_KEY")

        stt_result = perform_stt_from_local_audio(
            vfi.audio_bytes,
            transcript_level=vfi.transcript_level,
        )
        transcript_text = build_transcript_text(stt_result, vfi.transcript_level).strip()
        if not transcript_text:
            raise ValueError("STT produced no transcript text.")
        user_query = transcript_text

    # 2) Text2SQL
    sql = generate_sql(
        user_query=user_query,
        prompt_path=paths["text2sql_prompt"],
        model="gpt-5.2",
        machine_csv=paths["machine_csv"],
        sensor_csv=paths["sensor_csv"],
        telemetry_csv=paths["telemetry_csv"],
    )

    # 3) Execute
    df: pd.DataFrame = execute_sql(
        sql=sql,
        machine_csv=paths["machine_csv"],
        sensor_csv=paths["sensor_csv"],
        telemetry_csv=paths["telemetry_csv"],
    )

    # 4) Output-to-answer (answer text)
    answer_text = generate_tts_answer(
        user_query=user_query,
        sql_used=sql,
        df=df,
        prompt_path=paths["output2answer_prompt"],
        model="gpt-5.2",
        max_rows_for_model=20,
    ).strip()
    if not answer_text:
        raise ValueError("Answer generation returned empty text.")

    if not include_audio:
        return VoiceFactoryOutput(answer_text=answer_text)

    # 5) TTS (bytes)
    out_id = uuid.uuid4().hex
    wav_path = paths["outputs_dir"] / f"answer_{out_id}.wav"
    tts_res = await tts_from_text(
        answer_text,
        output_path=wav_path,
        voice_id="YTpq7expH9539ERJ",
        output_format="wav",
        model_name="default",
    )

    mime, _ = mimetypes.guess_type(tts_res.output_path.name)
    if not mime:
        mime = "audio/wav"

    return VoiceFactoryOutput(
        answer_text=answer_text,
        audio_bytes=tts_res.audio_bytes,
        audio_filename=tts_res.output_path.name,
        audio_mime_type=mime,
    )


app = FastAPI(title="voice_factory_backend", version="0.1.0")

class VoiceFactoryTextRequest(BaseModel):
    text: str
    transcript_level: TranscriptLevel = "turn"
    include_audio: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/voice-factory")
async def voice_factory(
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    transcript_level: TranscriptLevel = Form(default="turn"),
    include_audio: bool = Form(default=True),
) -> dict:
    """
    Accept either:
      - multipart form field `text`
      - multipart file `audio`

    Returns:
      - answer_text (string)
      - audio (optional): base64 of the TTS audio bytes (set include_audio=false to omit)
    """
    text_stripped = text.strip() if text else None

    # Important: if text is provided, do NOT read/consume audio and do NOT run STT.
    audio_bytes: Optional[bytes] = None
    audio_filename: Optional[str] = None
    if not text_stripped:
        if audio is None:
            raise HTTPException(status_code=400, detail="Missing input: provide either text or audio.")
        audio_bytes = await audio.read()
        audio_filename = audio.filename

    try:
        out = await run_voice_factory_pipeline(
            VoiceFactoryInput(
                text=text_stripped,
                audio_bytes=audio_bytes,
                audio_filename=audio_filename,
                transcript_level=transcript_level,
            ),
            include_audio=include_audio,
        )
        return out.to_api_payload()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Keep response small but actionable; full trace will be in server logs.
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {type(e).__name__}: {e}")


@app.post("/v1/voice-factory/text")
async def voice_factory_text(req: VoiceFactoryTextRequest) -> dict:
    """
    JSON-only variant (useful when you just want to send text).
    """
    try:
        out = await run_voice_factory_pipeline(
            VoiceFactoryInput(
                text=req.text,
                audio_bytes=None,
                audio_filename=None,
                transcript_level=req.transcript_level,
            ),
            include_audio=req.include_audio,
        )
        return out.to_api_payload()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {type(e).__name__}: {e}")
