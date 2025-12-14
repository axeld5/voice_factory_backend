from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    render_visualization_png_bytes,
)

load_dotenv()

TranscriptLevel = Literal["turn", "word", "both"]

logger = logging.getLogger("voice_factory_backend")
_log_level = (os.getenv("LOG_LEVEL") or "INFO").upper()
if _log_level in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
    logger.setLevel(getattr(logging, _log_level))

_log_text_content = (os.getenv("VOICE_FACTORY_LOG_TEXT") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

# NOTE: mock/forced test-audio mode removed. STT now always uses user-provided audio.


def _preview(s: Optional[str], *, max_len: int = 240) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


@dataclass(frozen=True)
class _Step:
    name: str
    t0: float

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


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
    - visualization_bytes: PNG bytes of the visualization
    """

    # The final natural-language question used by the pipeline (typed text or STT transcript)
    question_text: str
    # The final answer (natural language)
    answer_text: str
    visualization_bytes: Optional[bytes] = None
    visualization_filename: Optional[str] = None
    visualization_mime_type: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_filename: Optional[str] = None
    audio_mime_type: Optional[str] = None

    def to_api_payload(self) -> dict:
        payload: dict = {"answer_text": self.answer_text}
        payload["question_text"] = self.question_text
        if self.visualization_bytes is not None:
            payload["visualization"] = {
                "filename": self.visualization_filename or "visualization.png",
                "mime_type": self.visualization_mime_type or "image/png",
                "image_base64": base64.b64encode(self.visualization_bytes).decode("ascii"),
            }
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
    request_id: Optional[str] = None,
) -> VoiceFactoryOutput:
    rid = request_id or "-"
    t_start = time.perf_counter()

    # Keys needed for the pipeline
    ensure_env("OPENAI_API_KEY")
    if include_audio:
        ensure_env("GRADIUM_API_KEY")

    paths = _default_paths()

    # Pipeline is now decoupled from STT: it expects text only.
    user_query = (vfi.text or "").strip()
    if not user_query:
        raise ValueError("Missing input: provide text.")
    logger.info(
        "[%s] input=text include_audio=%s text_len=%d text_preview=%r",
        rid,
        include_audio,
        len(user_query),
        _preview(user_query) if _log_text_content else "",
    )

    # 2) Text2SQL
    s = _Step("text2sql", time.perf_counter())
    sql = generate_sql(
        user_query=user_query,
        prompt_path=paths["text2sql_prompt"],
        model="gpt-5.2",
        machine_csv=paths["machine_csv"],
        sensor_csv=paths["sensor_csv"],
        telemetry_csv=paths["telemetry_csv"],
    )
    logger.info(
        "[%s] text2sql_done ms=%.1f sql_len=%d sql_preview=%r",
        rid,
        s.ms(),
        len(sql or ""),
        _preview(sql) if _log_text_content else "",
    )

    # 3) Execute
    s = _Step("execute_sql", time.perf_counter())
    df: pd.DataFrame = execute_sql(
        sql=sql,
        machine_csv=paths["machine_csv"],
        sensor_csv=paths["sensor_csv"],
        telemetry_csv=paths["telemetry_csv"],
    )
    logger.info(
        "[%s] execute_sql_done ms=%.1f df_shape=%s df_cols=%d",
        rid,
        s.ms(),
        getattr(df, "shape", None),
        len(getattr(df, "columns", [])),
    )

    # 4) Output-to-answer (answer text)
    s = _Step("answer", time.perf_counter())
    ans = generate_tts_answer(
        user_query=user_query,
        sql_used=sql,
        df=df,
        prompt_path=paths["output2answer_prompt"],
        model="gpt-5.2",
        max_rows_for_model=20,
    )
    answer_text = ans.summary.strip()
    logger.info(
        "[%s] answer_done ms=%.1f answer_len=%d answer_preview=%r",
        rid,
        s.ms(),
        len(answer_text),
        _preview(answer_text) if _log_text_content else "",
    )
    if not answer_text:
        raise ValueError("Answer generation returned empty text.")

    # 4b) Visualization
    s = _Step("viz", time.perf_counter())
    out_id = uuid.uuid4().hex
    viz_png = render_visualization_png_bytes(
        answer_summary=answer_text,
        df=df,
        plan=ans,
    )
    viz_path = paths["outputs_dir"] / f"visualization_{out_id}.png"
    viz_path.write_bytes(viz_png)
    logger.info(
        "[%s] viz_done ms=%.1f png_bytes=%d file=%s",
        rid,
        s.ms(),
        len(viz_png),
        viz_path.name,
    )

    if not include_audio:
        logger.info("[%s] pipeline_done include_audio=false total_ms=%.1f", rid, (time.perf_counter() - t_start) * 1000.0)
        return VoiceFactoryOutput(
            question_text=user_query,
            answer_text=answer_text,
            visualization_bytes=viz_png,
            visualization_filename=viz_path.name,
            visualization_mime_type="image/png",
        )

    # 5) TTS (bytes)
    s = _Step("tts", time.perf_counter())
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

    logger.info(
        "[%s] tts_done ms=%.1f audio_bytes=%d file=%s mime=%s total_ms=%.1f",
        rid,
        s.ms(),
        len(tts_res.audio_bytes or b""),
        tts_res.output_path.name,
        mime,
        (time.perf_counter() - t_start) * 1000.0,
    )

    return VoiceFactoryOutput(
        question_text=user_query,
        answer_text=answer_text,
        visualization_bytes=viz_png,
        visualization_filename=viz_path.name,
        visualization_mime_type="image/png",
        audio_bytes=tts_res.audio_bytes,
        audio_filename=tts_res.output_path.name,
        audio_mime_type=mime,
    )


app = FastAPI(title="voice_factory_backend", version="0.1.0")

# CORS: allow the Next.js frontend to call this API directly from the browser.
# Configure with CORS_ALLOW_ORIGINS="http://localhost:3000,https://mydomain.com"
_cors_origins_env = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if _cors_origins_env:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    _cors_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceFactoryTextRequest(BaseModel):
    text: str
    transcript_level: TranscriptLevel = "turn"
    include_audio: bool = True


class VoiceFactoryAnswerRequest(BaseModel):
    text: str
    include_audio: bool = True


class VoiceFactorySTTResponse(BaseModel):
    question_text: str
    transcript_level: TranscriptLevel


@app.get("/health")
def health() -> dict:
    return {"ok": True}


async def run_stt_only(
    *,
    audio_bytes: bytes,
    audio_filename: Optional[str],
    transcript_level: TranscriptLevel,
    request_id: str,
) -> str:
    rid = request_id
    ensure_env("PYANNOTE_API_KEY")

    stt = _Step("stt", time.perf_counter())
    stt_result = perform_stt_from_local_audio(audio_bytes, transcript_level=transcript_level)
    transcript_text = build_transcript_text(stt_result, transcript_level).strip()
    logger.info(
        "[%s] stt_done ms=%.1f transcript_len=%d transcript_preview=%r",
        rid,
        stt.ms(),
        len(transcript_text),
        _preview(transcript_text) if _log_text_content else "",
    )
    if not transcript_text:
        raise HTTPException(status_code=400, detail="STT produced no transcript text.")
    return transcript_text


async def run_answer_only(
    *,
    question_text: str,
    include_audio: bool,
    request_id: str,
) -> VoiceFactoryOutput:
    # Reuse existing pipeline by passing text only (no STT step).
    return await run_voice_factory_pipeline(
        VoiceFactoryInput(text=question_text, audio_bytes=None, audio_filename=None, transcript_level="turn"),
        include_audio=include_audio,
        request_id=request_id,
    )


@app.post("/v1/voice-factory/stt")
async def voice_factory_stt(
    response: Response,
    audio: Optional[UploadFile] = File(default=None),
    transcript_level: TranscriptLevel = Form(default="turn"),
) -> VoiceFactorySTTResponse:
    """
    STT-only endpoint.
    Accepts multipart file `audio` and returns the recognized `question_text`.
    """
    rid = uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = rid
    t0 = time.perf_counter()

    if audio is None:
        logger.warning("[%s] /v1/voice-factory/stt bad_request missing_audio", rid)
        raise HTTPException(status_code=400, detail="Missing input: provide audio.")
    audio_bytes = await audio.read()
    audio_filename = audio.filename

    logger.info(
        "[%s] /v1/voice-factory/stt received audio filename=%r content_type=%r bytes=%d transcript_level=%s",
        rid,
        audio_filename,
        getattr(audio, "content_type", None),
        len(audio_bytes),
        transcript_level,
    )

    try:
        question_text = await run_stt_only(
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            transcript_level=transcript_level,
            request_id=rid,
        )
        logger.info("[%s] /v1/voice-factory/stt ok total_ms=%.1f", rid, (time.perf_counter() - t0) * 1000.0)
        return VoiceFactorySTTResponse(question_text=question_text, transcript_level=transcript_level)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] /v1/voice-factory/stt server_error total_ms=%.1f", rid, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=500, detail=f"STT failed: {type(e).__name__}: {e}")


@app.post("/v1/voice-factory/answer")
async def voice_factory_answer(req: VoiceFactoryAnswerRequest, response: Response) -> dict:
    """
    Answer-only endpoint (Text2SQL + answer + viz + optional TTS).
    Accepts JSON: { text, include_audio }.
    """
    rid = uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = rid
    t0 = time.perf_counter()

    q = (req.text or "").strip()
    if not q:
        logger.warning("[%s] /v1/voice-factory/answer bad_request missing_text", rid)
        raise HTTPException(status_code=400, detail="Missing input: provide text.")

    logger.info(
        "[%s] /v1/voice-factory/answer received len=%d include_audio=%s text_preview=%r",
        rid,
        len(q),
        req.include_audio,
        _preview(q) if _log_text_content else "",
    )

    try:
        out = await run_answer_only(question_text=q, include_audio=req.include_audio, request_id=rid)
        logger.info("[%s] /v1/voice-factory/answer ok total_ms=%.1f", rid, (time.perf_counter() - t0) * 1000.0)
        return out.to_api_payload()
    except ValueError as e:
        logger.warning("[%s] /v1/voice-factory/answer client_error=%s total_ms=%.1f", rid, str(e), (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[%s] /v1/voice-factory/answer server_error total_ms=%.1f", rid, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {type(e).__name__}: {e}")
