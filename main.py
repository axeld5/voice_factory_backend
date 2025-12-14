from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pyannote_stt import (
    perform_stt_from_local_audio,
    format_turn_level_transcript,
)
from text2sql import (
    get_data_path,
    get_prompt_path,
    generate_sql,
    execute_sql,
    save_result_csv,
    generate_tts_answer,
)
from gradium_tts import tts_from_text_sync


def build_transcript_text(stt_bundle: dict, transcript_level: str) -> str:
    transcript = (stt_bundle or {}).get("transcript") or {}

    if transcript_level in ("turn", "both"):
        turns = transcript.get("turnLevelTranscription") or []
        text = format_turn_level_transcript(turns).strip()
        if text:
            return text

    if transcript_level in ("word", "both"):
        words = transcript.get("wordLevelTranscription") or []
        text = " ".join((w.get("text") or "").strip() for w in words).strip()
        if text:
            return text

    return ""


def ensure_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise SystemExit(
            f"Missing environment variable {var_name}. "
            f"Set it in your shell or in a .env file."
        )


def main() -> None:
    default_text2sql_prompt = str(get_prompt_path("text2sql_prompt.txt"))
    default_output2answer_prompt = str(get_prompt_path("output2answer_prompt.txt"))

    default_machine_csv = str(get_data_path("Machine_Data.csv"))
    default_sensor_csv = str(get_data_path("Sensor_Data.csv"))
    default_telemetry_csv = str(get_data_path("Telemetry_Data.csv"))

    parser = argparse.ArgumentParser(
        description="Voice -> STT (pyannote) -> Text2SQL (OpenAI) -> Answer (OpenAI) -> TTS (Gradium) -> WAV"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to local audio file (e.g. .m4a)",
    )
    parser.add_argument(
        "--transcript-level",
        choices=["turn", "word", "both"],
        default="turn",
        help="Which transcript level to request from pyannote.",
    )

    # Text2SQL + data
    parser.add_argument("--text2sql-prompt", default=default_text2sql_prompt)
    parser.add_argument("--machine", default=default_machine_csv)
    parser.add_argument("--sensor", default=default_sensor_csv)
    parser.add_argument("--telemetry", default=default_telemetry_csv)
    parser.add_argument("--text2sql-model", default="gpt-5.2")

    # Output-to-answer
    parser.add_argument("--output2answer-prompt", default=default_output2answer_prompt)
    parser.add_argument("--output2answer-model", default="gpt-5.2")
    parser.add_argument(
        "--output2answer-max-rows",
        type=int,
        default=20,
        help="Max SQL result rows to include in the answer prompt.",
    )

    # Outputs
    parser.add_argument("--result-out-csv", default="result.csv")
    parser.add_argument("--wav-out", default="final_answer.wav")

    # TTS
    parser.add_argument("--voice-id", default="YTpq7expH9539ERJ")
    parser.add_argument("--tts-model-name", default="default")
    parser.add_argument("--tts-output-format", default="wav")

    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    # Keys needed for the pipeline
    ensure_env("PYANNOTE_API_KEY")
    ensure_env("OPENAI_API_KEY")

    print(f"[1/5] STT via pyannote from local audio: {audio_path.name}")
    stt_result = perform_stt_from_local_audio(
        audio_path,
        transcript_level=args.transcript_level,
    )

    transcript_text = build_transcript_text(stt_result, args.transcript_level)
    if not transcript_text:
        raise SystemExit("STT produced no transcript text.")

    # For Text2SQL we generally want *just* the spoken content.
    user_query = transcript_text.strip()
    print("\n--- Transcript (used as user query) ---")
    print(user_query)

    print("\n[2/5] Generate SQL (Text2SQL)")
    sql = generate_sql(
        user_query=user_query,
        prompt_path=args.text2sql_prompt,
        model=args.text2sql_model,
        machine_csv=args.machine,
        sensor_csv=args.sensor,
        telemetry_csv=args.telemetry,
    )
    print("\n--- Generated SQL ---")
    print(sql)

    print("\n[3/5] Execute SQL in DuckDB against CSVs")
    df: pd.DataFrame = execute_sql(
        sql=sql,
        machine_csv=args.machine,
        sensor_csv=args.sensor,
        telemetry_csv=args.telemetry,
    )

    saved_csv = save_result_csv(df, args.result_out_csv)
    print(f"\n--- Saved SQL result CSV ---\n{saved_csv}")
    print(f"\n--- Result preview (rows={len(df)}, cols={len(df.columns)}) ---")
    print(df.head(20).to_string(index=False) if not df.empty else "(empty)")

    print("\n[4/5] Generate spoken answer from SQL result (output2answer)")
    answer_text = generate_tts_answer(
        user_query=user_query,
        sql_used=sql,
        df=df,
        prompt_path=args.output2answer_prompt,
        model=args.output2answer_model,
        max_rows_for_model=args.output2answer_max_rows,
    )
    print("\n--- Answer text ---")
    print(answer_text)

    print("\n[5/5] Gradium TTS -> WAV")
    tts_res = tts_from_text_sync(
        answer_text,
        output_path=args.wav_out,
        voice_id=args.voice_id,
        output_format=args.tts_output_format,
        model_name=args.tts_model_name,
    )

    print("\n=== Done ===")
    print("WAV written:", str(tts_res.output_path))
    print("Sample rate:", tts_res.sample_rate)
    print("TTS request id:", tts_res.request_id)


if __name__ == "__main__":
    main()
