from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()


# ---------- 1) Load Text2SQL prompt from file ----------
def load_prompt(prompt_path: str) -> str:
    prompt_file = Path(prompt_path).expanduser().resolve()
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


# ---------- 2) Load CSVs into DuckDB ----------
def connect_and_register(machine_csv: str, sensor_csv: str, telemetry_csv: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")

    machine_path = str(Path(machine_csv).expanduser().resolve())
    sensor_path = str(Path(sensor_csv).expanduser().resolve())
    telemetry_path = str(Path(telemetry_csv).expanduser().resolve())

    con.execute(
        f"""
        CREATE OR REPLACE VIEW Machine_Data AS
        SELECT * FROM read_csv_auto('{machine_path}', header=True, all_varchar=False);
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW Sensor_Data AS
        SELECT * FROM read_csv_auto('{sensor_path}', header=True, all_varchar=False);
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW Telemetry_Data AS
        SELECT * FROM read_csv_auto('{telemetry_path}', header=True, all_varchar=False);
        """
    )

    return con


# ---------- 3) Call OpenAI to generate SQL ----------
def generate_sql(user_query: str, prompt_path: str, model: str = "gpt-5-nano") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = load_prompt(prompt_path)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {user_query}"},
        ],
    )

    sql = (response.output_text or "").strip()

    # Defensive cleanup (in case the model adds formatting)
    if sql.startswith("```"):
        sql = sql.strip("`").strip()
        if sql.lower().startswith("sql\n"):
            sql = sql[4:].strip()

    return sql


# ---------- 4) Execute SQL ----------
def execute_sql(
    sql: str,
    machine_csv: str,
    sensor_csv: str,
    telemetry_csv: str,
) -> pd.DataFrame:
    con = connect_and_register(machine_csv, sensor_csv, telemetry_csv)
    try:
        return con.execute(sql).df()
    except Exception as e:
        raise SystemExit(f"\nSQL execution failed:\n{e}\n\nSQL was:\n{sql}")


# ---------- 5) Save result for output2answer.py compatibility ----------
def save_result_csv(df: pd.DataFrame, out_path: str) -> str:
    p = Path(out_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)


# ---------- 6) Optional: generate TTS answer (same logic as output2answer.py) ----------
def load_output2answer_prompt(prompt_path: str) -> str:
    p = Path(prompt_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"output2answer prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def df_to_compact_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None:
        return "NO_RESULT"
    if df.empty:
        return "EMPTY_RESULT"
    return df.head(max_rows).to_string(index=False)


def render_output2answer_prompt(template: str, user_query: str, sql_used: str, sql_result: str) -> str:
    return (
        template.replace("{{USER_QUERY}}", user_query)
        .replace("{{SQL_USED}}", sql_used)
        .replace("{{SQL_RESULT}}", sql_result)
    )


def generate_tts_answer(
    user_query: str,
    sql_used: str,
    df: pd.DataFrame,
    prompt_path: str,
    model: str = "gpt-5-nano",
    max_rows_for_model: int = 20,
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    template = load_output2answer_prompt(prompt_path)
    sql_result_text = df_to_compact_text(df, max_rows=max_rows_for_model)
    full_prompt = render_output2answer_prompt(template, user_query, sql_used, sql_result_text)

    response = client.responses.create(
        model=model,
        input=full_prompt,
    )
    return (response.output_text or "").strip()


# ---------- 7) Pipeline ----------
def run_pipeline(
    user_query: str,
    text2sql_prompt_path: str,
    machine_csv: str,
    sensor_csv: str,
    telemetry_csv: str,
    model: str,
    result_out_csv: str,
    max_print_rows: int,
    # output2answer compatibility options
    also_generate_tts: bool,
    output2answer_prompt_path: Optional[str],
    output2answer_model: Optional[str],
    output2answer_max_rows: int,
) -> None:
    sql = generate_sql(user_query=user_query, prompt_path=text2sql_prompt_path, model=model)

    print("\n--- Generated SQL ---")
    print(sql)

    df = execute_sql(sql=sql, machine_csv=machine_csv, sensor_csv=sensor_csv, telemetry_csv=telemetry_csv)

    # Always save CSV so output2answer.py can consume it
    saved_path = save_result_csv(df, result_out_csv)
    print("\n--- Saved Result CSV ---")
    print(saved_path)

    # Keep original result preview behavior
    print(f"\n--- Results Preview (rows={len(df)}, cols={len(df.columns)}) ---")
    if len(df) > max_print_rows:
        print(df.head(max_print_rows).to_string(index=False))
        print(f"\n... (showing first {max_print_rows} rows)")
    else:
        print(df.to_string(index=False))

    # Optional: directly generate the spoken answer, using output2answer.txt
    if also_generate_tts:
        if not output2answer_prompt_path:
            raise SystemExit("--also-generate-tts requires --output2answer-prompt")
        tts_model = output2answer_model or model
        answer = generate_tts_answer(
            user_query=user_query,
            sql_used=sql,
            df=df,
            prompt_path=output2answer_prompt_path,
            model=tts_model,
            max_rows_for_model=output2answer_max_rows,
        )
        print("\n--- TTS Answer ---")
        print(answer)

    # Print a ready-to-run command for output2answer.py
    print("\n--- Next step (run output2answer.py) ---")
    print(
        "python output2answer.py "
        f"--prompt output2answer.txt "
        f'--user-query "{user_query.replace('"', r'\"')}" '
        f'--sql-used "{sql.replace('"', r'\"')}" '
        f"--result-csv {saved_path}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Text2SQL over CSVs using OpenAI + DuckDB (and save CSV for output2answer.py).")
    parser.add_argument("--prompt", required=True, help="Path to text2sql_prompt.txt")
    parser.add_argument("--machine", required=True, help="Path to Machine_Data.csv")
    parser.add_argument("--sensor", required=True, help="Path to Sensor_Data.csv")
    parser.add_argument("--telemetry", required=True, help="Path to Telemetry_Data.csv")
    parser.add_argument("--query", required=True, help="User natural-language query")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model for Text2SQL")
    parser.add_argument("--max-print-rows", type=int, default=50, help="Max rows to preview in terminal")

    # Compatibility: save results for output2answer.py
    parser.add_argument(
        "--result-out-csv",
        default="result.csv",
        help="Where to save the SQL result as CSV (consumed by output2answer.py). Default: result.csv",
    )

    # Optional: do output2answer step inside this script too
    parser.add_argument(
        "--also-generate-tts",
        action="store_true",
        help="If set, also generate a short TTS-friendly answer using output2answer prompt.",
    )
    parser.add_argument("--output2answer-prompt", default=None, help="Path to output2answer.txt (required if --also-generate-tts)")
    parser.add_argument("--output2answer-model", default=None, help="Model for output-to-answer step (defaults to --model)")
    parser.add_argument("--output2answer-max-rows", type=int, default=20, help="Max result rows to pass into output-to-answer step")

    args = parser.parse_args()

    run_pipeline(
        user_query=args.query,
        text2sql_prompt_path=args.prompt,
        machine_csv=args.machine,
        sensor_csv=args.sensor,
        telemetry_csv=args.telemetry,
        model=args.model,
        result_out_csv=args.result_out_csv,
        max_print_rows=args.max_print_rows,
        also_generate_tts=args.also_generate_tts,
        output2answer_prompt_path=args.output2answer_prompt,
        output2answer_model=args.output2answer_model,
        output2answer_max_rows=args.output2answer_max_rows,
    )


if __name__ == "__main__":
    main()
