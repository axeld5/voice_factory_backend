from __future__ import annotations

import argparse
import json
import io
from pathlib import Path
from typing import Literal, Optional

import duckdb
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

import os
from dotenv import load_dotenv
load_dotenv()


# ---------- Helper: Get project root (where this script is located) ----------
def get_project_root() -> Path:
    """Returns the directory containing this script (project root)."""
    return Path(__file__).parent.resolve()


def get_outputs_dir() -> Path:
    """Returns path to outputs/ folder (created by writers as needed)."""
    return get_project_root() / "outputs"


# ---------- Hardcoded paths ----------
def get_data_path(filename: str) -> Path:
    """Returns path to a data file in the data/ folder."""
    return get_project_root() / "data" / filename


def get_prompt_path(filename: str) -> Path:
    """Returns path to a prompt file in the prompts/ folder."""
    return get_project_root() / "prompts" / filename


# ---------- 1) Load Text2SQL prompt from file ----------
def load_prompt(prompt_path: str) -> str:
    prompt_file = Path(prompt_path).expanduser().resolve()
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def _escape_markdown_cell(value: object, *, max_len: int = 80) -> str:
    """Escape markdown table cells (keep it compact + safe)."""
    s = "" if value is None else str(value)
    s = s.replace("\n", " ").replace("|", "\\|").strip()
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table (no tabulate dependency)."""
    if df is None or df.empty:
        return "_(empty)_"

    cols = [str(c) for c in df.columns.tolist()]
    header = "| " + " | ".join(_escape_markdown_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_escape_markdown_cell(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def load_csv_head_markdown(csv_path: str, *, n: int = 5) -> str:
    """Read only the first n rows of a CSV and render as markdown."""
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        return f"_(missing file: {p})_"
    try:
        df = pd.read_csv(p, nrows=n)
    except Exception as e:
        return f"_(failed to read: {p.name}: {e})_"
    return df_to_markdown_table(df)


def render_text2sql_prompt(
    template: str,
    *,
    user_query: str,
    machine_head_md: str,
    sensor_head_md: str,
    telemetry_head_md: str,
) -> str:
    return (
        template.replace("{{USER_QUERY}}", user_query)
        .replace("{{MACHINE_DATA_HEAD_MD}}", machine_head_md)
        .replace("{{SENSOR_DATA_HEAD_MD}}", sensor_head_md)
        .replace("{{TELEMETRY_DATA_HEAD_MD}}", telemetry_head_md)
    )


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
def generate_sql(
    user_query: str,
    prompt_path: str,
    model: str = "gpt-5.2",
    *,
    machine_csv: Optional[str] = None,
    sensor_csv: Optional[str] = None,
    telemetry_csv: Optional[str] = None,
    data_head_rows: int = 5,
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    template = load_prompt(prompt_path)

    # Optional: inject markdown head() previews of each CSV into the system prompt.
    # This helps the model map the schema to real values (zones, ids, sensor types, etc.).
    machine_head_md = load_csv_head_markdown(machine_csv, n=data_head_rows) if machine_csv else "_(not provided)_"
    sensor_head_md = load_csv_head_markdown(sensor_csv, n=data_head_rows) if sensor_csv else "_(not provided)_"
    telemetry_head_md = load_csv_head_markdown(telemetry_csv, n=data_head_rows) if telemetry_csv else "_(not provided)_"

    system_prompt = render_text2sql_prompt(
        template,
        user_query=user_query,
        machine_head_md=machine_head_md,
        sensor_head_md=sensor_head_md,
        telemetry_head_md=telemetry_head_md,
    )

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


class Output2AnswerStructured(BaseModel):
    """
    Structured output for the output2answer step.

    summary:
      - TTS-friendly answer
      - If listing multiple items, use newline-separated lines
    viz_kind:
      - "text_image" for single-line summaries
      - otherwise one of bar/line/scatter/heatmap/table
    viz_x/viz_y:
      - optional column names from the SQL result to plot
    viz_title:
      - optional title
    """

    summary: str
    viz_kind: Literal["text_image", "bar", "line", "scatter", "heatmap", "table"]
    # For viz_kind="text_image": minimal label-free on-screen string (value + optional timestamp)
    text_box: Optional[str] = None
    viz_x: Optional[str] = None
    viz_y: Optional[str] = None
    viz_title: Optional[str] = None


class AnswerWithVisualizationPlan(BaseModel):
    summary: str
    viz_kind: Literal["text_image", "bar", "line", "scatter", "heatmap", "table"]
    text_box: Optional[str] = None
    viz_x: Optional[str] = None
    viz_y: Optional[str] = None
    viz_title: Optional[str] = None


def _coerce_datetime_if_possible(s: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=False)
        if dt.notna().mean() < 0.6:
            return s
        return dt
    except Exception:
        return s


def _pick_xy(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], str]:
    """Heuristic fallback when plan columns are missing."""
    if df is None or df.empty or len(df.columns) == 0:
        return None, None, "table"

    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == "object":
            df2[c] = _coerce_datetime_if_possible(df2[c])

    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    datetime_cols = [c for c in df2.columns if pd.api.types.is_datetime64_any_dtype(df2[c])]
    other_cols = [c for c in df2.columns if c not in numeric_cols]

    if datetime_cols and numeric_cols:
        return datetime_cols[0], numeric_cols[0], "line"
    if other_cols and numeric_cols:
        return other_cols[0], numeric_cols[0], "bar"
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1], "scatter"
    return None, None, "table"


def render_visualization_png_bytes(
    *,
    answer_summary: str,
    df: pd.DataFrame,
    plan: Output2AnswerStructured,
    dpi: int = 200,
) -> bytes:
    """
    Render the visualization to PNG bytes.

    Enforced rules:
    - single-line summary => black text on white image
    - multi-line summary => plotly visualization (PNG via kaleido)
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    summary = (answer_summary or "").strip()
    viz_kind = plan.viz_kind

    plan_x = plan.viz_x
    plan_y = plan.viz_y
    plan_title = plan.viz_title

    def _to_png(fig: go.Figure) -> bytes:
        # `scale` acts like DPI. 2 gives crisp output without huge images.
        return pio.to_image(fig, format="png", scale=max(1, int(round(dpi / 100))))

    def _text_card(text: str, title: Optional[str] = None) -> bytes:
        text = " ".join((text or "").splitlines()).strip()
        fig = go.Figure()
        fig.add_annotation(
            text=text or " ",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=22, color="#111827"),
            align="center",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            title=title or None,
            margin=dict(l=40, r=40, t=60 if title else 40, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=max(700, min(1400, 14 * max(len(text), 40))),
            height=260,
        )
        return _to_png(fig)

    if viz_kind == "text_image":
        return _text_card(((plan.text_box or "") or summary).strip(), title=plan_title)

    if df is None or df.empty:
        return _text_card("No data to visualize.", title=plan_title or "Result")

    df_plot = df.copy()
    for c in df_plot.columns:
        if df_plot[c].dtype == "object":
            df_plot[c] = _coerce_datetime_if_possible(df_plot[c])

    x_col = plan_x if (plan_x in df_plot.columns) else None
    y_col = plan_y if (plan_y in df_plot.columns) else None
    kind = viz_kind

    if kind in ("bar", "line", "scatter"):
        if x_col is None or y_col is None:
            x_col, y_col, kind = _pick_xy(df_plot)

    title = plan_title or None

    if kind == "heatmap":
        num = df_plot.select_dtypes(include="number")
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                color_continuous_scale="Viridis",
                title=title or "Correlation heatmap",
                aspect="auto",
            )
            fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="white")
            return _to_png(fig)
        x_col, y_col, kind = _pick_xy(df_plot)

    if kind == "bar" and x_col and y_col:
        df_small = df_plot.copy()
        try:
            if df_small[x_col].nunique(dropna=False) > 30:
                df_small = df_small.head(30)
        except Exception:
            df_small = df_small.head(30)
        fig = px.bar(df_small, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="white", plot_bgcolor="white")
        fig.update_xaxes(tickangle=25)
        return _to_png(fig)

    if kind == "line" and x_col and y_col:
        df_small = df_plot.sort_values(by=x_col) if x_col in df_plot.columns else df_plot
        fig = px.line(df_small, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}")
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="white", plot_bgcolor="white")
        return _to_png(fig)

    if kind == "scatter" and x_col and y_col:
        fig = px.scatter(df_plot, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}")
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="white", plot_bgcolor="white")
        return _to_png(fig)

    # Table fallback
    df_show = df_plot.head(15).copy()
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[str(c) for c in df_show.columns],
                    fill_color="#0f172a",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[df_show[c].astype(str).tolist() for c in df_show.columns],
                    fill_color="#ffffff",
                    font=dict(color="#111827", size=11),
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title=title or "Result table",
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        width=1100,
        height=max(300, 80 + 24 * len(df_show)),
    )
    return _to_png(fig)


def generate_tts_answer(
    user_query: str,
    sql_used: str,
    df: pd.DataFrame,
    prompt_path: str,
    model: str = "gpt-5.2",
    max_rows_for_model: int = 20,
) -> AnswerWithVisualizationPlan:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    template = load_output2answer_prompt(prompt_path)
    sql_result_text = df_to_compact_text(df, max_rows=max_rows_for_model)
    full_prompt = render_output2answer_prompt(template, user_query, sql_used, sql_result_text)

    # Prefer Structured Outputs parsing; fall back to JSON parsing if needed.
    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": full_prompt},
            ],
            text_format=Output2AnswerStructured,
        )
        plan: Output2AnswerStructured = response.output_parsed
    except Exception:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": full_prompt
                    + "\n\nIf structured outputs are unavailable, return a JSON object with keys:"
                    + " summary, viz_kind, text_box, viz_x, viz_y, viz_title (and nothing else).",
                }
            ],
        )
        raw = (response.output_text or "").strip()
        # Try to parse as JSON
        plan = Output2AnswerStructured.model_validate(json.loads(raw))

    return AnswerWithVisualizationPlan(
        summary=plan.summary.strip(),
        viz_kind=plan.viz_kind,
        text_box=(plan.text_box.strip() if plan.text_box else None),
        viz_x=plan.viz_x,
        viz_y=plan.viz_y,
        viz_title=plan.viz_title,
    )


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
    sql = generate_sql(
        user_query=user_query,
        prompt_path=text2sql_prompt_path,
        model=model,
        machine_csv=machine_csv,
        sensor_csv=sensor_csv,
        telemetry_csv=telemetry_csv,
    )

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
        print("\n--- Answer Summary (TTS) ---")
        print(answer.summary)
        viz_png = render_visualization_png_bytes(answer_summary=answer.summary, df=df, plan=answer)
        viz_out = get_outputs_dir() / "visualization.png"
        viz_out.parent.mkdir(parents=True, exist_ok=True)
        viz_out.write_bytes(viz_png)
        print("\n--- Visualization Image ---")
        print(str(viz_out))

    # Print a ready-to-run command for output2answer.py
    print("\n--- Next step (run output2answer.py) ---")
    escaped_user_query = user_query.replace('"', '\\"')
    escaped_sql = sql.replace('"', '\\"')
    print(
        "python output2answer.py "
        f"--prompt output2answer.txt "
        f'--user-query "{escaped_user_query}" '
        f'--sql-used "{escaped_sql}" '
        f"--result-csv {saved_path}"
    )

def main() -> None:
    # Hardcoded paths - data is always in data/ folder, prompts in prompts/ folder
    default_text2sql_prompt = str(get_prompt_path("text2sql_prompt.txt"))
    default_output2answer_prompt = str(get_prompt_path("output2answer_prompt.txt"))
    default_machine_csv = str(get_data_path("Machine_Data.csv"))
    default_sensor_csv = str(get_data_path("Sensor_Data.csv"))
    default_telemetry_csv = str(get_data_path("Telemetry_Data.csv"))
    default_result_csv = str(get_outputs_dir() / "result.csv")

    parser = argparse.ArgumentParser(description="Text2SQL over CSVs using OpenAI + DuckDB (and save CSV for output2answer.py).")
    parser.add_argument("--prompt", default=default_text2sql_prompt, help=f"Path to text2sql_prompt.txt (default: {default_text2sql_prompt})")
    parser.add_argument("--machine", default=default_machine_csv, help=f"Path to Machine_Data.csv (default: {default_machine_csv})")
    parser.add_argument("--sensor", default=default_sensor_csv, help=f"Path to Sensor_Data.csv (default: {default_sensor_csv})")
    parser.add_argument("--telemetry", default=default_telemetry_csv, help=f"Path to Telemetry_Data.csv (default: {default_telemetry_csv})")
    parser.add_argument("--query", required=True, help="User natural-language query")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for Text2SQL")
    parser.add_argument("--max-print-rows", type=int, default=50, help="Max rows to preview in terminal")

    # Compatibility: save results for output2answer.py
    parser.add_argument(
        "--result-out-csv",
        default=default_result_csv,
        help="Where to save the SQL result as CSV (consumed by output2answer.py). Default: outputs/result.csv",
    )

    # Optional: do output2answer step inside this script too
    parser.add_argument(
        "--also-generate-tts",
        action="store_true",
        default=True,
        help="If set, also generate a short TTS-friendly answer using output2answer prompt.",
    )
    parser.add_argument("--output2answer-prompt", default=default_output2answer_prompt, help=f"Path to output2answer_prompt.txt (default: {default_output2answer_prompt})")
    parser.add_argument("--output2answer-model", default="gpt-5.2", help="Model for output-to-answer step (defaults to --model)")
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
