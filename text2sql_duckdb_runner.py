from __future__ import annotations

import argparse
import duckdb
from pathlib import Path
from typing import Optional


def connect_and_register(
    machine_csv: str,
    sensor_csv: str,
    telemetry_csv: str,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Creates/uses a DuckDB connection and registers the three CSVs as tables:
    Machine_Data, Sensor_Data, Telemetry_Data.
    """
    con = con or duckdb.connect(database=":memory:")

    machine_path = str(Path(machine_csv).expanduser().resolve())
    sensor_path = str(Path(sensor_csv).expanduser().resolve())
    telemetry_path = str(Path(telemetry_csv).expanduser().resolve())

    # Read CSVs as relations and create views with stable names
    # all_varchar=False allows DuckDB to infer types where possible.
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


def run_sql(
    sql: str,
    machine_csv: str,
    sensor_csv: str,
    telemetry_csv: str,
    limit_print: int = 50,
) -> None:
    con = connect_and_register(machine_csv, sensor_csv, telemetry_csv)

    try:
        df = con.execute(sql).df()
    except Exception as e:
        raise SystemExit(f"SQL execution failed:\n{e}\n\nSQL was:\n{sql}")

    # Print shape + head to keep terminal readable
    print(f"Rows: {len(df)}  Cols: {len(df.columns)}")
    if len(df) > limit_print:
        print(df.head(limit_print).to_string(index=False))
        print(f"\n... (showing first {limit_print} rows)")
    else:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DuckDB SQL against Machine/Sensor/Telemetry CSVs.")
    parser.add_argument("--machine", required=True, help="Path to Machine_Data.csv")
    parser.add_argument("--sensor", required=True, help="Path to Sensor_Data.csv")
    parser.add_argument("--telemetry", required=True, help="Path to Telemetry_Data.csv")
    parser.add_argument("--sql", required=True, help="SQL query string to run (DuckDB dialect).")
    parser.add_argument("--limit-print", type=int, default=50, help="Max rows to print.")
    args = parser.parse_args()

    run_sql(
        sql=args.sql,
        machine_csv=args.machine,
        sensor_csv=args.sensor,
        telemetry_csv=args.telemetry,
        limit_print=args.limit_print,
    )


if __name__ == "__main__":
    main()
