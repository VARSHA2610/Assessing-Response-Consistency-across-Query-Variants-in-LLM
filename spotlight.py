#!/usr/bin/env python3

import duckdb
import pandas as pd
from renumics import spotlight


def load_duckdb_table(db_path: str, table_name: str = "responses"):
    """
    Load full table from DuckDB into a pandas DataFrame.
    You can modify the SELECT query if you want filtering.
    """
    con = duckdb.connect(db_path, read_only=True)
    query = f"SELECT * FROM {table_name}"
    df = con.execute(query).df()
    con.close()
    return df


def main():
    DB_PATH = "llm_variants_responses.duckdb"
    TABLE_NAME = "responses"

    df = load_duckdb_table(DB_PATH, TABLE_NAME)

    print(f"Loaded {len(df)} rows from {TABLE_NAME} in {DB_PATH}")
    print("Launching Renumics Spotlight...")

    # Optionally specify special column types:
    # dtype={"embedding_vector": spotlight.Embedding}
    spotlight.show(df)


if __name__ == "__main__":
    main()
