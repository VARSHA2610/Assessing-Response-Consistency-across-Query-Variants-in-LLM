#!/usr/bin/env python3
import os
import time
import pandas as pd
import duckdb
from openai import OpenAI

# ==========================
# CONFIG
# ==========================

INPUT_FILE = "Dataset(set1).csv"          # your CSV from OneDrive
OUTPUT_CSV = "questions_variant_responses.csv"
DB_PATH = "llm_variants_responses.duckdb"
TABLE_NAME = "responses"

# Set to a small integer while testing (e.g., 3). Use None for full run.
MAX_ROWS = None

# Small delay between calls to be gentle on the API
SLEEP_SECONDS = 0.2

# OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================
# OpenAI helper
# ==========================

def ask_chatgpt(question_text: str, user_id: str) -> str:
    """
    Send a single variant question to GPT and return the answer text.
    Returns an 'ERROR: ...' string if the call fails.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question_text}],
            user=user_id,
            max_tokens=300,
            temperature=0.7,
            timeout=30,  # seconds
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ==========================
# Main pipeline
# ==========================

def main():
    # 1. Read CSV (strip BOM, clean column names)
    df_questions = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df_questions.columns = [
        str(c).strip().lstrip("\ufeff") for c in df_questions.columns
    ]

    print("Columns:", df_questions.columns)

    # First column is the base question, rest are variants
    base_col = df_questions.columns[0]
    variant_cols = list(df_questions.columns[1:])

    # Optionally limit rows for testing
    if MAX_ROWS is not None:
        df_questions = df_questions.head(MAX_ROWS)

    # Count how many non-empty variants we will process
    total_variants = 0
    for _, row in df_questions.iterrows():
        base_q = row[base_col]
        if not isinstance(base_q, str) or not base_q.strip():
            continue
        for col in variant_cols:
            val = row[col]
            if isinstance(val, str) and val.strip():
                total_variants += 1

    print(f"Total variants to process: {total_variants}")

    rows_out = []
    processed = 0

    # 2. Loop over each row and each variant
    for row_idx, row in df_questions.iterrows():
        base_q = row[base_col]

        if not isinstance(base_q, str) or not base_q.strip():
            continue

        for col in variant_cols:
            variant_question = row[col]

            # Skip empty cells
            if not isinstance(variant_question, str) or not variant_question.strip():
                continue

            processed += 1
            print(
                f"[{processed}/{total_variants}] row {row_idx + 1} | "
                f"{col} ...",
                flush=True,
            )

            user_id = f"user_row{row_idx}_{col.replace(' ', '_')}"
            answer = ask_chatgpt(variant_question, user_id=user_id)

            rows_out.append(
                {
                    "base_question": base_q,
                    "variant_category": col,
                    "variant_question": variant_question,
                    "variant_response": answer,
                }
            )

            time.sleep(SLEEP_SECONDS)

    # Convert to DataFrame
    df_output = pd.DataFrame(rows_out)

    # ==========================
    # 3. SAVE TO CSV
    # ==========================
    df_output.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV} ({len(df_output)} rows)")

    # ==========================
    # 4. SAVE TO DUCKDB
    # ==========================
    con = duckdb.connect(DB_PATH)

    # Ensure table exists with correct schema
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            base_question TEXT,
            variant_category TEXT,
            variant_question TEXT,
            variant_response TEXT
        );
        """
    )

    # Register DataFrame and insert
    con.register("df_output", df_output)
    con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df_output;")
    con.close()

    print(
        f"Inserted {len(df_output)} rows into DuckDB table "
        f"'{TABLE_NAME}' in '{DB_PATH}'."
    )


if __name__ == "__main__":
    main()
