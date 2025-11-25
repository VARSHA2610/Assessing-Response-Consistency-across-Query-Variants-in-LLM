import os
import time
import pandas as pd
from openai import OpenAI
import duckdb

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input Excel file
input_xlsx = "questions.xlsx"
df_questions = pd.read_excel(input_xlsx)

def ask_chatgpt(question_text: str, user_id: str) -> str:
    """Send question to GPT and return the model's answer."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question_text}],
            user=user_id,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

rows = []

for idx, row in df_questions.iterrows():
    base_q = row["Base question"]

    # All other columns are variant categories
    for col in df_questions.columns:
        if col == "Base question":
            continue

        variant_category = col
        variant_question = row[col]

        if pd.isna(variant_question) or str(variant_question).strip() == "":
            continue

        # Unique per-variant user ID
        user_id = f"user_row{idx}_{variant_category.replace(' ', '_').lower()}"

        answer = ask_chatgpt(str(variant_question), user_id=user_id)

        rows.append({
            "base_question": base_q,
            "variant_category": variant_category,
            "variant_question": variant_question,
            "response": answer
        })

        time.sleep(0.2)

# Convert results to DataFrame
df_output = pd.DataFrame(rows)

# ============================
# SAVE TO CSV
# ============================
csv_path = "questions_variant_responses.csv"
df_output.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

# ============================
# SAVE TO DUCKDB
# ============================
db_path = "llm_variants.duckdb"
table_name = "responses"

con = duckdb.connect(db_path)

# Create table if missing
con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        base_question TEXT,
        variant_category TEXT,
        variant_question TEXT,
        response TEXT
    );
""")

# Insert new rows
con.register("df_output", df_output)
con.execute(f"INSERT INTO {table_name} SELECT * FROM df_output;")

con.close()

print(f"Inserted {len(df_output)} rows into DuckDB table '{table_name}' in '{db_path}'.")
