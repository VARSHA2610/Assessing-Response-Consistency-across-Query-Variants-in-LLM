import os
import time
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


questions_with_variations = [
    {
        "qid": 1,
        "base":"What are different kinds of cybersecurity attacks",
        "variations": [
            "Which top security threats pose the greatest risk to organizations today?",
            "Can you list and define the primary categories of malicious network activity?",
            "What are the most frequent online scams and hacks that regular users face?",
            "What are the most prevalent attack vectors used by threat actors currently?",
            "What are the trending cybersecurity vulnerabilities dominating the landscape right now?"
        ]
    }
]

def ask_chatgpt(question_text: str, user_id: str) -> str:
    """
    Send a question to the Chat Completions API and return the response text.
    `user_id` is used to tag the call (simulates different users in a ToS-friendly way).
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": question_text}
        ],
        # This `user` field is for your own tracking / abuse monitoring,
        # NOT an actual account switch – that’s the ToS-safe way to differentiate users.
        user=user_id,
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

rows = []

for q in questions_with_variations:
    qid = q["qid"]
    base = q["base"]
    variations = q["variations"]

    for idx, variation in enumerate(variations, start=1):
        # "Different user account" — here represented as different user IDs
        # (e.g. fake user IDs inside your system, NOT separate OpenAI accounts)
        user_id = f"user_{qid}_{idx}"  # e.g. user_1_1, user_1_2, ...

        try:
            answer = ask_chatgpt(variation, user_id=user_id)
        except Exception as e:
            answer = f"ERROR: {e}"

        rows.append({
            "question_id": qid,
            "base_question": base,
            "variation_index": idx,
            "variation_text": variation,
            "user_id": user_id,
            "response": answer,
        })

        # Small delay is polite / can help with rate limits
        time.sleep(0.2)

# Store in a table (Pandas DataFrame)
df = pd.DataFrame(rows)

# Save as CSV
output_path = "chatgpt_variations_results.csv"
df.to_csv(output_path, index=False)

print(f"Saved results to {output_path}")

