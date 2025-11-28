#!/usr/bin/env python3
"""
Compute LLM evaluation metrics between a gold answer (base_question)
and a model response (variant_response).

Metrics:
- Exact Match
- Token Precision / Recall / F1
- ROUGE-1 / ROUGE-2 (P/R/F1)
- BLEU
- Edit Distance (Levenshtein)
- Embedding cosine similarity (SentenceTransformer)
- BLEURT (via Hugging Face Elron/bleurt-base-512)
- NLI label (entailment / neutral / contradiction) via BART MNLI

Supports:
- Batch evaluation from a DuckDB file, updating the table in-place,
  and exporting the same rows with metrics to a CSV.
- (Optional) CSV-only mode function left in for flexibility.
"""

import os
import re
from collections import Counter
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import duckdb
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# -------------------------------------------------------------------
# Global env / device settings
# -------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Normalization & Tokenization
# -------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse spaces."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization after normalization."""
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


# -------------------------------------------------------------------
# Exact Match
# -------------------------------------------------------------------

def exact_match(gold: str, pred: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0


# -------------------------------------------------------------------
# Token Precision / Recall / F1
# -------------------------------------------------------------------

def prf1_token_overlap(gold: str, pred: str) -> Tuple[float, float, float]:
    """Precision/Recall/F1 based on token overlap (multiset)."""
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    if not gold_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0  # both empty -> perfect

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = sum((gold_counts & pred_counts).values())  # multiset intersection

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# -------------------------------------------------------------------
# ROUGE-N (N = 1, 2)
# -------------------------------------------------------------------

def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(gold: str, pred: str, n: int = 1) -> Tuple[float, float, float]:
    """
    ROUGE-N with precision/recall/F1 based on n-gram overlap.
    """
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    gold_ngrams = ngrams(gold_tokens, n)
    pred_ngrams = ngrams(pred_tokens, n)

    if not gold_ngrams and not pred_ngrams:
        return 1.0, 1.0, 1.0  # trivially perfect

    gold_counts = Counter(gold_ngrams)
    pred_counts = Counter(pred_ngrams)

    overlap = sum((gold_counts & pred_counts).values())

    precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
    recall = overlap / len(gold_ngrams) if gold_ngrams else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# -------------------------------------------------------------------
# BLEU
# -------------------------------------------------------------------

def bleu_score(gold: str, pred: str) -> float:
    """Sentence-level BLEU with smoothing."""
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothing)


# -------------------------------------------------------------------
# Edit Distance (Levenshtein)
# -------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (character-based)."""
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


# -------------------------------------------------------------------
# Embedding Cosine Similarity
# -------------------------------------------------------------------

_EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def embedding_similarity(gold: str, pred: str) -> float:
    """
    Cosine similarity between sentence embeddings.
    """
    embeddings = _EMB_MODEL.encode([gold, pred])
    return cosine_similarity(embeddings[0], embeddings[1])


# -------------------------------------------------------------------
# BLEURT via Hugging Face
# -------------------------------------------------------------------

_BLEURT_MODEL_NAME = "Elron/bleurt-base-512"
_BLEURT_DEVICE = torch.device("cpu")

_BLEURT_TOKENIZER = AutoTokenizer.from_pretrained(_BLEURT_MODEL_NAME)
_BLEURT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    _BLEURT_MODEL_NAME
).to(_BLEURT_DEVICE)
_BLEURT_MODEL.eval()


def bleurt_score(gold: str, pred: str) -> float:
    """
    BLEURT-like score using Hugging Face BLEURT model (PyTorch).
    Higher is better.
    """
    gold_n = normalize_text(gold)
    pred_n = normalize_text(pred)

    if not gold_n and not pred_n:
        return 1.0

    inputs = _BLEURT_TOKENIZER(
        gold_n,
        pred_n,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(_BLEURT_DEVICE)

    with torch.no_grad():
        outputs = _BLEURT_MODEL(**inputs)
        score = outputs.logits.squeeze(-1).item()

    return float(score)


# -------------------------------------------------------------------
# NLI (Entailment / Neutral / Contradiction)
# -------------------------------------------------------------------

_NLI_PIPE = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1,  # CPU
)


def nli_relation(gold: str, pred: str) -> Tuple[str, float]:
    """
    Classify relation between gold (premise) and pred (hypothesis)
    as entailment / neutral / contradiction.
    """
    result = _NLI_PIPE(
        {"text": gold, "text_pair": pred},
        truncation=True,
    )

    if isinstance(result, list):
        if len(result) == 0:
            return "UNKNOWN", 0.0
        best = result[0]
    elif isinstance(result, dict):
        best = result
    else:
        return "UNKNOWN", 0.0

    label = best.get("label", "UNKNOWN").upper()
    score = float(best.get("score", 0.0))
    return label, score


# -------------------------------------------------------------------
# Wrapper: compute all metrics
# -------------------------------------------------------------------

def compute_all_metrics(gold: str, pred: str, bleurt_checkpoint: str | None = None) -> Dict[str, Any]:
    """
    Compute all metrics between gold and pred.
    (bleurt_checkpoint is ignored; kept only for API compatibility.)
    """
    em = exact_match(gold, pred)
    p_tok, r_tok, f1_tok = prf1_token_overlap(gold, pred)

    r1_p, r1_r, r1_f = rouge_n(gold, pred, n=1)
    r2_p, r2_r, r2_f = rouge_n(gold, pred, n=2)

    bleu = bleu_score(gold, pred)
    edit_dist = levenshtein(gold, pred)
    emb_sim = embedding_similarity(gold, pred)

    nli_label, nli_score = nli_relation(gold, pred)
    bleurt = bleurt_score(gold, pred)

    return {
        "exact_match": em,
        "token_precision": p_tok,
        "token_recall": r_tok,
        "token_f1": f1_tok,
        "rouge1_precision": r1_p,
        "rouge1_recall": r1_r,
        "rouge1_f1": r1_f,
        "rouge2_precision": r2_p,
        "rouge2_recall": r2_r,
        "rouge2_f1": r2_f,
        "bleu": bleu,
        "edit_distance": edit_dist,
        "embedding_cosine_similarity": emb_sim,
        "nli_label": nli_label,
        "nli_label_score": nli_score,
        "bleurt_score": bleurt,
    }


# -------------------------------------------------------------------
# CSV batch mode (still available if you need it)
# -------------------------------------------------------------------

def evaluate_variants_from_csv(
    input_csv_path: str,
    output_csv_path: str,
    bleurt_checkpoint: str | None = None,
) -> None:
    """
    Read a CSV with:
      Base question, Variant 1 - Paraphrasing, Variant 2 - Add noise, ...
    and write metrics per (base, variant) row to output_csv_path.
    """
    df = pd.read_csv(input_csv_path)

    base_col = df.columns[0]
    variant_cols = df.columns[1:]

    rows_out = []

    for _, row in df.iterrows():
        base_question = str(row[base_col])

        for vc in variant_cols:
            variant_text = row[vc]

            if not isinstance(variant_text, str) or not variant_text.strip():
                continue

            metrics = compute_all_metrics(
                gold=base_question,
                pred=variant_text,
                bleurt_checkpoint=bleurt_checkpoint,
            )

            out_row = {
                "base_question": base_question,
                "variant_category": vc,
                "variant_question": variant_text,
                "variant_response": variant_text,
            }
            out_row.update(metrics)
            rows_out.append(out_row)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Saved metrics for {len(rows_out)} (base, variant) pairs to {output_csv_path}")


# -------------------------------------------------------------------
# DuckDB batch mode: read table, update columns, write CSV
# -------------------------------------------------------------------

def evaluate_responses_from_duckdb(
    db_path: str,
    table_name: str = "responses",
    bleurt_checkpoint: str | None = None,
    output_csv_path: str | None = None,
    only_missing: bool = False,
) -> None:
    """
    Read rows from a DuckDB table and store metrics back into the same table,
    and optionally also write all processed rows to a CSV.

    Assumes table schema includes at least:
      base_question       (gold text)
      variant_category    (category string)
      variant_question    (variant question text)
      variant_response    (model response text)

    And metric columns:
        exact_match                 BOOLEAN
        token_precision             DOUBLE
        token_recall                DOUBLE
        token_f1                    DOUBLE
        rouge1_precision            DOUBLE
        rouge1_recall               DOUBLE
        rouge1_f1                   DOUBLE
        rouge2_precision            DOUBLE
        rouge2_recall               DOUBLE
        rouge2_f1                   DOUBLE
        bleu                        DOUBLE
        edit_distance               INTEGER
        embedding_cosine_similarity DOUBLE
        bleurt                      DOUBLE
        nli_label                   VARCHAR
    """

    con = duckdb.connect(db_path)

    base_query = f"""
        SELECT
            rowid,
            base_question,
            variant_category,
            variant_question,
            variant_response
        FROM {table_name}
        WHERE variant_response IS NOT NULL
    """

    #if only_missing:
     #   base_query += " AND exact_match IS NULL"

    df = con.execute(base_query).df()

    if df.empty:
        print("No rows to process from DuckDB (query returned 0 rows).")
        con.close()
        return

    rows_to_update = []
    rows_for_csv: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        gold = str(row["base_question"])
        pred = str(row["variant_response"])

        metrics = compute_all_metrics(
            gold=gold,
            pred=pred,
            bleurt_checkpoint=bleurt_checkpoint,
        )

        # For DuckDB UPDATE
        rows_to_update.append(
            (
                metrics["exact_match"],
                metrics["token_precision"],
                metrics["token_recall"],
                metrics["token_f1"],
                metrics["rouge1_precision"],
                metrics["rouge1_recall"],
                metrics["rouge1_f1"],
                metrics["rouge2_precision"],
                metrics["rouge2_recall"],
                metrics["rouge2_f1"],
                metrics["bleu"],
                metrics["edit_distance"],
                metrics["embedding_cosine_similarity"],
                metrics["bleurt_score"],  # -> bleurt column
                metrics["nli_label"],      # -> nli_label column
                int(row["rowid"]),
            )
        )

        # For CSV
        csv_row = {
            "base_question":    row["base_question"],
            "variant_category": row["variant_category"],
            "variant_question": row["variant_question"],
            "variant_response": row["variant_response"],
        }
        csv_row.update(metrics)
        rows_for_csv.append(csv_row)

    update_sql = f"""
        UPDATE {table_name}
        SET
            exact_match                 = ?,
            token_precision             = ?,
            token_recall                = ?,
            token_f1                    = ?,
            rouge1_precision            = ?,
            rouge1_recall               = ?,
            rouge1_f1                   = ?,
            rouge2_precision            = ?,
            rouge2_recall               = ?,
            rouge2_f1                   = ?,
            bleu                        = ?,
            edit_distance               = ?,
            embedding_cosine_similarity = ?,
            bleurt                      = ?,
            nli_label                   = ?
        WHERE rowid = ?
    """

    con.executemany(update_sql, rows_to_update)
    con.close()
    print(f"Updated {len(rows_to_update)} rows in {table_name} in {db_path}")

    if output_csv_path is not None:
        out_df = pd.DataFrame(rows_for_csv)
        out_df.to_csv(output_csv_path, index=False)
        print(f"Also wrote metrics for {len(rows_for_csv)} rows to {output_csv_path}")


# -------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    # DuckDB-based evaluation
    DB_PATH = "llm_variants_responses.duckdb"
    TABLE_NAME = "responses"
    OUTPUT_CSV = "responses_metrics_from_duckdb.csv"

    evaluate_responses_from_duckdb(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        bleurt_checkpoint=None,  # kept for signature; BLEURT HF model ignores it
        output_csv_path=OUTPUT_CSV,
        only_missing=True,       # set False to recompute all rows
    )

    # If you still want CSV-only mode, uncomment:
    # INPUT_CSV = "Dataset(set1).csv"
    # OUTPUT_CSV_CSVMODE = "full_question_variants_metrics.csv"
    # evaluate_variants_from_csv(
    #     input_csv_path=INPUT_CSV,
    #     output_csv_path=OUTPUT_CSV_CSVMODE,
    # )
