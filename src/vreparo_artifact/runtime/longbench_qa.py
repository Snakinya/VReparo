"""Pinned minimal LongBench HotpotQA formatting and official token-F1 metric."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path


HOTPOTQA_TEMPLATE = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)


def load_jsonl(path: Path, rows: int) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if len(values) < rows:
        raise ValueError(f"Requested {rows} rows from {path}, found {len(values)}")
    return values[:rows]


def format_hotpotqa(tokenizer, item: dict, *, max_raw_tokens: int = 7500) -> tuple[str, dict]:
    raw = HOTPOTQA_TEMPLATE.format(**item)
    token_ids = tokenizer(raw, add_special_tokens=False, truncation=False)["input_ids"]
    original_tokens = len(token_ids)
    truncated = original_tokens > max_raw_tokens
    if truncated:
        half = max_raw_tokens // 2
        raw = tokenizer.decode(token_ids[:half], skip_special_tokens=True) + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    return raw, {"original_raw_tokens": original_tokens, "middle_truncated": truncated, "max_raw_tokens": max_raw_tokens}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, ground_truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: list[str]) -> float:
    return max((qa_f1(prediction, answer) for answer in answers), default=0.0)
