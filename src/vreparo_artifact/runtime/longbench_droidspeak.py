"""LongBench task formatting for DroidSpeak-style cache-sharing tests."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from .prompts import format_instruction


PROMPT_TEMPLATES = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
}

MAX_NEW_TOKENS = {
    "hotpotqa": 32,
    "2wikimqa": 32,
    "multifieldqa_en": 64,
}


def load_jsonl(path: Path, rows: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if rows < 1 or rows > len(values):
        raise ValueError(f"Requested 1..{len(values)} rows from {path}, got {rows}")
    return values[:rows]


def _middle_truncate_with_boundary(
    tokenizer,
    raw_prompt: str,
    boundary_chars: int,
    max_raw_tokens: int,
) -> tuple[str, int, int, bool]:
    encoded = tokenizer(
        raw_prompt,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    boundary_tokens = 0
    for start, end in offsets:
        if end <= boundary_chars and start < boundary_chars:
            boundary_tokens += 1
        else:
            break
    if len(ids) <= max_raw_tokens:
        return raw_prompt, boundary_chars, len(ids), False
    half = max_raw_tokens // 2
    left_ids, right_ids = ids[:half], ids[-half:]
    left_text = tokenizer.decode(left_ids, skip_special_tokens=True)
    right_text = tokenizer.decode(right_ids, skip_special_tokens=True)
    right_start = len(ids) - half
    if boundary_tokens <= half:
        tracked_boundary = len(tokenizer.decode(left_ids[:boundary_tokens], skip_special_tokens=True))
    elif boundary_tokens >= right_start:
        right_boundary = boundary_tokens - right_start
        tracked_boundary = len(left_text) + len(
            tokenizer.decode(right_ids[:right_boundary], skip_special_tokens=True)
        )
    else:
        tracked_boundary = len(left_text)
    return left_text + right_text, tracked_boundary, len(ids), True


def _coerce_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)]


def format_longbench(
    tokenizer,
    dataset: str,
    item: dict[str, Any],
    *,
    prompt_protocol: str,
    max_raw_tokens: int = 7500,
) -> tuple[str, int, dict[str, Any]]:
    """Return prompt, shared context-prefix token boundary, and metadata."""
    template = PROMPT_TEMPLATES[dataset]
    before_context, after_context = template.split("{context}", 1)
    raw_prefix = before_context.format(**item) + item["context"]
    raw_suffix = after_context.format(**item)
    raw_prompt, raw_boundary, original_raw_tokens, truncated = _middle_truncate_with_boundary(
        tokenizer,
        raw_prefix + raw_suffix,
        len(raw_prefix),
        max_raw_tokens,
    )
    if prompt_protocol == "raw":
        prompt = raw_prompt
        boundary_chars = raw_boundary
    else:
        prompt = format_instruction(tokenizer, raw_prompt, prompt_protocol)
        offset = prompt.find(raw_prompt)
        if offset < 0:
            raise ValueError("Prompt protocol transformed the LongBench text")
        boundary_chars = offset + raw_boundary
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    prefix_tokens = 0
    for start, end in encoded["offset_mapping"]:
        if end <= boundary_chars and start < boundary_chars:
            prefix_tokens += 1
        else:
            break
    if not 1 <= prefix_tokens < len(encoded["input_ids"]):
        raise ValueError(f"Invalid shared prefix {prefix_tokens}/{len(encoded['input_ids'])}")
    return prompt, prefix_tokens, {
        "answers": _coerce_answers(item["answers"]),
        "id": str(item.get("_id", item.get("id", ""))),
        "prompt_tokens": len(encoded["input_ids"]),
        "shared_prefix_tokens": prefix_tokens,
        "original_raw_tokens": original_raw_tokens,
        "middle_truncated": truncated,
        "max_raw_tokens": max_raw_tokens,
        "prompt_protocol": prompt_protocol,
    }


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
