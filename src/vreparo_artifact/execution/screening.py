"""Frozen response-form rules. Prior decisions apply only to exact text hashes."""
from collections import Counter
import re

def raw_flags(record: dict, budget: int) -> dict[str, bool]:
    text = record.get("response", "")
    words = text.lower().split()
    norm = " ".join(re.sub("[^a-z\\s]", "", text.lower()).split())
    prompt_norm = " ".join(re.sub("[^a-z\\s]", "", record.get("prompt", "").lower()).split())
    grams = Counter(tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3)))
    units = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s*)", "", line)
        for unit in re.split(r"(?<=[.!?])\s+", line):
            unit = " ".join(unit.lower().split())
            if len(unit) >= 20 and len(unit.split()) >= 3:
                units.append(unit)
    return {
        "empty": not text.strip(),
        "role_only": bool(text.strip()) and bool(re.fullmatch(r"(?:assistant|user|human|system|\s)+", text, re.I)),
        "content_free_template": bool(re.fullmatch(r"(?:the correct answer(?: is)?\s*)+", norm)),
        "prompt_tail_only_candidate": bool(norm)
        and len(norm.split()) <= 8
        and prompt_norm.endswith(norm)
        and bool(re.match(r"(?i)^(write|give|provide|describe|explain|create|generate)\b", record.get("prompt", ""))),
        "repeat4gram": len(words) >= 40 and bool(grams) and max(grams.values()) >= 10,
        "repeat_sentence": bool(units) and max(Counter(units).values()) >= 4,
        "capped_reasoning_candidate": record.get("generated_token_count") == budget
        and bool(
            re.search(
                r"(?i)(the user (?:wants|is asking|asked)|I need to|let me (?:think|see|break|analyze))",
                text[:400],
            )
        )
        and "</think>" not in text,
        "export_clipped": bool(record.get("response_clipped", False)),
    }
