"""Frozen safety-prefix boundary and quality-only recomputation selection."""
from typing import Any
from .prompts import format_instruction

def _select(summary: dict[str, Any], *, frozen: bool) -> dict[str, Any]:
    receiver = float(summary["receiver_score"])
    floor = 0.95 * receiver
    by_budget: dict[int, list[tuple[float, int, int, str]]] = {}
    for key, score in summary["group_scores"].items():
        start, end = (int(value) for value in key.split("_"))
        by_budget.setdefault(end - start + 1, []).append((float(score), start, end, key))
    pareto = []
    for budget in sorted(by_budget):
        score, start, end, key = sorted(by_budget[budget], key=lambda row: (-row[0], row[1], row[2]))[0]
        pareto.append({
            "recompute_layers": budget,
            "group": [start, end],
            "group_key": key,
            "profile_score": score,
            "within_5pct": score >= floor,
        })
    eligible = [row for row in pareto if row["within_5pct"]]
    global_key, global_score = max(
        summary["group_scores"].items(),
        key=lambda item: (float(item[1]), -int(item[0].split("_")[0])),
    )
    global_start, global_end = (int(value) for value in global_key.split("_"))
    return {
        "frozen": frozen,
        "selection_observes_safety": False,
        "selection_rule": "minimum recompute budget on per-budget max-quality frontier at >=0.95*receiver",
        "receiver_profile_score": receiver,
        "full_reuse_profile_score": float(summary["full_reuse_score"]),
        "quality_floor_5pct": floor,
        "selected": eligible[0] if eligible else None,
        "gate_pass": bool(eligible),
        "global_best_quality": {
            "group": [global_start, global_end],
            "group_key": global_key,
            "recompute_layers": global_end - global_start + 1,
            "profile_score": float(global_score),
            "within_5pct": float(global_score) >= floor,
        },
        "pareto_frontier": pareto,
    }


def safety_prefix(tokenizer, instruction: str, prompt_protocol: str) -> tuple[str, int, int]:
    prompt = format_instruction(tokenizer, instruction, prompt_protocol)
    if prompt_protocol == "lori_saferpaca_no_input":
        marker = "### Response:"
        boundary_chars = prompt.rfind(marker)
        if boundary_chars < 0:
            raise ValueError("Could not locate LoRI response marker")
    elif prompt_protocol == "tokenizer_chat_template":
        marker = "assistant"
        boundary_chars = prompt.rfind(marker)
        if boundary_chars < 0:
            boundary_chars = prompt.find(instruction) + len(instruction)
    else:
        boundary_chars = prompt.find(instruction) + len(instruction)
    encoded = tokenizer(prompt, truncation=False, return_offsets_mapping=True)
    prefix_tokens = 0
    for start, end in encoded["offset_mapping"]:
        if end <= boundary_chars and start < boundary_chars:
            prefix_tokens += 1
        else:
            break
    if not 1 <= prefix_tokens < len(encoded["input_ids"]):
        raise ValueError(f"Invalid safety shared prefix {prefix_tokens}/{len(encoded['input_ids'])}")
    return prompt, prefix_tokens, len(encoded["input_ids"])
