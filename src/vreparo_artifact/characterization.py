"""Recompute current characterization tables from frozen, text-free records."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

from .metrics import format_number, mean, paired_metrics


PAIRS = (
    ("qwen3_4b", "Qwen3-4B", "Base -> Instruct"),
    ("qwen3_8b", "Qwen3-8B", "Base -> Instruct"),
    ("llama32_3b", "Llama-3.2-3B", "Base -> Instruct"),
    ("llama3_8b", "Llama-3-8B", "Base -> Instruct"),
    ("gemma2_2b", "Gemma-2-2B", "Base -> Instruct"),
    ("mistral7b_v03", "Mistral-7B-v0.3", "Base -> Instruct"),
    ("code_to_safety", "LoRI-D", "Code -> Safety"),
    ("safety_to_code", "LoRI-D", "Safety -> Code"),
)
INVALID_FLAGS = frozenset((
    "empty", "role_only", "content_free_template", "prompt_tail_only_candidate",
    "repeat4gram", "repeat_sentence", "capped_reasoning_candidate", "export_clipped",
))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: dict, expected: set[str], location: str) -> None:
    _require(isinstance(value, dict) and set(value) == expected, f"Invalid fields: {location}")


def _hash(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_data(data: dict) -> dict[tuple[str, str, str], dict]:
    """Check pairing and recompute the frozen validity-intersection decision."""
    _keys(data, {"schema_version", "scope", "validity_note", "sources", "comparisons"}, "root")
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1,
             "Unsupported characterization schema")
    _require(isinstance(data["scope"], str) and isinstance(data["validity_note"], str),
             "Invalid characterization description")
    _require(isinstance(data["sources"], list) and bool(data["sources"]), "Invalid source list")
    source_paths = set()
    for source in data["sources"]:
        _keys(source, {"path", "sha256"}, "source")
        path = source["path"]
        _require(isinstance(path, str) and bool(path) and "\\" not in path and ":" not in path,
                 "Invalid source path")
        parsed = PurePosixPath(path)
        _require(not parsed.is_absolute() and ".." not in parsed.parts and bool(parsed.parts) and
                 parsed.as_posix() == path, "Invalid source path")
        _require(path not in source_paths, "Duplicate source path")
        source_paths.add(path)
        _require(_hash(source["sha256"]), "Invalid source SHA256")
    _require(isinstance(data["comparisons"], list), "Invalid comparison list")
    indexed = {}
    for comparison in data["comparisons"]:
        _keys(comparison, {"study", "pair", "benchmark", "conditions", "total_n", "retained_n", "samples"},
              "comparison")
        _require(all(isinstance(comparison[field], str) for field in ("study", "pair", "benchmark")),
                 "Invalid comparison identifiers")
        key = (comparison["study"], comparison["pair"], comparison["benchmark"])
        _require(key not in indexed, f"Duplicate comparison: {key}")
        _require(key[0] in ("rq1", "rq2") and key[2] in ("harmbench", "strongreject"),
                 f"Unknown study or benchmark: {key}")
        expected_conditions = (["native", "full_kv"] if key[0] == "rq1" else
                               ["native", "k_only", "v_only", "full_kv"])
        _require(comparison["conditions"] == expected_conditions, f"Condition order: {key}")
        samples = comparison["samples"]
        _require(isinstance(samples, list), f"Invalid sample list: {key}")
        _require(type(comparison["total_n"]) is int and type(comparison["retained_n"]) is int,
                 f"Invalid count type: {key}")
        total = 159 if key[2] == "harmbench" else 313
        _require(len(samples) == comparison["total_n"] == total, f"Total count: {key}")
        identifiers = set()
        for ordinal, sample in enumerate(samples):
            _keys(sample, {"ordinal", "prompt_sha256", "scores", "validity_flags", "review_labels", "retained"},
                  f"sample {key}/{ordinal}")
            _require(type(sample["ordinal"]) is int and sample["ordinal"] == ordinal, f"Sample order: {key}")
            identifier = sample["prompt_sha256"]
            _require(_hash(identifier), f"Prompt hash: {key}")
            _require(identifier not in identifiers, f"Duplicate prompt: {key}")
            identifiers.add(identifier)
            for field in ("scores", "validity_flags", "review_labels"):
                _require(isinstance(sample[field], list) and len(sample[field]) == len(expected_conditions),
                         f"Pairing: {key}/{field}")
            for score in sample["scores"]:
                _require(type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1,
                         f"Invalid score: {key}")
                if key[2] == "harmbench":
                    _require(score in (0, 1), f"Nonbinary HarmBench score: {key}")
            invalid = False
            for flags, label in zip(sample["validity_flags"], sample["review_labels"]):
                _require(isinstance(flags, list) and all(isinstance(flag, str) for flag in flags) and
                         len(flags) == len(set(flags)) and
                         set(flags) <= INVALID_FLAGS, f"Unknown validity flag: {key}")
                _require(label in (None, "A", "B", "C", "U"), f"Unknown review label: {key}")
                invalid |= bool(flags) or label not in (None, "A")
            _require(type(sample["retained"]) is bool and sample["retained"] == (not invalid),
                     f"Frozen validity mask mismatch: {key}")
        _require(sum(s["retained"] for s in samples) == comparison["retained_n"] > 0,
                 f"Retained count: {key}")
        indexed[key] = comparison
    expected = {("rq1", pair, benchmark) for pair, _, _ in PAIRS
                for benchmark in ("harmbench", "strongreject")}
    expected.update(("rq2", pair, "harmbench") for pair, _, _ in PAIRS)
    _require(set(indexed) == expected, "Missing or unexpected characterization comparisons")
    return indexed


def _vectors(comparison: dict) -> list[list[float]]:
    return [[sample["scores"][column] for sample in comparison["samples"] if sample["retained"]]
            for column in range(len(comparison["conditions"]))]


def _display(value: float, digits: int, signed: bool = False, leading_dot: bool = False) -> str:
    text = format_number(value, digits, signed=signed)
    if float(text) == 0:
        text = format_number(0, digits)
    if leading_dot:
        text = text.replace("+0.", "+.").replace("-0.", "-.")
        if text.startswith("0."):
            text = text[1:]
    return text


def build_tables(data_dir: Path) -> dict:
    data = json.loads((data_dir / "characterization.json").read_text(encoding="utf-8"))
    indexed = validate_data(data)
    combined, components, hb_details, sr_details = [], [], [], []
    for pair, model, direction in PAIRS:
        row = [model, direction]
        for benchmark in ("harmbench", "strongreject"):
            comparison = indexed[("rq1", pair, benchmark)]
            native, reuse = _vectors(comparison)
            metrics = paired_metrics(native, reuse)
            binary = benchmark == "harmbench"
            scale, digits = (100, 2) if binary else (1, 4)
            values = [_display(metrics[name] * scale, digits, name == "delta", not binary)
                      for name in ("delta", "sd", "si", "sdv")]
            row.extend(values)
            details = [model, direction, f"{metrics['n']}/{comparison['total_n']}"]
            details.extend(_display(metrics[name] * scale, digits, leading_dot=not binary)
                           for name in ("native", "condition"))
            details.extend(values)
            (hb_details if binary else sr_details).append(details)
        combined.append(row)
        comparison = indexed[("rq2", pair, "harmbench")]
        vectors = _vectors(comparison)
        components.append([model, direction, str(comparison["retained_n"])] +
                          [_display(mean(scores) * 100, 2) for scores in vectors] +
                          [_display(paired_metrics(vectors[0], scores)["delta"] * 100, 2, True)
                           for scores in vectors[1:]])
    return {
        "table_01": {
            "title": "Safety incompatibility under full KV cache reuse",
            "columns": ["Model", "Sender -> Receiver", "HB Delta S (pp)", "HB SD (%)",
                        "HB SI (%)", "HB SDV (%)", "SR Delta S", "SR SD", "SR SI", "SR SDV"],
            "rows": combined,
        },
        "table_02": {
            "title": "HarmBench ASRs and changes from Native",
            "columns": ["Model", "Sender -> Receiver", "n", "Native (%)", "K-only (%)",
                        "V-only (%)", "Full KV (%)", "K-N (pp)", "V-N (pp)", "KV-N (pp)"],
            "rows": components,
        },
        "table_05": {
            "title": "Full KV reuse on HarmBench",
            "columns": ["Model", "Sender -> Receiver", "n/total", "Native ASR (%)",
                        "Reuse ASR (%)", "Delta S (pp)", "SD (%)", "SI (%)", "SDV (%)"],
            "rows": hb_details,
        },
        "table_06": {
            "title": "StrongREJECT scores and incompatibility metrics under full KV reuse",
            "columns": ["Model", "Sender -> Receiver", "n/total", "Native", "Reuse",
                        "Delta S", "SD", "SI", "SDV"],
            "rows": sr_details,
        },
    }
