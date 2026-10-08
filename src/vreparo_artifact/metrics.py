"""Point estimates on paired, pre-aligned observations; no inference or judging."""

from __future__ import annotations

import math
from collections.abc import Iterable


def _numbers(values: Iterable[float]) -> list[float]:
    result = list(values)
    if not result:
        raise ValueError("At least one observation is required")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Observations must be numeric, not booleans or strings")
        if not math.isfinite(value):
            raise ValueError("Observations must be finite")
    return result


def mean(values: Iterable[float]) -> float:
    numbers = _numbers(values)
    return math.fsum(numbers) / len(numbers)


def paired_metrics(native: Iterable[float], condition: Iterable[float]) -> dict:
    baseline, observed = _numbers(native), _numbers(condition)
    if len(baseline) != len(observed):
        raise ValueError("Paired observations must have the same length")
    if any(value < 0 or value > 1 for value in baseline + observed):
        raise ValueError("Safety scores must be in [0, 1]")
    differences = [new - old for old, new in zip(baseline, observed)]
    return {
        "n": len(baseline),
        "native": mean(baseline),
        "condition": mean(observed),
        "delta": mean(differences),
        "sd": mean(max(value, 0.0) for value in differences),
        "si": mean(max(-value, 0.0) for value in differences),
        "sdv": mean(abs(value) for value in differences),
    }


def format_number(value: float, digits: int, signed: bool = False) -> str:
    _numbers([value])
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 12:
        raise ValueError("Decimal places must be an integer from 0 through 12")
    # Do not display a negative zero after rounding.
    rounded = float(f"{value:.{digits}f}")
    if rounded == 0:
        return f"{0:,.{digits}f}"
    return f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
