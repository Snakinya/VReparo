"""Summarize newly executed experiments without replacing frozen paper evidence."""

import csv
import hashlib
import math
from pathlib import Path

from .inputs import digest, read, validate_groups, write
from .screening import raw_flags
from ..metrics import mean, paired_metrics


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def scores_for(path, generation):
    scores = read(path)
    if scores.get("state") != "completed" or scores["input_sha256"] != digest(generation):
        raise ValueError("Scores do not belong to the completed generation file")
    groups = read(generation)["condition_records"]
    if set(scores["conditions"]) != set(groups):
        raise ValueError("Scored and generated conditions differ")
    for name, records in groups.items():
        values = scores["conditions"][name]
        if [r["prompt_id"] for r in records] != [r["prompt_id"] for r in values]:
            raise ValueError("Scores are not aligned by ID")
    return scores["conditions"]


def review_decisions(path):
    result = {}
    if path is None:
        return result
    values = read(path)
    if not isinstance(values, list):
        raise ValueError("Review file must be a list of decisions")
    for row in values:
        key = tuple(row[name] for name in ("target", "benchmark", "condition", "prompt_id", "prompt_sha256", "response_sha256"))
        if key in result or type(row.get("valid")) is not bool:
            raise ValueError("Duplicate review decision or non-boolean valid field")
        result[key] = row["valid"]
    return result


def screened_indices(root, study, target, benchmark, groups, decisions=None):
    reference = read(root / "data/screening_reference.json")["comparisons"][f"{study}/{target}/{benchmark}"]
    validate_groups(groups)
    retained, pending, excluded = [], [], []
    for index, row in enumerate(groups["native"]):
        reasons = {}
        for name, records in groups.items():
            current = records[index]
            prior = reference[name].get(hashed(current["prompt"]))
            if prior and hashed(current["response"]) == prior["response_sha256"]:
                flags = prior["flags"] + (["review_" + prior["review_label"]] if prior["review_label"] not in (None, "A") else [])
            else:
                flags = [k for k, v in raw_flags(dict(current, generated_token_count=len(current["token_ids"])), 512).items() if v]
                key = (target, benchmark, name, current["prompt_id"], hashed(current["prompt"]), hashed(current["response"]))
                if key in (decisions or {}):
                    if not decisions[key]:
                        flags.append("review_invalid")
                else:
                    pending.append({"target": target, "benchmark": benchmark, "condition": name,
                                "prompt_id": current["prompt_id"], "response_sha256": hashed(current["response"]),
                                "prompt_sha256": hashed(current["prompt"]),
                                "reason": "Fresh response differs from historical text; historical semantic review is not transferable"})
            if flags:
                reasons[name] = flags
        if reasons:
            excluded.append({"prompt_id": row["prompt_id"], "reasons": reasons})
        else:
            retained.append(index)
    if not retained:
        raise ValueError("No common-valid observations remain")
    return retained, pending, excluded


def safety_rows(target, dataset, groups, scores, indices):
    benchmark = "harmbench" if dataset == "harmbench159" else dataset
    scale = 100 if benchmark == "harmbench" else 1
    values = {name: [r["score"] for r in records] for name, records in scores.items()}
    for series in values.values():
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in series):
            raise ValueError("Invalid judge output; do not silently turn it into a negative label or omit it")
        if benchmark == "harmbench" and any(v not in (0, 1) for v in series):
            raise ValueError("HarmBench scores must be binary")
    selected = {name: [series[i] for i in indices] for name, series in values.items()}
    return [{"target": target, "dataset": benchmark, "condition": name, "n": len(indices),
             "mean": mean(series) * scale,
             **{key: paired_metrics(selected["native"], series)[key] * scale for key in ("delta", "sd", "si", "sdv")},
             "mean_unit": "%" if scale == 100 else "score", "delta_unit": "pp" if scale == 100 else "score"}
            for name, series in selected.items()]


def summarize(root, experiment, targets, review_path=None):
    base = root / "outputs/experiments"
    table, pending, exclusions = [], [], []
    decisions = review_decisions(review_path)
    invalid_judgments = 0
    for target in targets:
        folder = base / target
        if experiment in ("rq1", "rq2", "mitigation"):
            datasets = ("harmbench159", "strongreject") if experiment == "rq1" else ("harmbench159",) if experiment == "rq2" else ("harmbench", "strongreject", "orbench")
            for dataset in datasets:
                generation = folder / experiment / dataset / "generation_0.json"
                source = read(generation)
                if source.get("state") != "completed":
                    raise ValueError("Generation is incomplete")
                groups = source["condition_records"]
                validate_groups(groups)
                benchmark = "harmbench" if dataset == "harmbench159" else dataset
                scores = scores_for(generation.parent / "scored" / (benchmark + ".json"), generation)
                if dataset == "orbench":
                    for name, values in scores.items():
                        counts = {label: sum(r["label"] == label for r in values) for label in ("F", "P", "N", "U", "INVALID")}
                        if sum(counts.values()) != len(values):
                            raise ValueError("Unknown refusal label")
                        invalid_judgments += counts["INVALID"]
                        table.append({"target": target, "dataset": dataset, "condition": name, "n": len(values),
                                      "full_refusal_pct": 100 * counts["F"] / len(values), **counts})
                    continue
                indices = list(range(len(groups["native"])))
                if experiment in ("rq1", "rq2"):
                    indices, needs_review, removed = screened_indices(root, experiment, target, benchmark, groups, decisions)
                    pending.extend(needs_review)
                    exclusions.extend({"target": target, "dataset": benchmark, **r} for r in removed)
                table.extend(safety_rows(target, dataset, groups, scores, indices))
        elif experiment == "quality":
            for dataset in ("qa_droid" if target.endswith("droid") else "qa_full", "2wikimqa", "multifieldqa_en"):
                source = read(folder / "quality" / dataset / "generation_0.json")
                if source["state"] != "completed":
                    raise ValueError("QA generation is incomplete")
                for name, rows in source["condition_records"].items():
                    table.append({"target": target, "dataset": dataset, "condition": name, "n": len(rows), "f1_pct": mean(r["f1"] for r in rows) * 100})
        elif experiment == "calibration":
            selection = read(folder / "calibration/xstest/selected/selection.json")
            frozen = read(root / "experiments/paper_cases.json")["cases"][target]["alpha"]
            for alpha, gate in selection["gates"].items():
                table.append({"target": target, "alpha": alpha, "unsafe_gain_pp": 100 * gate["unsafe_refusal_gain"],
                              "safe_delta_pp": 100 * gate["safe_delta"], "max_subtype_delta_pp": 100 * max(gate["safe_type_deltas"].values()),
                              "eligible": gate["eligible"], "selected": float(alpha) == selection["selected_alpha"],
                              "paper_alpha": frozen, "selection_matches_paper": selection["selected_alpha"] == frozen})
        elif experiment == "performance":
            timing = read(folder / "performance/timing.json")
            if timing["state"] != "completed":
                raise ValueError("Timing is incomplete")
            means = timing["means"]
            for name in ("native", "raw", "vreparo"):
                table.append({"target": target, "condition": name, "ttft_ms": means["ttft_ms"][name],
                              "decode_ms_per_token": means["decode_ms_per_token"][name],
                              "speedup_vs_native": means["ttft_ms"]["native"] / means["ttft_ms"][name]})
        elif experiment == "profile":
            profile = read(folder / "profile/profile.json")
            if profile["state"] != "completed":
                raise ValueError("Profiling is incomplete")
            selected = profile["selection"]["selected"]
            table.append({"target": target, "selected_group": str(selected["group"]) if selected else "none",
                          "profile_score": selected["profile_score"] if selected else None,
                          "matches_paper_block": selected is not None and selected["group"] == [0, 1]})
    destination = root / "outputs/reports" / experiment / ("all" if len(targets) > 1 else targets[0])
    result = {"status": "needs_validity_review" if pending else "needs_judgment_review" if invalid_judgments else "completed", "experiment": experiment,
              "targets": list(targets), "rows": table, "pending_review": pending, "excluded": exclusions,
              "invalid_judgments": invalid_judgments, "review_sha256": digest(review_path) if review_path else None,
              "note": "Fresh results, not substituted frozen values. Hash-matched historical validity decisions are reused only for identical prompt/response text."}
    write(destination / "summary.json", result)
    columns = list(dict.fromkeys(key for row in table for key in row))
    with (destination / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = [f"# {experiment.upper()} Fresh Results", "", f"Status: {result['status']}", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |" for row in table)
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    return {"status": result["status"], "rows": len(table), "pending_validity_reviews": len(pending),
            "report": str(destination.relative_to(root))}
