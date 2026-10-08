"""Shared loops for paper fitting, characterization, calibration, and evaluation."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import statistics

import torch

from .inputs import digest, identifier, read, rows, validate_rows, write
from .runner import Runner, load_mapper
from ..runtime.cache import slice_cache_prefix
from ..runtime.longbench_qa import best_qa_f1
from ..runtime.prompts import format_instruction
from ..runtime.refusal import official_string_refusal
from ..runtime.ridge import ValueRidgeMapper, ValueRidgeStatistics


def signature(config, paths):
    source = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        code.update(path.relative_to(source).as_posix().encode())
        code.update(path.read_bytes())
    return {"config": config, "inputs": {key: digest(value) for key, value in paths.items()},
            "libraries": {name: version(name) for name in ("transformers", "numpy", "safetensors")},
            "torch": str(torch.__version__), "code_sha256": code.hexdigest(), "operator_schema": 1}


def checkpoint(path, fingerprint, **defaults):
    if path.exists():
        result = read(path)
        if result.get("fingerprint") != fingerprint:
            raise ValueError("Output belongs to different inputs/configuration; choose a new output directory")
        return result
    return {"state": "running", "fingerprint": fingerprint, **defaults}


def training_rows(runner, qa_path, safety_path, indices=None):
    qa, safety = rows(qa_path), rows(safety_path)
    if len(qa) < 64 or len(safety) < 64:
        raise ValueError("Historical fitting requires the first 64 rows of each source")
    selected = []
    for domain, values in (("qa", qa), ("safety", safety)):
        selected_indices = indices[domain] if indices is not None else [i for i in range(64) if i % 4 < 2]
        if len(selected_indices) != 32 or sorted(set(selected_indices)) != selected_indices:
            raise ValueError("Expected 32 unique source-sorted fitting indices per domain")
        for index in selected_indices:
            row = values[index]
            if domain == "qa" and not runner.droid:
                raw = f"Answer the question using the context.\n\nQuestion: {row['input']}\n\nContext:\n{row['context']}"
                text = format_instruction(runner.tokenizer, raw, runner.protocol)
                prefix = None
            else:
                text, prefix, _ = runner.formatted(row, "hotpotqa" if domain == "qa" else "safety")
            selected.append((domain, index, text, prefix))
    return selected


@torch.inference_mode()
def fit(args, config):
    fingerprint = signature(config, {"qa": args.qa, "safety": args.safety})
    report_path, map_path = args.output / "fit.json", args.output / "mapper.pt"
    if report_path.exists():
        old = read(report_path)
        if old["fingerprint"] != fingerprint or old["mapper_sha256"] != digest(map_path):
            raise ValueError("Existing fit does not match these inputs")
        return old
    runner = Runner(config, args.assets, args.device)
    data = training_rows(runner, args.qa, args.safety, config.get('fitting_indices'))
    stats = ValueRidgeStatistics.empty(layers=runner.layers,
                                     kv_heads=runner.receiver.config.num_key_value_heads,
                                     head_dim=runner.receiver.config.hidden_size // runner.receiver.config.num_attention_heads)
    records = []
    saved_path = args.output / "statistics.pt"
    if saved_path.exists():
        previous = torch.load(saved_path, map_location="cpu", weights_only=True)
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Partial fit belongs to different inputs")
        stats, records = ValueRidgeStatistics(**previous["stats"]), previous["records"]
    args.output.mkdir(parents=True, exist_ok=True)
    for domain, index, text, prefix in data[len(records):]:
        prepared = runner.prepare(text, prefix)
        source = prepared.selected if runner.droid else prepared.producer
        target = prepared.receiver
        if runner.droid:
            source = slice_cache_prefix(source, prepared.shared_prefix_tokens, runner.layers)
            target = slice_cache_prefix(target, prepared.shared_prefix_tokens, runner.layers)
        stats.update(source, target, runner.layers, max_tokens=128, component="value")
        records.append({"domain": domain, "source_index": index, "formatted_sha256": hashlib.sha256(text.encode()).hexdigest()})
        temporary = saved_path.with_suffix(".tmp.pt")
        torch.save({"fingerprint": fingerprint, "stats": vars(stats), "records": records}, temporary)
        temporary.replace(saved_path)
        del prepared, source, target
        print(f"fit {len(records)}/64", flush=True)
    mapper = ValueRidgeMapper.fit(stats, ridge=1.0, identity_centered=True)
    active = [i for i in range(runner.layers) if not runner.droid or not runner.group[0] <= i <= runner.group[1]]
    if not torch.isfinite(mapper.weights).all() or not torch.isfinite(mapper.bias).all():
        raise ValueError("Nonfinite fitted map")
    payload = {"kind": "dense", "weights": mapper.weights, "bias": mapper.bias, "active_layers": active}
    temporary = map_path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(map_path)
    result = {"state": "completed", "fingerprint": fingerprint, "mapper_sha256": digest(map_path),
              "texts": len(records), "positions_per_head": stats.count, "records": records}
    write(report_path, result)
    return result


@torch.inference_mode()
def generate(args, config):
    data = rows(args.input)
    validate_rows(data, args.task)
    experiments = {"rq1": ["native", "raw"], "rq2": ["native", "k_only", "v_only", "raw"],
                   "mitigation": ["native", "raw", "vreparo"], "quality": ["native", "raw", "vreparo"],
                   "calibration": ["native", "raw", *[f"vreparo_a{a:g}" for a in (0.1, 0.25, 0.5, 1.0)]]}
    conditions = experiments[args.experiment]
    if args.experiment == "rq2" and config["operator"] != "direct":
        raise ValueError("RQ2 compares direct K/V/full reuse, not a Droid operator")
    needs_map = any(name.startswith("vreparo") for name in conditions)
    if needs_map and args.mapper is None:
        raise ValueError("This experiment requires a fitted mapper")
    paths = {"dataset": args.input}
    if needs_map:
        paths["mapper"] = args.mapper
    if getattr(args, 'selection', None):
        paths['selection'] = args.selection
    fingerprint = signature(config, paths)
    fingerprint.update(experiment=args.experiment, task=args.task, budget=args.max_new_tokens,
                       shard=args.shard, shards=args.shards)
    path = args.output / f"generation_{args.shard}.json"
    result = checkpoint(path, fingerprint, condition_records={name: [] for name in conditions})
    if result["state"] == "completed":
        return result
    runner = Runner(config, args.assets, args.device)
    maps = {}
    for name in conditions:
        if name.startswith("vreparo"):
            alpha = float(name.split("_a")[1]) if "_a" in name else config["alpha"]
            maps[name] = load_mapper(args.mapper, alpha, runner.layers, runner.group, args.device)
    selected = data[args.shard::args.shards]
    done = {name: {r["prompt_id"] for r in values} for name, values in result["condition_records"].items()}
    for index, row in enumerate(selected):
        key = identifier(row)
        missing = [name for name in conditions if key not in done[name]]
        if not missing:
            continue
        text, prefix, metadata = runner.formatted(row, args.task)
        prepared = runner.prepare(text, prefix)
        for name in missing:
            generated = runner.decode(prepared, text, name, args.max_new_tokens, maps.get(name))
            record = {"prompt_id": key, "prompt": row.get("prompt", row.get("input")),
                      "response": generated.text, "token_ids": list(generated.token_ids),
                      "stopped_on_eos": generated.stopped_on_eos, "metadata": metadata}
            if args.task == "safety":
                record.update({k: row[k] for k in ("label", "type", "category") if k in row})
                record["string_refusal"] = official_string_refusal(generated.text)
            else:
                record["f1"] = best_qa_f1(generated.text, row["answers"])
            result["condition_records"][name].append(record)
            done[name].add(key)
            write(path, result)
        del prepared
        print(f"{args.experiment} shard={args.shard} {index + 1}/{len(selected)}", flush=True)
    result["state"] = "completed"
    write(path, result)
    return {"state": "completed", "rows_per_condition": len(selected), "conditions": conditions}


def merge(args):
    parts = [read(args.input / f"generation_{i}.json") for i in range(args.shards)]
    source = rows(args.dataset_file)
    expected = [identifier(row) for row in source]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate source IDs")
    common = {k: v for k, v in parts[0]["fingerprint"].items() if k != "shard"}
    if common["inputs"]["dataset"] != digest(args.dataset_file) or common["shards"] != args.shards:
        raise ValueError("Source dataset or shard count mismatch")
    for i, part in enumerate(parts):
        if part["state"] != "completed" or part["fingerprint"]["shard"] != i:
            raise ValueError("Missing or incomplete shard")
        if {k: v for k, v in part["fingerprint"].items() if k != "shard"} != common:
            raise ValueError("Shard fingerprints differ")
        if set(part["condition_records"]) != set(parts[0]["condition_records"]):
            raise ValueError("Shard conditions differ")
    groups = {}
    for name in parts[0]["condition_records"]:
        values = [r for part in parts for r in part["condition_records"][name]]
        by_id = {r["prompt_id"]: r for r in values}
        if len(by_id) != len(values) or set(by_id) != set(expected):
            raise ValueError("Missing, duplicate, or extra generated rows")
        groups[name] = [by_id[key] for key in expected]
    result = {"state": "completed", "fingerprint": common, "condition_records": groups}
    write(args.output / "generation.json", result)
    return {"state": "completed", "conditions": list(groups), "n": len(expected)}


@torch.inference_mode()
def profile(args, config):
    from ..runtime.boundaries import _select
    from ..runtime.droidspeak import profile_two_layer_groups

    if config["operator"] != "droid":
        raise ValueError("Profiling applies to DroidSpeak cases")
    data = rows(args.input)
    validate_rows(data, "hotpotqa")
    if len(data) != 50:
        raise ValueError("Paper profile uses exactly 50 QA examples")
    path = args.output / "profile.json"
    result = checkpoint(path, signature(config, {"profile": args.input}), records=[])
    if result["state"] == "completed":
        return result["selection"]
    runner = Runner(config, args.assets, args.device)
    for row in data[len(result["records"]):]:
        text, prefix, _ = runner.formatted(row, "hotpotqa")
        native = runner.native(text, 32)
        groups = profile_two_layer_groups(runner.sender, runner.receiver, runner.tokenizer, text, prefix,
                                         num_layers=runner.layers, max_new_tokens=32)
        result["records"].append({"prompt_id": identifier(row), "native_f1": best_qa_f1(native.text, row["answers"]),
                                  "groups": {k: best_qa_f1(v[0], row["answers"]) for k, v in groups.items()}})
        write(path, result)
        print(f"profile {len(result['records'])}/50", flush=True)
    summary = {"receiver_score": statistics.mean(r["native_f1"] for r in result["records"]), "full_reuse_score": 0,
               "group_scores": {k: statistics.mean(r["groups"][k] for r in result["records"]) for k in result["records"][0]["groups"]}}
    selected = _select(summary, frozen=True)
    selected["full_reuse_profile_score"] = None
    result.update(state="completed", selection=selected)
    write(path, result)
    return selected
