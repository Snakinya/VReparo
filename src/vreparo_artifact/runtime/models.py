"""Model loading and strict compatibility gates for cache sharing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    name: str
    foundation: Path
    adapter: Path | None = None
    tokenizer: Path | None = None
    repository: str | None = None
    revision: str | None = None
    training_task: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], root: Path) -> "ModelSpec":
        adapter = value.get("adapter")
        tokenizer = value.get("tokenizer")
        return cls(
            name=value["name"],
            foundation=(root / value["foundation"]).resolve(),
            adapter=(root / adapter).resolve() if adapter else None,
            tokenizer=(root / tokenizer).resolve() if tokenizer else None,
            repository=value.get("repository"),
            revision=value.get("revision"),
            training_task=value.get("training_task"),
        )


def adapter_config(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    return json.loads(config_path.read_text())


def adapter_saved_modules(path: Path) -> dict[str, Any]:
    """Describe full modules bundled with an adapter checkpoint."""
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights}")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        full = [key for key in keys if ".lora_" not in key]
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in full}
        dtypes = {key: str(handle.get_slice(key).get_dtype()) for key in full}
    vocab_rows = {
        shape[0]
        for key, shape in shapes.items()
        if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")
    }
    if len(vocab_rows) > 1:
        raise ValueError(f"Inconsistent saved vocabulary sizes in {path}")
    return {
        "full_module_keys": full,
        "full_module_shapes": shapes,
        "full_module_dtypes": dtypes,
        "vocab_size": next(iter(vocab_rows), None),
    }


def assert_adapter_family(specs: Iterable[ModelSpec]) -> dict[str, Any]:
    """Require the same foundation and LoRI/LoRA structural recipe."""
    adapter_specs = [spec for spec in specs if spec.adapter is not None]
    if not adapter_specs:
        raise ValueError("At least one adapter is required")
    foundations = {spec.foundation for spec in adapter_specs}
    if len(foundations) != 1:
        raise ValueError(f"Adapter foundations differ: {sorted(map(str, foundations))}")
    configs = {spec.name: adapter_config(spec.adapter) for spec in adapter_specs}
    fields = ("base_model_name_or_path", "peft_type", "r", "lora_alpha", "bias")
    mismatches: list[str] = []
    reference = configs[adapter_specs[0].name]
    for spec in adapter_specs[1:]:
        current = configs[spec.name]
        for field in fields:
            if current.get(field) != reference.get(field):
                mismatches.append(
                    f"{spec.name}.{field}={current.get(field)!r} != {reference.get(field)!r}"
                )
        if set(current.get("target_modules", ())) != set(reference.get("target_modules", ())):
            mismatches.append(f"{spec.name}.target_modules differs")
    if mismatches:
        raise ValueError("Adapter-family mismatch: " + "; ".join(mismatches))
    modules = {
        spec.name: adapter_saved_modules(spec.adapter) for spec in adapter_specs
    }
    saved_vocab_sizes = {
        value["vocab_size"] for value in modules.values() if value["vocab_size"]
    }
    if len(saved_vocab_sizes) > 1:
        raise ValueError(f"Adapter saved vocabularies differ: {saved_vocab_sizes}")
    return {
        "foundation": str(adapter_specs[0].foundation),
        "base_model_name_or_path": reference.get("base_model_name_or_path"),
        "peft_type": reference.get("peft_type"),
        "rank": reference.get("r"),
        "alpha": reference.get("lora_alpha"),
        "target_modules": sorted(reference.get("target_modules", ())),
        "saved_vocab_size": next(iter(saved_vocab_sizes), None),
        "saved_full_modules": modules,
    }


def load_shared_tokenizer(specs: Iterable[ModelSpec]):
    specs = list(specs)
    if not specs:
        raise ValueError("No model specs supplied")
    tokenizer_path = specs[0].tokenizer or specs[0].foundation
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    required_sizes = {
        adapter_saved_modules(spec.adapter)["vocab_size"]
        for spec in specs
        if spec.adapter is not None
    }
    required_sizes.discard(None)
    if len(required_sizes) > 1:
        raise ValueError(f"Conflicting adapter vocabulary sizes: {required_sizes}")
    if required_sizes:
        required = next(iter(required_sizes))
        if len(tokenizer) == required - 1:
            added = tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            if added != 1:
                raise RuntimeError("Failed to reproduce LoRI's one-token <PAD> extension")
        if len(tokenizer) != required:
            raise ValueError(f"Tokenizer size {len(tokenizer)} != adapter size {required}")
    return tokenizer


def load_model(
    spec: ModelSpec,
    *,
    tokenizer_size: int,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str | None = None,
    merge_adapter_on_device: bool = False,
):
    """Load one private model instance and merge its adapter if present."""
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(str(spec.foundation), **load_kwargs)
    if model.get_input_embeddings().num_embeddings != tokenizer_size:
        model.resize_token_embeddings(tokenizer_size)
    if spec.adapter is not None:
        from peft import PeftModel

        if merge_adapter_on_device:
            model = model.to(device)
        model = PeftModel.from_pretrained(model, str(spec.adapter))
        model = model.merge_and_unload()
        assert_saved_full_modules_loaded(model, spec.adapter)
    return model.to(device).eval()


def assert_saved_full_modules_loaded(model, adapter_path: Path) -> None:
    """Fail if PEFT silently ignored adapter-bundled embeddings or LM head."""
    weights = adapter_path / "adapter_model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        checks = (
            ("embed_tokens.weight", model.get_input_embeddings().weight),
            ("lm_head.weight", model.get_output_embeddings().weight),
        )
        for suffix, actual in checks:
            matching = [key for key in handle.keys() if key.endswith(suffix)]
            if not matching:
                continue
            expected = handle.get_slice(matching[0])[:2, :16].to(actual.dtype)
            observed = actual[:2, :16].detach().cpu()
            if not torch.equal(observed, expected):
                raise RuntimeError(
                    f"Adapter-bundled {suffix} was not loaded from {adapter_path}"
                )


def assert_tokenizers_identical(path_a: Path, path_b: Path) -> None:
    a = AutoTokenizer.from_pretrained(str(path_a), trust_remote_code=True)
    b = AutoTokenizer.from_pretrained(str(path_b), trust_remote_code=True)
    if a.get_vocab() != b.get_vocab():
        raise ValueError(f"Tokenizer token-to-id mappings differ: {path_a} vs {path_b}")


def assert_kv_compatible(producer, receiver) -> int:
    """Check every structural field that affects cached attention tensors."""
    a, b = producer.config, receiver.config
    fields = (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rope_scaling",
        "attention_bias",
        "use_sliding_window",
        "sliding_window",
    )
    differences = []
    for field in fields:
        left, right = getattr(a, field, None), getattr(b, field, None)
        if field == "sliding_window":
            left = left if getattr(a, "use_sliding_window", False) else None
            right = right if getattr(b, "use_sliding_window", False) else None
        if left != right:
            differences.append(f"{field}: {left!r} vs {right!r}")
    head_a = int(getattr(a, "head_dim", 0) or a.hidden_size // a.num_attention_heads)
    head_b = int(getattr(b, "head_dim", 0) or b.hidden_size // b.num_attention_heads)
    if head_a != head_b:
        differences.append(f"head_dim: {head_a} vs {head_b}")
    if differences:
        raise ValueError("KV-incompatible models: " + "; ".join(differences))
    return int(a.num_hidden_layers)
