"""Typed, side-effect-free primitives for cross-model KV-cache reuse.

The public protocol is deliberately small:

1. tokenize one prompt once;
2. prefill producer and receiver with those exact token IDs;
3. choose the owner of K and V independently at every layer;
4. remove the final prompt position; and
5. let the receiver recompute that boundary token before decoding.

Keeping cache construction separate from decoding makes provenance assignments
directly testable and prevents experiment scripts from implementing subtly
different reuse semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Literal

import torch
from transformers.cache_utils import DynamicCache


Owner = Literal["producer", "receiver"]
TruncationMode = Literal["right", "left", "middle"]


def truncate_prompt_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
    mode: TruncationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate one unpadded prompt while optionally preserving both boundaries."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected matching rank-2 input IDs and attention mask")
    length = int(input_ids.shape[1])
    if length <= max_tokens:
        return input_ids, attention_mask
    if mode == "right":
        indices = torch.arange(max_tokens, device=input_ids.device)
    elif mode == "left":
        indices = torch.arange(length - max_tokens, length, device=input_ids.device)
    elif mode == "middle":
        left = (max_tokens + 1) // 2
        right = max_tokens - left
        indices = torch.cat((
            torch.arange(left, device=input_ids.device),
            torch.arange(length - right, length, device=input_ids.device),
        )) if right else torch.arange(left, device=input_ids.device)
    else:
        raise ValueError(f"Unsupported truncation mode: {mode}")
    return (
        input_ids.index_select(1, indices),
        attention_mask.index_select(1, indices),
    )


@dataclass(frozen=True)
class CacheSources:
    """Layer sets whose K or V tensors come from the producer."""

    producer_k_layers: frozenset[int]
    producer_v_layers: frozenset[int]

    @classmethod
    def native(cls) -> "CacheSources":
        return cls(frozenset(), frozenset())

    @classmethod
    def full(cls, num_layers: int) -> "CacheSources":
        layers = frozenset(range(num_layers))
        return cls(layers, layers)

    @classmethod
    def k_only(cls, num_layers: int) -> "CacheSources":
        return cls(frozenset(range(num_layers)), frozenset())

    @classmethod
    def v_only(cls, num_layers: int) -> "CacheSources":
        return cls(frozenset(), frozenset(range(num_layers)))

    @classmethod
    def restore_receiver_v(
        cls, num_layers: int, restored_layers: AbstractSet[int]
    ) -> "CacheSources":
        """Start with full foreign KV and restore receiver V on selected layers."""
        all_layers = frozenset(range(num_layers))
        restored = frozenset(restored_layers)
        _validate_layers(restored, num_layers)
        return cls(all_layers, all_layers - restored)

    @classmethod
    def restore_receiver_k(
        cls, num_layers: int, restored_layers: AbstractSet[int]
    ) -> "CacheSources":
        """Matched control: restore receiver K while keeping producer V."""
        all_layers = frozenset(range(num_layers))
        restored = frozenset(restored_layers)
        _validate_layers(restored, num_layers)
        return cls(all_layers - restored, all_layers)

    def validate(self, num_layers: int) -> None:
        _validate_layers(self.producer_k_layers, num_layers)
        _validate_layers(self.producer_v_layers, num_layers)


@dataclass(frozen=True)
class PreparedCaches:
    """Prompt IDs and immutable producer/receiver prefill caches."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    producer: object
    receiver: object
    num_layers: int


def _validate_layers(layers: AbstractSet[int], num_layers: int) -> None:
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range [0, {num_layers}): {invalid}")


def layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one layer's K and V across supported Transformers cache APIs."""
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer][0], cache[layer][1]


def cache_length(cache) -> int:
    """Return the cached sequence length without relying on model internals."""
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    key, _ = layer_kv(cache, 0)
    return int(key.shape[-2])


def mix_caches(
    producer_cache,
    receiver_cache,
    sources: CacheSources,
    num_layers: int,
) -> DynamicCache:
    """Build a fresh cache container with explicit per-component provenance.

    Tensor storage is referenced, not copied. A subsequent Transformers forward
    appends by replacing entries in this new container, so the two source cache
    objects remain unchanged.
    """
    sources.validate(num_layers)
    if cache_length(producer_cache) != cache_length(receiver_cache):
        raise ValueError("Producer and receiver caches have different sequence lengths")
    mixed = DynamicCache()
    for layer in range(num_layers):
        producer_k, producer_v = layer_kv(producer_cache, layer)
        receiver_k, receiver_v = layer_kv(receiver_cache, layer)
        key = producer_k if layer in sources.producer_k_layers else receiver_k
        value = producer_v if layer in sources.producer_v_layers else receiver_v
        if key.shape != receiver_k.shape or value.shape != receiver_v.shape:
            raise ValueError(f"KV tensor shape mismatch at layer {layer}")
        mixed.update(key, value, layer)
    return mixed


def truncate_boundary(cache, num_layers: int) -> DynamicCache:
    """Drop the last prompt position so the receiver owns the boundary token."""
    length = cache_length(cache)
    if length < 1:
        raise ValueError("Cannot truncate an empty cache")
    truncated = DynamicCache()
    for layer in range(num_layers):
        key, value = layer_kv(cache, layer)
        truncated.update(
            key[..., :-1, :].contiguous(),
            value[..., :-1, :].contiguous(),
            layer,
        )
    return truncated


def slice_cache_prefix(cache, prefix_length: int, num_layers: int) -> DynamicCache:
    """Return the first ``prefix_length`` positions of every cache layer.

    The helper deliberately rejects an empty prefix because Transformers cache
    implementations do not represent it consistently.  Callers should use a
    normal receiver-native prefill when the receiver owns the complete prompt.
    """
    length = cache_length(cache)
    if prefix_length < 1 or prefix_length > length:
        raise ValueError(
            f"prefix_length must be in [1, {length}], got {prefix_length}"
        )
    prefix = DynamicCache()
    for layer in range(num_layers):
        key, value = layer_kv(cache, layer)
        prefix.update(
            key[..., :prefix_length, :].contiguous(),
            value[..., :prefix_length, :].contiguous(),
            layer,
        )
    return prefix


def prepare_caches(
    producer_model,
    receiver_model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    num_layers: int,
    max_input_tokens: int | None = None,
    truncation_mode: TruncationMode = "right",
) -> PreparedCaches:
    """Tokenize once and prefill both models on identical prompt positions."""
    if max_input_tokens is not None and max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive when provided")
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if max_input_tokens is not None:
        input_ids, attention_mask = truncate_prompt_tensors(
            input_ids, attention_mask, max_input_tokens, truncation_mode
        )
    if input_ids.shape[0] != 1:
        raise ValueError("The reference runtime supports batch size one only")
    if not torch.all(attention_mask == 1):
        raise ValueError("Reference reuse requires an unpadded single prompt")
    with torch.inference_mode():
        producer_out = producer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        if producer_model is receiver_model:
            receiver_out = producer_out
        else:
            receiver_out = receiver_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
    return PreparedCaches(
        input_ids=input_ids,
        attention_mask=attention_mask,
        producer=producer_out.past_key_values,
        receiver=receiver_out.past_key_values,
        num_layers=num_layers,
    )
