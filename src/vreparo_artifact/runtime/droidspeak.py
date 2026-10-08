"""DroidSpeak-style contiguous receiver-block cache reuse.

This module implements the cache semantics used by DroidSpeak-style selective
recomputation experiments for same-architecture model variants.  A producer
prefills the shared prefix, a contiguous receiver layer block may be recomputed
from the producer's intermediate state, producer KV is reused outside that
block, and the receiver processes the non-shared prompt suffix before decoding.

The implementation is deliberately local to our current package rather than
importing the frozen paper-v1 reproduction tree, whose project root is fixed to
that archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers.cache_utils import DynamicCache

try:
    from transformers.masking_utils import create_causal_mask as _create_causal_mask
except ImportError:  # Transformers 4.45 uses LlamaModel._update_causal_mask.
    _create_causal_mask = None

from .cache import cache_length, layer_kv
from .generation import greedy_from_boundary_cache, terminal_token_ids


LayerGroup = tuple[int, int]


@dataclass(frozen=True)
class PreparedDroidCaches:
    """Prompt caches for one fixed DroidSpeak recomputation operator."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    receiver: object
    selected: object
    num_layers: int
    shared_prefix_tokens: int
    recompute_group: LayerGroup


def _receiver_causal_mask(
    receiver_body,
    attention_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
):
    """Build the Llama decoder mask across supported Transformers versions."""
    legacy_update = getattr(receiver_body, "_update_causal_mask", None)
    if callable(legacy_update):
        return legacy_update(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            False,
        )
    if _create_causal_mask is None:
        raise RuntimeError(
            "This Transformers version exposes neither LlamaModel._update_causal_mask "
            "nor transformers.masking_utils.create_causal_mask"
        )
    return _create_causal_mask(
        config=receiver_body.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def decoder_layers(model) -> torch.nn.ModuleList:
    """Return the decoder layer container for supported HF causal LMs."""
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise TypeError(
        f"Unsupported model layout {type(model).__name__}: "
        "expected model.layers or transformer.h"
    )


def validate_group(group: LayerGroup | None, num_layers: int) -> None:
    if group is None:
        return
    start, end = group
    if start < 0 or end < start or end >= num_layers:
        raise ValueError(f"Invalid inclusive recompute group {group}")


def _copy_cache(cache, num_layers: int) -> DynamicCache:
    copied = DynamicCache()
    for layer in range(num_layers):
        key, value = layer_kv(cache, layer)
        copied.update(key, value, layer)
    return copied


def _mix_sender_and_block(
    sender_cache,
    block_cache,
    group: LayerGroup,
    num_layers: int,
) -> DynamicCache:
    start, end = group
    mixed = DynamicCache()
    for layer in range(num_layers):
        owner = block_cache if start <= layer <= end else sender_cache
        key, value = layer_kv(owner, layer)
        mixed.update(key, value, layer)
    return mixed


def _prefill_receiver_suffix(receiver_model, suffix_ids: torch.Tensor, past_key_values):
    """Append receiver-owned prompt suffix positions to an imported cache."""
    if suffix_ids.shape[1] == 0:
        return past_key_values
    with torch.inference_mode():
        output = receiver_model(
            input_ids=suffix_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return output.past_key_values


def _decode_from_prompt_cache(
    receiver_model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_cache,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    receiver_residual=None,
):
    # ``greedy_from_boundary_cache`` expects a cache that is missing the final
    # prompt token.  DroidSpeak suffix prefill has already processed the full
    # prompt, so we drop one position here and let the receiver own the boundary
    # token in the same validated Variant-B decode path used elsewhere.
    num_layers = int(receiver_model.config.num_hidden_layers)
    boundary = DynamicCache()
    if cache_length(prompt_cache) < 1:
        raise ValueError("Cannot decode from an empty prompt cache")
    for layer in range(num_layers):
        key, value = layer_kv(prompt_cache, layer)
        boundary.update(key[..., :-1, :].contiguous(), value[..., :-1, :].contiguous(), layer)
    stop_ids = terminal_token_ids(receiver_model, tokenizer, eos_token_ids)
    ids = greedy_from_boundary_cache(
        receiver_model,
        input_ids,
        boundary,
        max_new_tokens=max_new_tokens,
        eos_token_ids=stop_ids,
        receiver_residual=receiver_residual,
    )
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), list(ids)


def _receiver_block_cache_from_sender_state(
    producer_model,
    receiver_model,
    prefix_ids: torch.Tensor,
    group: LayerGroup,
    *,
    num_layers: int,
) -> tuple[object, DynamicCache]:
    """Return producer prefix KV and receiver-block KV from producer E state."""
    validate_group(group, num_layers)
    prefix_len = int(prefix_ids.shape[1])
    attention_mask = torch.ones_like(prefix_ids)
    producer_body = getattr(producer_model, "model", None)
    receiver_body = getattr(receiver_model, "model", None)
    if producer_body is None or receiver_body is None:
        raise TypeError("DroidSpeak runtime currently requires Llama-style .model bodies")
    receiver_layers = decoder_layers(receiver_model)
    start, end = group
    with torch.inference_mode():
        producer_out = producer_body(
            input_ids=prefix_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        producer_cache = producer_out.past_key_values
        hidden_states = producer_out.hidden_states[start]
        block_cache = DynamicCache()
        cache_position = torch.arange(prefix_len, device=prefix_ids.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = _receiver_causal_mask(
            receiver_body,
            attention_mask,
            hidden_states,
            cache_position,
            block_cache,
            position_ids,
        )
        position_embeddings = receiver_body.rotary_emb(hidden_states, position_ids)
        for layer_idx in range(start, end + 1):
            layer_outputs = receiver_layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=block_cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
    return producer_cache, block_cache


def prepare_droidspeak_caches(
    producer_model,
    receiver_model,
    tokenizer,
    prompt: str,
    shared_prefix_tokens: int,
    recompute_group: LayerGroup,
    *,
    num_layers: int,
) -> PreparedDroidCaches:
    """Prefill native and selected-recompute caches for matched calibration.

    The selected cache follows the exact same operator as online DroidSpeak
    generation: producer KV outside ``recompute_group``, receiver-block KV
    inside it, followed by receiver-owned prefill of the non-shared suffix.
    """
    validate_group(recompute_group, num_layers)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(
        receiver_model.device
    )
    input_ids = encoded.input_ids
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[0] != 1 or not torch.all(attention_mask == 1):
        raise ValueError("DroidSpeak calibration requires one unpadded prompt")
    prefix_len = int(shared_prefix_tokens)
    prompt_len = int(input_ids.shape[1])
    if not 1 <= prefix_len < prompt_len:
        raise ValueError(f"Invalid shared prefix {prefix_len}/{prompt_len}")

    producer_cache, block_cache = _receiver_block_cache_from_sender_state(
        producer_model,
        receiver_model,
        input_ids[:, :prefix_len],
        recompute_group,
        num_layers=num_layers,
    )
    selected_cache = _mix_sender_and_block(
        producer_cache,
        block_cache,
        recompute_group,
        num_layers,
    )
    selected_cache = _prefill_receiver_suffix(
        receiver_model,
        input_ids[:, prefix_len:],
        selected_cache,
    )
    with torch.inference_mode():
        receiver_output = receiver_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return PreparedDroidCaches(
        input_ids=input_ids,
        attention_mask=attention_mask,
        receiver=receiver_output.past_key_values,
        selected=selected_cache,
        num_layers=num_layers,
        shared_prefix_tokens=prefix_len,
        recompute_group=recompute_group,
    )


def generate_droidspeak_conditions(
    producer_model,
    receiver_model,
    tokenizer,
    prompt: str,
    shared_prefix_tokens: int,
    recompute_group: LayerGroup,
    *,
    num_layers: int,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    cache_mapper=None,
    receiver_residual=None,
    include_full: bool = True,
) -> dict[str, tuple[str, list[int]]]:
    """Generate full reuse, selected reuse, and optionally corrected selected reuse."""
    validate_group(recompute_group, num_layers)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(receiver_model.device)
    input_ids = encoded.input_ids
    n_tokens = int(input_ids.shape[1])
    prefix_len = int(shared_prefix_tokens)
    if not 1 <= prefix_len < n_tokens:
        raise ValueError(f"Invalid shared prefix {prefix_len}/{n_tokens}")
    prefix_ids = input_ids[:, :prefix_len]
    receiver_suffix_ids = input_ids[:, prefix_len:]

    producer_cache, block_cache = _receiver_block_cache_from_sender_state(
        producer_model,
        receiver_model,
        prefix_ids,
        recompute_group,
        num_layers=num_layers,
    )
    results: dict[str, tuple[str, list[int]]] = {}
    if include_full:
        full_cache = _copy_cache(producer_cache, num_layers)
        full_cache = _prefill_receiver_suffix(receiver_model, receiver_suffix_ids, full_cache)
        results["full_reuse"] = _decode_from_prompt_cache(
            receiver_model,
            tokenizer,
            input_ids,
            full_cache,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    selected_cache = _mix_sender_and_block(
        producer_cache,
        block_cache,
        recompute_group,
        num_layers,
    )
    selected_cache = _prefill_receiver_suffix(receiver_model, receiver_suffix_ids, selected_cache)
    results["selected"] = _decode_from_prompt_cache(
        receiver_model,
        tokenizer,
        input_ids,
        selected_cache,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )

    if receiver_residual is not None:
        results["selected_residual"] = _decode_from_prompt_cache(
            receiver_model,
            tokenizer,
            input_ids,
            selected_cache,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            receiver_residual=receiver_residual,
        )

    if cache_mapper is not None:
        mapped_producer = cache_mapper.transform_cache(producer_cache, num_layers)
        corrected_cache = _mix_sender_and_block(
            mapped_producer,
            block_cache,
            recompute_group,
            num_layers,
        )
        corrected_cache = _prefill_receiver_suffix(
            receiver_model,
            receiver_suffix_ids,
            corrected_cache,
        )
        results["selected_corrected"] = _decode_from_prompt_cache(
            receiver_model,
            tokenizer,
            input_ids,
            corrected_cache,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
    return results


def profile_two_layer_groups(
    producer_model,
    receiver_model,
    tokenizer,
    prompt: str,
    shared_prefix_tokens: int,
    *,
    num_layers: int,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
) -> dict[str, tuple[str, list[int]]]:
    """Generate all contiguous two-layer-boundary receiver-block groups."""
    if num_layers % 2:
        raise ValueError("Two-layer profiling requires an even layer count")
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(receiver_model.device)
    input_ids = encoded.input_ids
    n_tokens = int(input_ids.shape[1])
    prefix_len = int(shared_prefix_tokens)
    if not 1 <= prefix_len < n_tokens:
        raise ValueError(f"Invalid shared prefix {prefix_len}/{n_tokens}")
    prefix_ids = input_ids[:, :prefix_len]
    receiver_suffix_ids = input_ids[:, prefix_len:]
    attention_mask = torch.ones_like(prefix_ids)

    producer_body = getattr(producer_model, "model", None)
    receiver_body = getattr(receiver_model, "model", None)
    if producer_body is None or receiver_body is None:
        raise TypeError("DroidSpeak profiler currently requires Llama-style .model bodies")
    receiver_layers = decoder_layers(receiver_model)
    results: dict[str, tuple[str, list[int]]] = {}
    stop_ids = terminal_token_ids(receiver_model, tokenizer, eos_token_ids)

    with torch.inference_mode():
        producer_out = producer_body(
            input_ids=prefix_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        producer_cache = producer_out.past_key_values
        producer_states = producer_out.hidden_states
        cache_position = torch.arange(prefix_len, device=prefix_ids.device)
        position_ids = cache_position.unsqueeze(0)
        for start in range(0, num_layers, 2):
            hidden_states = producer_states[start]
            block_cache = DynamicCache()
            causal_mask = _receiver_causal_mask(
                receiver_body,
                attention_mask,
                hidden_states,
                cache_position,
                block_cache,
                position_ids,
            )
            position_embeddings = receiver_body.rotary_emb(hidden_states, position_ids)
            for layer_idx in range(start, num_layers):
                layer_outputs = receiver_layers[layer_idx](
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=block_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
                if layer_idx % 2 == 0:
                    continue
                group = (start, layer_idx)
                mixed = _mix_sender_and_block(producer_cache, block_cache, group, num_layers)
                mixed = _prefill_receiver_suffix(receiver_model, receiver_suffix_ids, mixed)
                text, ids = _decode_from_prompt_cache(
                    receiver_model,
                    tokenizer,
                    input_ids,
                    mixed,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=stop_ids,
                )
                results[f"{start}_{layer_idx}"] = (text, ids)
    expected = (num_layers // 2) * (num_layers // 2 + 1) // 2
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} groups, produced {len(results)}")
    return results
