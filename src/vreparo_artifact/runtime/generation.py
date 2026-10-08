"""Deterministic generation from explicitly provenance-labelled KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .cache import (
    CacheSources,
    cache_length,
    mix_caches,
    slice_cache_prefix,
    truncate_boundary,
)


@dataclass(frozen=True)
class DecodeResult:
    text: str
    token_ids: tuple[int, ...]
    stopped_on_eos: bool


def terminal_token_ids(model, tokenizer, extra: Iterable[int] = ()) -> frozenset[int]:
    ids = {int(token) for token in extra if token is not None}
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for owner in (getattr(model, "generation_config", None), model.config):
        value = getattr(owner, "eos_token_id", None)
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        ids.update(int(token) for token in values if token is not None)
    return frozenset(ids)


def greedy_from_boundary_cache(
    receiver_model,
    prompt_ids: torch.Tensor,
    boundary_cache,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int],
    receiver_residual=None,
) -> tuple[int, ...]:
    """Decode after a Variant-B boundary with explicit masks and positions."""
    if prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("Expected one non-empty prompt")
    stop_ids = frozenset(int(token) for token in eos_token_ids)
    generated: list[int] = []
    current = prompt_ids[:, -1:]
    cache = boundary_cache
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            past_length = cache_length(cache)
            attention_mask = torch.ones(
                (1, past_length + 1), dtype=torch.long, device=current.device
            )
            cache_position = torch.tensor(
                [past_length], dtype=torch.long, device=current.device
            )
            extra = {"output_hidden_states": True} if receiver_residual is not None else {}
            output = receiver_model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                **extra,
            )
            logits = output.logits[:, -1]
            if receiver_residual is not None:
                hidden = output.hidden_states[-1][:, -1, :]
                corrected = receiver_residual.correct(hidden)
                logits = receiver_model.get_output_embeddings()(corrected)
            next_id = int(logits[0].argmax().item())
            generated.append(next_id)
            cache = output.past_key_values
            if next_id in stop_ids:
                break
            current = torch.tensor([[next_id]], device=current.device)
    return tuple(generated)


def decode_condition(
    receiver_model,
    tokenizer,
    prepared,
    sources: CacheSources,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    value_mapper=None,
    receiver_residual=None,
) -> DecodeResult:
    """Decode one cache-provenance condition from a shared prompt prefill."""
    producer_cache = (
        value_mapper.transform_cache(prepared.producer, prepared.num_layers)
        if value_mapper is not None
        else prepared.producer
    )
    mixed = mix_caches(
        producer_cache,
        prepared.receiver,
        sources,
        prepared.num_layers,
    )
    boundary = truncate_boundary(mixed, prepared.num_layers)
    stop_ids = terminal_token_ids(receiver_model, tokenizer, eos_token_ids)
    ids = greedy_from_boundary_cache(
        receiver_model,
        prepared.input_ids,
        boundary,
        max_new_tokens=max_new_tokens,
        eos_token_ids=stop_ids,
        receiver_residual=receiver_residual,
    )
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return DecodeResult(text, ids, bool(ids and ids[-1] in stop_ids))


def generate_native(
    receiver_model,
    tokenizer,
    prepared,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
) -> DecodeResult:
    """Receiver-native canonical decode using the same boundary path as reuse."""
    return decode_condition(
        receiver_model,
        tokenizer,
        prepared,
        CacheSources.native(),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )


def decode_with_receiver_suffix(
    receiver_model,
    tokenizer,
    prepared,
    *,
    receiver_suffix_tokens: int,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    cache_mapper=None,
) -> DecodeResult:
    """Reuse a mapped producer prefix and natively recompute a prompt suffix.

    ``receiver_suffix_tokens=1`` is exactly the validated boundary protocol:
    every prompt position except the last comes from the producer cache and the
    receiver owns the boundary token.  Larger values let the receiver recompute
    a longer contiguous suffix.  If the requested suffix covers the complete
    prompt, this function delegates to the canonical receiver-native path.
    """
    prompt_length = int(prepared.input_ids.shape[1])
    if receiver_suffix_tokens < 1:
        raise ValueError("receiver_suffix_tokens must be positive")
    suffix_length = min(receiver_suffix_tokens, prompt_length)
    prefix_length = prompt_length - suffix_length
    if prefix_length == 0:
        return generate_native(
            receiver_model,
            tokenizer,
            prepared,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    prefix = slice_cache_prefix(prepared.producer, prefix_length, prepared.num_layers)
    if cache_mapper is not None:
        prefix = cache_mapper.transform_cache(prefix, prepared.num_layers)
    suffix_ids = prepared.input_ids[:, prefix_length:]
    stop_ids = terminal_token_ids(receiver_model, tokenizer, eos_token_ids)
    generated: list[int] = []

    with torch.inference_mode():
        positions = torch.arange(
            prefix_length,
            prompt_length,
            dtype=torch.long,
            device=suffix_ids.device,
        )
        output = receiver_model(
            input_ids=suffix_ids,
            attention_mask=torch.ones(
                (1, prompt_length), dtype=torch.long, device=suffix_ids.device
            ),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=prefix,
            use_cache=True,
        )
        next_id = int(output.logits[0, -1].argmax().item())
        generated.append(next_id)
        cache = output.past_key_values

        while len(generated) < max_new_tokens and next_id not in stop_ids:
            past_length = cache_length(cache)
            current = torch.tensor([[next_id]], device=suffix_ids.device)
            position = torch.tensor([past_length], dtype=torch.long, device=current.device)
            output = receiver_model(
                input_ids=current,
                attention_mask=torch.ones(
                    (1, past_length + 1), dtype=torch.long, device=current.device
                ),
                position_ids=position.unsqueeze(0),
                cache_position=position,
                past_key_values=cache,
                use_cache=True,
            )
            next_id = int(output.logits[0, -1].argmax().item())
            generated.append(next_id)
            cache = output.past_key_values

    ids = tuple(generated[:max_new_tokens])
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return DecodeResult(text, ids, bool(ids and ids[-1] in stop_ids))
