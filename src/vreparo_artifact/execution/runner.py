"""One model pair and one cache operator, reused by every experiment."""

from __future__ import annotations

from pathlib import Path

import torch

from ..runtime.boundaries import safety_prefix
from ..runtime.cache import CacheSources, prepare_caches, slice_cache_prefix
from ..runtime.correction import ValueCorrection
from ..runtime.droidspeak import prepare_droidspeak_caches, _decode_from_prompt_cache, _prefill_receiver_suffix
from ..runtime.generation import DecodeResult, decode_condition, terminal_token_ids
from ..runtime.longbench_droidspeak import PROMPT_TEMPLATES, format_longbench
from ..runtime.longbench_qa import format_hotpotqa
from ..runtime.models import ModelSpec, assert_kv_compatible, load_model, load_shared_tokenizer
from ..runtime.prompts import format_instruction


class Runner:
    def __init__(self, config: dict, assets: Path, device: str):
        self.config = config
        self.device = device
        self.droid = config["operator"] == "droid"
        if config["operator"] not in ("direct", "droid"):
            raise ValueError("Unknown cache operator")
        self.group = tuple(config["recompute_group"]) if self.droid else None
        self.protocol = config["protocol"]
        receiver = ModelSpec.from_dict(config["receiver"], assets)
        sender = ModelSpec.from_dict(config["sender"], assets)
        self.tokenizer = load_shared_tokenizer([receiver, sender])
        self.receiver = load_model(receiver, tokenizer_size=len(self.tokenizer), device=device, dtype=torch.float16)
        self.sender = load_model(sender, tokenizer_size=len(self.tokenizer), device=device, dtype=torch.float16)
        self.layers = assert_kv_compatible(self.sender, self.receiver)
        self.stop = terminal_token_ids(self.receiver, self.tokenizer)

    def formatted(self, row: dict, task: str):
        if task == "safety":
            if self.droid:
                text, prefix, _ = safety_prefix(self.tokenizer, row["prompt"], self.protocol)
                return text, prefix, {}
            return format_instruction(self.tokenizer, row["prompt"], self.protocol), None, {}
        if self.droid:
            return format_longbench(self.tokenizer, task, row, prompt_protocol=self.protocol,
                                   max_raw_tokens=self.config["max_raw_tokens"])
        if task == "hotpotqa":
            raw, metadata = format_hotpotqa(self.tokenizer, row, max_raw_tokens=self.config["max_raw_tokens"])
        else:
            raw = PROMPT_TEMPLATES[task].format(**row)
            ids = self.tokenizer(raw, add_special_tokens=False)["input_ids"]
            limit = self.config["max_raw_tokens"]
            metadata = {"original_raw_tokens": len(ids), "middle_truncated": len(ids) > limit}
            if len(ids) > limit:
                raw = self.tokenizer.decode(ids[:limit // 2], skip_special_tokens=True) + self.tokenizer.decode(ids[-limit // 2:], skip_special_tokens=True)
        return format_instruction(self.tokenizer, raw, self.protocol), None, metadata

    def prepare(self, text: str, prefix=None):
        if self.droid:
            return prepare_droidspeak_caches(self.sender, self.receiver, self.tokenizer, text, prefix,
                                            self.group, num_layers=self.layers)
        return prepare_caches(self.sender, self.receiver, self.tokenizer, text, device=self.device,
                              num_layers=self.layers, max_input_tokens=self.config["max_input_tokens"],
                              truncation_mode="middle")

    def native(self, text, budget):
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        ids = self.receiver.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                     pad_token_id=self.tokenizer.eos_token_id,
                                     eos_token_id=list(self.stop))[0, encoded.input_ids.shape[1]:].tolist()
        return DecodeResult(self.tokenizer.decode(ids, skip_special_tokens=True).strip(), tuple(ids), bool(ids and ids[-1] in self.stop))

    def corrected_cache(self, prepared, mapper):
        prefix = slice_cache_prefix(prepared.selected, prepared.shared_prefix_tokens, self.layers)
        return _prefill_receiver_suffix(self.receiver, prepared.input_ids[:, prepared.shared_prefix_tokens:],
                                        mapper.transform_cache(prefix, self.layers))

    def decode(self, prepared, text, condition, budget, mapper=None):
        if self.droid:
            if condition in ("k_only", "v_only"):
                raise ValueError("The component experiment uses full-prompt direct reuse")
            if condition == "native":
                return self.native(text, budget)
            cache = self.corrected_cache(prepared, mapper) if mapper else prepared.selected
            text, ids = _decode_from_prompt_cache(self.receiver, self.tokenizer, prepared.input_ids, cache,
                                                 max_new_tokens=budget)
            return DecodeResult(text, tuple(ids), bool(ids and ids[-1] in self.stop))
        sources = {"native": CacheSources.native, "k_only": lambda: CacheSources.k_only(self.layers),
                   "v_only": lambda: CacheSources.v_only(self.layers)}.get(condition, lambda: CacheSources.full(self.layers))()
        return decode_condition(self.receiver, self.tokenizer, prepared, sources,
                                max_new_tokens=budget, value_mapper=mapper)


def load_mapper(path: Path, alpha: float, layers: int, group, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    active = [i for i in range(layers) if group is None or not group[0] <= i <= group[1]]
    if payload.get("kind") != "dense" or payload.get("active_layers") != active:
        raise ValueError("Mapper does not match this operator's active layers")
    if payload["weights"].shape[0] != layers or not 0 <= alpha <= 1:
        raise ValueError("Invalid mapper dimensions or interpolation strength")
    return ValueCorrection(payload, alpha).to(device)
