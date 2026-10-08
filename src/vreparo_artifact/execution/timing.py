"""Resident-cache timing with the paper's boundaries, warmups, and repetitions."""

import statistics
import time

import torch
from transformers.cache_utils import DynamicCache

from .experiments import checkpoint, signature
from .inputs import identifier, rows, validate_rows, write
from .runner import Runner, load_mapper
from ..runtime.cache import layer_kv, slice_cache_prefix, truncate_boundary
from ..runtime.droidspeak import _mix_sender_and_block, _prefill_receiver_suffix, _receiver_causal_mask
from ..runtime.generation import greedy_from_boundary_cache


@torch.inference_mode()
def measure(args, config):
    data = rows(args.input)
    validate_rows(data, "hotpotqa")
    if len(data) != 50:
        raise ValueError("Paper timing uses the 50-example profile set")
    if config["operator"] == "droid" and config["recompute_group"] != [0, 1]:
        raise ValueError("The paper resident-embedding timing path is defined for block [0,1]")
    path = args.output / "timing.json"
    result = checkpoint(path, signature(config, {"profile": args.input, "mapper": args.mapper}), records=[])
    if result["state"] == "completed":
        return result["means"]
    runner = Runner(config, args.assets, args.device)
    mapper = load_mapper(args.mapper, config["alpha"], runner.layers, runner.group, args.device)
    names = ["native", "raw", "vreparo"]

    def timed(function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000, value

    for row_index, row in enumerate(data):
        if row_index < len(result["records"]):
            continue
        torch.cuda.empty_cache()
        text, prefix, _ = runner.formatted(row, "hotpotqa")
        prepared = runner.prepare(text, prefix)
        ids = prepared.input_ids
        cached = prepared.selected if runner.droid else prepared.producer
        resident_e = resident_kv = None
        if runner.droid:
            prefix_ids = ids[:, :prefix]
            sender = runner.sender.model(input_ids=prefix_ids, use_cache=True, output_hidden_states=True, return_dict=True)
            resident_e, resident_kv = sender.hidden_states[0], sender.past_key_values
            del sender

        def raw_cache():
            if not runner.droid:
                return cached
            body = runner.receiver.model
            block = DynamicCache()
            positions = torch.arange(prefix, device=ids.device)
            position_ids = positions.unsqueeze(0)
            hidden = resident_e
            mask = _receiver_causal_mask(body, torch.ones_like(prefix_ids), hidden, positions, block, position_ids)
            embeddings = body.rotary_emb(hidden, position_ids)
            for layer in range(runner.group[0], runner.group[1] + 1):
                hidden = body.layers[layer](hidden, attention_mask=mask, position_ids=position_ids,
                                            past_key_value=block, output_attentions=False, use_cache=True,
                                            cache_position=positions, position_embeddings=embeddings)[0]
            mixed = _mix_sender_and_block(resident_kv, block, runner.group, runner.layers)
            return _prefill_receiver_suffix(runner.receiver, ids[:, prefix:], mixed)

        def mapped_cache(cache, correction):
            if correction is None:
                return cache
            if not runner.droid:
                return correction.transform_cache(cache, runner.layers)
            part = slice_cache_prefix(cache, prefix, runner.layers)
            return _prefill_receiver_suffix(runner.receiver, ids[:, prefix:], correction.transform_cache(part, runner.layers))

        def decode(cache, budget):
            return greedy_from_boundary_cache(runner.receiver, ids, truncate_boundary(cache, runner.layers),
                                              max_new_tokens=budget, eos_token_ids=())

        check = raw_cache()
        if any(not torch.equal(a, b) for layer in range(runner.layers) for a, b in zip(layer_kv(check, layer), layer_kv(cached, layer))):
            raise ValueError("Timing cache differs from generation cache")
        del check
        record = {"prompt_id": identifier(row), "prompt_tokens": ids.shape[1], "ttft_ms": {}, "decode_ms_per_token": {}}
        order = names[row_index % len(names):] + names[:row_index % len(names)]
        for name in order:
            correction = mapper if name == "vreparo" else None
            ready = prepared.receiver if name == "native" else mapped_cache(cached, correction)

            def first():
                if name == "native":
                    return runner.receiver.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                                    max_new_tokens=1, pad_token_id=runner.tokenizer.eos_token_id,
                                                    eos_token_id=list(runner.stop))[0, -1:].tolist()
                return decode(mapped_cache(raw_cache(), correction), 1)

            if name != "native" and tuple(first()) != runner.decode(prepared, text, name, 1, correction).token_ids:
                raise ValueError("Timing first token differs from generation")
            for _ in range(5):
                first()
                decode(ready, 32)
            ttft, continuation = [], []
            for _ in range(5):
                elapsed, _ = timed(first)
                ttft.append(elapsed)
                elapsed, tokens = timed(lambda: decode(ready, 32))
                if len(tokens) != 32:
                    raise ValueError("Timing decode must generate 32 forced tokens")
                continuation.append(elapsed / 32)
            record["ttft_ms"][name], record["decode_ms_per_token"][name] = ttft, continuation
            del ready
        result["records"].append(record)
        write(path, result)
        print(f"timing {row_index + 1}/50", flush=True)
        del prepared, cached, resident_e, resident_kv
    result["means"] = {metric: {name: statistics.mean(statistics.mean(row[metric][name]) for row in result["records"]) for name in names}
                       for metric in ("ttft_ms", "decode_ms_per_token")}
    result.update(state="completed", gpu=torch.cuda.get_device_name(),
                  scope="Resident sender KV/E; excludes sender prefill, transfer, tokenization and scheduling. Decode includes boundary step.")
    write(path, result)
    return result["means"]
