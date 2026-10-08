"""Optional CPU numerical checks, including a tiny randomly initialized Llama."""

import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "Optional Torch/Transformers environment not installed")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def cache(self, values):
        import torch
        from transformers.cache_utils import DynamicCache
        result = DynamicCache()
        for index, value in enumerate(values):
            result.update(torch.ones_like(value) * (index + 1), value, index)
        return result

    def test_identity_centered_fit_matches_direct_solve(self):
        import torch
        from vreparo_artifact.runtime.ridge import ValueRidgeMapper, ValueRidgeStatistics
        torch.manual_seed(12)
        x = torch.randn(1, 2, 13, 4, dtype=torch.float64)
        y = 1.2 * x + 0.4
        stats = ValueRidgeStatistics.empty(layers=1, kv_heads=2, head_dim=4)
        stats.update(self.cache([x]), self.cache([y]), 1)
        fitted = ValueRidgeMapper.fit(stats, ridge=1, identity_centered=True)
        for head in range(2):
            xx, yy = x[0, head], y[0, head]
            xc, yc = xx - xx.mean(0), yy - yy.mean(0)
            eye = torch.eye(4, dtype=torch.float64)
            w = torch.linalg.solve(xc.T @ xc + eye, xc.T @ yc + eye)
            b = yy.mean(0) - xx.mean(0) @ w
            torch.testing.assert_close(fitted.weights[0, head], w.float())
            torch.testing.assert_close(fitted.bias[0, head], b.float())

    def test_correction_preserves_keys_block_and_source(self):
        import torch
        from vreparo_artifact.runtime.cache import layer_kv
        from vreparo_artifact.runtime.correction import ValueCorrection
        value = torch.randn(1, 2, 5, 4).half()
        source = self.cache([value.clone(), value.clone()])
        payload = {"kind": "dense", "weights": torch.eye(4).repeat(2, 2, 1, 1) * 1.3,
                   "bias": torch.ones(2, 2, 4) * 0.03, "active_layers": [1]}
        corrected = ValueCorrection(payload, 0.25).transform_cache(source, 2)
        for layer in range(2):
            self.assertTrue(torch.equal(layer_kv(source, layer)[0], layer_kv(corrected, layer)[0]))
            self.assertTrue(torch.equal(layer_kv(source, layer)[1], value))
        self.assertTrue(torch.equal(layer_kv(corrected, 0)[1], value))
        aligned = (torch.einsum('bhtd,hde->bhte', value.float(), payload['weights'][1]) + 0.03).half()
        expected = (value.float() + 0.25 * (aligned.float() - value.float())).half()
        self.assertTrue(torch.equal(layer_kv(corrected, 1)[1], expected))
        self.assertIs(ValueCorrection(payload, 0).transform_tensor(value, 1), value)

    def test_boundary_and_component_ownership(self):
        import torch
        from vreparo_artifact.runtime.cache import CacheSources, layer_kv, mix_caches, truncate_boundary
        a, b = self.cache([torch.zeros(1, 2, 7, 4)]), self.cache([torch.ones(1, 2, 7, 4)])
        a.key_cache[0].fill_(5)
        mixed = mix_caches(a, b, CacheSources.k_only(1), 1)
        self.assertTrue(torch.equal(layer_kv(mixed, 0)[0], layer_kv(a, 0)[0]))
        self.assertTrue(torch.equal(layer_kv(mixed, 0)[1], layer_kv(b, 0)[1]))
        self.assertEqual(layer_kv(truncate_boundary(mixed, 1), 0)[0].shape[2], 6)
        self.assertEqual(layer_kv(a, 0)[0].shape[2], 7)

    def test_tiny_llama_cache_decode_matches_native_generate(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from vreparo_artifact.runtime.cache import truncate_boundary
        from vreparo_artifact.runtime.generation import greedy_from_boundary_cache
        torch.manual_seed(1)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=64, eos_token_id=None, pad_token_id=0)).eval()
        ids = torch.tensor([[1, 5, 8, 9]])
        with torch.inference_mode():
            cache = model(ids, use_cache=True).past_key_values
            tokens = greedy_from_boundary_cache(model, ids, truncate_boundary(cache, 2),
                                                max_new_tokens=4, eos_token_ids=())
            expected = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=4)[0, ids.shape[1]:].tolist()
        self.assertEqual(list(tokens), expected)

    def test_droid_same_model_prefix_and_suffix(self):
        import torch
        from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM
        from vreparo_artifact.runtime.cache import layer_kv
        from vreparo_artifact.runtime.droidspeak import prepare_droidspeak_caches

        class Tokenizer:
            def __call__(self, text, **kwargs):
                ids = torch.tensor([[1, 4, 8, 2, 9]])
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        torch.manual_seed(3)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=64, eos_token_id=None, pad_token_id=0)).eval()
        with torch.inference_mode():
            prepared = prepare_droidspeak_caches(model, model, Tokenizer(), "synthetic", 3, (0, 1), num_layers=4)
        self.assertEqual(prepared.shared_prefix_tokens, 3)
        for layer in range(4):
            for current, native in zip(layer_kv(prepared.selected, layer), layer_kv(prepared.receiver, layer)):
                torch.testing.assert_close(current, native, rtol=1e-5, atol=1e-6)

    def test_tiny_gemma_mistral_native_cache_parity(self):
        import torch
        from transformers import Gemma2Config, Gemma2ForCausalLM, MistralConfig, MistralForCausalLM
        from vreparo_artifact.runtime.cache import truncate_boundary
        from vreparo_artifact.runtime.generation import greedy_from_boundary_cache
        for config, cls in ((Gemma2Config, Gemma2ForCausalLM), (MistralConfig, MistralForCausalLM)):
            with self.subTest(family=cls.__name__):
                torch.manual_seed(1)
                model = cls(config(vocab_size=37, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=64, sliding_window=32, eos_token_id=None, pad_token_id=0)).eval()
                self.compare_native_cache(model)

    def test_tiny_qwen3_native_cache_parity_when_available(self):
        import transformers
        import torch
        if not hasattr(transformers, "Qwen3ForCausalLM"):
            self.skipTest("Qwen3 requires the separate semantic environment")
        torch.manual_seed(1)
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, eos_token_id=None, pad_token_id=0)).eval()
        self.compare_native_cache(model)

    def compare_native_cache(self, model):
        import torch
        from vreparo_artifact.runtime.cache import truncate_boundary
        from vreparo_artifact.runtime.generation import greedy_from_boundary_cache
        ids = torch.tensor([[1, 5, 8, 9]])
        with torch.inference_mode():
            cache = model(ids, use_cache=True).past_key_values
            tokens = greedy_from_boundary_cache(model, ids, truncate_boundary(cache, 2),
                                                max_new_tokens=4, eos_token_ids=())
            expected = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=4)[0, ids.shape[1]:].tolist()
        self.assertEqual(list(tokens), expected)


if __name__ == "__main__":
    unittest.main()
