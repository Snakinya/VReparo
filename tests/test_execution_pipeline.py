"""CPU-only orchestration tests; no checkpoints, downloads, or GPU allocation."""

import argparse
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vreparo_artifact.execution.inputs import read, write

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class FakeRunner:
    def __init__(self, *args):
        self.layers, self.group = 2, None

    def formatted(self, row, task):
        return row["prompt"], None, {}

    def prepare(self, text, prefix):
        return text

    def decode(self, prepared, text, name, budget, mapper):
        return SimpleNamespace(text="I cannot assist.", token_ids=(1, 2), stopped_on_eos=True)


@unittest.skipUnless(AVAILABLE, "Optional Torch/Transformers environment not installed")
class ExecutionPipelineTests(unittest.TestCase):
    def arguments(self, root, shard=0, shards=1):
        return argparse.Namespace(input=root / "dataset.json", output=root / "run", experiment="rq2",
                                  task="safety", max_new_tokens=8, mapper=None, assets=root, device="cpu",
                                  shard=shard, shards=shards)

    def test_generation_shard_merge_resume_and_input_mismatch(self):
        from vreparo_artifact.execution import experiments
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "dataset.json", [{"id": str(i), "prompt": "Synthetic test"} for i in range(5)])
            config = {"operator": "direct"}
            with patch.object(experiments, "Runner", FakeRunner):
                for shard in range(2):
                    args = self.arguments(root, shard, 2)
                    experiments.generate(args, config)
                    before = (args.output / f"generation_{shard}.json").read_bytes()
                    experiments.generate(args, config)
                    self.assertEqual(before, (args.output / f"generation_{shard}.json").read_bytes())
                merge = argparse.Namespace(input=root / "run", dataset_file=root / "dataset.json", shards=2,
                                           output=root / "merged")
                experiments.merge(merge)
                groups = read(root / "merged/generation.json")["condition_records"]
                self.assertEqual(set(groups), {"native", "k_only", "v_only", "raw"})
                self.assertEqual([r["prompt_id"] for r in groups["native"]], list(map(str, range(5))))
                self.assertTrue(all(r["string_refusal"] for r in groups["raw"]))
                write(root / "dataset.json", [{"id": "other", "prompt": "Different"}])
                with self.assertRaises(ValueError):
                    experiments.generate(self.arguments(root, 0, 2), config)
                with self.assertRaises(ValueError):
                    experiments.merge(merge)

    def test_incomplete_and_duplicate_shards_are_rejected(self):
        from vreparo_artifact.execution import experiments
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "dataset.json", [{"id": str(i), "prompt": "Synthetic test"} for i in range(2)])
            with patch.object(experiments, "Runner", FakeRunner):
                experiments.generate(self.arguments(root), {"operator": "direct"})
            shard = root / "run/generation_0.json"
            saved = read(shard)
            saved["condition_records"]["raw"].append(saved["condition_records"]["raw"][0])
            write(shard, saved)
            merge = argparse.Namespace(input=root / "run", dataset_file=root / "dataset.json", shards=1,
                                       output=root / "merged")
            with self.assertRaises(ValueError):
                experiments.merge(merge)


if __name__ == "__main__":
    unittest.main()
