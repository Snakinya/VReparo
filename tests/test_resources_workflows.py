"""Portable acquisition, workflow and fresh-report contracts without GPUs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vreparo_artifact.execution import reporting, resources, workflows
from vreparo_artifact.execution.inputs import digest, read, write

ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    def test_cached_download_checks_content_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "source"
            target.write_bytes(b"test")
            with patch.object(resources, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(resources.download("unused", target, digest(target), 4), target)
                with self.assertRaises(ValueError):
                    resources.download("unused", target, "0" * 64)
                with self.assertRaises(ValueError):
                    resources.download("unused", target, size=5)

    def test_reconstruction_preserves_order_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"id": "b", "prompt": "Second"}, {"id": "a", "prompt": "First"}]
            expected = (json.dumps(rows[::-1], indent=2) + "\n").encode()
            recipe = {"filename": "test.json", "source": "synthetic", "member": None,
                      "rows": [resources.canonical(r) for r in rows[::-1]], "n": 2,
                      "sha256": hashlib.sha256(expected).hexdigest(), "wrapper_json": "null",
                      "wrapper_order": None, "serialization": {"sort_keys": False, "ensure_ascii": True, "newline": "\n"}}
            write(root / "experiments/resources.json", {"datasets": {"test": recipe},
                  "sources": {"synthetic": {"format": "json", "url": "unused", "sha256": "unused"}}})
            with patch.object(resources, "download"), patch.object(resources, "source_rows", return_value=(rows, None)):
                resources.prepare_data(root, root / "assets")
                self.assertEqual((root / "assets/data/test.json").read_bytes(), expected)
                resources.prepare_data(root, root / "assets")
                (root / "assets/data/test.json").write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    resources.prepare_data(root, root / "assets")
                recipe["rows"] = ["missing"]
                write(root / "experiments/resources.json", {"datasets": {"test": recipe},
                      "sources": {"synthetic": {"format": "json", "url": "unused", "sha256": "unused"}}})
                with self.assertRaises(ValueError):
                    resources.prepare_data(root, root / "other")

    def test_model_download_storage_preflight_and_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            content = b"{}"
            write(root / "experiments/resources.json", {"models": {"models/test": {
                "provider": "modelscope", "repo": "example/test", "revisions_by_file": {"config.json": "pin"},
                "files": {"config.json": {"size": 2, "sha256": hashlib.sha256(content).hexdigest()}}}}})
            assets = root / "assets"
            plan = resources.download_models(root, assets, ["models/test"], dry_run=True)
            self.assertEqual(plan["missing_weight_and_tokenizer_bytes"], 2)
            self.assertFalse(assets.exists())
            with patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "Insufficient storage"):
                    resources.download_models(root, assets, ["models/test"])
            with self.assertRaisesRegex(ValueError, "missing"):
                resources.download_models(root, assets, ["models/test"], verify_only=True)

            def fake_download(url, path, expected, size):
                self.assertIn("Revision=pin", url)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            with patch.object(resources, "download", side_effect=fake_download):
                resources.download_models(root, assets, ["models/test"])
            resources.download_models(root, assets, ["models/test"], verify_only=True)
            (assets / "models/test/config.json").write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "differs"):
                resources.download_models(root, assets, ["models/test"], verify_only=True)


class WorkflowTests(unittest.TestCase):
    def recipe(self, experiment, *arguments):
        return workflows.plan(ROOT, experiment, workflows.parser(experiment).parse_args(arguments))

    def test_all_experiments_reference_pinned_resources(self):
        resources_manifest = read(ROOT / "experiments/resources.json")
        for experiment in ("rq1", "rq2", "mitigation", "quality", "calibration", "performance", "profile"):
            with self.subTest(experiment=experiment):
                recipe = self.recipe(experiment, "--pair", "all", "--case", "all")
                self.assertTrue(set(recipe["datasets"]) <= set(resources_manifest["datasets"]))
                self.assertTrue(set(recipe["models"]) <= set(resources_manifest["models"]))
                self.assertTrue(recipe["commands"])
                from vreparo_artifact.execution.cli import add_arguments
                cli = argparse.ArgumentParser()
                add_arguments(cli.add_subparsers(dest="command", required=True))
                for command in recipe["commands"]:
                    cli.parse_args(command[2:])
                    self.assertEqual(command[2], "experiment")
                    self.assertTrue(command[-1])
                    if command[3] == "generate":
                        target = command[command.index("--output") + 1]
                        is_qwen = "/qwen3_" in target
                        self.assertEqual(".venv-semantic" in command[0], is_qwen)

    def test_recipe_shapes(self):
        self.assertEqual(len(self.recipe("rq1")["commands"]), 4)
        self.assertEqual(len(self.recipe("rq2")["commands"]), 2)
        self.assertEqual(len(self.recipe("mitigation")["commands"]), 7)
        self.assertEqual(len(self.recipe("quality")["commands"]), 4)
        self.assertEqual(len(self.recipe("calibration")["commands"]), 3)
        self.assertEqual(len(self.recipe("performance")["commands"]), 2)
        self.assertEqual(self.recipe("profile", "--case", "all")["targets"], ["code_droid", "base_droid"])
        self.assertEqual(self.recipe("profile")["targets"], ["code_droid"])
        with self.assertRaises(ValueError):
            self.recipe("profile", "--case", "base_direct")

    def test_standalone_dry_run_has_no_asset_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder) / "absent"
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_mitigation.py"),
                                     "--dry-run", "--assets", str(assets)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["experiment"], "mitigation")
            self.assertFalse(assets.exists())


class ReportingTests(unittest.TestCase):
    def row(self, response="A complete synthetic response."):
        return {"prompt_id": "1", "prompt": "Synthetic request", "response": response, "token_ids": [1, 2]}

    def test_hash_bound_screening_and_fresh_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = self.row()
            prior = {reporting.hashed(row["prompt"]): {"response_sha256": reporting.hashed(row["response"]), "flags": [], "review_label": "A"}}
            write(root / "data/screening_reference.json", {"comparisons": {"rq1/test/harmbench": {"native": prior, "raw": prior}}})
            groups = {"native": [row], "raw": [dict(row)]}
            retained, pending, excluded = reporting.screened_indices(root, "rq1", "test", "harmbench", groups)
            self.assertEqual((retained, pending, excluded), ([0], [], []))
            groups["raw"][0]["response"] = "A different complete response."
            retained, pending, _ = reporting.screened_indices(root, "rq1", "test", "harmbench", groups)
            self.assertEqual(len(pending), 1)
            review = root / "review.json"
            write(review, [dict(pending[0], valid=True)])
            decisions = reporting.review_decisions(review)
            self.assertEqual(reporting.screened_indices(root, "rq1", "test", "harmbench", groups, decisions)[1], [])
            write(review, [dict(pending[0], valid="yes")])
            with self.assertRaises(ValueError):
                reporting.review_decisions(review)

    def test_score_alignment_and_units(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generated = root / "generated.json"
            groups = {"native": [self.row()], "raw": [self.row()]}
            write(generated, {"condition_records": groups})
            values = {"native": [{"prompt_id": "1", "score": 0}], "raw": [{"prompt_id": "1", "score": 1}]}
            scored = root / "scored.json"
            write(scored, {"state": "completed", "input_sha256": digest(generated), "conditions": values})
            self.assertEqual(reporting.scores_for(scored, generated), values)
            hb = reporting.safety_rows("test", "harmbench159", groups, values, [0])
            sr = reporting.safety_rows("test", "strongreject", groups, values, [0])
            self.assertEqual((hb[1]["delta"], hb[1]["delta_unit"]), (100, "pp"))
            self.assertEqual((sr[1]["delta"], sr[1]["delta_unit"]), (1, "score"))
            values["raw"][0]["score"] = -1
            with self.assertRaises(ValueError):
                reporting.safety_rows("test", "harmbench159", groups, values, [0])
            values["raw"][0]["prompt_id"] = "bad"
            write(scored, {"state": "completed", "input_sha256": digest(generated), "conditions": values})
            with self.assertRaises(ValueError):
                reporting.scores_for(scored, generated)

    def test_complete_fresh_mitigation_report_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dataset in ("harmbench", "strongreject", "orbench"):
                generation = root / f"outputs/experiments/code_direct/mitigation/{dataset}/generation_0.json"
                groups = {name: [self.row()] for name in ("native", "raw", "vreparo")}
                write(generation, {"state": "completed", "condition_records": groups})
                values = {name: [{"prompt_id": "1", **({"label": "INVALID"} if dataset == "orbench" else {"score": 0})}] for name in groups}
                write(generation.parent / f"scored/{dataset}.json", {
                    "state": "completed", "input_sha256": digest(generation), "conditions": values})
            summary = reporting.summarize(root, "mitigation", ["code_direct"])
            self.assertEqual(summary["rows"], 9)
            self.assertEqual(summary["status"], "needs_judgment_review")
            self.assertTrue((root / summary["report"] / "summary.csv").is_file())


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "Optional Torch/Transformers environment not installed")
class ScoringResumeTests(unittest.TestCase):
    def test_completed_scoring_skips_loading_and_changed_model_fails(self):
        from vreparo_artifact.execution import scoring
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "model/config.json", {"test": 1})
            write(root / "generated.json", {"state": "completed", "condition_records": {"native": [
                {"prompt_id": "1", "prompt": "Test", "response": "Synthetic answer"}]}})
            args = argparse.Namespace(input=root / "generated.json", output=root / "scored", model=root / "model",
                                      base_model=None, evaluator="harmbench", batch_size=8)
            tokenizer = SimpleNamespace(pad_token_id=0)
            model = SimpleNamespace(eval=lambda: "model")
            with patch.object(scoring.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                 patch.object(scoring.AutoModelForCausalLM, "from_pretrained", return_value=model) as loader, \
                 patch.object(scoring, "hf_scores", return_value=[{"score": 0}]):
                scoring.score(args)
                self.assertEqual(loader.call_count, 1)
                self.assertTrue(scoring.score(args)["resumed"])
                self.assertEqual(loader.call_count, 1)
                write(root / "model/config.json", {"test": 2})
                with self.assertRaises(ValueError):
                    scoring.score(args)


if __name__ == "__main__":
    unittest.main()
