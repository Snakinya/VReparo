import copy
import json
from pathlib import Path
import unittest

from vreparo_artifact.characterization import build_tables, validate_data


ROOT = Path(__file__).resolve().parents[1]


class CharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "data/characterization.json").read_text())

    def test_every_paper_table_cell(self):
        expected = json.loads((ROOT / "expected/characterization.json").read_text())
        actual = build_tables(ROOT / "data")
        self.assertEqual(expected, actual)
        self.assertEqual(sum(len(t["rows"]) for t in actual.values()), 32)

    def test_current_comparison_coverage(self):
        comparisons = validate_data(self.data)
        self.assertEqual(len(comparisons), 24)
        self.assertEqual(comparisons[("rq2", "gemma2_2b", "harmbench")]["retained_n"], 121)
        self.assertEqual(comparisons[("rq1", "qwen3_8b", "harmbench")]["retained_n"], 132)
        self.assertEqual(comparisons[("rq2", "qwen3_8b", "harmbench")]["retained_n"], 129)

    def test_frozen_mask_must_match_flags_and_reviews(self):
        data = copy.deepcopy(self.data)
        sample = data["comparisons"][0]["samples"][0]
        sample["retained"] = not sample["retained"]
        with self.assertRaisesRegex(ValueError, "Frozen validity mask"):
            validate_data(data)

    def test_pairing_and_hash_uniqueness(self):
        data = copy.deepcopy(self.data)
        samples = data["comparisons"][0]["samples"]
        samples[1]["prompt_sha256"] = samples[0]["prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            validate_data(data)
        data = copy.deepcopy(self.data)
        data["comparisons"][0]["samples"][0]["scores"].pop()
        with self.assertRaisesRegex(ValueError, "Pairing"):
            validate_data(data)

    def test_scores_are_finite_and_binary_for_harmbench(self):
        for score in (float("nan"), 0.5, 2, True):
            data = copy.deepcopy(self.data)
            data["comparisons"][0]["samples"][0]["scores"][0] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate_data(data)

    def test_invalid_hashes_and_source_paths(self):
        for field, value in (("path", "/private/file.json"), ("path", "../file.json"),
                             ("path", "C:\\private\\file.json"), ("path", "a/./file.json"),
                             ("sha256", "g" * 64), ("sha256", "0" * 63)):
            data = copy.deepcopy(self.data)
            data["sources"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_data(data)
        data = copy.deepcopy(self.data)
        data["comparisons"][0]["samples"][0]["prompt_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "Prompt hash"):
            validate_data(data)

    def test_unknown_fields_and_invalid_collection_types(self):
        for field, value in (("unexpected_text", "not permitted"), ("scores", None),
                             ("validity_flags", {}), ("ordinal", False)):
            data = copy.deepcopy(self.data)
            data["comparisons"][0]["samples"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_data(data)
        data = copy.deepcopy(self.data)
        data["comparisons"][0]["retained_n"] = True
        with self.assertRaisesRegex(ValueError, "Invalid count type"):
            validate_data(data)

    def test_duplicates_and_unknown_screening_labels(self):
        data = copy.deepcopy(self.data)
        data["comparisons"].append(data["comparisons"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate comparison"):
            validate_data(data)
        data = copy.deepcopy(self.data)
        data["sources"].append(data["sources"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            validate_data(data)
        for field, value in (("validity_flags", [["unknown"], []]),
                             ("validity_flags", [["empty", "empty"], []]),
                             ("review_labels", ["unknown", None])):
            data = copy.deepcopy(self.data)
            data["comparisons"][0]["samples"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_data(data)

    def test_export_has_only_whitelisted_record_fields(self):
        expected = {"ordinal", "prompt_sha256", "scores", "validity_flags", "review_labels", "retained"}
        for comparison in self.data["comparisons"]:
            for sample in comparison["samples"]:
                self.assertEqual(set(sample), expected)
        for source in self.data["sources"]:
            self.assertFalse(Path(source["path"]).is_absolute())
            self.assertNotIn("..", Path(source["path"]).parts)


if __name__ == "__main__":
    unittest.main()
