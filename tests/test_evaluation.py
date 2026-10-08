import copy
import json
import tempfile
import unittest
from pathlib import Path

from vreparo_artifact.evaluation import build_tables, calibration_rows, validate_data


ARTIFACT = Path(__file__).resolve().parents[1]


class EvaluationTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ARTIFACT / "data/evaluation.json").read_text())

    def changed(self, path, value):
        payload = copy.deepcopy(self.payload)
        target = payload
        for field in path[:-1]:
            target = target[field]
        target[path[-1]] = value
        return payload

    def test_every_displayed_evaluation_value_matches_paper(self):
        expected = json.loads((ARTIFACT / "expected/evaluation.json").read_text())
        actual = build_tables(ARTIFACT / "data")
        self.assertEqual(set(actual), {"table_03", "table_04", "table_07", "table_08", "table_09", "table_10"})
        for table_id, table in expected.items():
            with self.subTest(table=table_id):
                self.assertEqual(actual[table_id], table)

    def test_rejects_nonfinite_scores(self):
        payload = json.loads((ARTIFACT / "data/evaluation.json").read_text())
        payload["cases"]["code_direct"]["safety"]["harmbench"]["native"][0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "finite"):
                build_tables(root)

    def test_rejects_duplicate_prompt_ids(self):
        payload = json.loads((ARTIFACT / "data/evaluation.json").read_text())
        observations = payload["cases"]["code_direct"]["safety"]["harmbench"]
        observations["prompt_ids"][1] = observations["prompt_ids"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_tables(root)

    def test_development_selection_preserves_frozen_alpha(self):
        payload = json.loads((ARTIFACT / "data/evaluation.json").read_text())
        for case_id, case in payload["cases"].items():
            chosen = [row for row in calibration_rows(case["calibration"]) if row["selected"]]
            self.assertEqual(len(chosen), 1)
            self.assertEqual(chosen[0]["alpha"], "1.0" if case_id.startswith("code") else "0.25")
            changed = copy.deepcopy(case["calibration"])
            changed["selected_alpha"] = 0.5
            with self.assertRaisesRegex(ValueError, "Frozen alpha"):
                calibration_rows(changed)

    def test_unscorable_refusal_labels_keep_fixed_denominator(self):
        payload = json.loads((ARTIFACT / "data/evaluation.json").read_text())
        observations = payload["cases"]["base_direct"]["orbench"]["raw"]
        self.assertEqual(len(observations), 200)
        self.assertEqual(observations.count("U"), 1)
        self.assertEqual(observations.count("F") / 200 * 100, 12.5)

    def test_rejects_extra_fields_at_every_schema_layer(self):
        paths = [
            (), ("cases",), ("sources", 0), ("cases", "code_direct"),
            ("cases", "code_direct", "safety"),
            ("cases", "code_direct", "safety", "harmbench"),
            ("cases", "code_direct", "quality"),
            ("cases", "code_direct", "quality", "hotpotqa"),
            ("cases", "code_direct", "orbench"),
            ("cases", "code_direct", "timing", 0),
            ("cases", "code_direct", "timing", 0, "ttft_ms"),
            ("cases", "code_direct", "timing", 0, "decode_ms_per_token"),
            ("cases", "code_droid", "timing", 0),
            ("cases", "code_direct", "calibration"),
            ("cases", "code_direct", "calibration", "refusal"),
        ]
        for path in paths:
            with self.subTest(layer=path):
                with self.assertRaisesRegex(ValueError, "Invalid fields"):
                    validate_data(self.changed((*path, "raw_text"), "not permitted"))

    def test_rejects_missing_fields_and_malformed_containers(self):
        for path, value in [
            (("cases",), []), (("sources",), {}), (("sources", 0), []),
            (("cases", "code_direct"), []),
            (("cases", "code_direct", "timing"), {}),
            (("cases", "code_direct", "timing", 0), None),
            (("cases", "code_direct", "safety", "harmbench", "native"), {}),
            (("cases", "code_direct", "calibration", "labels"), {}),
            (("cases", "code_direct", "calibration", "subtypes"), {}),
            (("cases", "code_direct", "calibration", "refusal"), []),
            (("cases", "code_direct", "orbench", "raw"), {}),
        ]:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_data(self.changed(path, value))
        payload = copy.deepcopy(self.payload)
        del payload["cases"]["code_direct"]["safety"]["harmbench"]["native"]
        with self.assertRaisesRegex(ValueError, "Invalid fields"):
            validate_data(payload)
        with self.assertRaisesRegex(ValueError, "Invalid fields"):
            validate_data([])

    def test_rejects_boolean_numbers_and_counts(self):
        paths = [
            ("schema_version",),
            ("cases", "code_direct", "safety", "harmbench", "native", 0),
            ("cases", "code_direct", "quality", "hotpotqa", "raw", 0),
            ("cases", "code_direct", "timing", 0, "prompt_tokens"),
            ("cases", "code_direct", "timing", 0, "ttft_ms", "native", 0),
            ("cases", "code_droid", "timing", 0, "shared_prefix_tokens"),
            ("cases", "code_direct", "calibration", "fitting_texts"),
            ("cases", "code_direct", "calibration", "positions_per_head"),
            ("cases", "code_direct", "calibration", "selected_alpha"),
            ("cases", "code_direct", "calibration", "refusal", "native", 0),
        ]
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_data(self.changed(path, True))

    def test_rejects_malformed_or_substituted_prompt_ids(self):
        for value in (None, True, 1, {}, [], "", "contains spaces", "line\nbreak", "a" * 65, "unexpected_id"):
            with self.subTest(value=value):
                path = ("cases", "code_direct", "safety", "harmbench", "prompt_ids", 0)
                with self.assertRaises(ValueError):
                    validate_data(self.changed(path, value))
        for path in [
            ("cases", "code_direct", "quality", "hotpotqa", "prompt_ids", 0),
            ("cases", "code_direct", "orbench", "prompt_ids", 0),
            ("cases", "code_direct", "calibration", "prompt_ids", 0),
            ("cases", "code_direct", "timing", 0, "prompt_id"),
        ]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "Unexpected prompt IDs"):
                    validate_data(self.changed(path, "unexpected_id"))

    def test_rejects_malformed_or_duplicate_source_metadata(self):
        for value in (None, True, [], "", "../source.json", "/source.json", "C:\\source.json", "a//b.json", "a/./b.json", "a\x00b", "unknown.json"):
            with self.subTest(path=value):
                with self.assertRaises(ValueError):
                    validate_data(self.changed(("sources", 0, "path"), value))
        for value in (None, True, [], "", "0" * 63, "g" * 64, "A" * 64):
            with self.subTest(sha=value):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    validate_data(self.changed(("sources", 0, "sha256"), value))
        for field in ("path", "sha256"):
            with self.subTest(duplicate=field):
                duplicate = self.payload["sources"][1][field]
                with self.assertRaisesRegex(ValueError, "Duplicate source"):
                    validate_data(self.changed(("sources", 0, field), duplicate))

    def test_rejects_unexpected_text_in_allowed_metadata(self):
        for path, value in [
            (("scope",), "unexpected text"),
            (("limitations", 0), "unexpected text"),
            (("cases", "code_direct", "calibration", "labels", 0), "unexpected text"),
            (("cases", "code_direct", "calibration", "subtypes", 0), "unexpected text"),
            (("cases", "code_direct", "orbench", "native", 0), "unexpected text"),
        ]:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_data(self.changed(path, value))

    def test_rejects_nonfinite_or_extreme_timing_values(self):
        for value in (float("nan"), float("inf"), 0, -1, 10 ** 1000):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    validate_data(self.changed(("cases", "code_direct", "timing", 0, "ttft_ms", "raw", 0), value))

    def test_rejects_duplicate_json_keys(self):
        text = (ARTIFACT / "data/evaluation.json").read_text()
        for content in (text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1),
                        text.replace('"fitting_texts": 64,', '"fitting_texts": 64, "fitting_texts": 64,', 1)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "evaluation.json").write_text(content)
                with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                    build_tables(root)


if __name__ == "__main__":
    unittest.main()
