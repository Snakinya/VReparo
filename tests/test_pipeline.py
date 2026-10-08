import copy
import json
import tempfile
import unittest
from pathlib import Path

from vreparo_artifact.pipeline import build_tables, compare_tables, load_registry, write_reports


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = build_tables(ROOT)

    def test_all_eleven_tables_match(self):
        self.assertEqual(len(self.tables), 11)
        self.assertEqual(compare_tables(ROOT, self.tables), [])

    def test_experiments_partition_tables(self):
        registry = load_registry(ROOT)
        for key, group in registry.items():
            with self.subTest(experiment=key):
                self.assertEqual(set(build_tables(ROOT, key)), set(group["tables"]))

    def test_modified_display_value_fails(self):
        changed = copy.deepcopy(self.tables)
        changed["table_01"]["rows"][0][2] = "999.99"
        self.assertEqual(len(compare_tables(ROOT, changed)), 1)

    def test_empty_table_set_fails(self):
        self.assertEqual(compare_tables(ROOT, {}), ["No tables were computed"])

    def test_unknown_experiment_fails(self):
        with self.assertRaises(ValueError):
            build_tables(ROOT, "not-a-study")

    def test_reports_are_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated"
            report = {"generator": "vreparo-artifact", "status": "passed"}
            write_reports(ROOT, output, self.tables, report)
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            write_reports(ROOT, output, self.tables, report)
            second = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(json.loads(first["tables.json"]), self.tables)
            self.assertEqual(len(first), 24)

    def test_unrelated_output_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            unrelated = output / "important.txt"
            unrelated.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_reports(ROOT, output, self.tables, {})
            self.assertEqual(unrelated.read_text(), "preserve")

    def test_source_directories_are_not_output_directories(self):
        for output in (ROOT, ROOT.parent, ROOT / "data" / "generated"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                write_reports(ROOT, output, self.tables, {})

    def test_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "destination").mkdir()
            (base / "link").symlink_to(base / "destination", target_is_directory=True)
            with self.assertRaises(ValueError):
                write_reports(ROOT, base / "link", self.tables, {})


if __name__ == "__main__":
    unittest.main()
