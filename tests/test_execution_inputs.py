import copy
from pathlib import Path
import tempfile
import unittest

from vreparo_artifact.execution.inputs import identifier, read, select_alpha, validate_groups, validate_rows, write


class InputContractTests(unittest.TestCase):
    def test_no_implicit_missing_id(self):
        with self.assertRaises(ValueError):
            identifier({"id": None})
        self.assertEqual(identifier({"id": 0}), "0")

    def test_duplicate_and_missing_fields(self):
        with self.assertRaises(ValueError):
            validate_rows([{"id": "a", "prompt": "x"}] * 2, "safety")
        with self.assertRaises(ValueError):
            validate_rows([{"id": "a"}], "hotpotqa")

    def test_alignment_is_not_just_length(self):
        with self.assertRaises(ValueError):
            validate_groups({"native": [{"id": "a", "prompt": "x"}], "raw": [{"id": "b", "prompt": "x"}]})

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "child/report.json"
            write(path, {"state": "running"})
            write(path, {"state": "completed"})
            self.assertEqual(read(path), {"state": "completed"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def groups(self):
        native = [{"prompt_id": str(i), "prompt": "synthetic", "label": "safe" if i < 250 else "unsafe",
                   "type": f"safe_{i // 25}" if i < 250 else "unsafe", "string_refusal": False} for i in range(450)]
        return {name: copy.deepcopy(native) for name in ("native", "raw", "vreparo_a0.1", "vreparo_a0.25", "vreparo_a0.5", "vreparo_a1")}

    def test_selection_rejects_over_refusal_and_breaks_ties(self):
        groups = self.groups()
        for name in ("vreparo_a0.25", "vreparo_a0.5", "vreparo_a1"):
            groups[name][250]["string_refusal"] = True
        for row in groups["vreparo_a1"][:6]:
            row["string_refusal"] = True
        result = select_alpha(groups)
        self.assertEqual(result["selected_alpha"], 0.25)
        self.assertFalse(result["gates"]["1.0"]["eligible"])

    def test_no_candidate_and_label_mismatch(self):
        groups = self.groups()
        self.assertIsNone(select_alpha(groups)["selected_alpha"])
        groups["raw"][0]["type"] = "changed"
        with self.assertRaises(ValueError):
            select_alpha(groups)


if __name__ == "__main__":
    unittest.main()
