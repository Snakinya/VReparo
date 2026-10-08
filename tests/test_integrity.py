"""Synthetic, offline tests for release-integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vreparo_artifact.integrity import is_release_path, scan_release_tree, verify_manifest


class IntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative: str, content: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def manifest(self, files: dict[str, object], **overrides: object) -> None:
        document = {"schema_version": 1, "paper_commit": "a" * 40, "files": files}
        document.update(overrides)
        self.write("manifests/files.json", json.dumps(document))

    def test_valid_manifest(self) -> None:
        content = b"summary\n"
        self.write("data/summary.csv", content)
        self.manifest({"data/summary.csv": hashlib.sha256(content).hexdigest()})
        self.assertEqual(verify_manifest(self.root), [])

    def test_missing_manifest(self) -> None:
        self.assertTrue(verify_manifest(self.root))

    def test_directory_is_not_a_manifest(self) -> None:
        (self.root / "manifests/files.json").mkdir(parents=True)
        self.assertIn("regular file", " ".join(verify_manifest(self.root)))

    def test_empty_manifest_is_not_success(self) -> None:
        self.manifest({})
        self.assertIn("nonempty", " ".join(verify_manifest(self.root)))

    def test_unsupported_and_boolean_schema(self) -> None:
        for version in (None, 0, 2, True, "1"):
            with self.subTest(version=version):
                self.manifest({"data.csv": "a" * 64}, schema_version=version)
                self.assertIn("schema_version", " ".join(verify_manifest(self.root)))

    def test_invalid_paper_commit(self) -> None:
        self.write("data.csv", "")
        for commit in (None, "", " ", 123):
            with self.subTest(commit=commit):
                self.manifest({"data.csv": hashlib.sha256(b"").hexdigest()}, paper_commit=commit)
                self.assertIn("paper_commit", " ".join(verify_manifest(self.root)))

    def test_duplicate_keys_are_rejected(self) -> None:
        self.write("manifests/files.json", '{"schema_version":1,"schema_version":1}')
        self.assertTrue(verify_manifest(self.root))

    def test_nonobject_manifest(self) -> None:
        self.write("manifests/files.json", "[]")
        self.assertIn("JSON object", " ".join(verify_manifest(self.root)))

    def test_unsafe_manifest_paths(self) -> None:
        for relative in ("../escape", "/absolute", "C:/absolute", "a/../b", "a//b", "./a", "a\\b", "a\x00b"):
            with self.subTest(relative=relative):
                self.manifest({relative: "a" * 64})
                self.assertTrue(verify_manifest(self.root))

    def test_missing_and_malformed_hash(self) -> None:
        self.write("data.csv", "data")
        for digest in (None, "", "g" * 64, "a" * 63, 42):
            with self.subTest(digest=digest):
                self.manifest({"data.csv": digest})
                self.assertIn("malformed", " ".join(verify_manifest(self.root)))

    def test_missing_file(self) -> None:
        self.manifest({"missing.csv": "a" * 64})
        self.assertIn("missing", " ".join(verify_manifest(self.root)))

    def test_wrong_hash(self) -> None:
        self.write("data.csv", "data")
        self.manifest({"data.csv": "a" * 64})
        self.assertIn("mismatch", " ".join(verify_manifest(self.root)))

    def test_symlink_file_and_directory_components(self) -> None:
        self.write("actual/data.csv", "data")
        (self.root / "linked").symlink_to(self.root / "actual", target_is_directory=True)
        self.manifest({"linked/data.csv": hashlib.sha256(b"data").hexdigest()})
        self.assertIn("symlink", " ".join(verify_manifest(self.root)))
        (self.root / "file.csv").symlink_to(self.root / "actual/data.csv")
        self.manifest({"file.csv": hashlib.sha256(b"data").hexdigest()})
        self.assertIn("symlink", " ".join(verify_manifest(self.root)))

    def test_symlink_manifest_is_rejected(self) -> None:
        self.write("external.json", "{}")
        (self.root / "manifests").mkdir()
        (self.root / "manifests/files.json").symlink_to(self.root / "external.json")
        self.assertIn("symlink", " ".join(verify_manifest(self.root)))

    def test_large_manifest_fails_closed(self) -> None:
        self.write("manifests/files.json", " " * 20)
        with patch("vreparo_artifact.integrity._MANIFEST_LIMIT", 10):
            self.assertIn("size limit", " ".join(verify_manifest(self.root)))

    def test_clean_tree(self) -> None:
        self.write("README.md", "Offline aggregate metrics.\n")
        self.write("data/summary.csv", "count,rate\n2,0.5\n")
        self.write("src/module.py", "VALUE = 1\n")
        self.assertEqual(scan_release_tree(self.root), [])

    def test_credential_classes_and_redaction(self) -> None:
        credentials = (
            "gh" + "p_" + "A" * 36,
            "github" + "_pat_" + "A" * 60,
            "h" + "f_" + "A" * 34,
            "s" + "k-proj-" + "A" * 40,
            "AK" + "IA" + "A" * 16,
            "AI" + "za" + "A" * 35,
            "xo" + "xb-" + "A" * 20,
            "-----BEGIN " + "PRIVATE KEY-----",
        )
        for credential in credentials:
            with self.subTest(prefix=credential[:2]):
                self.write("docs/example.txt", credential)
                issues = scan_release_tree(self.root)
                self.assertTrue(issues)
                self.assertNotIn(credential, " ".join(issues))

    def test_personal_paths(self) -> None:
        paths = (
            "/" + "home/researcher/project",
            "/" + "Users/researcher/project",
            "/" + "mnt/hdd/researcher/project",
            "C:" + "\\Users\\researcher\\project",
        )
        for personal_path in paths:
            with self.subTest(path_style=personal_path[:2]):
                self.write("docs/example.txt", personal_path)
                issues = scan_release_tree(self.root)
                self.assertIn("personal-machine", " ".join(issues))
                self.assertNotIn(personal_path, " ".join(issues))

    def test_binary_model_and_raw_response_files(self) -> None:
        for relative in ("data/weights.safetensors", "data/cache.pt", "data/state.pkl", "data/raw_responses.jsonl"):
            with self.subTest(relative=relative):
                path = self.write(relative, "{}")
                self.assertTrue(scan_release_tree(self.root))
                path.unlink()

    def test_symlink_scan_does_not_follow_target(self) -> None:
        (self.root / "docs").symlink_to("missing-target", target_is_directory=True)
        self.assertIn("symlink", " ".join(scan_release_tree(self.root)))

    def test_non_utf8_requires_review(self) -> None:
        self.write("data/unknown.dat", b"\xff\xfe")
        self.assertIn("non-UTF-8", " ".join(scan_release_tree(self.root)))

    def test_nul_binary_requires_review(self) -> None:
        self.write("data/unknown.dat", b"abc\x00def")
        self.assertIn("binary", " ".join(scan_release_tree(self.root)))

    def test_oversized_text_requires_review(self) -> None:
        self.write("docs/example.txt", "x" * 20)
        with patch("vreparo_artifact.integrity._TEXT_LIMIT", 10):
            self.assertIn("size limit", " ".join(scan_release_tree(self.root)))

    def test_generated_and_environment_dirs_are_out_of_scope(self) -> None:
        for directory in ("outputs", "__pycache__", ".git", "venv", ".venv"):
            self.write(f"{directory}/model.pt", b"\x00")
        self.assertEqual(scan_release_tree(self.root), [])

    def test_release_path_membership(self) -> None:
        for relative in ("README.md", ".gitignore", "src/module.py", "tests/test_module.py",
                         "docs/guide.md", "data/observations.json", "expected/table.json",
                         "experiments/registry.json", "manifests/files.json"):
            with self.subTest(relative=relative):
                self.assertTrue(is_release_path(relative))
        for relative in ("", "../outside", "/absolute", "C:/absolute", "docs/../file",
                         "docs//file", "docs\\file", "unknown/file.txt"):
            with self.subTest(relative=relative):
                self.assertFalse(is_release_path(relative))

    def test_nested_generated_and_environment_paths_are_out_of_scope(self) -> None:
        for excluded in ("outputs", "__pycache__", ".git", "venv", ".venv"):
            for relative in (f"{excluded}/model.pt", f"docs/{excluded}/model.pt",
                             f"src/package/{excluded}/model.pt"):
                with self.subTest(relative=relative):
                    self.assertFalse(is_release_path(relative))
                    self.write(relative, b"\x00")
        self.assertEqual(scan_release_tree(self.root), [])

    def test_excluded_symlink_is_still_rejected(self) -> None:
        (self.root / "outputs").symlink_to("missing-target", target_is_directory=True)
        self.assertIn("symlink", " ".join(scan_release_tree(self.root)))

    def test_symlink_root_is_rejected(self) -> None:
        actual = self.root / "actual"
        actual.mkdir()
        linked = self.root / "linked"
        linked.symlink_to(actual, target_is_directory=True)
        self.assertIn("symlink", " ".join(scan_release_tree(linked)))
        self.assertIn("symlink", " ".join(verify_manifest(linked)))


if __name__ == "__main__":
    unittest.main()
