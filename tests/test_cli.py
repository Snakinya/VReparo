import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StandaloneCLITests(unittest.TestCase):
    def test_allowlisted_copy_works_without_site_packages(self):
        manifest = json.loads((ROOT / "manifests/files.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            standalone = Path(folder) / "standalone"
            for relative in [*manifest["files"], "manifests/files.json"]:
                target = standalone / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            environment = {**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"}

            def run(*args):
                return subprocess.run([sys.executable, "-S", "-m", "vreparo_artifact", *args],
                                      cwd=standalone, env=environment, capture_output=True,
                                      text=True, timeout=30)

            listing = run("list")
            self.assertEqual(listing.returncode, 0, listing.stderr)
            self.assertIn("calibration", listing.stdout)
            verification = run("verify")
            self.assertEqual(verification.returncode, 0, verification.stdout + verification.stderr)
            self.assertEqual(json.loads(verification.stdout)["status"], "passed")
            reproduction = run("reproduce")
            self.assertEqual(reproduction.returncode, 0, reproduction.stdout + reproduction.stderr)
            self.assertEqual(len(list((standalone / "outputs/all").glob("table_*.csv"))), 11)
            subset = run("reproduce", "--experiment", "rq2", "--output", "outputs/rq2")
            self.assertEqual(subset.returncode, 0, subset.stdout + subset.stderr)
            self.assertEqual(json.loads(subset.stdout)["tables"], ["table_02"])
            repeated = run("reproduce")
            self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
            data = standalone / "data/characterization.json"
            data.write_bytes(data.read_bytes() + b"\n")
            tampered = run("verify")
            self.assertEqual(tampered.returncode, 1)
            self.assertIn("SHA-256 mismatch", tampered.stdout)


if __name__ == "__main__":
    unittest.main()
