"""Single source-checkout entry point; no installation required for offline use."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vreparo_artifact.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
