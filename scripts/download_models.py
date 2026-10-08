"""Acquire only the pinned model files required by the requested experiment."""

import argparse
import json
from pathlib import Path

from _entry import ROOT
from vreparo_artifact.execution.resources import download_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/assets")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-model-licenses", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.accept_model_licenses:
        parser.error("Accept upstream access/license terms first, then pass --accept-model-licenses")
    print(json.dumps(download_models(ROOT, args.assets.absolute(), args.paths, dry_run=args.dry_run), indent=2))
