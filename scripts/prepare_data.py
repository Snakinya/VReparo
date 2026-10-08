"""Acquire public source files and rebuild exactly the paper's subsets."""

import argparse
import json
from pathlib import Path

from _entry import ROOT
from vreparo_artifact.execution.resources import prepare_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/assets")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    print(json.dumps(prepare_data(ROOT, args.assets.absolute(), args.datasets), indent=2))
