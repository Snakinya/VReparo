"""Rebuild fresh summaries, optionally applying hash-bound validity reviews."""

import argparse
import json
from pathlib import Path

from _entry import ROOT
from vreparo_artifact.execution.reporting import summarize
from vreparo_artifact.execution.workflows import CASES, PAIRS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Review JSON is a list copied from summary.json pending_review entries. "
        "Inspect each corresponding generation response and add valid: true or false. "
        "Keep target, benchmark, condition, prompt_id, prompt_sha256 and response_sha256 unchanged. "
        "Decisions apply only to matching hashes; automatic form exclusions remain active."))
    parser.add_argument("--experiment", choices=("rq1", "rq2", "mitigation", "calibration", "quality", "performance", "profile"), required=True)
    parser.add_argument("--targets", nargs="+", required=True, choices=(*CASES, *PAIRS, "all"))
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    allowed = PAIRS if args.experiment in ("rq1", "rq2") else tuple(c for c in CASES if c.endswith("droid")) if args.experiment == "profile" else CASES
    targets = allowed if args.targets == ["all"] else args.targets
    if len(set(targets)) != len(targets) or any(t not in allowed for t in targets):
        parser.error("Targets do not belong to this experiment or contain duplicates")
    print(json.dumps(summarize(ROOT, args.experiment, targets, args.reviews), indent=2))
