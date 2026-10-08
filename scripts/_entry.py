"""Common bootstrap for source-checkout experiment scripts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(experiment):
    from vreparo_artifact.execution.workflows import main
    try:
        raise SystemExit(main(experiment, ROOT))
    except (ValueError, OSError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        raise SystemExit(1)
