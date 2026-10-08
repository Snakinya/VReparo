"""Install isolated, pinned model and semantic environments."""

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3.10", help="Python 3.10 interpreter available on PATH")
    parser.add_argument("--environment", choices=("all", "models", "semantic"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    executable = shutil.which(args.python)
    if executable is None:
        raise SystemExit("Python interpreter not found; install Python 3.10 or specify --python")
    for name in ("models", "semantic") if args.environment == "all" else (args.environment,):
        environment = ROOT / (".venv-" + name)
        commands = [[executable, "-m", "venv", str(environment)],
                    [str(environment / "bin/python"), "-m", "pip", "install", "--upgrade", "pip"],
                    [str(environment / "bin/python"), "-m", "pip", "install", "-r", str(ROOT / f"requirements-{name}.txt")]]
        for command in commands:
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
    print("For gated Hugging Face resources, accept their access terms and run .venv-models/bin/huggingface-cli login. Never put tokens in scripts.")
