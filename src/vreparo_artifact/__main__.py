"""Run from the artifact directory with PYTHONPATH=src python -m vreparo_artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrity import scan_release_tree, verify_manifest
from .pipeline import build_tables, compare_tables, load_registry, write_reports
from .execution.cli import add_arguments, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify frozen measurements and recompute the paper tables offline.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Artifact directory (default: current directory)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List experiment groups and paper tables")
    subparsers.add_parser("verify", help="Check integrity, disclosure scan, and every paper table")
    reproduce = subparsers.add_parser("reproduce", help="Recompute and write verified CSV/Markdown tables")
    reproduce.add_argument("--experiment", default="all", help="Experiment group from list, or all")
    reproduce.add_argument("--output", type=Path, default=Path("outputs/all"), help="Report folder (default: outputs/all)")
    add_arguments(subparsers)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    try:
        if args.command == "list":
            for key, group in load_registry(root).items():
                print(f"{key:12} {', '.join(group['tables'])}: {group['title']}")
            return 0
        issues = verify_manifest(root) + scan_release_tree(root)
        if issues:
            print(json.dumps({"status": "failed", "issues": issues}, indent=2))
            return 1
        if args.command == "experiment":
            print(json.dumps(run(args, root), indent=2))
            return 0
        tables = build_tables(root, getattr(args, "experiment", "all"))
        issues = compare_tables(root, tables)
        manifest = json.loads((root / "manifests" / "files.json").read_text(encoding="utf-8"))
        report = {
            "generator": "vreparo-artifact",
            "status": "failed" if issues else "passed",
            "scope": "Offline aggregation of frozen observations, not fresh model generation or judging",
            "paper_commit": manifest["paper_commit"],
            "tables": list(tables),
            "rows": sum(len(table["rows"]) for table in tables.values()),
            "displayed_cells": sum(sum(len(row) for row in table["rows"]) for table in tables.values()),
            "issues": issues,
            "release_status": "research-artifact",
        }
        if args.command == "reproduce" and not issues:
            output = args.output if args.output.is_absolute() else root / args.output
            write_reports(root, output, tables, report)
        print(json.dumps(report, indent=2))
        return 1 if issues else 0
    except ImportError:
        print(json.dumps({"status": "failed", "message": "Optional execution dependencies are missing. See docs/EXECUTION.md; offline reproduce needs none."}))
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Validators use field-level diagnostics rather than source text or secrets.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": "Invalid artifact input or output path; check schema, completeness, and directory ownership."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
