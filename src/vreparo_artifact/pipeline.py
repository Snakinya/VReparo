"""Experiment selection, independent paper comparisons, and portable reports."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from . import characterization, evaluation, repeated


def load_registry(root: Path) -> dict:
    payload = json.loads((root / "experiments" / "registry.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("experiments"):
        raise ValueError("Invalid experiment registry")
    registry = payload["experiments"]
    table_ids = [table for group in registry.values() for table in group["tables"]]
    if len(table_ids) != 11 or set(table_ids) != {f"table_{n:02d}" for n in range(1, 12)}:
        raise ValueError("Experiment registry must partition the eleven manuscript tables")
    return registry


def build_tables(root: Path, experiment: str = "all") -> dict:
    registry = load_registry(root)
    if experiment != "all" and experiment not in registry:
        raise ValueError(f"Unknown experiment: {experiment}")
    selected = (registry.values() if experiment == "all" else [registry[experiment]])
    requested = {table for group in selected for table in group["tables"]}
    result = {}
    if requested & {"table_01", "table_02", "table_05", "table_06"}:
        result.update(characterization.build_tables(root / "data"))
    if requested & {"table_03", "table_04", "table_07", "table_08", "table_09", "table_10"}:
        result.update(evaluation.build_tables(root / "data"))
    if requested & {'table_03', 'table_10', 'table_11'}:
        result.update(repeated.build_tables(root / 'data'))
    return {key: result[key] for key in sorted(requested)}


def compare_tables(root: Path, tables: dict) -> list[str]:
    """Compare every displayed cell to separately transcribed manuscript values."""
    expected = {}
    for name in ("characterization", "evaluation", "repeated"):
        data = json.loads((root / "expected" / f"{name}.json").read_text(encoding="utf-8"))
        if expected.keys() & data.keys():
            raise ValueError("Duplicate reference table")
        expected.update(data)
    if set(expected) != {f"table_{n:02d}" for n in range(1, 12)}:
        raise ValueError("Expected values must cover exactly eleven paper tables")
    issues = []
    if not tables:
        return ["No tables were computed"]
    for key, actual in tables.items():
        reference = expected.get(key)
        if reference is None:
            issues.append(f"{key}: missing manuscript reference")
            continue
        for field in ("title", "columns"):
            if actual.get(field) != reference.get(field):
                issues.append(f"{key}: {field} differs from manuscript reference")
        if len(actual["rows"]) != len(reference["rows"]):
            issues.append(f"{key}: row count differs from manuscript reference")
            continue
        for index, (row, target) in enumerate(zip(actual["rows"], reference["rows"]), 1):
            if len(row) != len(actual["columns"]) or len(row) != len(target):
                issues.append(f"{key}: row {index} has the wrong number of columns")
                continue
            for column, (value, expected_value) in enumerate(zip(row, target), 1):
                if value != expected_value:
                    issues.append(f"{key}: row {index}, column {column} differs from manuscript reference")
    return issues


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(table: dict) -> str:
    lines = [f"# {table['title']}", ""]
    lines.append("| " + " | ".join(_markdown_cell(v) for v in table["columns"]) + " |")
    lines.append("| " + " | ".join("---" for _ in table["columns"]) + " |")
    lines.extend("| " + " | ".join(_markdown_cell(v) for v in row) + " |" for row in table["rows"])
    return "\n".join(lines) + "\n"


def write_reports(root: Path, output: Path, tables: dict, report: dict) -> None:
    """Only write to a new folder or a folder containing our own prior outputs."""
    root = root.resolve()
    output = output.absolute()
    if any(component.is_symlink() for component in (*output.parents, output)):
        raise ValueError("Output directory cannot use symlinks")
    output = output.resolve()
    if output == root or output in root.parents:
        raise ValueError("Output cannot replace the artifact root or one of its parents")
    if root in output.parents and output.relative_to(root).parts[0] != "outputs":
        raise ValueError("Inside the artifact, generated reports must be under outputs/")
    names = {"tables.json", "verification.json"}
    names.update(f"{key}.{suffix}" for key in tables for suffix in ("csv", "md"))
    if output.exists():
        if not output.is_dir():
            raise ValueError("Output path must be a directory")
        existing = {path.name for path in output.iterdir()}
        if existing:
            marker = output / "verification.json"
            if not marker.is_file() or marker.is_symlink():
                raise ValueError("Nonempty output directory is not an artifact report directory")
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("generator") != "vreparo-artifact" or existing - names:
                raise ValueError("Output directory contains unrelated files; choose a fresh directory")
        for name in names:
            target = output / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Output target is not a regular file")
    output.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(table["columns"])
        writer.writerows(table["rows"])
        (output / f"{key}.csv").write_text(buffer.getvalue(), encoding="utf-8")
        (output / f"{key}.md").write_text(render_markdown(table), encoding="utf-8")
    (output / "tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
