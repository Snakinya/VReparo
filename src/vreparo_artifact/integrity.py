"""Read-only integrity and disclosure checks for the offline release bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


_MANIFEST_LIMIT = 8 * 1024 * 1024
_TEXT_LIMIT = 16 * 1024 * 1024
_RELEASE_DIRS = frozenset(
    {"src", "source", "scripts", "tests", "docs", "data", "expected", "experiments", "manifests"}
)
_SKIP_DIRS = frozenset({"outputs", "__pycache__", ".git", "venv", ".venv"})
_PROHIBITED_SUFFIXES = frozenset(
    {
        ".bin", ".cache", ".ckpt", ".dill", ".ggml", ".gguf", ".h5", ".hdf5",
        ".joblib", ".msgpack", ".npy", ".npz", ".onnx", ".pickle", ".pkl",
        ".pt", ".pth", ".safetensors",
    }
)
_RAW_RESPONSE_NAME = re.compile(
    r"(?:^|[_.-])(?:raw[_.-]?(?:responses?|completions?|outputs?)|"
    r"responses?|completions?|generations?)(?:[_.-]|$)",
    re.IGNORECASE,
)
_CREDENTIAL_PATTERNS = (
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
    ("API secret key", re.compile(r"\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")),
)
_PERSONAL_PATH_PATTERNS = (
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)"),
    re.compile(r"/mnt/(?:hdd|ssd|nvme|data)/[A-Za-z0-9_.-]+/"),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+", re.IGNORECASE),
)


def _label(path: str) -> str:
    """Keep diagnostics single-line and avoid repeating credentials in names."""
    for _, pattern in _CREDENTIAL_PATTERNS:
        path = pattern.sub("[redacted]", path)
    return json.dumps(path, ensure_ascii=True)


def _relative_path_issue(relative: object) -> str | None:
    if not isinstance(relative, str) or not relative:
        return "path must be a nonempty POSIX-relative string"
    if "\\" in relative or "\x00" in relative:
        return "path contains a backslash or NUL"
    if PurePosixPath(relative).is_absolute() or re.match(r"^[A-Za-z]:", relative):
        return "absolute paths are not permitted"
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        return "path contains an empty, dot, or traversal component"
    return None


def is_release_path(relative_posix: str) -> bool:
    """Identify the intended file scope shared by scanning and packaging."""
    if _relative_path_issue(relative_posix):
        return False
    parts = relative_posix.split("/")
    if any(part in _SKIP_DIRS for part in parts):
        return False
    return len(parts) == 1 or parts[0] in _RELEASE_DIRS


def _symlink_issue(root: Path, relative: str = "") -> str | None:
    # Inspect lexical components without resolving away a symlink first.
    target = root.absolute()
    if relative:
        target = target.joinpath(*relative.split("/"))
    for component in (*reversed(target.parents), target):
        if component.is_symlink():
            return "symlink path components are not permitted"
    return None


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def verify_manifest(root: Path) -> list[str]:
    """Check every declared SHA-256; return issues without changing any files."""
    root = Path(root)
    manifest_name = "manifests/files.json"
    manifest_path = root / manifest_name
    try:
        issue = _symlink_issue(root, manifest_name)
        if issue:
            return [f"Manifest: {issue}."]
        if not manifest_path.is_file():
            return ["Manifest is missing or not a regular file."]
        if manifest_path.stat().st_size > _MANIFEST_LIMIT:
            return ["Manifest exceeds the size limit; manual review required."]
        with manifest_path.open("rb") as handle:
            raw = handle.read(_MANIFEST_LIMIT + 1)
        if len(raw) > _MANIFEST_LIMIT:
            return ["Manifest exceeds the size limit; manual review required."]
        manifest = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError, UnicodeError):
        return ["Manifest is missing, unreadable, or invalid JSON."]

    if not isinstance(manifest, dict):
        return ["Manifest must be a JSON object."]
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        return ["Manifest has a missing or unsupported schema_version."]
    issues: list[str] = []
    if not isinstance(manifest.get("paper_commit"), str) or not manifest["paper_commit"].strip():
        issues.append("Manifest paper_commit must be a nonempty string.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        issues.append("Manifest files must be a nonempty path-to-SHA-256 object.")
        return issues

    for relative, expected in sorted(files.items()):
        path_issue = _relative_path_issue(relative)
        label = _label(relative)
        if path_issue:
            issues.append(f"Manifest entry {label}: {path_issue}.")
            continue
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            issues.append(f"Manifest entry {label}: missing or malformed SHA-256.")
            continue
        try:
            path_issue = _symlink_issue(root, relative)
            if path_issue:
                issues.append(f"Manifest entry {label}: {path_issue}.")
                continue
            path = root.joinpath(*relative.split("/"))
            if not path.is_file():
                issues.append(f"Manifest entry {label}: missing or not a regular file.")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected.lower():
                issues.append(f"Manifest entry {label}: SHA-256 mismatch.")
        except OSError:
            issues.append(f"Manifest entry {label}: unable to read file.")
    return issues


def scan_release_tree(root: Path) -> list[str]:
    """Audit the offline bundle's source, docs, data, and root-level files.

    Generated outputs, virtual environments, Git internals, and Python caches
    are outside this scan. Files above the bounded text limit fail closed.
    """
    root = Path(root)
    try:
        if _symlink_issue(root):
            return ["Release root: symlink path components are not permitted."]
        if not root.is_dir():
            return ["Release root is missing or not a directory."]
        pending = sorted(root.iterdir(), reverse=True)
    except OSError:
        return ["Release root cannot be read."]

    issues: list[str] = []
    while pending:
        path = pending.pop()
        relative = path.relative_to(root).as_posix()
        label = _label(relative)
        try:
            if path.is_symlink():
                issues.append(f"Release file {label}: symlinks are not permitted.")
                continue
            if not is_release_path(relative):
                continue
            if path.is_dir():
                if len(path.relative_to(root).parts) == 1 and path.name not in _RELEASE_DIRS:
                    continue
                pending.extend(sorted(path.iterdir(), reverse=True))
                continue
            if not path.is_file():
                issues.append(f"Release file {label}: not a regular file.")
                continue
            if path.suffix.lower() in _PROHIBITED_SUFFIXES:
                issues.append(f"Release file {label}: model/cache/serialized binary artifact is prohibited.")
                continue
            # A generation implementation is source code, not an output dump.
            if _RAW_RESPONSE_NAME.search(path.name) and path.suffix != ".py":
                issues.append(f"Release file {label}: raw-response filename requires removal or review.")
            if path.stat().st_size > _TEXT_LIMIT:
                issues.append(f"Release file {label}: exceeds text audit size limit; manual review required.")
                continue
            with path.open("rb") as handle:
                raw = handle.read(_TEXT_LIMIT + 1)
            if len(raw) > _TEXT_LIMIT:
                issues.append(f"Release file {label}: exceeds text audit size limit; manual review required.")
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeError:
                issues.append(f"Release file {label}: non-UTF-8 content requires manual review.")
                continue
            if "\x00" in text:
                issues.append(f"Release file {label}: binary content requires manual review.")
                continue
            for description, pattern in _CREDENTIAL_PATTERNS:
                if pattern.search(text):
                    issues.append(f"Release file {label}: possible {description}; value redacted.")
            if any(pattern.search(text) for pattern in _PERSONAL_PATH_PATTERNS):
                issues.append(f"Release file {label}: personal-machine absolute path; value redacted.")
        except OSError:
            issues.append(f"Release file {label}: unable to inspect file.")
    return issues
