"""Pinned downloads and exact reconstruction of the historical paper subsets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from .inputs import digest, read, write


def canonical(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def download(url, target: Path, expected=None, size=None):
    if target.exists():
        if (expected and digest(target) != expected) or (size is not None and target.stat().st_size != size):
            raise ValueError(f"Cached resource failed verification: {target.name}. Move it aside before retrying.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "vreparo-reproduction/0.3"}), timeout=90) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if (expected and digest(partial) != expected) or (size is not None and partial.stat().st_size != size):
                raise ValueError("Downloaded resource checksum/size mismatch")
            partial.replace(target)
            return target
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def source_rows(name, path, member=None):
    if name == "longbench":
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if PurePosixPath(n).name == member]
            if len(candidates) != 1:
                raise ValueError("LongBench member is missing or ambiguous")
            raw = archive.read(candidates[0])
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()], raw
    if name == "strongreject":
        import pyarrow.parquet as pq
        result = []
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            content = row["content"]
            text = content[0] if isinstance(content, list) else content
            if text.strip():
                result.append({"category": "strongreject", "prompt": text.strip(), "behavior_id": f"strongreject_{index:04d}"})
        return result, None
    if name == "hexphi":
        original = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [{"behavior_id": f"hexphi_{i:03d}", "category": r["category"], "prompt": r["instruction"]} for i, r in enumerate(original)], None
    original = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if name == "harmbench":
        return [{"category": r["SemanticCategory"], "prompt": r["Behavior"], "behavior_id": r["BehaviorID"]} for r in original], None
    if name == "orbench":
        return [{"id": "orbench-" + hashlib.sha256(r["prompt"].encode()).hexdigest()[:20], "prompt": r["prompt"],
                 "category": r["category"], "label": "safe", "type": r["category"]} for r in original], None
    if name == "xstest":
        return original, None
    raise ValueError("Unknown upstream data transformation")


def prepare_data(root: Path, assets: Path, names=None):
    manifest = read(root / "experiments/resources.json")
    selected = list(manifest["datasets"]) if names is None else list(dict.fromkeys(names))
    report = {}
    decoded = {}
    for name in selected:
        recipe = manifest["datasets"][name]
        target = assets / "data" / recipe["filename"]
        if target.exists():
            if digest(target) != recipe["sha256"]:
                raise ValueError("Existing dataset differs from the paper: " + name)
            report[name] = {"n": recipe["n"], "sha256": recipe["sha256"]}
            continue
        spec = manifest["sources"][recipe["source"]]
        cache = assets / "downloads" / (recipe["source"] + "." + spec["format"])
        print(f"Preparing {name} from pinned {recipe['source']}", flush=True)
        download(spec["url"], cache, spec["sha256"])
        key = (recipe["source"], recipe["member"])
        if key not in decoded:
            values, raw = source_rows(recipe["source"], cache, recipe["member"])
            decoded[key] = ({canonical(row): row for row in values}, raw)
        by_hash, raw = decoded[key]
        if any(key not in by_hash for key in recipe["rows"]):
            raise ValueError("An upstream row does not match its historical content fingerprint: " + name)
        values = [by_hash[key] for key in recipe["rows"]]
        if recipe["serialization"] == "verbatim_member":
            output = raw
        else:
            wrapper = json.loads(recipe["wrapper_json"])
            value = dict(wrapper, prompts=values) if wrapper is not None else values
            settings = recipe["serialization"]
            # Frozen files with sort_keys=False retain the recorded object key order.
            if "key_order" in recipe:
                values = [{k: row[k] for k in recipe["key_order"]} for row in values]
                value = dict(wrapper, prompts=values) if wrapper is not None else values
            if recipe["wrapper_order"] is not None:
                value = {key: value[key] for key in recipe["wrapper_order"]}
            output = (json.dumps(value, indent=2, sort_keys=settings["sort_keys"], ensure_ascii=settings["ensure_ascii"]) + settings["newline"]).encode()
        if hashlib.sha256(output).hexdigest() != recipe["sha256"]:
            raise ValueError("Reconstructed file differs from frozen bytes: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(output)
        report[name] = {"n": len(values), "sha256": digest(target)}
        print(f"Verified {name}: {len(values)} rows, exact file SHA-256", flush=True)
    write(assets / "data/preparation.json", {"status": "passed", "datasets": report})
    return report


def model_paths(config):
    return sorted({config[role][key] for role in ("sender", "receiver") for key in ("foundation", "adapter", "tokenizer") if config[role].get(key)})


def download_models(root: Path, assets: Path, paths, *, dry_run=False, verify_only=False):
    manifest = read(root / "experiments/resources.json")["models"]
    paths = sorted(set(paths))
    total = sum(value["size"] for path in paths for name, value in manifest[path]["files"].items()
                if not (assets / path / name).exists())
    plan = {"models": paths, "missing_weight_and_tokenizer_bytes": total}
    if dry_run:
        return plan
    assets.mkdir(parents=True, exist_ok=True)
    if not verify_only and shutil.disk_usage(assets).free < total + 2 * 1024 ** 3:
        raise ValueError(f"Insufficient storage: need approximately {total / 1024**3:.1f} GiB plus 2 GiB reserve. Use --assets on a larger volume.")
    for relative in paths:
        spec = manifest[relative]
        target = assets / relative
        needed = []
        for name, expected in spec["files"].items():
            path = target / name
            if path.exists():
                if path.stat().st_size != expected["size"] or (expected["sha256"] and digest(path) != expected["sha256"]):
                    raise ValueError("Existing model file differs from the recorded checkpoint: " + relative + "/" + name)
            else:
                needed.append(name)
        if not needed:
            print(f"Model present: {relative}", flush=True)
            continue
        if verify_only:
            raise ValueError("Required model files are missing: " + relative + "/" + needed[0])
        print(f"Downloading {spec['repo']} ({len(needed)} files)", flush=True)
        if spec["provider"] == "huggingface":
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=target,
                              allow_patterns=needed, max_workers=2)
        else:
            for name in needed:
                revision = spec["revisions_by_file"][name]
                url = f"https://modelscope.cn/api/v1/models/{spec['repo']}/repo?Revision={revision}&FilePath={quote(name)}"
                expected = spec["files"][name]
                download(url, target / name, expected["sha256"], expected["size"])
        for name, expected in spec["files"].items():
            path = target / name
            if not path.is_file() or path.stat().st_size != expected["size"] or (expected["sha256"] and digest(path) != expected["sha256"]):
                raise ValueError("Downloaded model failed validation: " + relative + "/" + name)
    return plan
