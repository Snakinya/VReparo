"""One-command experiment recipes over shared, resumable stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .inputs import read
from .resources import download_models, model_paths

CASES = ("code_direct", "code_droid", "base_direct", "base_droid")
PAIRS = ("qwen3_4b", "qwen3_8b", "llama32_3b", "llama3_8b", "gemma2_2b", "mistral7b_v03", "code_to_safety", "safety_to_code")
JUDGES = {"harmbench": "models/evaluators/HarmBench-Llama-2-13b-cls",
          "strongreject": "models/evaluators/strongreject-15k-v1", "orbench": "models/Qwen/Qwen3-8B"}


def parser(experiment):
    result = argparse.ArgumentParser(description=f"Download, execute, and summarize the paper's {experiment} experiment.")
    result.add_argument("--assets", type=Path, help="Shared download/data directory (default: outputs/assets)")
    result.add_argument("--gpu", default="0", help="One idle GPU index; all stages run sequentially on it")
    result.add_argument("--model-python", type=Path, help="Override .venv-models/bin/python")
    result.add_argument("--semantic-python", type=Path, help="Override .venv-semantic/bin/python")
    result.add_argument("--dry-run", action="store_true", help="Print exact stages and download sizes without running or downloading")
    result.add_argument("--skip-download", action="store_true", help="Verify existing models without downloading weights; datasets are still prepared automatically")
    result.add_argument("--accept-model-licenses", action="store_true", help="Confirm you have accepted applicable model/dataset terms")
    result.add_argument("--case", choices=(*CASES, "all"), default="code_droid" if experiment == "profile" else "code_direct")
    result.add_argument("--pair", choices=(*PAIRS, "all"), default="code_to_safety")
    return result


def plan(root, experiment, args):
    if experiment == 'repeated_mitigation':
        from .repeated import plan as repeated_plan
        return repeated_plan(root, args)
    assets = (args.assets or root / "outputs/assets").absolute()
    models_python = (args.model_python or root / ".venv-models/bin/python").absolute()
    semantic_python = (args.semantic_python or root / ".venv-semantic/bin/python").absolute()
    stages, models, datasets = [], set(), set()
    cases = read(root / "experiments/paper_cases.json")["cases"]
    output_base = root / "outputs/experiments"

    def add(stage, *arguments, semantic=False):
        executable = semantic_python if semantic else models_python
        stages.append([str(executable), str(root / "run.py"), "experiment", stage, *map(str, arguments)])

    def fit(case):
        datasets.update(("train_qa", "train_safety"))
        target = output_base / case / "fit"
        add("fit", "--case", case, "--assets", assets, "--qa", assets / "data/train_qa.jsonl",
            "--safety", assets / "data/train_safety.json", "--output", target)
        return target / "mapper.pt"

    def generation(case, exp, dataset, task="safety", config=None, mapper=None):
        datasets.add(dataset)
        target = output_base / case / exp / dataset
        args_list = ["--experiment", exp, "--assets", assets, "--input", assets / "data" / (dataset + ".json"),
                     "--task", task, "--max-new-tokens", 512 if task == "safety" else 64 if task == "multifieldqa_en" else 32,
                     "--output", target]
        args_list += ["--config", config] if config else ["--case", case]
        if dataset != "harmbench159":
            args_list += ["--dataset", dataset]
        if mapper:
            args_list += ["--mapper", mapper]
        add("generate", *args_list, semantic=case.startswith("qwen"))
        return target / "generation_0.json"

    def score(case, dataset, generated):
        judge = "harmbench" if dataset == "harmbench159" else dataset
        models.add(JUDGES[judge])
        extra = []
        if judge == "strongreject":
            models.add("models/evaluators/gemma-2b")
            extra = ["--base-model", assets / "models/evaluators/gemma-2b"]
        add("score", "--evaluator", judge, "--model", assets / JUDGES[judge], "--input", generated,
            "--output", generated.parent / "scored", *extra, semantic=judge == "orbench")

    selected = PAIRS if args.pair == "all" else (args.pair,)
    if experiment in ("rq1", "rq2"):
        for pair in selected:
            config = root / "experiments/pairs" / (pair + ".json")
            models.update(model_paths(read(config)))
            for dataset in ("harmbench159", "strongreject") if experiment == "rq1" else ("harmbench159",):
                generated = generation(pair, experiment, dataset, config=config)
                score(pair, dataset, generated)
    else:
        selected = CASES if args.case == "all" else (args.case,)
        if experiment == "profile" and args.case == "all":
            selected = tuple(case for case in selected if case.endswith("droid"))
        for case in selected:
            models.update(model_paths(cases[case]))
            if experiment == "profile":
                if not case.endswith("droid"):
                    raise ValueError("Profile requires --case code_droid or base_droid")
                datasets.add("profile")
                add("profile", "--case", case, "--assets", assets, "--input", assets / "data/profile.json",
                    "--output", output_base / case / "profile")
                continue
            mapper = fit(case)
            if experiment == "calibration":
                generated = generation(case, "calibration", "xstest", mapper=mapper)
                add("select", "--input", generated, "--output", generated.parent / "selected")
            elif experiment == "mitigation":
                for dataset in ("harmbench", "strongreject", "orbench"):
                    generated = generation(case, "mitigation", dataset, mapper=mapper)
                    score(case, dataset, generated)
            elif experiment == "quality":
                for dataset, task in (("qa_droid" if case.endswith("droid") else "qa_full", "hotpotqa"),
                                      ("2wikimqa", "2wikimqa"), ("multifieldqa_en", "multifieldqa_en")):
                    generation(case, "quality", dataset, task=task, mapper=mapper)
            elif experiment == "performance":
                datasets.add("profile")
                add("timing", "--case", case, "--assets", assets, "--input", assets / "data/profile.json",
                    "--mapper", mapper, "--output", output_base / case / "performance")
            else:
                raise ValueError("Unknown workflow")
    return {"experiment": experiment, "targets": list(selected), "assets": str(assets),
            "datasets": sorted(datasets), "models": sorted(models), "commands": stages}


def main(experiment, root, argv=None):
    args = parser(experiment).parse_args(argv)
    recipe = plan(root, experiment, args)
    assets = Path(recipe["assets"])
    recipe["download_plan"] = download_models(root, assets, recipe["models"], dry_run=True)
    if args.dry_run:
        print(json.dumps(recipe, indent=2))
        return 0
    from ..integrity import scan_release_tree, verify_manifest
    issues = verify_manifest(root) + scan_release_tree(root)
    if issues:
        raise ValueError("Artifact integrity check failed; run python3 run.py verify")
    for command in recipe["commands"]:
        if not Path(command[0]).is_file():
            raise ValueError("Execution environment missing; first run python3 scripts/setup.py")
    # Keep shared mapper and download checkpoints under a single writer.
    import fcntl
    lock_path = root / "outputs/workflow.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another experiment workflow is active in this artifact directory") from error
        return execute(root, args, recipe, assets)


def execute(root, args, recipe, assets):
    if not args.skip_download:
        if not args.accept_model_licenses:
            raise ValueError("Accept the upstream licenses/access agreements, then pass --accept-model-licenses. This flag does not grant access to gated models.")
        acquisition_python = args.model_python or root / ".venv-models/bin/python"
        subprocess.run([str(acquisition_python.absolute()), str(root / "scripts/download_models.py"), "--assets", str(assets),
                        "--paths", *recipe["models"], "--accept-model-licenses"], cwd=root, check=True)
    else:
        download_models(root, assets, recipe["models"], verify_only=True)
    # Data preparation runs in the acquisition environment, which includes pyarrow.
    acquisition_python = args.model_python or root / ".venv-models/bin/python"
    subprocess.run([str(acquisition_python.absolute()), str(root / "scripts/prepare_data.py"), "--assets", str(assets),
                    "--datasets", *recipe["datasets"]], cwd=root, check=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for command in recipe["commands"]:
        print("Stage: " + " ".join(command[2:]), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
    if recipe['experiment'] == 'repeated_mitigation':
        from .repeated import summarize
        report = summarize(root, recipe['targets'])
    else:
        from .reporting import summarize
        report = summarize(root, recipe["experiment"], recipe["targets"])
    print(json.dumps(report, indent=2))
    return 0
