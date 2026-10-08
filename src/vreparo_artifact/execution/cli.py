"""CLI schema stays dependency-free; Torch imports occur only after validation."""

from pathlib import Path

from .inputs import digest, model_identity, read, rows, select_alpha, write


def add_arguments(parent):
    parser = parent.add_parser("experiment", help="Opt-in fresh model experiments (optional GPU dependencies)")
    commands = parser.add_subparsers(dest="stage", required=True)

    def paired(name, help_text):
        child = commands.add_parser(name, help=help_text)
        child.add_argument("--case", choices=("code_direct", "code_droid", "base_direct", "base_droid"), default="code_direct")
        child.add_argument("--config", type=Path, help="Optional single-pair config for characterization")
        child.add_argument("--assets", type=Path, required=True, help="Local directory containing models/")
        child.add_argument("--device", default="cuda")
        child.add_argument("--output", type=Path, required=True, help="Directory below artifact outputs/")
        return child

    fitting = paired("fit", "Fit the paper's 32 QA + 32 HEx-PHI Value maps")
    fitting.add_argument("--qa", type=Path, required=True)
    fitting.add_argument("--safety", type=Path, required=True)
    fitting.add_argument("--sampling-seed", type=int, choices=range(5), help="Use one of the five stratified 32+32 selections")
    generation = paired("generate", "Run RQ1, RQ2, calibration, mitigation or QA")
    generation.add_argument("--experiment", choices=("rq1", "rq2", "calibration", "mitigation", "quality"), required=True)
    generation.add_argument("--input", type=Path, required=True)
    generation.add_argument("--task", choices=("safety", "hotpotqa", "2wikimqa", "multifieldqa_en"), default="safety")
    generation.add_argument("--max-new-tokens", type=int, required=True)
    generation.add_argument("--mapper", type=Path)
    generation.add_argument("--selection", type=Path, help="Use alpha selected by this map's completed XSTest run")
    generation.add_argument("--shard", type=int, default=0)
    generation.add_argument("--shards", type=int, default=1)
    generation.add_argument("--dataset", help="Optional exact input hash check against experiments/datasets.json")
    profile = paired("profile", "Repeat the 50-QA DroidSpeak block search")
    profile.add_argument("--input", type=Path, required=True)
    timing = paired("timing", "Measure paper resident-cache timing on an idle GPU")
    timing.add_argument("--input", type=Path, required=True)
    timing.add_argument("--mapper", type=Path, required=True)
    for stage in ("select", "merge", "score"):
        child = commands.add_parser(stage)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if stage == "merge":
            child.add_argument("--dataset-file", type=Path, required=True)
            child.add_argument("--shards", type=int, required=True)
        if stage == "score":
            child.add_argument("--evaluator", choices=("harmbench", "strongreject", "orbench"), required=True)
            child.add_argument("--model", type=Path, required=True)
            child.add_argument("--base-model", type=Path)
            child.add_argument("--batch-size", type=int, default=8)
    check = commands.add_parser("check-input", help="Verify a locally acquired source without distributing its text")
    check.add_argument("--dataset", required=True)
    check.add_argument("--input", type=Path, required=True)


def check_input(root, name, path):
    spec = read(root / "experiments/datasets.json")["datasets"][name]
    if digest(path) != spec["source_sha256"] or len(rows(path)) != spec["n"]:
        raise ValueError("Input differs from the frozen paper source; no automatic substitution")
    return {"status": "passed", "dataset": name, "n": spec["n"]}


def run(args, root):
    if args.stage == "check-input":
        return check_input(root, args.dataset, args.input)
    output = args.output.resolve()
    allowed = (root / "outputs").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Fresh output must be a subdirectory of artifact outputs/")
    args.output = output
    if getattr(args, "dataset", None):
        check_input(root, args.dataset, args.input)
    if args.stage == "select":
        source = read(args.input)
        if source.get("state") != "completed":
            raise ValueError("Selection requires completed calibration")
        result = select_alpha(source["condition_records"])
        result["input_sha256"] = digest(args.input)
        write(output / "selection.json", result)
        if result["selected_alpha"] is None:
            raise ValueError("No eligible alpha; candidate results saved, do not discard this run")
        return result
    if hasattr(args, "shards") and (args.shards < 1 or not 0 <= getattr(args, "shard", 0) < args.shards):
        raise ValueError("Invalid shard index/count")
    if getattr(args, "max_new_tokens", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        raise ValueError("Token budget and batch size must be positive")
    import torch
    torch.set_num_threads(4)
    if args.stage == "merge":
        from .experiments import merge
        return merge(args)
    if not torch.cuda.is_available():
        raise ValueError("Fresh paper experiments require CUDA; offline reproduce needs no GPU")
    if args.stage == "score":
        from .scoring import score
        return score(args)
    config = read(args.config) if args.config else read(root / "experiments/paper_cases.json")["cases"][args.case]
    if args.stage in ("fit", "profile", "timing") and args.config:
        raise ValueError("Custom pair configurations apply only to characterization generation")
    if args.config and args.stage == "generate" and args.experiment not in ("rq1", "rq2"):
        raise ValueError("Custom pairs are only supported for RQ1 and RQ2")
    if args.stage == "fit":
        check_input(root, "train_qa", args.qa)
        check_input(root, "train_safety", args.safety)
        if args.sampling_seed is not None:
            from .sampling import selected_indices
            config = dict(config, sampling_seed=args.sampling_seed,
                          fitting_indices=selected_indices(root, args.assets, args.sampling_seed))
    if args.stage == "generate" and args.selection:
        if args.experiment not in ("mitigation", "quality") or args.mapper is None:
            raise ValueError("Selected alpha requires mitigation/quality with a fitted map")
        selection = read(args.selection)
        calibration_file = args.selection.parent.parent / "generation_0.json"
        calibration = read(calibration_file)
        fingerprint = calibration.get("fingerprint", {})
        if (calibration.get("state") != "completed" or selection.get("input_sha256") != digest(calibration_file)
                or fingerprint.get("inputs", {}).get("mapper") != digest(args.mapper)):
            raise ValueError("Selection is not bound to this map's completed calibration")
        selected = select_alpha(calibration["condition_records"])
        if selected['selected_alpha'] is None or selection['selected_alpha'] != selected['selected_alpha']:
            raise ValueError("Invalid or ineligible selected alpha")
        if any(fingerprint['config'][key] != config[key] for key in ('sender', 'receiver', 'operator', 'recompute_group')):
            raise ValueError("Calibration belongs to a different model configuration")
        config = dict(config, alpha=selected['selected_alpha'], selection_sha256=digest(args.selection))
    if args.stage in ("profile", "timing"):
        check_input(root, "profile", args.input)
    config = dict(config, local_model_sha256=model_identity(config, args.assets))
    if args.stage == "timing":
        from .timing import measure
        return measure(args, config)
    from . import experiments
    return getattr(experiments, args.stage)(args, config)
