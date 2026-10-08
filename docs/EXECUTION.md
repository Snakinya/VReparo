# Execution Guide

## Setup

Use Linux, Python 3.10 with `venv`, and a compatible NVIDIA driver. Run commands
from the extracted artifact root. No Slurm or paid evaluator API is needed.

```bash
python3 scripts/setup.py
.venv-models/bin/huggingface-cli login
python3 scripts/run_repeated_mitigation.py --case code_direct --dry-run
python3 scripts/run_repeated_mitigation.py --case code_direct --gpu 0 --accept-model-licenses
```

Setup creates `.venv-models` (Torch 2.4 / Transformers 4.45.2) and `.venv-semantic`
(Torch 2.4 / Transformers 4.53.3). Qwen3 generation and OR-Bench judging use the
latter; other stages use the former. Original mitigation used CUDA 12.1; choose
an appropriate PyTorch wheel source for your driver when installing manually.
`--model-python` and `--semantic-python` override these interpreter paths.

Eight scripts are listed in the README. They acquire inputs, execute stages
sequentially on the selected GPU, and summarize fresh results. Base Python can
invoke the workflows; heavy stages run in their designated environments.
`--dry-run` has no download/install/GPU side effects. The artifact never starts
background jobs or terminates another process.

## Outputs and Resume

```text
outputs/experiments/<case>/fit/{fit.json,mapper.pt,statistics.pt}
outputs/experiments/<target>/<experiment>/<dataset>/generation_0.json
outputs/experiments/<target>/<experiment>/<dataset>/scored/<judge>.json
outputs/experiments/<case>/calibration/xstest/selected/selection.json
outputs/experiments/<case>/profile/profile.json
outputs/experiments/<case>/performance/timing.json
outputs/reports/<experiment>/<target-or-all>/summary.{json,csv,md}
```

Repeat the same command to resume. Fitting checkpoints per text, generation
per condition/example, scoring per batch. Inputs, model content, configuration
and code must match. Full-weight hashing adds startup I/O. Changed protocols
need a separate artifact copy/output path, not reuse of an old checkpoint.
A workflow lock prevents shared checkpoint collisions and is released by the
OS after a crash. Low-level stages require one writer per output/shard.

Use `--assets` for a larger disk. `--skip-download` verifies existing model
files without downloading weights; datasets are still prepared/checked. Missing
or corrupt inputs fail explicitly. Only load tensor checkpoints you trust.

## Selection and Evaluation

The main mitigation script is `run_repeated_mitigation.py`. It reconstructs
five stratified selections, fits a separate map for each seed, calibrates on
XSTest and applies the selected alpha to that run's evaluation. It uses 248
HotpotQA prompts for all configurations. See [Five Runs](FIVE_RUNS.md).

The following fixed settings apply to `run_mitigation.py` (historical single-fit
check), `run_calibration.py`, `run_quality.py`, and `run_performance.py`:

Fitting uses the historical 32 QA + 32 HEx-PHI texts, ridge 1, identity-centered
regularization and at most 128 positions/text. Direct/Droid formatting differences
are preserved. XSTest selects among alpha 0.1/0.25/0.5/1 by unsafe string-refusal
gain over Raw, with safe-refusal increases at most 5 pp overall and 20 pp per
subtype over Native. Ties choose smaller alpha; QA is not a selection gate.

Fresh selections are compared against Code=1/Base=0.25 and Droid block [0,1],
but never edit downstream paper configs. This permits independent replication
of fixed paper settings. A fresh selection mismatch must be investigated.

QA uses 149 HotpotQA examples for direct reuse, 248 for Droid, 200 2WikiMQA
and 150 MultiFieldQA-en examples. New answers are scored by token F1. Budgets:
32 tokens for QA, 64 for MultiFieldQA-en, and 512 for safety benchmarks.

RQ1 compares Native/full reuse; RQ2 compares Native/K-only/V-only/full reuse.
Each uses its own intersection of valid responses. Automatic form rules are
included; historical semantic decisions apply only to exact prompt/response
hash matches. Changed responses are listed in `pending_review` and summaries
say `needs_validity_review`; their numbers are provisional. Inspect generated
texts and record fresh decisions following the paper's validity protocol.
See `python3 scripts/summarize.py --help` for the review-file format. No
relaxed-screening analysis is included.

HarmBench uses strict yes/no. Invalid judgments stop scoring, never become
zero. StrongREJECT uses continuous [0,1], BF16 loading/merging then FP32 forward
and softmax. Clipping keeps the first 512 evaluator tokens for HarmBench and
last 512 for StrongREJECT. OR-Bench uses condition-blind Qwen3 F/P/N/U/INVALID;
full refusal is F divided by fixed dataset size. INVALID is reported and requires
review, never silently excluded. These are automated labels, not human labels.

Fresh evaluator parity is explicitly unverified until checked on the target
environment. Different GPU kernels can change greedy outputs. Frozen scores
never substitute for fresh measurements.

## Profiling and Timing

Profiling searches 136 contiguous two-layer-boundary blocks on 50 QA examples.
Timing uses five warmups/five repetitions, 32 forced decode tokens and CUDA
synchronization. Sender KV/embeddings are resident. Sender prefill, transfer,
tokenization and scheduling are excluded. Mapped Droid retains its historical
two suffix-prefill calls. Measure on an idle GPU; latency varies by hardware.

## Lower-Level Stages

`python3 run.py experiment --help` exposes fit, generate, merge, select, score,
profile and timing. Workflow dry-runs print exact invocations. For independent
generation shards, add `--shards 3` and `--shard 0`, then 1 and 2 on separate
GPUs to the printed generate command. Merge completed shards with:

```bash
.venv-models/bin/python run.py experiment merge \
  --input outputs/my-shards --shards 3 \
  --dataset-file outputs/assets/data/harmbench.json \
  --output outputs/my-merged
```

Merge produces `generation.json` for a low-level score command. High-level
workflows expect unsharded `generation_0.json`; they do not discover custom
shard paths. Fresh outputs must be below artifact `outputs/`; acquired assets
may reside externally.
