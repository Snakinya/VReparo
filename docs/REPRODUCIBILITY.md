# Reproducibility Guide

Companion to [**To Cache or Not to Cache: Understanding and Mitigating Safety
Incompatibility in Cross-Model KV Cache Reuse**](https://github.com/Snakinya/kv-cache-provenance-overleaf/blob/d751d395f3b6f46e928095df66eceade0f531383/paper.pdf).

One script per paper experiment: acquire pinned models/datasets, run the required
fitting/generation/evaluation stages, and write fresh JSON, CSV and Markdown
summaries. Model loading, cache operators, VReparo and scoring share one
implementation. This release includes the five sampling runs used in the
current manuscript's Tables 3, 10 and 11, alongside its fixed-map timing and
additional QA experiments.

## Release Metadata

`AUTHORS.json` records the author order and affiliations: Zhiwei Lin,
Jiahao Liu, Yan Liu, and Chengpeng Wang. `CITATION.cff` provides the same
attribution. Provided emails are zhiweil@u.nus.edu (Zhiwei Lin),
ljiahaomail@gmail.com (Jiahao Liu), bencao.ly@antgroup.com (Yan Liu),
and wang-chengpeng@nus.edu.sg (Chengpeng Wang). Original code and software documentation use the MIT License; see
[LICENSE](../LICENSE) and [NOTICE](../NOTICE.md). Third-party terms remain separate.

The numerical data and experimental implementations retain the original
artifact baseline. `manifests/paper.json` identifies that frozen baseline;
`manifests/release.json` additionally binds this preparation to the cleaned
paper sources and records the packaging changes. See [Release Notes](RELEASE.md).

## Reproduce an Experiment

Use Linux, Python 3.10 with `venv`, a CUDA-compatible NVIDIA driver and an idle
GPU. The original mitigation runs used a 48 GB A40; memory requirements depend
on the model pair and prompt lengths. Run from the extracted artifact directory.

```bash
# Inspect stages and missing model bytes without downloading or installing.
python3 scripts/run_repeated_mitigation.py --case code_direct --dry-run

# Create two isolated, pinned execution environments.
python3 scripts/setup.py

# First accept gated models' terms on their upstream pages; then authenticate.
.venv-models/bin/huggingface-cli login

# Download inputs, fit and calibrate five maps, evaluate safety/refusal and QA.
python3 scripts/run_repeated_mitigation.py --case code_direct --gpu 0 --accept-model-licenses
```

The license flag acknowledges terms; it does not grant access to gated models.
Credentials belong in the standard Hugging Face login cache or `HF_TOKEN`,
never in the repository. Original ModelScope snapshots are pinned too; mirrors
do not remove upstream license obligations.

Downloads are shared in `outputs/assets/`. Pass `--assets /your/large/volume`
to store them elsewhere. `--dry-run` reports missing weight/tokenizer bytes;
allow extra space for environments, caches and outputs. Files are downloaded
at pinned revisions and checked by size, plus SHA-256 for small files. Full
local model hashes bind resumed runs to their actual weights.

## One Script per Experiment

| Script | Stages | Paper tables |
| --- | --- | --- |
| `scripts/run_rq1.py` | Native/full reuse, HarmBench159 and StrongREJECT, scoring and common-valid screening | 1, 5, 6 |
| `scripts/run_rq2.py` | Native/K-only/V-only/full reuse, HarmBench159, scoring and screening | 2 |
| `scripts/run_repeated_mitigation.py` | Five sampling seeds, separate fitting and XSTest selection, safety/refusal and HotpotQA, per-run results and means | 3, 10, 11 |
| `scripts/run_mitigation.py` | Historical single-fit safety check with fixed alpha | Reference experiment, not current main-table averages |
| `scripts/run_calibration.py` | Fit, four alpha candidates on XSTest, selection gates | 7 |
| `scripts/run_quality.py` | Fit, HotpotQA, 2WikiMQA and MultiFieldQA-en, token F1 | 9 |
| `scripts/run_performance.py` | Fit, resident-cache TTFT and decode measurement | 4, 8 |
| `scripts/run_profile.py` | 50-QA DroidSpeak recomputation-block search | Method selection |

```bash
python3 scripts/run_rq1.py --pair all --gpu 0 --accept-model-licenses
python3 scripts/run_rq2.py --pair all --gpu 0 --accept-model-licenses
python3 scripts/run_calibration.py --case all --gpu 0 --accept-model-licenses
python3 scripts/run_repeated_mitigation.py --case all --gpu 0 --accept-model-licenses
python3 scripts/run_quality.py --case all --gpu 0 --accept-model-licenses
python3 scripts/run_performance.py --case all --gpu 0 --accept-model-licenses
python3 scripts/run_profile.py --case all --gpu 0 --accept-model-licenses
```

Run these sequentially. `--case` accepts `code_direct`, `code_droid`,
`base_direct`, `base_droid`, or `all`; profiling accepts only Droid cases.
`--pair` accepts `qwen3_4b`, `qwen3_8b`, `llama32_3b`, `llama3_8b`,
`gemma2_2b`, `mistral7b_v03`, `code_to_safety`, `safety_to_code`, or `all`.
Every script supports `--help` and `--dry-run` without optional dependencies.

Repeat the same command to resume. Completed stages are reused only when their
fingerprints match. A workflow lock prevents concurrent writes to shared
checkpoints. `--skip-download` verifies existing models without downloading
weights; dataset preparation still runs. Changed resources fail explicitly.

Fresh outputs: `outputs/experiments/<target>/<experiment>/`.
Summaries: `outputs/reports/<experiment>/<target-or-all>/summary.{json,csv,md}`.
The five-run workflow applies each run's freshly selected alpha to its own
evaluation. Its report contains `summary.json`, `individual.{csv,md}` and
`means.{csv,md}`. Timing and additional QA retain their separately documented
fixed-map settings. See [Five Runs](FIVE_RUNS.md) and [Execution Guide](EXECUTION.md).

## Verify Published Numbers Without a GPU

This separate shortcut recomputes tables from included sanitized observations.
It does **not** rerun models or evaluators.

```bash
python3 run.py verify
python3 run.py reproduce
mkdir -p outputs/test-tmp
TMPDIR="$(pwd -P)/outputs/test-tmp" PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`verify` checks file hashes, schemas, units and 856 displayed cells across eleven
tables. `reproduce` writes CSV/Markdown tables to `outputs/all/`. This shortcut
needs no packages, accounts or network. Stored observations never substitute
for new measurements in the fresh experiment scripts.

## Data and Execution Flow

The checked-in `data/` directory contains sanitized experimental observations,
including scores, timing measurements, retained sample identifiers and five-run
results. `expected/` holds independently transcribed paper table values for
comparison. These files support offline verification of the published numbers.

Raw benchmark prompt/answer files and model weights are acquired separately.
`experiments/resources.json` records pinned download sources, revisions, sizes
and available hashes; `experiments/datasets.json` and `sampling.json` record
serialization rules and sampling selections. The acquisition scripts rebuild
the required data under `outputs/assets/data/` and place models under
`outputs/assets/models/`.

Fresh experiments follow this path:

```text
scripts/run_*.py + experiments/ configurations
    -> execution/ downloads data and models, plans and resumes stages
    -> runtime/ fits maps, reuses caches, generates and evaluates
    -> outputs/ stores checkpoints, individual results and summary reports
```

Offline table verification follows `data/ -> pipeline.py -> expected/`.
Local `outputs/`, model weights, fitted tensors, virtual environments and caches
are excluded from the upload bundle.

## Organization

```text
scripts/                   Experiment entry points, setup and acquisition tools
experiments/               Registry, case/pair configs, pinned resource recipes
src/vreparo_artifact/
  runtime/                 Cache operators, VReparo, decoding, judge kernels
  execution/               Acquisition, shared runner, workflows and reports
  metrics.py               Shared metric definitions
  characterization.py      Frozen RQ1/RQ2 table calculations
  evaluation.py            Frozen mitigation/quality/calibration/timing calculations
  repeated.py              Five-run safety means and individual results
  pipeline.py              Offline reproduction and paper-value comparison
  integrity.py             File manifest and disclosure checks
data/                      Sanitized observations, five-run scores and screening references
expected/                  Independently transcribed paper table values
manifests/                 Paper baseline and artifact file hashes
docs/                      Protocols, provenance and validation boundaries
tests/                     Offline, orchestration and CPU numerical tests
outputs/                   Local assets/checkpoints/results; excluded from release
```

## Verified Scope

Automatic acquisition has reconstructed all twelve dataset files from pinned
public sources with exact historical SHA-256 matches. The five stratified
32+32 selections are reconstructed and checked against their frozen IDs.
CPU tests cover operators and orchestration; the ZIP is extracted and tested outside the research
workspace. The full-sized GPU matrix has **not** been rerun for this release.

RQ1/RQ2 reuse historical validity decisions only for identical prompt/response
hashes. Changed responses require review and produce `needs_validity_review`.
Automated judge labels are not human judgments. Timing varies by hardware/load.
Conceptual figures' editable-source regeneration is not included.

See [Data and Models](DATA_AND_MODELS.md), [Validation](VALIDATION.md),
[Evidence Map](EVIDENCE_MAP.md), [Protocol](PROTOCOL.md), and
[Third-Party Components](THIRD_PARTY.md).

When running tests on macOS, use a physical temporary directory as above.
The integrity checks reject symlink path components, including a symlinked
system temporary-directory root. Model experiments use the Linux environments
documented in the execution guide.
