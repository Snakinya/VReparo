# Five Sampling Runs

This experiment supplies the current paper's mean VReparo rows in Tables 3
and 10, and all 20 rows of Table 11. There are four model/reuse configurations
and five sampling seeds (0–4). Each run fits maps to 32 HotpotQA and 32 HEx-PHI
examples, then selects alpha independently on the full XSTest set.

## Run from downloaded public inputs

```bash
python3 scripts/setup.py
.venv-models/bin/huggingface-cli login
python3 scripts/run_repeated_mitigation.py --case all --dry-run
python3 scripts/run_repeated_mitigation.py --case all --gpu 0 --accept-model-licenses
```

Use `--case code_direct`, `code_droid`, `base_direct` or `base_droid` for one
configuration. All stages execute sequentially and resume using fingerprints.
The first run downloads pinned models and datasets. Shared assets can be
placed on another disk with `--assets`. No fitted maps or model weights are
included in the ZIP.

Each of the 20 runs has its own fit, calibration, selected alpha, generation
and evaluator checkpoints in `outputs/experiments/<case>/repeated_mitigation/seed_<n>/`.
Calibration output is bound to the fitted map by hash. Evaluation uses that
run's selected alpha, not the historical fixed alpha in `paper_cases.json`.
An ineligible or incomplete seed stops aggregation rather than being omitted.

Outputs are in `outputs/reports/repeated_mitigation/<case-or-all>/`:

- `summary.json`: unrounded individual metrics and five-run means.
- `individual.csv` and `individual.md`: one row per sampling seed/configuration.
- `means.csv` and `means.md`: means over all five seeds for each configuration.

The summary JSON records metric units. HarmBench and OR-Bench rates and
HotpotQA F1 are stored as fractions, multiplied by 100 for the paper.

## Sampling and fixed settings

The source pools have 200 HotpotQA and 330 HEx-PHI examples. Excluding exact
ID or normalized-text overlaps with profiling/evaluation leaves 143 QA and
330 safety candidates. QA is balanced across three length strata and safety
across all 11 categories. Selections may overlap across seeds. Each seed uses
the same selected examples in all four configurations, fitted separately.

`execution/sampling.py` reconstructs the pools from acquired public sources,
replays the seeded sampling algorithm, and checks every selected index against
`experiments/sampling.json`. Fitting preserves source order, uses ridge 1 and
at most 128 positions per text. DroidSpeak keeps receiver layers [0,1] fixed.

Each run evaluates the same 127 HarmBench, 313 StrongREJECT, 200 OR-Bench and
248 HotpotQA prompts. The research jobs also scored the larger HarmBench159
population, but the paper and this workflow use the specified 127 prompts.
XSTest's 450 prompts are used only for alpha selection.

Compute paired Delta/SD/SI/SDV separately for each seed, then average those
metrics. Averaging prompt scores before computing SDV is incorrect because
opposite changes can cancel. All seeds contribute equally. These seeds vary
the fitting examples, not decoding randomness. Decoding is greedy.

## Recompute the published tables without a GPU

```bash
python3 run.py verify
python3 run.py reproduce --experiment mitigation --output outputs/mitigation
python3 run.py reproduce --experiment repeated_mitigation --output outputs/seeds
```

`data/repeated.json` contains aligned, text-free per-example safety scores,
refusal labels, QA F1 and XSTest refusal observations. The verifier recalculates
all metrics and all 20 alpha choices. Expected cells are separately transcribed
from the manuscript in `expected/repeated.json`.

Tables 4/8 (timing), 7 (fixed-map calibration) and 9 (additional QA) retain
their existing fixed-map measurements in `data/evaluation.json`. TTFT averages
50 prompts and five timing repetitions per prompt, not five sampling seeds.
Unfinished budget sweeps and extra experiments outside the current manuscript
are not included. This packaging pass verifies records and CPU/orchestration
tests. It does not rerun the full GPU matrix or claim fresh evaluator parity.
