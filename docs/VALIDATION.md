# Validation Boundary

## Offline Evidence

The standalone verifier checks 11 tables, 94 rows, and 856 displayed cells
against independent paper reference values. It also checks source-file hashes,
schema/ID alignment, retained subsets, units, and disclosure patterns. It runs
without site packages and without any files from the research workspace.

## Method Tests

The optional CPU test suite checks:

- Identity-centered ridge against a direct centered-matrix solve.
- Exact Value interpolation with the historical intermediate dtype rounding.
- Unchanged Keys, inactive receiver block, source tensors, and alpha-zero path.
- Independent K/V ownership and boundary-cache truncation.
- Greedy cached decoding versus native generation on a tiny random Llama model.
- Same-model DroidSpeak prefix recomputation and receiver suffix caches.
- Input IDs, condition alignment, strict XSTest selection, and atomic JSON output.
- Generation resume, shard ordering, changed-input detection, and duplicate rejection.
- Cached download checks, exact dataset serialization and insufficient-storage gates.
- All eight workflow plans, interpreter selection, parser compatibility and dry-run isolation.
- All five stratified selections, 20 alpha choices, per-seed metrics and mean aggregation.
- Fresh score alignment/units, hash-bound screening decisions and invalid-label reporting.
- Scoring resume without reloading models; changed checkpoints are rejected.
- Tiny Gemma2/Mistral native versus cache decoding; tiny Qwen3 in Transformers 4.53.3.

An author-side CPU comparison also checked all 64 formatted fitting texts and
prefix boundaries in each of the four cases against the historical saved
training inputs: 256/256 matched exactly. That check does not ship the texts.
The five-run extension additionally matches all 1,280 formatted fitting inputs
against the saved per-seed experiment hashes, without distributing their text.

Run both layers with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-models/bin/python -m unittest discover -s tests -v
```

The first command skips numerical tests if optional libraries are absent.
The second deliberately disables GPU visibility. Tiny random models are test
fixtures, not evidence of paper-level quality or full checkpoint token parity.

## Not Claimed

Acquisition was tested from the six pinned public sources: all twelve rebuilt
dataset files matched their historical file SHA-256 exactly. The extracted ZIP
is tested again using those downloaded source files as a cache, not author-made
subsets. Model endpoint checks compare small configuration files; gated resources
require login. Full weights were not downloaded again for this packaging pass.

This packaging pass does not rerun the full GPU experiment matrix, independently
re-judge historical form-validity labels, or revalidate every model architecture
and evaluator environment. Fresh scoring outputs explicitly mark their parity
as unverified. Exact existing numbers are supported by the frozen observations,
not inferred from new smoke tests. The completed five-run 32+32 experiment is
included. Unfinished budget sweeps and other results outside the paper are not.

Timing tests require an idle GPU and licensed models; CPU tests do not certify
published latency. Figure regeneration is not included.
