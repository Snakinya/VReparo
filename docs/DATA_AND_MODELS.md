# Data and Models

## Acquisition

Experiment scripts acquire only the models/datasets they need.
`experiments/resources.json` pins six dataset sources, twelve exact input
reconstruction recipes and seventeen model/tokenizer/adapter directories.

```bash
.venv-models/bin/python scripts/prepare_data.py --assets outputs/assets
.venv-models/bin/python scripts/prepare_data.py --assets outputs/assets --datasets xstest harmbench
python3 scripts/run_rq1.py --pair all --dry-run
```

Preparation downloads a pinned file, verifies SHA-256, normalizes source fields,
selects recorded content hashes in order, and reproduces the original
serialization. Output must match the historical file SHA-256 exactly. All
twelve recipes have passed this test using actual public-source downloads.

| Name | Rows | Role |
| --- | ---: | --- |
| `train_qa` | 200 | Original HotpotQA fitting source, JSONL |
| `train_safety` | 330 | Original HEx-PHI fitting source |
| `profile` | 50 | QA-only recomputation profiling and timing |
| `xstest` | 450 | Alpha selection, 250 safe and 200 unsafe |
| `harmbench` | 127 | Mitigation confirmation |
| `harmbench159` | 159 | RQ1/RQ2 before common-valid screening |
| `strongreject` | 313 | Continuous safety scoring |
| `orbench` | 200 | Benign full refusal |
| `qa_full` | 149 | Fixed-map additional QA for direct reuse |
| `qa_droid` | 248 | HotpotQA evaluation for all five-run cases and fixed-map DroidSpeak |
| `2wikimqa` | 200 | Additional QA evaluation |
| `multifieldqa_en` | 150 | Additional QA evaluation |

Files: `<assets>/data/<name>.json`, except `train_qa.jsonl`. Download cache:
`<assets>/downloads/`; model layout: `<assets>/models/`. Existing checksum
mismatches are rejected. Move a corrupted file aside and retry.

Historical fitting selects offsets 0/1 modulo four from each source's first
64 rows, yielding 32+32. This is not new stratified sampling. Profiling uses
50 rows of the original 200-row QA source, not the held-out QA pool.

The current five-run experiment instead draws 32 QA and 32 safety examples
for each seed from filtered pools of 143 and 330 examples. It balances QA
length groups and safety categories. See [FIVE_RUNS.md](FIVE_RUNS.md) for the
filtering, sampling and independent calibration procedure.

## Public Sources

- HarmBench: https://github.com/centerforaisafety/HarmBench
- StrongREJECT: https://huggingface.co/datasets/AlignmentResearch/StrongREJECT
- XSTest: https://github.com/paul-rottger/xstest
- HEx-PHI source used by LoRI: https://github.com/juzhengz/LoRI
- LongBench: https://huggingface.co/datasets/THUDM/LongBench
- OR-Bench: https://huggingface.co/datasets/orbench-llm/or-bench

Exact commits, paths and hashes are in `resources.json`. No raw benchmark
prompts, completions or weights are redistributed. Review upstream terms before
downloading or redistributing derivatives.

## Model Pins and Storage

Included configurations cover LoRI Code/Safety, Llama-3-8B, Qwen3-4B/8B,
Llama-3.2-3B, Gemma-2-2B, Mistral-7B-v0.3, and their evaluation models. LoRI's
saved embeddings/LM head and pad-token extension are retained.

HF acquisition uses explicit immutable revisions and filename lists. Original
Llama-3.2/Gemma-2 ModelScope snapshots retain their per-file revisions. All
files are checked by size and files below 100 MB also by SHA-256. Full local
weight hashes are recorded in execution fingerprints. Revision plus size is
not a cryptographic comparison of large weights against the authors' copies.

No credentials are bundled. Accept applicable upstream agreements and use
normal Hugging Face login. Authentication failures must be resolved with the
provider; no alternative checkpoint or access bypass is attempted. A public
mirror does not remove upstream model license obligations.

Dry-run reports missing model bytes. Acquisition reserves another 2 GiB; allow
more for environments, library caches and checkpoints. Full-matrix storage is
much larger than a single case. Downloads and fitted maps are shared across
experiments. No measured full-matrix runtime for this portable package is claimed.

Empty-cache model storage estimates (excluding environments and generated files):
Code/direct QA 19.5 GiB; Code/direct mitigation including all judges 63.8 GiB;
all four mitigation cases 78.7 GiB; all eight RQ1 pairs and judges 162.7 GiB.
