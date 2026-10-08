# Third-Party Components

The implementation uses external libraries by import; their sources and model
weights are not vendored. Torch, Transformers, PEFT, Accelerate, safetensors,
NumPy, SentencePiece, PyArrow, tokenizers and huggingface-hub retain their own
licenses. The setup script installs them into isolated environments.

The following text/protocol components have upstream origins:

| Component | Source | Recorded reference |
| --- | --- | --- |
| HarmBench classifier prompt and response clipping | centerforaisafety/HarmBench | 8e1604d1171fe8a48d8febecd22f600e462bdcdd |
| StrongREJECT fine-tuned judge prompt and scoring | dsbowen/strong_reject | 7a551d5b440ec7b75d4f6f5bb7c1719965b76b47 |
| XSTest string-refusal matching | paul-rottger/xstest | d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d |
| LoRI prompt formatting | juzhengz/LoRI | f57931df86d4bbc49bdd76dedcbaa6ce06c1de40 |
| LongBench prompts and QA F1 conventions | THUDM/LongBench | 2e00731f8d0bff23dc4325161044d0ed8af94c1e |

These references identify the upstream components. Their licenses and
redistribution terms apply separately, as do model and dataset access terms.

`manifests/operators.json` records the exact local frozen source hash and the
exported operator hash. Most kernels are copied verbatim; selected definitions
are extracted with explicit package-local imports. New orchestration lives in
`execution/` and is not represented as historically executed code.

## Licensing scope

Original project code and software documentation use MIT. The manuscript,
figures, external datasets/model weights, and borrowed third-party material
are outside that grant.

HarmBench, StrongREJECT, and LongBench use MIT at the pinned revisions above.
Their full copyright and license notices are retained in `licenses/`.
Source license files:

- https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/LICENSE
- https://github.com/dsbowen/strong_reject/blob/7a551d5b440ec7b75d4f6f5bb7c1719965b76b47/LICENSE
- https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LICENSE

XSTest identifies its prompts as CC BY 4.0; attribution is to Paul Roettger
and the XSTest authors. Its imported refusal-matching protocol is not
relicensed under the project MIT grant. See https://github.com/paul-rottger/xstest
and https://creativecommons.org/licenses/by/4.0/ . No benchmark prompt text
or model completions are bundled.

No explicit repository license was identified for LoRI. Its imported prompt
formatting/protocol text remains outside the project MIT grant; public
repository access is not represented as permission to relicense it. See
https://github.com/juzhengz/LoRI . Model adapters are acquired separately.
