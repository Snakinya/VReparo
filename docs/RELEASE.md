# Release Notes

This preparation preserves the original paper's observations, expected table
values, model and dataset pins, and experiment algorithms.

The paper repository removes unused collaborator-comment macros and commented
out drafts. The PDF displays the named author block from `authors.tex`, with no
anonymous-review text, review line numbers, or conference header. The
local `preprint.sty` applies these layout settings while preserving the
upstream style file. Font sizes, text dimensions, and title spacing are
unchanged. The author block follows the supplied visual reference with names on one
line, numbered affiliations, and email addresses below. `AUTHORS.json` records the same author information.
The author order is Zhiwei Lin, Jiahao Liu, Yan Liu, and Chengpeng Wang.
Zhiwei Lin and Chengpeng Wang are affiliated with the National University of
Singapore; Jiahao Liu and Yan Liu are affiliated with Ant Group. The provided emails for
all four authors are recorded. The statements following the conclusion now flow without
a manual page break; template font sizes and spacing remain unchanged.

The artifact adds author/citation metadata and changes the CLI's release label
from the anonymous submission label to `research-artifact`. Hashes are updated
for the changed packaging files. The frozen experiment baseline in
`manifests/paper.json` and `manifests/operators.json` is preserved.

`manifests/release.json` records cleaned paper source hashes separately from
the frozen baseline. Removing source comments does not constitute a new
experimental measurement. The full GPU matrix has not been rerun for this
preparation. Licensing remains governed by `NOTICE.md` and upstream terms.

The paper entry point and active sections use `main.tex` and `sections/`.
Historical path/hash records in `manifests/paper.json` remain unchanged;
`manifests/release.json` binds the current standard paths. Original code and
software documentation use MIT, with upstream notices and exclusions retained.

## VReparo GitHub Project

The public project name and upload directory are `VReparo` and `vreparo/`.
The README presents the method, paper results, installation, and experiments;
detailed audit records are retained in `REPRODUCIBILITY.md` and `VALIDATION.md`.
The distribution name is `vreparo`; the existing `vreparo_artifact` Python
namespace and historical report identifiers remain compatible.
Experiment implementations, numerical observations, and frozen provenance
records are unchanged.
