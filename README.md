# Tree completion barriers

Reproducible computational checks accompanying **Exact Observation-Loss Barriers for Rank-Two Tree Completions**, by Kecen Li (University of Bristol), an unsubmitted manuscript (2026).

## Reproduce

Tested with Python 3.12 on Linux. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The command regenerates the six files in `outputs/` and runs 76 regression tests. The dependencies are pinned in `requirements.txt`. On Windows activate `.venv\Scripts\activate` instead. No external dataset, GPU or model weights are required.

## Contents and scope

| Program | Checks |
| --- | --- |
| `scripts/tree_planner.py` | Directed tree factor curves, complete endpoints, folds through singular neighboring blocks and two-hop reconstruction |
| `scripts/noisy_tree.py` | Inexact endpoint repair and the nearby target set |
| `scripts/operator_sensitivity.py` | Observation-operator perturbation, including symbolic identities |
| `scripts/cycle_scope.py` | Four-cycle construction and explicit separation certificate |

The CSV tables contain synthetic constructions, not empirical training results. Returned arrays sample factor curves: linear interpolation between their matrices need not preserve rank. The all-path lower bounds and exact attainment claims are proved in the manuscript; numerical sampling alone does not establish them. The PSD separation proof is in the manuscript, not a separate numerical certificate in this repository.

## Provenance and AI assistance

The research question and the overall strategy and structure of the main theorem's proof were proposed by the author. OpenAI Codex contributed substantially to draft mathematical arguments, construction code, tests, literature comparisons and manuscript preparation. The author has reviewed and verified the complete manuscript, made the final mathematical and editorial decisions, and accepts responsibility for its correctness and integrity.

This standalone distribution retains the research programs and tests, with a small CSV-writing helper replacing an import from the wider research workspace. It contains no private discussions or third-party paper files.

## Citation and license

Please use `CITATION.cff` and identify the exact commit used. The manuscript is not claimed to be published or accepted. Code and generated synthetic tables are provided under the MIT license.
