from __future__ import annotations

import json
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = LAB_DIR / "05_mineral_prospectivity.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


cells = [
    md(
        """
# LAB05 - Mineral Prospectivity Mapping

**Public portfolio notebook**

This notebook is a reproducible scientific walkthrough of the validated LAB05
workflow. It does not reimplement or retrain the production experiment.

Instead, it reads the persisted artifacts produced by Gates 0-6, checks the
frozen headline metrics, and presents the evidence used to support the public
portfolio conclusions.

All mineral-system, geochemical, occurrence, and prospectivity data used here
are synthetic.
"""
    ),
    md(
        """
## Scientific question

Can spatially validated machine learning integrate geological, structural,
geophysical, and compositional geochemical evidence to rank mineral
prospectivity while minimizing spatial leakage and exploration area?

The validated experiment uses:

- EPSG:32719;
- an 80 km x 80 km synthetic study area;
- a 1 km grid with 6,400 cells;
- 100 known synthetic occurrences;
- 115 sealed hidden deposits;
- spatially grouped cross-validation;
- positive-unlabeled background sampling;
- Average Precision as the primary model-selection metric;
- a sealed hidden-oracle evaluation only after model freeze.
"""
    ),
    code(
        """
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd

LAB_DIR = Path.cwd()
if LAB_DIR.name != "lab05_mineral_prospectivity":
    candidate = Path("labs/lab05_mineral_prospectivity")
    if candidate.exists():
        LAB_DIR = candidate.resolve()

RESULTS_DIR = LAB_DIR / "results"
README_PATH = LAB_DIR / "README.md"

required = [
    RESULTS_DIR / "benchmark_oof_aggregate_metrics.csv",
    RESULTS_DIR / "benchmark_oof_metrics.csv",
    RESULTS_DIR / "success_rate_curve_oof.csv",
    RESULTS_DIR / "prospectivity_ranking_summary_oos.csv",
    RESULTS_DIR / "prospectivity_scores_oos.csv",
    README_PATH,
]

missing = [str(path) for path in required if not path.exists()]
assert not missing, f"Missing validated artifacts: {missing}"

print(f"LAB_DIR: {LAB_DIR}")
print("Validated artifact inventory: PASS")
"""
    ),
    md(
        """
## Frozen experimental design

Feature families were frozen before hidden-oracle evaluation.

The selected model was determined only from the predeclared primary criterion:
mean pooled spatial out-of-fold Average Precision.

Random stratified cross-validation is retained only as a diagnostic comparison;
it is not used to select the final model.
"""
    ),
    code(
        """
benchmark = pd.read_csv(
    RESULTS_DIR / "benchmark_oof_aggregate_metrics.csv"
)

print("Aggregate benchmark shape:", benchmark.shape)
display(benchmark)
"""
    ),
    md(
        """
## Model-selection result

The frozen selection is **Feature A + Logistic Regression**.

Validated spatial pooled out-of-fold metrics:

- mean Average Precision: 0.393412;
- median Average Precision: 0.380985;
- mean ROC-AUC: 0.734202.

Competing configurations were close. Secondary metrics did not replace the
predeclared primary selection rule.
"""
    ),
    code(
        """
readme = README_PATH.read_text(encoding="utf-8")

headline_metrics = {
    "mean_spatial_oof_ap": "0.393412",
    "median_spatial_oof_ap": "0.380985",
    "mean_spatial_oof_roc_auc": "0.734202",
}

for name, value in headline_metrics.items():
    assert value in readme, f"{name} is inconsistent with the validated README"

print("Frozen model-selection metrics: PASS")
"""
    ),
    md(
        """
## Out-of-sample prospectivity ranking

After model selection, LAB05 generates ranking scores for the 6,300 unlabeled
candidate cells.

A candidate cell is scored only by replicate models for which that cell was not
sampled as training background. The resulting `prospectivity_score` is an
ensemble ranking score, **not a calibrated deposit probability**.
"""
    ),
    code(
        """
ranking_summary = pd.read_csv(
    RESULTS_DIR / "prospectivity_ranking_summary_oos.csv"
)
scores = pd.read_csv(
    RESULTS_DIR / "prospectivity_scores_oos.csv"
)

print("Ranking summary shape:", ranking_summary.shape)
print("Candidate score table shape:", scores.shape)
display(ranking_summary.head(20))
"""
    ),
    code(
        """
def find_column(frame, exact=(), contains=()):
    lookup = {column.lower(): column for column in frame.columns}

    for candidate in exact:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for column in frame.columns:
        low = column.lower()
        if all(token.lower() in low for token in contains):
            return column

    return None


x_col = find_column(scores, exact=("Easting", "easting", "x"))
y_col = find_column(scores, exact=("Northing", "northing", "y"))

score_col = find_column(
    scores,
    exact=(
        "prospectivity_score",
        "prospectivity_score_mean",
        "mean_score",
        "score_mean",
    ),
)

if score_col is None:
    score_col = find_column(scores, contains=("score", "mean"))

if x_col and y_col and score_col:
    plot_data = scores[[x_col, y_col, score_col]].dropna()

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        plot_data[x_col],
        plot_data[y_col],
        c=plot_data[score_col],
        s=12,
    )
    ax.set_title("LAB05 out-of-sample prospectivity ranking")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.colorbar(scatter, ax=ax, label=score_col)
    plt.show()
else:
    print(
        "Map preview skipped because coordinate/score column names "
        "could not be resolved safely."
    )
"""
    ),
    md(
        """
## Sealed hidden-oracle evaluation

The hidden oracle remained sealed until after model selection and ranking logic
were frozen.

Validated hidden-oracle metrics:

| Metric | Result |
| --- | ---: |
| Hidden deposits | 115 |
| Hidden Average Precision | 0.056650 |
| Hidden ROC-AUC | 0.709070 |
| Hidden captured in top 1% | 6 / 115 |
| Lift in top 1% | 5.22x |
| Hidden captured in top 5% | 25 / 115 |
| Lift in top 5% | 4.35x |
| Hidden captured in top 10% | 44 / 115 |
| Lift in top 10% | 3.83x |
| Median hidden rank | 1117 / 6300 |

The hidden Average Precision must be interpreted relative to the very low
hidden-deposit prevalence. The useful result is enrichment and search-space
reduction, not a claim of calibrated deposit detection.
"""
    ),
    code(
        """
hidden_metrics = [
    "0.056650",
    "0.709070",
    "5.22x",
    "4.35x",
    "3.83x",
]

for value in hidden_metrics:
    assert value in readme, f"Hidden-oracle metric missing from validated README: {value}"

print("Frozen hidden-oracle metrics: PASS")
"""
    ),
    md(
        """
## Validation and reproducibility

The public LAB05 milestone was validated with:

- 163 passing project tests at the v0.2.0 checkpoint;
- pre-commit quality controls;
- Gitleaks secret scanning;
- signed milestone commits;
- a signed v0.2.0 release tag;
- public synthetic data only.

This notebook is intentionally downstream of the validated workflow. Its role
is communication, inspection, and reproducible review of persisted evidence.
"""
    ),
    code(
        """
assert "163 passed" in readme
assert "not a" in readme.lower() and "calibrated" in readme.lower()

print("Public documentation consistency: PASS")
"""
    ),
    md(
        """
## Limitations

1. The entire mineral system is synthetic.
2. Background cells are unlabeled, not confirmed negatives.
3. Scores rank prospectivity; they are not calibrated occurrence probabilities.
4. Spatial validation is methodologically required even though random-CV
   inflation was not empirically demonstrated in every LAB05 comparison.
5. The workflow does not replace field mapping, drilling, geophysical
   acquisition, laboratory assays, or competent geological interpretation.
6. Hidden-oracle performance is evidence for this frozen synthetic experiment,
   not a guarantee of real-world exploration performance.
"""
    ),
    md(
        """
## Conclusion

LAB05 demonstrates an end-to-end mineral prospectivity workflow in which
spatial validation, leakage control, compositional evidence, out-of-fold
evaluation, out-of-sample ranking, model freeze, and hidden-oracle evaluation
are treated as part of the scientific design.

The notebook presents the validated evidence without introducing a second
implementation of the modelling pipeline.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"NOTEBOOK_WRITTEN={NOTEBOOK_PATH}")
