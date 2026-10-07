"""Compare the regenerated LAB04 outputs with the pre-refactor baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
RESULTS = LAB_DIR / "results"
BASELINE = json.loads((LAB_DIR / "baseline_sha256.json").read_text(encoding="utf-8"))
FILES = {
    "dataset": LAB_DIR / "data" / "synthetic_geochemistry.csv",
    "analysis_summary": RESULTS / "analysis_summary.json",
    "lod_sensitivity_summary": RESULTS / "lod_sensitivity_summary.json",
    "pca_explained_variance": RESULTS / "pca_explained_variance.csv",
}


def main() -> None:
    matches = {}
    for name, path in FILES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        matches[name] = digest == BASELINE["files"][name]
        print(f"SHA256_{name.upper()} = {'MATCH' if matches[name] else 'DIFFERENT'}")

    analysis = json.loads((RESULTS / "analysis_summary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((RESULTS / "lod_sensitivity_summary.json").read_text(encoding="utf-8"))
    expected = {
        "input_rows": 1200,
        "complete_rows": 1092,
        "excluded_missing_rows": 108,
        "elements": 10,
        "clr_dimension": 10,
        "clr_rank": 9,
        "ilr_dimension": 9,
    }
    for name, value in expected.items():
        if analysis[name] != value:
            raise AssertionError(f"Analysis regression: {name} differs.")
    if analysis["coda_eigenvalues_match"] is not True:
        raise AssertionError("CLR/ILR PCA eigenvalue equivalence failed.")
    numeric = {
        "raw_pc1_variance": 0.304756782810416,
        "closed_pc1_variance": 0.3104532518375104,
        "clr_pc1_variance": 0.2255979771466171,
        "ilr_pc1_variance": 0.225597977146663,
    }
    for name, value in numeric.items():
        if not np.isclose(analysis[name], value, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"Analysis regression: {name} differs.")
    if sensitivity["common_samples"] != 1092:
        raise AssertionError("Sensitivity sample count differs.")
    for key, target in (
        ("pc1_score_correlation", 0.9836860998093131),
        ("pc2_score_correlation", 0.9812981793421425),
        ("pc3_score_correlation", 0.9544328110360077),
        ("pc1_explained_variance_absolute_difference", 0.02089763195085495),
    ):
        if not np.isclose(sensitivity[key], target, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"Sensitivity regression: {key} differs.")
    explained = pd.read_csv(RESULTS / "pca_explained_variance.csv")
    assert len(explained) == 39
    assert set(explained["representation"]) == {"raw", "closed", "clr", "ilr"}
    print("LAB04_NUMERIC_REGRESSION = PASS")
    if all(matches.values()):
        print("LAB04_BYTEWISE_REPRODUCIBILITY = PASS")
    else:
        print("LAB04_BYTEWISE_REPRODUCIBILITY = REVIEW_PLATFORM_OR_SERIALIZATION")


if __name__ == "__main__":
    main()
