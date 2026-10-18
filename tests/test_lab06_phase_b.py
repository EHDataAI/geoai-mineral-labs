"""Phase B tests for LABORATORY 06."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB06 = ROOT / "labs" / "lab06_explainable_geoai"
RESULTS = LAB06 / "results"


def test_phase_b_reconstruction_manifest_passes():
    manifest = json.loads(
        (RESULTS / "model_reconstruction_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "PASS"
    assert manifest["primary_feature_set"] == "A"
    assert manifest["primary_model"] == "logistic_regression"
    assert manifest["evidence_rows"] == 6400
    assert manifest["candidate_rows"] == 6300
    assert manifest["background_replicates"] == 10
    assert manifest["oracle_loaded"] is False
    assert manifest["model_reselected"] is False
    assert max(
        manifest["maximum_absolute_differences"].values()
    ) <= manifest["absolute_tolerance"]


def test_phase_b_coefficients_are_complete_and_stable():
    frame = pd.read_csv(RESULTS / "transformed_coefficients.csv")

    assert frame["replicate"].nunique() == 10
    assert not frame["coefficient"].isna().any()
    assert not frame["absolute_coefficient"].isna().any()

    counts = frame.groupby("replicate")["transformed_feature"].nunique()
    assert counts.nunique() == 1
    assert int(counts.iloc[0]) >= 9


def test_phase_b_summary_has_stability_metrics():
    summary = pd.read_csv(RESULTS / "coefficient_summary.csv")

    assert len(summary) >= 9
    assert summary["rank_by_mean_absolute_coefficient"].tolist() == list(
        range(1, len(summary) + 1)
    )
    assert summary["sign_stability"].between(0.0, 1.0).all()
    assert summary["absolute_coefficient_mean"].is_monotonic_decreasing
