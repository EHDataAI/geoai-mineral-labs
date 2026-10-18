"""Phase D tests for LABORATORY 06."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "labs" / "lab06_explainable_geoai" / "results"


def test_phase_d_local_case_contract():
    cases = pd.read_csv(R / "local_explanation_cases.csv")

    assert len(cases) == 5
    assert cases["candidate_rank"].tolist() == [1, 63, 315, 3150, 6300]
    assert cases["cell_id"].nunique() == 5
    assert cases["selection_rule"].str.startswith(
        "candidate_rank_equals_"
    ).all()


def test_phase_d_local_contributions_complete():
    cases = pd.read_csv(R / "local_explanation_cases.csv")
    local = pd.read_csv(R / "local_shap_contributions.csv")

    assert set(local["cell_id"].astype(str)) == set(
        cases["cell_id"].astype(str)
    )
    assert local["feature"].nunique() == 9
    assert local.groupby("cell_id")["feature"].nunique().eq(9).all()
    assert local["oos_explanation_count"].between(4, 10).all()
    assert local["direction"].isin(
        ["positive", "negative", "neutral"]
    ).all()


def test_phase_d_spatial_contract():
    manifest = json.loads(
        (R / "spatial_explainability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    spatial = pd.read_csv(R / "spatial_shap_contributions.csv.gz")
    dominant = pd.read_csv(R / "dominant_driver_by_cell.csv")

    assert manifest["status"] == "PASS"
    assert manifest["total_oos_explanations"] == 58000
    assert manifest["candidate_rows"] == 6300
    assert manifest["raw_feature_count"] == 9
    assert manifest["spatial_long_rows"] == 56700
    assert manifest["dominant_driver_rows"] == 6300
    assert manifest["support_min"] >= 4
    assert manifest["support_max"] <= 10
    assert (
        manifest["maximum_additivity_error"]
        <= manifest["additivity_tolerance"]
    )
    assert manifest["oracle_loaded"] is False
    assert manifest["model_reselected"] is False
    assert len(spatial) == 56700
    assert len(dominant) == 6300


def test_phase_d_dominant_driver_is_valid():
    spatial = pd.read_csv(R / "spatial_shap_contributions.csv.gz")
    dominant = pd.read_csv(R / "dominant_driver_by_cell.csv")

    assert dominant["cell_id"].nunique() == 6300
    assert set(dominant["dominant_feature"]).issubset(
        set(spatial["feature"])
    )
    assert set(dominant["dominant_family"]).issubset(
        {
            "lithology",
            "intrusive",
            "structure",
            "geophysics",
            "alteration",
        }
    )
    assert dominant["candidate_rank"].between(1, 6300).all()
