"""Contract tests for LAB05 Mineral Prospectivity Mapping."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "labs" / "lab05_mineral_prospectivity"
CONTRACT_PATH = LAB_DIR / "prospectivity_contract.json"


def load_contract() -> dict:
    """Load LAB05 contract."""
    return json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8")
    )


def test_lab05_contract_exists() -> None:
    """LAB05 must define an explicit machine-readable contract."""
    assert CONTRACT_PATH.is_file()


def test_lab05_grid_definition_is_consistent() -> None:
    """Grid dimensions must reproduce the expected number of cells."""
    contract = load_contract()
    study = contract["study_area"]

    nx = (
        study["easting_max"] - study["easting_min"]
    ) // study["grid_cell_size_m"]

    ny = (
        study["northing_max"] - study["northing_min"]
    ) // study["grid_cell_size_m"]

    assert nx == 80
    assert ny == 80
    assert nx * ny == 6400
    assert study["expected_grid_cells"] == 6400


def test_oracle_columns_are_forbidden_features() -> None:
    """Synthetic truth must never be available to the model."""
    contract = load_contract()

    forbidden = set(contract["forbidden_model_features"])
    diagnostic = set(contract["diagnostic_only_columns"])

    assert diagnostic.issubset(forbidden)


def test_geometry_is_not_a_model_feature() -> None:
    """Absolute coordinates are reserved for geometry and validation."""
    contract = load_contract()

    forbidden = set(contract["forbidden_model_features"])

    assert "Easting" in forbidden
    assert "Northing" in forbidden


def test_spatial_group_is_not_a_predictor() -> None:
    """Spatial fold identifiers must not leak into modelling."""
    contract = load_contract()

    forbidden = set(contract["forbidden_model_features"])

    assert "spatial_block_id" in forbidden


def test_primary_validation_is_spatial() -> None:
    """Spatial validation must be the primary evaluation design."""
    contract = load_contract()

    validation = contract["validation"]

    assert validation["primary"] == "spatial_group_cross_validation"
    assert validation["primary_splitter"] == "StratifiedGroupKFold"
    assert validation["require_zero_group_overlap"] is True


def test_model_scope_is_intentionally_small() -> None:
    """LAB05 v1 must retain the preregistered baseline model scope."""
    contract = load_contract()

    models = contract["models"]

    assert models == {
        "baseline_null": "DummyClassifier",
        "baseline_statistical": "LogisticRegression",
        "primary_nonlinear": "RandomForestClassifier",
    }


def test_oracle_is_sealed_until_model_freeze() -> None:
    """Synthetic oracle may only be opened after model development."""
    contract = load_contract()

    oracle = contract["synthetic_oracle"]

    assert oracle["allowed_during_model_selection"] is False
    assert oracle["allowed_during_hyperparameter_selection"] is False
    assert oracle["allowed_during_feature_selection"] is False
    assert oracle["allowed_after_model_freeze"] is True
