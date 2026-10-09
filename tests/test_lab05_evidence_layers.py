"""Tests for LAB05 Gate 3 evidence layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

EVIDENCE_PATH = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

MANIFEST_PATH = (
    LAB_DIR
    / "results"
    / "evidence_manifest.json"
)

DIAGNOSTICS_PATH = (
    LAB_DIR
    / "results"
    / "geochemistry_diagnostics.json"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

ELEMENTS = [
    "Cu",
    "Mo",
    "Au",
    "As",
    "Sb",
    "Fe",
    "Mg",
    "Ca",
    "Al",
    "K",
]


def load_evidence() -> pd.DataFrame:
    return pd.read_csv(
        EVIDENCE_PATH
    )


def load_manifest() -> dict:
    return json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_gate03_outputs_exist() -> None:
    assert EVIDENCE_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    assert DIAGNOSTICS_PATH.is_file()


def test_gate03_preserves_6400_cells() -> None:
    data = load_evidence()

    assert len(data) == 6400
    assert data["cell_id"].nunique() == 6400


def test_lod_treated_geochemistry_is_positive() -> None:
    data = load_evidence()

    values = data[
        ELEMENTS
    ].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert (values > 0.0).all()


def test_closed_composition_sums_to_one() -> None:
    data = load_evidence()

    columns = [
        f"{element}_closed"
        for element in ELEMENTS
    ]

    row_sums = data[
        columns
    ].sum(axis=1)

    assert np.allclose(
        row_sums,
        1.0,
        atol=1e-10,
    )


def test_clr_rows_sum_to_zero() -> None:
    data = load_evidence()

    columns = [
        f"clr_{element}"
        for element in ELEMENTS
    ]

    row_sums = data[
        columns
    ].sum(axis=1)

    assert np.allclose(
        row_sums,
        0.0,
        atol=1e-10,
    )


def test_ilr_dimension_is_d_minus_one() -> None:
    data = load_evidence()

    columns = [
        column
        for column in data.columns
        if column.startswith("ilr_")
    ]

    assert len(columns) == 9


def test_feature_sets_exist() -> None:
    manifest = load_manifest()

    sets = manifest[
        "feature_sets"
    ]

    assert set(sets) == {
        "A_non_geochemical",
        "B_raw_geochemistry",
        "C_ilr_geochemistry",
    }


def test_raw_feature_set_contains_ten_elements() -> None:
    manifest = load_manifest()

    raw_set = set(
        manifest[
            "feature_sets"
        ][
            "B_raw_geochemistry"
        ]
    )

    assert set(ELEMENTS).issubset(
        raw_set
    )


def test_ilr_feature_set_contains_nine_coordinates() -> None:
    manifest = load_manifest()

    ilr_set = set(
        manifest[
            "feature_sets"
        ][
            "C_ilr_geochemistry"
        ]
    )

    expected = {
        f"ilr_{index}"
        for index in range(
            1,
            10,
        )
    }

    assert expected.issubset(
        ilr_set
    )


def test_no_forbidden_features_enter_model_sets() -> None:
    manifest = load_manifest()

    forbidden = set(
        manifest[
            "forbidden_model_features"
        ]
    )

    for columns in manifest[
        "feature_sets"
    ].values():
        assert not (
            set(columns)
            & forbidden
        )


def test_cell_id_is_model_forbidden() -> None:
    contract = json.loads(
        CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "cell_id"
        in contract[
            "forbidden_model_features"
        ]
    )


def test_oracle_did_not_generate_geochemistry() -> None:
    manifest = load_manifest()

    assert (
        manifest[
            "oracle_used_to_generate_geochemistry"
        ]
        is False
    )


def test_lod_censoring_is_exercised_but_limited() -> None:
    diagnostics = json.loads(
        DIAGNOSTICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        diagnostics["total_censored_values"]
        > 0
    )

    assert (
        diagnostics["censored_fraction"]
        < 0.10
    )


def test_ilr_is_primary_coda_representation() -> None:
    manifest = load_manifest()

    assert (
        manifest[
            "primary_geochemical_representation"
        ]
        == "ILR"
    )

    assert (
        manifest["clr_role"]
        == "diagnostic_comparison"
    )
