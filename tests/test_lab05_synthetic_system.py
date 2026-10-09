"""Tests for LAB05 Gate 1 synthetic mineral system."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

DATA_PATH = (
    LAB_DIR
    / "data"
    / "synthetic_mineral_system.csv"
)

SUMMARY_PATH = (
    LAB_DIR
    / "results"
    / "synthetic_system_summary.json"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)


def load_data() -> pd.DataFrame:
    """Load Gate 1 synthetic data."""
    return pd.read_csv(DATA_PATH)


def test_gate01_outputs_exist() -> None:
    assert DATA_PATH.is_file()
    assert SUMMARY_PATH.is_file()


def test_grid_has_expected_size() -> None:
    data = load_data()

    assert len(data) == 6400
    assert data["cell_id"].nunique() == 6400


def test_coordinates_are_unique() -> None:
    data = load_data()

    duplicates = data[
        ["Easting", "Northing"]
    ].duplicated()

    assert not duplicates.any()


def test_oracle_ranges_are_valid() -> None:
    data = load_data()

    assert data[
        "latent_favourability"
    ].between(0.0, 1.0).all()

    assert data[
        "true_deposit_probability"
    ].between(0.0, 1.0).all()


def test_true_deposit_is_rare_but_present() -> None:
    data = load_data()

    count = int(
        data["true_deposit"].sum()
    )

    assert 5 <= count <= 500


def test_known_positives_are_true_deposits() -> None:
    data = load_data()

    known = data.loc[
        data["known_positive"] == 1
    ]

    assert len(known) > 0
    assert known["true_deposit"].eq(1).all()


def test_hidden_deposits_exist() -> None:
    data = load_data()

    hidden = data.loc[
        (data["true_deposit"] == 1)
        & (data["known_positive"] == 0)
    ]

    assert len(hidden) > 0


def test_spatial_blocks_are_available() -> None:
    data = load_data()

    assert (
        data["spatial_block_id"].nunique()
        == 64
    )


def test_oracle_columns_are_model_forbidden() -> None:
    contract = json.loads(
        CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    forbidden = set(
        contract[
            "forbidden_model_features"
        ]
    )

    required = {
        "true_deposit",
        "latent_favourability",
        "true_deposit_probability",
    }

    assert required.issubset(forbidden)


def test_summary_matches_dataset() -> None:
    data = load_data()

    summary = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    true_count = int(
        data["true_deposit"].sum()
    )

    known_count = int(
        data["known_positive"].sum()
    )

    assert summary["rows"] == len(data)
    assert summary["true_deposits"] == true_count
    assert summary["known_positives"] == known_count
    assert (
        summary["hidden_deposits"]
        == true_count - known_count
    )
