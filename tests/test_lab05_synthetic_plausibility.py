"""Tests for LAB05 Gate 3.1 synthetic plausibility audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

MODULE_PATH = (
    LAB_DIR
    / "05a1_audit_synthetic_plausibility.py"
)

RESULTS_DIR = LAB_DIR / "results"

METRICS_PATH = (
    RESULTS_DIR
    / "synthetic_plausibility_metrics.csv"
)

SPATIAL_PATH = (
    RESULTS_DIR
    / "synthetic_spatial_diagnostics.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "synthetic_plausibility_manifest.json"
)


def load_module():
    spec = (
        importlib.util.spec_from_file_location(
            "lab05_gate03_1",
            MODULE_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load Gate 3.1 module."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_outputs_exist() -> None:
    for path in (
        METRICS_PATH,
        SPATIAL_PATH,
        MANIFEST_PATH,
    ):
        assert path.is_file()


def test_safe_loader_contract() -> None:
    module = load_module()

    frame = (
        module.load_safe_data()
    )

    assert len(frame) == 6400

    assert int(
        frame[
            "known_positive"
        ].sum()
    ) == 100

    assert not (
        set(frame.columns)
        & module.FORBIDDEN_ORACLE_FIELDS
    )


def test_manifest_has_no_oracle_access() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "oracle_values_loaded"
        ]
        is False
    )

    assert (
        manifest[
            "dataset_modified"
        ]
        is False
    )

    assert (
        manifest[
            "model_selected"
        ]
        is False
    )


def test_marginal_rows_present() -> None:
    metrics = pd.read_csv(
        METRICS_PATH
    )

    marginal = metrics.loc[
        metrics[
            "analysis"
        ]
        == "marginal_distribution"
    ]

    assert len(
        marginal
    ) == 18


def test_all_geochemistry_is_audited() -> None:
    metrics = pd.read_csv(
        METRICS_PATH
    )

    variables = set(
        metrics.loc[
            metrics[
                "analysis"
            ]
            == "marginal_distribution",
            "variable",
        ]
    )

    assert {
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
    }.issubset(
        variables
    )


def test_spatial_diagnostics_present() -> None:
    spatial = pd.read_csv(
        SPATIAL_PATH
    )

    expected = {
        "magnetic_anomaly",
        "gravity_anomaly",
        "alteration_index",
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
    }

    assert set(
        spatial[
            "variable"
        ]
    ) == expected


def test_spatial_metrics_finite() -> None:
    spatial = pd.read_csv(
        SPATIAL_PATH
    )

    for column in (
        "rook_moran_like",
        "lag1_semivariance",
        "global_variance",
        "lag1_semivariance_variance_ratio",
    ):
        assert spatial[
            column
        ].notna().all()


def test_manifest_row_counts() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "rows"
        ]
        == 6400
    )

    assert (
        manifest[
            "known_positive_rows"
        ]
        == 100
    )

    assert (
        manifest[
            "unlabeled_rows"
        ]
        == 6300
    )
