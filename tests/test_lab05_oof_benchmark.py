"""Tests for LAB05 Gate 4.1 pooled OOF benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

RESULTS_DIR = LAB_DIR / "results"

METRICS_PATH = (
    RESULTS_DIR
    / "benchmark_oof_metrics.csv"
)

AGGREGATE_PATH = (
    RESULTS_DIR
    / "benchmark_oof_aggregate_metrics.csv"
)

CURVE_PATH = (
    RESULTS_DIR
    / "success_rate_curve_oof.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "benchmark_oof_manifest.json"
)


def test_gate04_1_outputs_exist() -> None:
    for path in (
        METRICS_PATH,
        AGGREGATE_PATH,
        CURVE_PATH,
        MANIFEST_PATH,
    ):
        assert path.is_file()


def test_oof_metrics_have_180_rows() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    assert len(frame) == 180


def test_oof_aggregate_has_18_rows() -> None:
    frame = pd.read_csv(
        AGGREGATE_PATH
    )

    assert len(frame) == 18


def test_oof_curve_has_1080_rows() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    assert len(frame) == 1080


def test_each_configuration_has_ten_replicates() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    counts = frame.groupby(
        [
            "validation_strategy",
            "feature_set",
            "model",
        ]
    ).size()

    assert len(counts) == 18

    assert (
        counts == 10
    ).all()


def test_exact_oof_coverage() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    assert (
        frame["n_rows"]
        == 600
    ).all()

    assert (
        frame["known_positives"]
        == 100
    ).all()

    assert (
        frame["oof_count_min"]
        == 1
    ).all()

    assert (
        frame["oof_count_max"]
        == 1
    ).all()

    assert (
        frame[
            "missing_oof_predictions"
        ]
        == 0
    ).all()

    assert (
        frame[
            "duplicate_oof_predictions"
        ]
        == 0
    ).all()


def test_oof_metric_ranges() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    for column in (
        "average_precision",
        "roc_auc",
    ):
        assert frame[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_spatial_group_overlap_is_zero() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    spatial = frame.loc[
        frame[
            "validation_strategy"
        ]
        == "spatial"
    ]

    assert (
        spatial[
            "group_overlap_max"
        ]
        == 0
    ).all()


def test_expected_models_features_and_validation() -> None:
    frame = pd.read_csv(
        METRICS_PATH
    )

    assert set(
        frame["model"]
    ) == {
        "dummy",
        "logistic_regression",
        "random_forest",
    }

    assert set(
        frame["feature_set"]
    ) == {
        "A",
        "B",
        "C",
    }

    assert set(
        frame[
            "validation_strategy"
        ]
    ) == {
        "spatial",
        "random",
    }


def test_success_curve_top_k_counts() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    expected = {
        0.01: 6,
        0.05: 30,
        0.10: 60,
        0.20: 120,
        0.30: 180,
        0.50: 300,
    }

    for fraction, n_select in (
        expected.items()
    ):
        subset = frame.loc[
            frame[
                "top_fraction"
            ].sub(
                fraction
            ).abs()
            < 1e-12
        ]

        assert len(subset) == 180

        assert (
            subset[
                "top_k_n"
            ]
            == n_select
        ).all()


def test_success_metrics_are_valid() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    assert frame[
        "known_positive_capture_rate"
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

    assert frame[
        "precision_at_k"
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

    assert (
        frame[
            "positives_captured"
        ]
        >= 0
    ).all()

    assert (
        frame[
            "positives_captured"
        ]
        <= 100
    ).all()


def test_capture_is_monotonic_with_fraction() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    groups = frame.groupby(
        [
            "replicate",
            "validation_strategy",
            "feature_set",
            "model",
        ],
        sort=False,
    )

    for _, group in groups:
        ordered = group.sort_values(
            "top_fraction"
        )

        captured = ordered[
            "positives_captured"
        ].to_numpy()

        assert (
            captured[1:]
            >= captured[:-1]
        ).all()


def test_manifest_semantics_and_invariants() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["gate"]
        == "4.1"
    )

    assert (
        manifest[
            "background_is_confirmed_negative"
        ]
        is False
    )

    assert (
        manifest[
            "oracle_used_for_training"
        ]
        is False
    )

    assert (
        manifest[
            "oracle_used_for_feature_selection"
        ]
        is False
    )

    assert (
        manifest[
            "oracle_used_for_model_selection"
        ]
        is False
    )

    assert (
        manifest[
            "observed_model_fits"
        ]
        == 900
    )

    assert (
        manifest[
            "oof_metric_rows"
        ]
        == 180
    )

    assert (
        manifest[
            "aggregate_rows"
        ]
        == 18
    )

    assert (
        manifest[
            "success_curve_rows"
        ]
        == 1080
    )

    assert (
        manifest[
            "oof_prediction_count_min"
        ]
        == 1
    )

    assert (
        manifest[
            "oof_prediction_count_max"
        ]
        == 1
    )

    assert (
        manifest[
            "missing_oof_predictions_total"
        ]
        == 0
    )

    assert (
        manifest[
            "duplicate_oof_predictions_total"
        ]
        == 0
    )

    assert (
        manifest[
            "spatial_group_overlap_max"
        ]
        == 0
    )
