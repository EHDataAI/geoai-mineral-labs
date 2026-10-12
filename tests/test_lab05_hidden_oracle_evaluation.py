"""Tests for LAB05 Gate 6 post-freeze oracle evaluation."""

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

RESULTS_DIR = LAB_DIR / "results"

SUMMARY_PATH = (
    RESULTS_DIR
    / "oracle_evaluation_summary.csv"
)

HIDDEN_RANKS_PATH = (
    RESULTS_DIR
    / "oracle_hidden_deposit_ranks.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "oracle_evaluation_manifest.json"
)


def test_gate06_outputs_exist() -> None:
    for path in (
        SUMMARY_PATH,
        HIDDEN_RANKS_PATH,
        MANIFEST_PATH,
    ):
        assert path.is_file()


def test_oracle_accounting() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "candidate_rows"
        ]
        == 6300
    )

    assert (
        manifest[
            "known_positive_rows"
        ]
        == 100
    )

    assert (
        manifest[
            "true_deposit_rows"
        ]
        == 215
    )

    assert (
        manifest[
            "hidden_deposit_rows"
        ]
        == 115
    )


def test_oracle_is_post_freeze_only() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "oracle_loaded_for_final_evaluation"
        ]
        is True
    )

    assert (
        manifest[
            "model_or_features_changed_after_oracle"
        ]
        is False
    )

    assert (
        manifest[
            "model_selection_after_oracle"
        ]
        is False
    )

    assert (
        manifest[
            "feature_selection_after_oracle"
        ]
        is False
    )

    assert (
        manifest[
            "threshold_selection_after_oracle"
        ]
        is False
    )


def test_frozen_model_is_preserved() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "primary_feature_set"
        ]
        == "A"
    )

    assert (
        manifest[
            "primary_model"
        ]
        == "logistic_regression"
    )


def test_hidden_rank_rows_are_exact() -> None:
    frame = pd.read_csv(
        HIDDEN_RANKS_PATH
    )

    assert len(frame) == 115

    assert frame[
        "cell_id"
    ].is_unique

    assert frame[
        "candidate_rank"
    ].between(
        1,
        6300,
        inclusive="both",
    ).all()


def test_tier_counts_are_exact() -> None:
    frame = pd.read_csv(
        SUMMARY_PATH
    )

    expected = {
        "top_1pct": 63,
        "top_5pct": 315,
        "top_10pct": 630,
    }

    assert set(
        frame[
            "tier"
        ]
    ) == set(
        expected
    )

    for tier, count in (
        expected.items()
    ):
        row = frame.loc[
            frame[
                "tier"
            ]
            == tier
        ].iloc[0]

        assert int(
            row[
                "candidate_cells"
            ]
        ) == count


def test_capture_is_monotonic() -> None:
    frame = pd.read_csv(
        SUMMARY_PATH
    ).set_index(
        "tier"
    )

    captured = [
        int(
            frame.loc[
                tier,
                "hidden_deposits_captured",
            ]
        )
        for tier in (
            "top_1pct",
            "top_5pct",
            "top_10pct",
        )
    ]

    assert (
        captured[0]
        <= captured[1]
        <= captured[2]
    )


def test_final_metrics_are_bounded() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        0.0
        <= manifest[
            "final_hidden_average_precision"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= manifest[
            "final_hidden_roc_auc"
        ]
        <= 1.0
    )


def test_enrichment_metrics_are_valid() -> None:
    frame = pd.read_csv(
        SUMMARY_PATH
    )

    assert (
        frame[
            "hidden_capture_rate"
        ].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ).all()

    assert (
        frame[
            "precision_hidden_deposit"
        ].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ).all()

    assert (
        frame[
            "lift_over_random"
        ]
        >= 0.0
    ).all()

    assert (
        frame[
            "enrichment_factor"
        ]
        >= 0.0
    ).all()
