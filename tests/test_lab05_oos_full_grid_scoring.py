"""Tests for LAB05 Gate 5.1 OOS-only candidate scoring."""

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

SCORES_PATH = (
    RESULTS_DIR
    / "prospectivity_scores_oos.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "prospectivity_ranking_summary_oos.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "prospectivity_model_manifest_oos.json"
)


def load_scores() -> pd.DataFrame:
    return pd.read_csv(
        SCORES_PATH
    )


def test_outputs_exist() -> None:
    for path in (
        SCORES_PATH,
        SUMMARY_PATH,
        MANIFEST_PATH,
    ):
        assert path.is_file()


def test_scores_have_6400_rows() -> None:
    frame = load_scores()

    assert len(frame) == 6400

    assert frame[
        "cell_id"
    ].is_unique


def test_candidate_contract() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    known = frame.loc[
        frame[
            "known_positive"
        ]
        == 1
    ]

    assert len(candidate) == 6300
    assert len(known) == 100

    assert (
        known[
            "candidate_rank"
        ].isna()
    ).all()

    assert (
        known[
            "oos_model_count"
        ]
        == 0
    ).all()


def test_candidate_rank_is_exact_permutation() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    ranks = candidate[
        "candidate_rank"
    ].astype(
        int
    )

    assert set(
        ranks
    ) == set(
        range(
            1,
            6301,
        )
    )


def test_candidate_top_counts_are_exact() -> None:
    frame = load_scores()

    assert int(
        frame[
            "candidate_top_1pct"
        ].sum()
    ) == 63

    assert int(
        frame[
            "candidate_top_5pct"
        ].sum()
    ) == 315

    assert int(
        frame[
            "candidate_top_10pct"
        ].sum()
    ) == 630


def test_candidate_top_sets_are_nested() -> None:
    frame = load_scores()

    assert (
        ~frame[
            "candidate_top_1pct"
        ]
        | frame[
            "candidate_top_5pct"
        ]
    ).all()

    assert (
        ~frame[
            "candidate_top_5pct"
        ]
        | frame[
            "candidate_top_10pct"
        ]
    ).all()


def test_oos_support_is_valid() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    assert (
        candidate[
            "oos_model_count"
        ]
        >= 1
    ).all()

    assert (
        candidate[
            "oos_model_count"
        ]
        <= 10
    ).all()


def test_oos_score_statistics_are_valid() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    for column in (
        "prospectivity_score_oos_mean",
        "prospectivity_score_oos_median",
        "prospectivity_score_oos_p25",
        "prospectivity_score_oos_p75",
    ):
        assert candidate[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()

    assert (
        candidate[
            "prospectivity_score_oos_std"
        ]
        >= 0.0
    ).all()

    assert (
        candidate[
            "prospectivity_score_oos_iqr"
        ]
        >= 0.0
    ).all()


def test_quantiles_are_ordered() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    assert (
        candidate[
            "prospectivity_score_oos_p25"
        ]
        <= candidate[
            "prospectivity_score_oos_median"
        ]
    ).all()

    assert (
        candidate[
            "prospectivity_score_oos_median"
        ]
        <= candidate[
            "prospectivity_score_oos_p75"
        ]
    ).all()


def test_frequencies_are_valid() -> None:
    frame = load_scores()

    candidate = frame.loc[
        frame[
            "known_positive"
        ]
        == 0
    ]

    for column in (
        "oos_top_1pct_frequency",
        "oos_top_5pct_frequency",
        "oos_top_10pct_frequency",
    ):
        assert candidate[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_no_oracle_columns_are_exported() -> None:
    frame = pd.read_csv(
        SCORES_PATH,
        nrows=0,
    )

    oracle = {
        "true_deposit",
        "latent_favourability",
        "true_deposit_probability",
        "deposit_id",
        "distance_to_known_deposit",
    }

    assert not (
        set(
            frame.columns
        )
        & oracle
    )


def test_manifest_model_freeze() -> None:
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

    assert (
        manifest[
            "model_frozen"
        ]
        is True
    )

    assert (
        manifest[
            "oracle_values_loaded"
        ]
        is False
    )

    assert (
        manifest[
            "oracle_used_for_scoring"
        ]
        is False
    )


def test_manifest_oos_contract() -> None:
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
            "oos_candidates_per_replicate"
        ]
        == 5800
    )

    assert (
        manifest[
            "candidate_top_1pct_count"
        ]
        == 63
    )

    assert (
        manifest[
            "candidate_top_5pct_count"
        ]
        == 315
    )

    assert (
        manifest[
            "candidate_top_10pct_count"
        ]
        == 630
    )

    assert (
        manifest[
            "frequency_denominator"
        ]
        == "cell_specific_oos_model_count"
    )

def test_known_positives_have_no_oos_statistics() -> None:
    frame = load_scores()

    known = frame.loc[
        frame[
            "known_positive"
        ]
        == 1
    ]

    columns = [
        "prospectivity_score_oos_mean",
        "prospectivity_score_oos_median",
        "prospectivity_score_oos_std",
        "prospectivity_score_oos_p25",
        "prospectivity_score_oos_p75",
        "prospectivity_score_oos_iqr",
        "oos_replicate_rank_mean",
        "oos_replicate_rank_std",
        "oos_top_1pct_frequency",
        "oos_top_5pct_frequency",
        "oos_top_10pct_frequency",
    ]

    assert len(known) == 100

    assert known[
        columns
    ].isna().all().all()
