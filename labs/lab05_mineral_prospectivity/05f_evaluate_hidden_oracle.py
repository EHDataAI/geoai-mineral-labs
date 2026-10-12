"""LAB05 Gate 6: post-freeze hidden-oracle evaluation.

The operational model and candidate ranking were frozen before
loading oracle labels. Oracle information is used here exclusively
for final evaluation and never for model, feature, or threshold
selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

RESULTS_DIR = LAB_DIR / "results"

SYNTHETIC_PATH = (
    LAB_DIR
    / "data"
    / "synthetic_mineral_system.csv"
)

SCORES_PATH = (
    RESULTS_DIR
    / "prospectivity_scores_oos.csv"
)

MODEL_MANIFEST_PATH = (
    RESULTS_DIR
    / "prospectivity_model_manifest_oos.json"
)

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

CANDIDATE_COUNT = 6300

TIERS = {
    "top_1pct": 63,
    "top_5pct": 315,
    "top_10pct": 630,
}


def load_json(
    path: Path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def validate_freeze(
    manifest: dict,
) -> None:
    """Ensure Gate 6 starts only after the operational freeze."""
    if manifest.get(
        "gate"
    ) != "5.1":
        raise RuntimeError(
            "Gate 5.1 manifest required."
        )

    if manifest.get(
        "model_frozen"
    ) is not True:
        raise RuntimeError(
            "Operational model is not frozen."
        )

    if manifest.get(
        "primary_feature_set"
    ) != "A":
        raise RuntimeError(
            "Unexpected frozen feature set."
        )

    if manifest.get(
        "primary_model"
    ) != "logistic_regression":
        raise RuntimeError(
            "Unexpected frozen model."
        )

    if manifest.get(
        "oracle_values_loaded"
    ) is not False:
        raise RuntimeError(
            "Gate 5.1 oracle boundary violated."
        )


def load_final_evaluation_data() -> pd.DataFrame:
    """Join frozen candidate ranking to minimal oracle labels."""
    scores = pd.read_csv(
        SCORES_PATH
    )

    if len(scores) != 6400:
        raise RuntimeError(
            "Unexpected Gate 5.1 score rows."
        )

    oracle = pd.read_csv(
        SYNTHETIC_PATH,
        usecols=[
            "cell_id",
            "known_positive",
            "true_deposit",
        ],
    )

    if len(oracle) != 6400:
        raise RuntimeError(
            "Unexpected oracle row count."
        )

    if oracle[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "Oracle cell_id is not unique."
        )

    merged = scores.merge(
        oracle[
            [
                "cell_id",
                "true_deposit",
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    if merged[
        "true_deposit"
    ].isna().any():
        raise RuntimeError(
            "Oracle join produced missing labels."
        )

    if (
        merged.loc[
            merged[
                "known_positive"
            ]
            == 1,
            "true_deposit",
        ]
        != 1
    ).any():
        raise RuntimeError(
            "Known positive not backed by true deposit."
        )

    return merged


def build_tier_summary(
    candidate: pd.DataFrame,
    hidden_total: int,
) -> pd.DataFrame:
    """Measure hidden-deposit capture and enrichment."""
    baseline_prevalence = (
        hidden_total
        / len(candidate)
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        tier,
        expected_count,
    ) in TIERS.items():
        column = (
            f"candidate_{tier}"
        )

        selected = candidate.loc[
            candidate[
                column
            ]
        ]

        if len(selected) != expected_count:
            raise RuntimeError(
                f"Unexpected {tier} candidate count."
            )

        hidden_captured = int(
            selected[
                "true_deposit"
            ].sum()
        )

        capture_rate = float(
            hidden_captured
            / hidden_total
        )

        precision = float(
            hidden_captured
            / len(selected)
        )

        lift = float(
            precision
            / baseline_prevalence
        )

        expected_random_hits = float(
            len(selected)
            * baseline_prevalence
        )

        area_fraction = float(
            len(selected)
            / len(candidate)
        )

        rows.append(
            {
                "tier": tier,
                "candidate_cells": int(
                    len(selected)
                ),
                "area_fraction": (
                    area_fraction
                ),
                "area_reduction": float(
                    1.0
                    - area_fraction
                ),
                "hidden_deposits_total": (
                    hidden_total
                ),
                "hidden_deposits_captured": (
                    hidden_captured
                ),
                "hidden_capture_rate": (
                    capture_rate
                ),
                "precision_hidden_deposit": (
                    precision
                ),
                "baseline_hidden_prevalence": (
                    baseline_prevalence
                ),
                "expected_random_hits": (
                    expected_random_hits
                ),
                "lift_over_random": (
                    lift
                ),
                "enrichment_factor": (
                    lift
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> int:
    print(
        "LAB05_GATE06_ORACLE_START"
    )

    model_manifest = load_json(
        MODEL_MANIFEST_PATH
    )

    validate_freeze(
        model_manifest
    )

    data = load_final_evaluation_data()

    candidate = data.loc[
        data[
            "known_positive"
        ]
        == 0
    ].copy()

    known = data.loc[
        data[
            "known_positive"
        ]
        == 1
    ].copy()

    if len(candidate) != CANDIDATE_COUNT:
        raise RuntimeError(
            "Expected 6300 candidate cells."
        )

    if len(known) != 100:
        raise RuntimeError(
            "Expected 100 known positives."
        )

    hidden = candidate.loc[
        candidate[
            "true_deposit"
        ]
        == 1
    ].copy()

    hidden_total = int(
        len(hidden)
    )

    if hidden_total <= 0:
        raise RuntimeError(
            "No hidden deposits available."
        )

    true_total = int(
        data[
            "true_deposit"
        ].sum()
    )

    if (
        true_total
        != hidden_total
        + len(known)
    ):
        raise RuntimeError(
            "Deposit accounting invariant failed."
        )

    hidden_labels = candidate[
        "true_deposit"
    ].to_numpy(
        dtype=int
    )

    scores = candidate[
        "prospectivity_score_oos_mean"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        scores
    ).all():
        raise RuntimeError(
            "Candidate OOS scores must be finite."
        )

    final_ap = float(
        average_precision_score(
            hidden_labels,
            scores,
        )
    )

    final_roc_auc = float(
        roc_auc_score(
            hidden_labels,
            scores,
        )
    )

    summary = build_tier_summary(
        candidate,
        hidden_total,
    )

    hidden_ranks = hidden[
        [
            "cell_id",
            "candidate_rank",
            "prospectivity_score_oos_mean",
            "prospectivity_score_oos_median",
            "prospectivity_score_oos_std",
            "oos_model_count",
            "oos_top_1pct_frequency",
            "oos_top_5pct_frequency",
            "oos_top_10pct_frequency",
        ]
    ].sort_values(
        "candidate_rank",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    hidden_ranks.to_csv(
        HIDDEN_RANKS_PATH,
        index=False,
    )

    manifest = {
        "gate": "6",
        "purpose": (
            "post_freeze_hidden_oracle_evaluation"
        ),
        "model_or_features_changed_after_oracle": False,
        "model_selection_after_oracle": False,
        "feature_selection_after_oracle": False,
        "threshold_selection_after_oracle": False,
        "oracle_loaded_for_final_evaluation": True,
        "primary_feature_set": (
            model_manifest[
                "primary_feature_set"
            ]
        ),
        "primary_model": (
            model_manifest[
                "primary_model"
            ]
        ),
        "candidate_rows": int(
            len(candidate)
        ),
        "known_positive_rows": int(
            len(known)
        ),
        "true_deposit_rows": (
            true_total
        ),
        "hidden_deposit_rows": (
            hidden_total
        ),
        "hidden_prevalence": float(
            hidden_total
            / len(candidate)
        ),
        "final_hidden_average_precision": (
            final_ap
        ),
        "final_hidden_roc_auc": (
            final_roc_auc
        ),
        "hidden_rank_min": int(
            hidden_ranks[
                "candidate_rank"
            ].min()
        ),
        "hidden_rank_median": float(
            hidden_ranks[
                "candidate_rank"
            ].median()
        ),
        "hidden_rank_mean": float(
            hidden_ranks[
                "candidate_rank"
            ].mean()
        ),
        "hidden_rank_max": int(
            hidden_ranks[
                "candidate_rank"
            ].max()
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"TRUE_DEPOSITS = {true_total}"
    )

    print(
        f"KNOWN_POSITIVES = {len(known)}"
    )

    print(
        f"HIDDEN_DEPOSITS = {hidden_total}"
    )

    print(
        "FINAL_HIDDEN_AP = "
        f"{final_ap:.6f}"
    )

    print(
        "FINAL_HIDDEN_ROC_AUC = "
        f"{final_roc_auc:.6f}"
    )

    for row in (
        summary.to_dict(
            orient="records"
        )
    ):
        print(
            f"{row['tier'].upper()} "
            f"HIDDEN_CAPTURE = "
            f"{row['hidden_deposits_captured']}/"
            f"{hidden_total} "
            f"({row['hidden_capture_rate']:.6f}), "
            f"LIFT = "
            f"{row['lift_over_random']:.6f}"
        )

    print(
        "MODEL_CHANGED_AFTER_ORACLE = NO"
    )

    print(
        "LAB05_GATE06_ORACLE = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
