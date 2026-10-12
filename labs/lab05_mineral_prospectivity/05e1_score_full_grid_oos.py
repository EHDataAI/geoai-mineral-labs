"""LAB05 Gate 5.1: out-of-sample-only prospectivity ranking.

The operational model is frozen after Gate 4.1:
Feature Set A + Logistic Regression.

Only unlabeled cells that were NOT sampled as background training data
in a replicate receive an OOS score from that replicate.

The synthetic oracle remains sealed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prospectivity_config import (
    BACKGROUND_REPLICATES,
    KNOWN_POSITIVE_COUNT,
    ROWS_PER_REPLICATE,
    SEED,
)


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

RESULTS_DIR = LAB_DIR / "results"

LEGACY_GATE04_PATH = (
    LAB_DIR
    / "05d_run_spatial_benchmark.py"
)

EVIDENCE_PATH = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

EVIDENCE_MANIFEST_PATH = (
    RESULTS_DIR
    / "evidence_manifest.json"
)

OOF_AGGREGATE_PATH = (
    RESULTS_DIR
    / "benchmark_oof_aggregate_metrics.csv"
)

OOF_MANIFEST_PATH = (
    RESULTS_DIR
    / "benchmark_oof_manifest.json"
)

SCORES_PATH = (
    RESULTS_DIR
    / "prospectivity_scores_oos.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "prospectivity_ranking_summary_oos.csv"
)

MODEL_MANIFEST_PATH = (
    RESULTS_DIR
    / "prospectivity_model_manifest_oos.json"
)

TARGET = "known_positive"

FEATURE_SET = "A"

MODEL_NAME = "logistic_regression"

CANDIDATE_COUNT = 6300

CANDIDATE_TOP_COUNTS = {
    "top_1pct": 63,
    "top_5pct": 315,
    "top_10pct": 630,
}

REPLICATE_OOS_COUNT = 5800

REPLICATE_TOP_COUNTS = {
    "top_1pct": 58,
    "top_5pct": 290,
    "top_10pct": 580,
}

OUTPUT_IDENTIFIERS = [
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "known_positive",
]

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "deposit_id",
    "distance_to_known_deposit",
}


def load_json(
    path: Path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_gate04_module():
    if str(LAB_DIR) not in sys.path:
        sys.path.insert(
            0,
            str(LAB_DIR),
        )

    spec = importlib.util.spec_from_file_location(
        "lab05_gate04_legacy_for_gate05_1",
        LEGACY_GATE04_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load Gate 4 utilities."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def validate_model_freeze(
    aggregate: pd.DataFrame,
    manifest: dict,
) -> None:
    """Enforce the predeclared Gate 4.1 model-selection rule."""
    if manifest.get(
        "selection_metric"
    ) != "mean_spatial_oof_average_precision":
        raise RuntimeError(
            "Unexpected Gate 4.1 selection metric."
        )

    spatial = aggregate.loc[
        aggregate[
            "validation_strategy"
        ]
        == "spatial"
    ].copy()

    if len(spatial) != 9:
        raise RuntimeError(
            "Expected nine spatial configurations."
        )

    winner = spatial.sort_values(
        [
            "average_precision_mean",
            "average_precision_median",
        ],
        ascending=False,
    ).iloc[0]

    if (
        winner["feature_set"]
        != FEATURE_SET
        or winner["model"]
        != MODEL_NAME
    ):
        raise RuntimeError(
            "Frozen operational model no longer "
            "matches Gate 4.1 evidence."
        )


def load_safe_evidence(
    gate04,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """Load only identifiers and frozen Feature Set A."""
    evidence_manifest = load_json(
        EVIDENCE_MANIFEST_PATH
    )

    feature_sets = (
        gate04.feature_sets_from_manifest(
            evidence_manifest
        )
    )

    if FEATURE_SET not in feature_sets:
        raise RuntimeError(
            "Feature Set A missing."
        )

    feature_columns = list(
        feature_sets[
            FEATURE_SET
        ]
    )

    if len(feature_columns) != 9:
        raise RuntimeError(
            "Expected nine Feature Set A columns."
        )

    if (
        set(feature_columns)
        & ORACLE_FIELDS
    ):
        raise RuntimeError(
            "Oracle leakage in operational features."
        )

    usecols = list(
        dict.fromkeys(
            [
                *OUTPUT_IDENTIFIERS,
                *feature_columns,
            ]
        )
    )

    frame = pd.read_csv(
        EVIDENCE_PATH,
        usecols=usecols,
    )

    if len(frame) != 6400:
        raise RuntimeError(
            "Expected 6400 evidence rows."
        )

    if int(
        frame[
            TARGET
        ].sum()
    ) != KNOWN_POSITIVE_COUNT:
        raise RuntimeError(
            "Unexpected known-positive count."
        )

    if frame[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "cell_id must be unique."
        )

    return (
        frame,
        feature_columns,
    )


def score_oos_replicates(
    gate04,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict],
    dict[str, np.ndarray],
]:
    """Create OOS-only score and rank matrices for unlabeled cells."""
    candidate_mask = (
        frame[
            TARGET
        ].to_numpy(
            dtype=int
        )
        == 0
    )

    if int(
        candidate_mask.sum()
    ) != CANDIDATE_COUNT:
        raise RuntimeError(
            "Expected 6300 candidates."
        )

    scores = np.full(
        (
            len(frame),
            BACKGROUND_REPLICATES,
        ),
        np.nan,
        dtype=float,
    )

    ranks = np.full(
        (
            len(frame),
            BACKGROUND_REPLICATES,
        ),
        np.nan,
        dtype=float,
    )

    top_hits = {
        name: np.zeros(
            len(frame),
            dtype=float,
        )
        for name in REPLICATE_TOP_COUNTS
    }

    audit: list[dict] = []

    cell_ids = frame[
        "cell_id"
    ].astype(
        str
    ).to_numpy()

    for replicate in range(
        BACKGROUND_REPLICATES
    ):
        training = gate04.sample_background(
            frame,
            replicate,
        )

        if len(training) != ROWS_PER_REPLICATE:
            raise RuntimeError(
                "Unexpected training size."
            )

        training_background_ids = set(
            training.loc[
                training[
                    TARGET
                ]
                == 0,
                "cell_id",
            ].astype(
                str
            )
        )

        if len(
            training_background_ids
        ) != 500:
            raise RuntimeError(
                "Unexpected sampled background count."
            )

        candidate_indices = np.flatnonzero(
            candidate_mask
        )

        candidate_ids = cell_ids[
            candidate_indices
        ]

        eligible_local_mask = np.array(
            [
                cell_id
                not in training_background_ids
                for cell_id in candidate_ids
            ],
            dtype=bool,
        )

        eligible_indices = (
            candidate_indices[
                eligible_local_mask
            ]
        )

        if len(
            eligible_indices
        ) != REPLICATE_OOS_COUNT:
            raise RuntimeError(
                "Expected 5800 OOS candidates "
                "per replicate."
            )

        pipeline = (
            gate04.build_model_pipeline(
                MODEL_NAME,
                feature_columns,
                SEED
                + replicate
                * 1000,
            )
        )

        pipeline.fit(
            training[
                feature_columns
            ],
            training[
                TARGET
            ].to_numpy(
                dtype=int
            ),
        )

        replicate_scores = (
            pipeline.predict_proba(
                frame.iloc[
                    eligible_indices
                ][
                    feature_columns
                ]
            )[:, 1]
        )

        if not np.isfinite(
            replicate_scores
        ).all():
            raise RuntimeError(
                "Non-finite OOS score."
            )

        scores[
            eligible_indices,
            replicate,
        ] = replicate_scores

        eligible_cell_ids = cell_ids[
            eligible_indices
        ]

        order_local = np.lexsort(
            (
                eligible_cell_ids,
                -replicate_scores,
            )
        )

        ordered_indices = (
            eligible_indices[
                order_local
            ]
        )

        one_ranks = np.arange(
            1,
            REPLICATE_OOS_COUNT + 1,
            dtype=float,
        )

        ranks[
            ordered_indices,
            replicate,
        ] = one_ranks

        for (
            name,
            top_n,
        ) in REPLICATE_TOP_COUNTS.items():
            top_hits[
                name
            ][
                ordered_indices[
                    :top_n
                ]
            ] += 1.0

        audit.append(
            {
                "replicate": replicate,
                "seed": (
                    SEED
                    + replicate
                ),
                "model_seed": (
                    SEED
                    + replicate
                    * 1000
                ),
                "training_rows": int(
                    len(training)
                ),
                "training_background_rows": (
                    len(
                        training_background_ids
                    )
                ),
                "oos_candidate_rows": int(
                    len(
                        eligible_indices
                    )
                ),
            }
        )

        print(
            "OOS_SCORING_REPLICATE_COMPLETE = "
            f"{replicate + 1}/"
            f"{BACKGROUND_REPLICATES}"
        )

    return (
        scores,
        ranks,
        [
            {
                **item,
            }
            for item in audit
        ],
        top_hits,
    )


def rowwise_nan_std(
    values: np.ndarray,
) -> np.ndarray:
    result = np.full(
        values.shape[0],
        np.nan,
        dtype=float,
    )

    counts = np.sum(
        np.isfinite(
            values
        ),
        axis=1,
    )

    valid = counts >= 2

    result[
        valid
    ] = np.nanstd(
        values[
            valid
        ],
        axis=1,
        ddof=1,
    )

    return result


def build_output(
    frame: pd.DataFrame,
    scores: np.ndarray,
    ranks: np.ndarray,
    top_hits: dict[
        str,
        np.ndarray
    ],
) -> pd.DataFrame:
    """Aggregate OOS predictions and build discovery ranking."""
    candidate_mask = (
        frame[
            TARGET
        ].to_numpy(
            dtype=int
        )
        == 0
    )

    oos_count = np.sum(
        np.isfinite(
            scores
        ),
        axis=1,
    ).astype(
        int
    )

    result = frame[
        OUTPUT_IDENTIFIERS
    ].copy()

    result[
        "oos_model_count"
    ] = oos_count

    score_mean = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    score_median = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    score_p25 = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    score_p75 = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    rank_mean = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    score_mean[
        candidate_mask
    ] = np.nanmean(
        scores[
            candidate_mask,
            :
        ],
        axis=1,
    )

    score_median[
        candidate_mask
    ] = np.nanmedian(
        scores[
            candidate_mask,
            :
        ],
        axis=1,
    )

    score_p25[
        candidate_mask
    ] = np.nanquantile(
        scores[
            candidate_mask,
            :
        ],
        0.25,
        axis=1,
    )

    score_p75[
        candidate_mask
    ] = np.nanquantile(
        scores[
            candidate_mask,
            :
        ],
        0.75,
        axis=1,
    )

    rank_mean[
        candidate_mask
    ] = np.nanmean(
        ranks[
            candidate_mask,
            :
        ],
        axis=1,
    )

    result[
        "prospectivity_score_oos_mean"
    ] = score_mean

    result[
        "prospectivity_score_oos_median"
    ] = score_median

    result[
        "prospectivity_score_oos_std"
    ] = rowwise_nan_std(
        scores
    )

    result[
        "prospectivity_score_oos_p25"
    ] = score_p25

    result[
        "prospectivity_score_oos_p75"
    ] = score_p75

    result[
        "prospectivity_score_oos_iqr"
    ] = (
        score_p75
        - score_p25
    )

    result[
        "oos_replicate_rank_mean"
    ] = rank_mean

    result[
        "oos_replicate_rank_std"
    ] = rowwise_nan_std(
        ranks
    )

    for name in (
        "top_1pct",
        "top_5pct",
        "top_10pct",
    ):
        numerator = top_hits[
            name
        ]

        frequency = np.full(
            len(frame),
            np.nan,
            dtype=float,
        )

        valid = (
            oos_count > 0
        )

        frequency[
            valid
        ] = (
            numerator[
                valid
            ]
            / oos_count[
                valid
            ]
        )

        result[
            f"oos_{name}_frequency"
        ] = frequency

    known_mask = ~candidate_mask

    oos_columns = [
        column
        for column in result.columns
        if (
            column.startswith(
                "prospectivity_score_oos_"
            )
            or column.startswith(
                "oos_replicate_"
            )
            or column.startswith(
                "oos_top_"
            )
        )
    ]

    result.loc[
        known_mask,
        oos_columns,
    ] = np.nan

    result.loc[
        known_mask,
        "oos_model_count",
    ] = 0

    candidate = result.loc[
        candidate_mask
    ].copy()

    if (
        candidate[
            "oos_model_count"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Candidate without OOS support."
        )

    candidate = candidate.sort_values(
        [
            "prospectivity_score_oos_mean",
            "prospectivity_score_oos_median",
            "prospectivity_score_oos_std",
            "cell_id",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    candidate_rank = pd.Series(
        np.arange(
            1,
            CANDIDATE_COUNT + 1,
            dtype=int,
        ),
        index=candidate.index,
    )

    result[
        "candidate_rank"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result.loc[
        candidate.index,
        "candidate_rank",
    ] = candidate_rank

    result[
        "candidate_top_1pct"
    ] = False

    result[
        "candidate_top_5pct"
    ] = False

    result[
        "candidate_top_10pct"
    ] = False

    for (
        name,
        top_n,
    ) in CANDIDATE_TOP_COUNTS.items():
        result.loc[
            candidate.index[
                :top_n
            ],
            f"candidate_{name}",
        ] = True

    return result.sort_values(
        "cell_id",
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def build_summary(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Create compact candidate-ranking summary."""
    candidate = scores.loc[
        scores[
            TARGET
        ]
        == 0
    ].copy()

    rows: list[
        dict[str, object]
    ] = []

    for (
        label,
        top_n,
    ) in CANDIDATE_TOP_COUNTS.items():
        top = candidate.loc[
            candidate[
                f"candidate_{label}"
            ]
        ]

        rows.append(
            {
                "tier": label,
                "candidate_count": int(
                    len(top)
                ),
                "expected_count": (
                    top_n
                ),
                "mean_oos_score": float(
                    top[
                        "prospectivity_score_oos_mean"
                    ].mean()
                ),
                "median_oos_score": float(
                    top[
                        "prospectivity_score_oos_median"
                    ].median()
                ),
                "mean_oos_support": float(
                    top[
                        "oos_model_count"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> int:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate04 = load_gate04_module()

    aggregate = pd.read_csv(
        OOF_AGGREGATE_PATH
    )

    oof_manifest = load_json(
        OOF_MANIFEST_PATH
    )

    validate_model_freeze(
        aggregate,
        oof_manifest,
    )

    (
        frame,
        feature_columns,
    ) = load_safe_evidence(
        gate04
    )

    print(
        "LAB05_GATE05_1_OOS_START"
    )

    (
        score_matrix,
        rank_matrix,
        audit,
        top_hits,
    ) = score_oos_replicates(
        gate04,
        frame,
        feature_columns,
    )

    scores = build_output(
        frame,
        score_matrix,
        rank_matrix,
        top_hits,
    )

    candidate = scores.loc[
        scores[
            TARGET
        ]
        == 0
    ]

    known = scores.loc[
        scores[
            TARGET
        ]
        == 1
    ]

    if len(candidate) != CANDIDATE_COUNT:
        raise RuntimeError(
            "Unexpected candidate count."
        )

    support_min = int(
        candidate[
            "oos_model_count"
        ].min()
    )

    support_max = int(
        candidate[
            "oos_model_count"
        ].max()
    )

    support_mean = float(
        candidate[
            "oos_model_count"
        ].mean()
    )

    if (
        known[
            "oos_model_count"
        ]
        != 0
    ).any():
        raise RuntimeError(
            "Known positives must not receive "
            "candidate OOS support."
        )

    if not (
        known[
            "candidate_rank"
        ].isna()
    ).all():
        raise RuntimeError(
            "Known positives entered discovery ranking."
        )

    if int(
        scores[
            "candidate_top_1pct"
        ].sum()
    ) != 63:
        raise RuntimeError(
            "Unexpected candidate top-1% count."
        )

    if int(
        scores[
            "candidate_top_5pct"
        ].sum()
    ) != 315:
        raise RuntimeError(
            "Unexpected candidate top-5% count."
        )

    if int(
        scores[
            "candidate_top_10pct"
        ].sum()
    ) != 630:
        raise RuntimeError(
            "Unexpected candidate top-10% count."
        )

    summary = build_summary(
        scores
    )

    scores.to_csv(
        SCORES_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    manifest = {
        "gate": "5.1",
        "method": (
            "out_of_sample_only_candidate_scoring"
        ),
        "dgp_v2_frozen": True,
        "model_frozen": True,
        "primary_feature_set": FEATURE_SET,
        "primary_feature_set_name": (
            "A_non_geochemical"
        ),
        "primary_model": MODEL_NAME,
        "selection_metric": (
            "mean_spatial_oof_average_precision"
        ),
        "oracle_values_loaded": False,
        "oracle_used_for_training": False,
        "oracle_used_for_scoring": False,
        "background_is_confirmed_negative": False,
        "score_semantics": (
            "prospectivity_score_not_calibrated_deposit_probability"
        ),
        "candidate_rows": CANDIDATE_COUNT,
        "known_positive_rows": (
            KNOWN_POSITIVE_COUNT
        ),
        "background_replicates": (
            BACKGROUND_REPLICATES
        ),
        "training_rows_per_replicate": (
            ROWS_PER_REPLICATE
        ),
        "oos_candidates_per_replicate": (
            REPLICATE_OOS_COUNT
        ),
        "candidate_top_1pct_count": 63,
        "candidate_top_5pct_count": 315,
        "candidate_top_10pct_count": 630,
        "replicate_top_1pct_count": 58,
        "replicate_top_5pct_count": 290,
        "replicate_top_10pct_count": 580,
        "oos_model_count_min": (
            support_min
        ),
        "oos_model_count_max": (
            support_max
        ),
        "oos_model_count_mean": (
            support_mean
        ),
        "frequency_denominator": (
            "cell_specific_oos_model_count"
        ),
        "candidate_ranking_order": [
            "prospectivity_score_oos_mean_desc",
            "prospectivity_score_oos_median_desc",
            "prospectivity_score_oos_std_asc",
            "cell_id_asc",
        ],
        "replicates": audit,
    }

    MODEL_MANIFEST_PATH.write_text(
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
        f"CANDIDATES = {len(candidate)}"
    )

    print(
        f"KNOWN_POSITIVES = {len(known)}"
    )

    print(
        f"OOS_SUPPORT_MIN = {support_min}"
    )

    print(
        f"OOS_SUPPORT_MAX = {support_max}"
    )

    print(
        "OOS_SUPPORT_MEAN = "
        f"{support_mean:.6f}"
    )

    print(
        "CANDIDATE_TOP_1PCT = "
        f"{int(scores['candidate_top_1pct'].sum())}"
    )

    print(
        "CANDIDATE_TOP_5PCT = "
        f"{int(scores['candidate_top_5pct'].sum())}"
    )

    print(
        "CANDIDATE_TOP_10PCT = "
        f"{int(scores['candidate_top_10pct'].sum())}"
    )

    print(
        "ORACLE_USED = NO"
    )

    print(
        "LAB05_GATE05_1_OOS = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
