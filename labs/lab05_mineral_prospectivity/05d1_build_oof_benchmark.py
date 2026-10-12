"""LAB05 Gate 4.1: pooled out-of-fold prospectivity benchmark.

Each replicate/configuration receives exactly one out-of-fold score
per sampled observation. Ranking metrics are calculated only after
pooling all fold-level OOF predictions for the replicate.

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
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from prospectivity_config import (
    BACKGROUND_REPLICATES,
    CV_FOLDS,
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

OOF_METRICS_PATH = (
    RESULTS_DIR
    / "benchmark_oof_metrics.csv"
)

OOF_AGGREGATE_PATH = (
    RESULTS_DIR
    / "benchmark_oof_aggregate_metrics.csv"
)

OOF_CURVE_PATH = (
    RESULTS_DIR
    / "success_rate_curve_oof.csv"
)

OOF_MANIFEST_PATH = (
    RESULTS_DIR
    / "benchmark_oof_manifest.json"
)

TOP_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
)

VALIDATION_STRATEGIES = (
    "spatial",
    "random",
)

MODEL_NAMES = (
    "dummy",
    "logistic_regression",
    "random_forest",
)


def load_gate04_module():
    """Load the validated Gate 4 utilities without running its main()."""
    if str(LAB_DIR) not in sys.path:
        sys.path.insert(
            0,
            str(LAB_DIR),
        )

    spec = importlib.util.spec_from_file_location(
        "lab05_gate04_legacy",
        LEGACY_GATE04_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load Gate 4 utility module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def top_fraction_record(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    """Calculate pooled OOF ranking metrics at one inspection fraction."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    positives = int(
        y_true.sum()
    )

    if positives <= 0:
        raise RuntimeError(
            "Top-fraction metrics require positives."
        )

    n_select = max(
        1,
        int(
            math.ceil(
                len(y_true)
                * fraction
            )
        ),
    )

    ranked = np.argsort(
        -scores,
        kind="mergesort",
    )

    selected = ranked[
        :n_select
    ]

    positives_captured = int(
        y_true[
            selected
        ].sum()
    )

    return {
        "top_fraction": float(
            fraction
        ),
        "top_k_n": int(
            n_select
        ),
        "positives_captured": (
            positives_captured
        ),
        "known_positive_capture_rate": float(
            positives_captured
            / positives
        ),
        "precision_at_k": float(
            positives_captured
            / n_select
        ),
    }


def aggregate_oof_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate replicate-level OOF metrics across ten backgrounds."""
    grouped = metrics.groupby(
        [
            "validation_strategy",
            "feature_set",
            "model",
        ],
        sort=True,
    )

    rows: list[dict[str, object]] = []

    for keys, frame in grouped:
        (
            validation_strategy,
            feature_set,
            model,
        ) = keys

        if len(frame) != BACKGROUND_REPLICATES:
            raise RuntimeError(
                "Unexpected replicate count "
                "during OOF aggregation."
            )

        rows.append(
            {
                "validation_strategy": (
                    validation_strategy
                ),
                "feature_set": (
                    feature_set
                ),
                "model": model,
                "n_replicates": int(
                    len(frame)
                ),
                "average_precision_mean": float(
                    frame[
                        "average_precision"
                    ].mean()
                ),
                "average_precision_std": float(
                    frame[
                        "average_precision"
                    ].std(
                        ddof=1
                    )
                ),
                "average_precision_median": float(
                    frame[
                        "average_precision"
                    ].median()
                ),
                "roc_auc_mean": float(
                    frame[
                        "roc_auc"
                    ].mean()
                ),
                "roc_auc_std": float(
                    frame[
                        "roc_auc"
                    ].std(
                        ddof=1
                    )
                ),
                "roc_auc_median": float(
                    frame[
                        "roc_auc"
                    ].median()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        [
            "validation_strategy",
            "feature_set",
            "model",
        ]
    ).reset_index(
        drop=True
    )

    if len(result) != 18:
        raise RuntimeError(
            "Expected 18 OOF aggregate configurations."
        )

    return result


def main() -> int:
    """Run pooled OOF Gate 4.1."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate04 = load_gate04_module()

    evidence = pd.read_csv(
        gate04.EVIDENCE_PATH
    )

    evidence_manifest = (
        gate04.load_json(
            gate04.MANIFEST_PATH
        )
    )

    contract = gate04.load_json(
        gate04.CONTRACT_PATH
    )

    feature_sets = (
        gate04.feature_sets_from_manifest(
            evidence_manifest
        )
    )

    forbidden = set(
        contract[
            "forbidden_model_features"
        ]
    )

    gate04.validate_feature_contract(
        evidence,
        feature_sets,
        forbidden,
    )

    if int(
        evidence[
            gate04.TARGET
        ].sum()
    ) != KNOWN_POSITIVE_COUNT:
        raise RuntimeError(
            "Unexpected known-positive count."
        )

    metric_rows: list[
        dict[str, object]
    ] = []

    curve_rows: list[
        dict[str, object]
    ] = []

    total_fits = 0
    global_spatial_overlap_max = 0

    print(
        "LAB05_GATE04_1_OOF_START"
    )

    for replicate in range(
        BACKGROUND_REPLICATES
    ):
        sampled = gate04.sample_background(
            evidence,
            replicate,
        )

        if len(sampled) != ROWS_PER_REPLICATE:
            raise RuntimeError(
                "Unexpected sampled replicate size."
            )

        y = sampled[
            gate04.TARGET
        ].to_numpy(
            dtype=int
        )

        if int(
            y.sum()
        ) != KNOWN_POSITIVE_COUNT:
            raise RuntimeError(
                "Unexpected replicate positive count."
            )

        for validation_strategy in (
            VALIDATION_STRATEGIES
        ):
            splits = list(
                gate04.get_splits(
                    sampled,
                    validation_strategy,
                    replicate,
                )
            )

            if len(splits) != CV_FOLDS:
                raise RuntimeError(
                    "Unexpected fold count."
                )

            for (
                feature_set_name,
                feature_columns,
            ) in feature_sets.items():
                X = sampled[
                    feature_columns
                ]

                for model_name in MODEL_NAMES:
                    oof_scores = np.full(
                        len(sampled),
                        np.nan,
                        dtype=float,
                    )

                    oof_count = np.zeros(
                        len(sampled),
                        dtype=int,
                    )

                    configuration_overlap_max = 0

                    for fold_number, (
                        train_index,
                        test_index,
                    ) in enumerate(
                        splits,
                        start=1,
                    ):
                        (
                            _train_pos,
                            _test_pos,
                            overlap,
                        ) = gate04.validate_fold(
                            sampled,
                            train_index,
                            test_index,
                            validation_strategy,
                        )

                        configuration_overlap_max = max(
                            configuration_overlap_max,
                            overlap,
                        )

                        if (
                            validation_strategy
                            == "spatial"
                        ):
                            global_spatial_overlap_max = max(
                                global_spatial_overlap_max,
                                overlap,
                            )

                        seed = (
                            SEED
                            + replicate
                            * 1000
                            + fold_number
                            * 10
                        )

                        pipeline = (
                            gate04.build_model_pipeline(
                                model_name,
                                feature_columns,
                                seed,
                            )
                        )

                        pipeline.fit(
                            X.iloc[
                                train_index
                            ],
                            y[
                                train_index
                            ],
                        )

                        scores = (
                            pipeline.predict_proba(
                                X.iloc[
                                    test_index
                                ]
                            )[:, 1]
                        )

                        if not np.isfinite(
                            scores
                        ).all():
                            raise RuntimeError(
                                "Non-finite OOF scores."
                            )

                        if (
                            oof_count[
                                test_index
                            ]
                            != 0
                        ).any():
                            raise RuntimeError(
                                "Duplicate OOF prediction "
                                "assignment detected."
                            )

                        oof_scores[
                            test_index
                        ] = scores

                        oof_count[
                            test_index
                        ] += 1

                        total_fits += 1

                    oof_count_min = int(
                        oof_count.min()
                    )

                    oof_count_max = int(
                        oof_count.max()
                    )

                    missing_oof = int(
                        np.isnan(
                            oof_scores
                        ).sum()
                    )

                    duplicate_oof = int(
                        (
                            oof_count > 1
                        ).sum()
                    )

                    if (
                        oof_count_min != 1
                        or oof_count_max != 1
                        or missing_oof != 0
                        or duplicate_oof != 0
                    ):
                        raise RuntimeError(
                            "OOF coverage invariant failed."
                        )

                    average_precision = float(
                        average_precision_score(
                            y,
                            oof_scores,
                        )
                    )

                    roc_auc = float(
                        roc_auc_score(
                            y,
                            oof_scores,
                        )
                    )

                    metric_rows.append(
                        {
                            "replicate": replicate,
                            "validation_strategy": (
                                validation_strategy
                            ),
                            "feature_set": (
                                feature_set_name
                            ),
                            "model": model_name,
                            "n_rows": int(
                                len(sampled)
                            ),
                            "known_positives": int(
                                y.sum()
                            ),
                            "average_precision": (
                                average_precision
                            ),
                            "roc_auc": (
                                roc_auc
                            ),
                            "oof_count_min": (
                                oof_count_min
                            ),
                            "oof_count_max": (
                                oof_count_max
                            ),
                            "missing_oof_predictions": (
                                missing_oof
                            ),
                            "duplicate_oof_predictions": (
                                duplicate_oof
                            ),
                            "group_overlap_max": int(
                                configuration_overlap_max
                            ),
                        }
                    )

                    for fraction in TOP_FRACTIONS:
                        ranking = (
                            top_fraction_record(
                                y,
                                oof_scores,
                                fraction,
                            )
                        )

                        curve_rows.append(
                            {
                                "replicate": (
                                    replicate
                                ),
                                "validation_strategy": (
                                    validation_strategy
                                ),
                                "feature_set": (
                                    feature_set_name
                                ),
                                "model": (
                                    model_name
                                ),
                                **ranking,
                            }
                        )

        print(
            "BACKGROUND_REPLICATE_COMPLETE = "
            f"{replicate + 1}/"
            f"{BACKGROUND_REPLICATES}"
        )

    expected_fits = (
        BACKGROUND_REPLICATES
        * len(
            VALIDATION_STRATEGIES
        )
        * len(
            feature_sets
        )
        * len(
            MODEL_NAMES
        )
        * CV_FOLDS
    )

    if total_fits != expected_fits:
        raise RuntimeError(
            f"Expected {expected_fits} fits, "
            f"observed {total_fits}."
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    if len(metrics) != 180:
        raise RuntimeError(
            "Expected exactly 180 "
            "replicate-level OOF rows."
        )

    curve = pd.DataFrame(
        curve_rows
    )

    if len(curve) != 1080:
        raise RuntimeError(
            "Expected exactly 1080 "
            "OOF success-curve rows."
        )

    aggregate = aggregate_oof_metrics(
        metrics
    )

    if global_spatial_overlap_max != 0:
        raise RuntimeError(
            "Spatial group leakage detected."
        )

    metrics.to_csv(
        OOF_METRICS_PATH,
        index=False,
    )

    aggregate.to_csv(
        OOF_AGGREGATE_PATH,
        index=False,
    )

    curve.to_csv(
        OOF_CURVE_PATH,
        index=False,
    )

    manifest = {
        "gate": "4.1",
        "method": (
            "pooled_out_of_fold_evaluation"
        ),
        "dgp_v2_frozen": True,
        "seed": SEED,
        "target_semantics": (
            "known_positive_vs_sampled_unlabeled_background"
        ),
        "background_is_confirmed_negative": False,
        "score_semantics": (
            "prospectivity_score_not_calibrated_deposit_probability"
        ),
        "selection_metric": (
            "mean_spatial_oof_average_precision"
        ),
        "random_cv_role": (
            "diagnostic_only"
        ),
        "oracle_used_for_training": False,
        "oracle_used_for_feature_selection": False,
        "oracle_used_for_model_selection": False,
        "background_replicates": (
            BACKGROUND_REPLICATES
        ),
        "known_positives_per_replicate": (
            KNOWN_POSITIVE_COUNT
        ),
        "rows_per_replicate": (
            ROWS_PER_REPLICATE
        ),
        "cv_folds": CV_FOLDS,
        "feature_sets": sorted(
            feature_sets
        ),
        "models": list(
            MODEL_NAMES
        ),
        "validation_strategies": list(
            VALIDATION_STRATEGIES
        ),
        "top_fractions": list(
            TOP_FRACTIONS
        ),
        "planned_model_fits": (
            expected_fits
        ),
        "observed_model_fits": (
            total_fits
        ),
        "oof_metric_rows": int(
            len(metrics)
        ),
        "aggregate_rows": int(
            len(aggregate)
        ),
        "success_curve_rows": int(
            len(curve)
        ),
        "oof_prediction_count_min": int(
            metrics[
                "oof_count_min"
            ].min()
        ),
        "oof_prediction_count_max": int(
            metrics[
                "oof_count_max"
            ].max()
        ),
        "missing_oof_predictions_total": int(
            metrics[
                "missing_oof_predictions"
            ].sum()
        ),
        "duplicate_oof_predictions_total": int(
            metrics[
                "duplicate_oof_predictions"
            ].sum()
        ),
        "spatial_group_overlap_max": int(
            global_spatial_overlap_max
        ),
    }

    OOF_MANIFEST_PATH.write_text(
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
        f"OOF_METRIC_ROWS = {len(metrics)}"
    )

    print(
        f"OOF_AGGREGATE_ROWS = {len(aggregate)}"
    )

    print(
        f"OOF_SUCCESS_CURVE_ROWS = {len(curve)}"
    )

    print(
        f"OBSERVED_MODEL_FITS = {total_fits}"
    )

    print(
        "OOF_COUNT_MIN = "
        f"{manifest['oof_prediction_count_min']}"
    )

    print(
        "OOF_COUNT_MAX = "
        f"{manifest['oof_prediction_count_max']}"
    )

    print(
        "MISSING_OOF = "
        f"{manifest['missing_oof_predictions_total']}"
    )

    print(
        "DUPLICATE_OOF = "
        f"{manifest['duplicate_oof_predictions_total']}"
    )

    print(
        "SPATIAL_GROUP_OVERLAP_MAX = "
        f"{global_spatial_overlap_max}"
    )

    print(
        "ORACLE_USED = NO"
    )

    print(
        "LAB05_GATE04_1_OOF = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
