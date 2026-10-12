"""Build and run LAB05 Gate 4 spatial benchmarking.

Engineering policies:
- deterministic execution
- non-interactive subprocesses
- no oracle leakage
- fold-safe preprocessing
- spatial CV is primary
- random CV is diagnostic only
- background observations are unlabeled/pseudo-negative
- generated outputs remain lightweight
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

SCRIPT_DIR = (
    ROOT
    / "scripts"
    / "lab05"
)

TEST_DIR = ROOT / "tests"

PYTHON = (
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

BENCHMARK_PATH = (
    LAB_DIR
    / "05d_run_spatial_benchmark.py"
)

RUNNER_PATH = (
    SCRIPT_DIR
    / "run_gate04.py"
)

TEST_PATH = (
    TEST_DIR
    / "test_lab05_spatial_benchmark.py"
)

EXPECTED_BRANCH = (
    "feat/lab05-mineral-prospectivity"
)

SELF_RELATIVE = (
    "scripts/lab05/setup_and_run_gate04.py"
)


BENCHMARK_CODE = r'''"""LAB05 Gate 4: spatially validated prospectivity benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parents[2]

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

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

RESULTS_DIR = LAB_DIR / "results"

FOLD_METRICS_PATH = (
    RESULTS_DIR
    / "benchmark_fold_metrics.csv"
)

AGGREGATE_METRICS_PATH = (
    RESULTS_DIR
    / "benchmark_aggregate_metrics.csv"
)

GAP_PATH = (
    RESULTS_DIR
    / "spatial_random_gap.csv"
)

SUCCESS_CURVE_PATH = (
    RESULTS_DIR
    / "success_rate_curve.csv"
)

BENCHMARK_MANIFEST_PATH = (
    RESULTS_DIR
    / "benchmark_manifest.json"
)

TARGET = "known_positive"

GROUP = "spatial_block_id"

SEED = 12345

BACKGROUND_REPLICATES = 10

BACKGROUND_PER_REPLICATE = 500

CV_FOLDS = 5

TOP_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
)

FEATURE_SET_KEYS = {
    "A": "A_non_geochemical",
    "B": "B_raw_geochemistry",
    "C": "C_ilr_geochemistry",
}

FORBIDDEN_ORACLE = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "deposit_id",
    "distance_to_known_deposit",
}

MODEL_NAMES = (
    "dummy",
    "logistic_regression",
    "random_forest",
)

VALIDATION_STRATEGIES = (
    "spatial",
    "random",
)


def load_json(
    path: Path,
) -> dict:
    """Load one JSON object."""
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def feature_sets_from_manifest(
    manifest: dict,
) -> dict[str, list[str]]:
    """Read canonical Gate 3 feature definitions."""
    raw = manifest["feature_sets"]

    return {
        short_name: list(
            raw[canonical_name]
        )
        for short_name, canonical_name
        in FEATURE_SET_KEYS.items()
    }


def validate_feature_contract(
    evidence: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    forbidden: set[str],
) -> None:
    """Reject missing, duplicate or forbidden model columns."""
    evidence_columns = set(
        evidence.columns
    )

    for name, columns in (
        feature_sets.items()
    ):
        missing = (
            set(columns)
            - evidence_columns
        )

        leakage = (
            set(columns)
            & forbidden
        )

        if missing:
            raise RuntimeError(
                f"Feature set {name} has "
                f"missing columns: "
                f"{sorted(missing)}"
            )

        if leakage:
            raise RuntimeError(
                f"Feature set {name} leaks "
                f"forbidden columns: "
                f"{sorted(leakage)}"
            )

        if len(columns) != len(
            set(columns)
        ):
            raise RuntimeError(
                f"Feature set {name} "
                "contains duplicates."
            )

    all_features = set().union(
        *(
            set(columns)
            for columns
            in feature_sets.values()
        )
    )

    oracle_overlap = (
        all_features
        & FORBIDDEN_ORACLE
    )

    if oracle_overlap:
        raise RuntimeError(
            "Oracle leakage detected: "
            f"{sorted(oracle_overlap)}"
        )


def sample_background(
    evidence: pd.DataFrame,
    replicate: int,
) -> pd.DataFrame:
    """Sample known positives plus unlabeled background only."""
    positives = evidence.loc[
        evidence[TARGET] == 1
    ]

    background = evidence.loc[
        evidence[TARGET] == 0
    ]

    if len(positives) != 100:
        raise RuntimeError(
            "Expected exactly 100 known positives."
        )

    if len(background) < (
        BACKGROUND_PER_REPLICATE
    ):
        raise RuntimeError(
            "Insufficient background rows."
        )

    rng = np.random.default_rng(
        SEED + replicate
    )

    positions = rng.choice(
        len(background),
        size=BACKGROUND_PER_REPLICATE,
        replace=False,
    )

    sampled_background = (
        background.iloc[positions]
    )

    sampled = pd.concat(
        [
            positives,
            sampled_background,
        ],
        axis=0,
        ignore_index=True,
    )

    sampled = sampled.sample(
        frac=1.0,
        random_state=SEED + replicate,
    ).reset_index(
        drop=True
    )

    if int(
        sampled[TARGET].sum()
    ) != 100:
        raise RuntimeError(
            "Positive count changed during "
            "background sampling."
        )

    if len(sampled) != 600:
        raise RuntimeError(
            "Expected 600 rows per replicate."
        )

    return sampled


def build_preprocessor(
    feature_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build fold-local categorical/numeric preprocessing."""
    categorical = [
        column
        for column in feature_columns
        if column == "lithology"
    ]

    numerical = [
        column
        for column in feature_columns
        if column not in categorical
    ]

    if categorical != ["lithology"]:
        raise RuntimeError(
            "Expected lithology as the sole "
            "categorical feature."
        )

    numeric_transformer: object

    if scale_numeric:
        numeric_transformer = (
            StandardScaler()
        )
    else:
        numeric_transformer = (
            "passthrough"
        )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
            (
                "numerical",
                numeric_transformer,
                numerical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    model_name: str,
    feature_columns: list[str],
    seed: int,
) -> Pipeline:
    """Create deterministic fold-safe sklearn pipeline."""
    if model_name == "dummy":
        estimator = DummyClassifier(
            strategy="prior"
        )

        scale_numeric = False

    elif model_name == (
        "logistic_regression"
    ):
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )

        scale_numeric = True

    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=64,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )

        scale_numeric = False

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    preprocessor = build_preprocessor(
        feature_columns,
        scale_numeric=scale_numeric,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def recall_at_fraction(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> float:
    """Recall obtained by inspecting the top fraction of ranked cells."""
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
            "Recall-at-fraction requires "
            "at least one positive."
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

    hits = int(
        y_true[selected].sum()
    )

    return float(
        hits / positives
    )


def metric_record(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate benchmark metrics."""
    predicted = (
        scores >= 0.5
    ).astype(int)

    return {
        "average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "recall_top_1pct": (
            recall_at_fraction(
                y_true,
                scores,
                0.01,
            )
        ),
        "recall_top_5pct": (
            recall_at_fraction(
                y_true,
                scores,
                0.05,
            )
        ),
        "recall_top_10pct": (
            recall_at_fraction(
                y_true,
                scores,
                0.10,
            )
        ),
    }


def get_splits(
    sampled: pd.DataFrame,
    strategy: str,
    replicate: int,
):
    """Generate deterministic CV folds."""
    y = sampled[TARGET].to_numpy(
        dtype=int
    )

    if strategy == "spatial":
        splitter = StratifiedGroupKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=SEED + replicate,
        )

        return splitter.split(
            sampled,
            y,
            groups=sampled[
                GROUP
            ].to_numpy(),
        )

    if strategy == "random":
        splitter = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=SEED + replicate,
        )

        return splitter.split(
            sampled,
            y,
        )

    raise ValueError(
        f"Unknown validation strategy: "
        f"{strategy}"
    )


def validate_fold(
    sampled: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    strategy: str,
) -> tuple[int, int, int]:
    """Validate fold class support and spatial overlap."""
    train = sampled.iloc[
        train_index
    ]

    test = sampled.iloc[
        test_index
    ]

    train_pos = int(
        train[TARGET].sum()
    )

    test_pos = int(
        test[TARGET].sum()
    )

    if train_pos <= 0:
        raise RuntimeError(
            "Training fold has zero positives."
        )

    if test_pos <= 0:
        raise RuntimeError(
            "Test fold has zero positives."
        )

    overlap = len(
        set(
            train[GROUP]
        )
        & set(
            test[GROUP]
        )
    )

    if (
        strategy == "spatial"
        and overlap != 0
    ):
        raise RuntimeError(
            "Spatial group leakage detected."
        )

    return (
        train_pos,
        test_pos,
        overlap,
    )


def aggregate_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold metrics over replicates and folds."""
    metric_columns = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "recall_top_1pct",
        "recall_top_5pct",
        "recall_top_10pct",
    ]

    grouped = fold_metrics.groupby(
        [
            "validation_strategy",
            "feature_set",
            "model",
        ],
        sort=True,
    )

    rows: list[dict] = []

    for keys, group in grouped:
        strategy, feature_set, model = (
            keys
        )

        row: dict[str, object] = {
            "validation_strategy": (
                strategy
            ),
            "feature_set": feature_set,
            "model": model,
            "n_fits": int(
                len(group)
            ),
        }

        for metric in metric_columns:
            row[
                f"{metric}_mean"
            ] = float(
                group[metric].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                group[metric].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
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


def build_spatial_random_gap(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Compare random-CV optimism against spatial CV."""
    metrics = [
        "average_precision",
        "roc_auc",
        "recall_top_1pct",
        "recall_top_5pct",
        "recall_top_10pct",
    ]

    rows: list[dict] = []

    for feature_set in (
        "A",
        "B",
        "C",
    ):
        for model in MODEL_NAMES:
            spatial = aggregate.loc[
                (
                    aggregate[
                        "validation_strategy"
                    ]
                    == "spatial"
                )
                & (
                    aggregate[
                        "feature_set"
                    ]
                    == feature_set
                )
                & (
                    aggregate[
                        "model"
                    ]
                    == model
                )
            ]

            random = aggregate.loc[
                (
                    aggregate[
                        "validation_strategy"
                    ]
                    == "random"
                )
                & (
                    aggregate[
                        "feature_set"
                    ]
                    == feature_set
                )
                & (
                    aggregate[
                        "model"
                    ]
                    == model
                )
            ]

            if (
                len(spatial) != 1
                or len(random) != 1
            ):
                raise RuntimeError(
                    "Incomplete spatial/random "
                    "aggregate comparison."
                )

            row: dict[str, object] = {
                "feature_set": feature_set,
                "model": model,
            }

            for metric in metrics:
                spatial_value = float(
                    spatial[
                        f"{metric}_mean"
                    ].iloc[0]
                )

                random_value = float(
                    random[
                        f"{metric}_mean"
                    ].iloc[0]
                )

                row[
                    f"{metric}_spatial"
                ] = spatial_value

                row[
                    f"{metric}_random"
                ] = random_value

                row[
                    f"{metric}_gap_random_minus_spatial"
                ] = (
                    random_value
                    - spatial_value
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def main() -> int:
    """Run the complete Gate 4 benchmark."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence = pd.read_csv(
        EVIDENCE_PATH
    )

    manifest = load_json(
        MANIFEST_PATH
    )

    contract = load_json(
        CONTRACT_PATH
    )

    feature_sets = (
        feature_sets_from_manifest(
            manifest
        )
    )

    forbidden = set(
        contract[
            "forbidden_model_features"
        ]
    )

    validate_feature_contract(
        evidence,
        feature_sets,
        forbidden,
    )

    if int(
        evidence[TARGET].sum()
    ) != 100:
        raise RuntimeError(
            "Unexpected known-positive count."
        )

    fold_rows: list[dict] = []

    curve_rows: list[dict] = []

    total_fits = 0

    print(
        "LAB05_GATE04_BENCHMARK_START"
    )

    for replicate in range(
        BACKGROUND_REPLICATES
    ):
        sampled = sample_background(
            evidence,
            replicate,
        )

        for strategy in (
            VALIDATION_STRATEGIES
        ):
            splits = list(
                get_splits(
                    sampled,
                    strategy,
                    replicate,
                )
            )

            if len(splits) != CV_FOLDS:
                raise RuntimeError(
                    "Unexpected fold count."
                )

            for feature_set_name, feature_columns in (
                feature_sets.items()
            ):
                X = sampled[
                    feature_columns
                ]

                y = sampled[
                    TARGET
                ].to_numpy(
                    dtype=int
                )

                for model_name in (
                    MODEL_NAMES
                ):
                    for fold_number, (
                        train_index,
                        test_index,
                    ) in enumerate(
                        splits,
                        start=1,
                    ):
                        (
                            train_pos,
                            test_pos,
                            overlap,
                        ) = validate_fold(
                            sampled,
                            train_index,
                            test_index,
                            strategy,
                        )

                        seed = (
                            SEED
                            + replicate * 1000
                            + fold_number * 10
                        )

                        pipeline = (
                            build_model_pipeline(
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

                        fold_metric = (
                            metric_record(
                                y[
                                    test_index
                                ],
                                scores,
                            )
                        )

                        train_groups = int(
                            sampled.iloc[
                                train_index
                            ][GROUP].nunique()
                        )

                        test_groups = int(
                            sampled.iloc[
                                test_index
                            ][GROUP].nunique()
                        )

                        fold_rows.append(
                            {
                                "replicate": (
                                    replicate
                                ),
                                "validation_strategy": (
                                    strategy
                                ),
                                "feature_set": (
                                    feature_set_name
                                ),
                                "model": (
                                    model_name
                                ),
                                "fold": (
                                    fold_number
                                ),
                                "n_train": int(
                                    len(
                                        train_index
                                    )
                                ),
                                "n_test": int(
                                    len(
                                        test_index
                                    )
                                ),
                                "train_positives": (
                                    train_pos
                                ),
                                "test_positives": (
                                    test_pos
                                ),
                                "train_groups": (
                                    train_groups
                                ),
                                "test_groups": (
                                    test_groups
                                ),
                                "group_overlap": (
                                    overlap
                                ),
                                **fold_metric,
                            }
                        )

                        y_test = y[
                            test_index
                        ]

                        for fraction in (
                            TOP_FRACTIONS
                        ):
                            curve_rows.append(
                                {
                                    "replicate": (
                                        replicate
                                    ),
                                    "validation_strategy": (
                                        strategy
                                    ),
                                    "feature_set": (
                                        feature_set_name
                                    ),
                                    "model": (
                                        model_name
                                    ),
                                    "fold": (
                                        fold_number
                                    ),
                                    "top_fraction": (
                                        fraction
                                    ),
                                    "recall": (
                                        recall_at_fraction(
                                            y_test,
                                            scores,
                                            fraction,
                                        )
                                    ),
                                }
                            )

                        total_fits += 1

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

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    if len(fold_metrics) != 900:
        raise RuntimeError(
            "Expected exactly 900 "
            "fold-level benchmark rows."
        )

    spatial_overlap = fold_metrics.loc[
        fold_metrics[
            "validation_strategy"
        ]
        == "spatial",
        "group_overlap",
    ]

    if not (
        spatial_overlap == 0
    ).all():
        raise RuntimeError(
            "Spatial CV contains group overlap."
        )

    aggregate = aggregate_metrics(
        fold_metrics
    )

    if len(aggregate) != 18:
        raise RuntimeError(
            "Expected 18 aggregate "
            "benchmark configurations."
        )

    gap = build_spatial_random_gap(
        aggregate
    )

    if len(gap) != 9:
        raise RuntimeError(
            "Expected 9 spatial/random "
            "comparisons."
        )

    success_curve = pd.DataFrame(
        curve_rows
    )

    expected_curve_rows = (
        900
        * len(TOP_FRACTIONS)
    )

    if (
        len(success_curve)
        != expected_curve_rows
    ):
        raise RuntimeError(
            "Unexpected success-rate "
            "curve row count."
        )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    aggregate.to_csv(
        AGGREGATE_METRICS_PATH,
        index=False,
    )

    gap.to_csv(
        GAP_PATH,
        index=False,
    )

    success_curve.to_csv(
        SUCCESS_CURVE_PATH,
        index=False,
    )

    benchmark_manifest = {
        "seed": SEED,
        "target_semantics": (
            "known_positive_vs_sampled_unlabeled_background"
        ),
        "background_is_confirmed_negative": False,
        "score_semantics": (
            "prospectivity_score_not_calibrated_deposit_probability"
        ),
        "background_replicates": (
            BACKGROUND_REPLICATES
        ),
        "background_per_replicate": (
            BACKGROUND_PER_REPLICATE
        ),
        "known_positives_per_replicate": 100,
        "rows_per_replicate": 600,
        "cv_folds": CV_FOLDS,
        "validation_primary": "spatial",
        "validation_diagnostic": "random",
        "models": list(
            MODEL_NAMES
        ),
        "feature_sets": (
            FEATURE_SET_KEYS
        ),
        "planned_model_fits": (
            expected_fits
        ),
        "observed_model_fits": (
            total_fits
        ),
        "spatial_group_overlap_max": int(
            spatial_overlap.max()
        ),
        "oracle_used_for_training": False,
        "oracle_used_for_background_sampling": False,
        "top_fractions": list(
            TOP_FRACTIONS
        ),
        "random_forest": {
            "n_estimators": 64,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": 1,
        },
        "logistic_regression": {
            "solver": "liblinear",
            "max_iter": 2000,
        },
        "categorical_preprocessing": (
            "OneHotEncoder fitted inside each fold"
        ),
        "numeric_scaling_logistic": (
            "StandardScaler fitted inside each fold"
        ),
    }

    BENCHMARK_MANIFEST_PATH.write_text(
        json.dumps(
            benchmark_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "LAB05_GATE04 = PASS"
    )

    print(
        f"MODEL_FITS = {total_fits}"
    )

    print(
        f"FOLD_METRIC_ROWS = "
        f"{len(fold_metrics)}"
    )

    print(
        f"AGGREGATE_ROWS = "
        f"{len(aggregate)}"
    )

    print(
        f"SPATIAL_RANDOM_GAP_ROWS = "
        f"{len(gap)}"
    )

    print(
        f"SUCCESS_CURVE_ROWS = "
        f"{len(success_curve)}"
    )

    print(
        "SPATIAL_GROUP_OVERLAP_MAX = "
        f"{int(spatial_overlap.max())}"
    )

    print(
        "ORACLE_USED_FOR_TRAINING = NO"
    )

    print(
        "ORACLE_USED_FOR_BACKGROUND_SAMPLING = NO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
'''


TEST_CODE = r'''"""Tests for LAB05 Gate 4 spatial benchmarking."""

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
    / "05d_run_spatial_benchmark.py"
)

FOLD_PATH = (
    LAB_DIR
    / "results"
    / "benchmark_fold_metrics.csv"
)

AGG_PATH = (
    LAB_DIR
    / "results"
    / "benchmark_aggregate_metrics.csv"
)

GAP_PATH = (
    LAB_DIR
    / "results"
    / "spatial_random_gap.csv"
)

CURVE_PATH = (
    LAB_DIR
    / "results"
    / "success_rate_curve.csv"
)

MANIFEST_PATH = (
    LAB_DIR
    / "results"
    / "benchmark_manifest.json"
)


def load_module():
    spec = (
        importlib.util.spec_from_file_location(
            "lab05_gate04",
            MODULE_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load Gate 4 module."
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


def test_gate04_outputs_exist() -> None:
    for path in (
        FOLD_PATH,
        AGG_PATH,
        GAP_PATH,
        CURVE_PATH,
        MANIFEST_PATH,
    ):
        assert path.is_file()


def test_fold_metrics_have_900_rows() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    assert len(frame) == 900


def test_aggregate_has_18_rows() -> None:
    frame = pd.read_csv(
        AGG_PATH
    )

    assert len(frame) == 18


def test_gap_has_9_rows() -> None:
    frame = pd.read_csv(
        GAP_PATH
    )

    assert len(frame) == 9


def test_success_curve_has_5400_rows() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    assert len(frame) == 5400


def test_all_three_models_present() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    assert set(
        frame["model"]
    ) == {
        "dummy",
        "logistic_regression",
        "random_forest",
    }


def test_all_feature_sets_present() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    assert set(
        frame["feature_set"]
    ) == {
        "A",
        "B",
        "C",
    }


def test_both_validation_strategies_present() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    assert set(
        frame["validation_strategy"]
    ) == {
        "spatial",
        "random",
    }


def test_spatial_group_overlap_is_zero() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    spatial = frame.loc[
        frame[
            "validation_strategy"
        ]
        == "spatial"
    ]

    assert (
        spatial[
            "group_overlap"
        ]
        == 0
    ).all()


def test_each_configuration_has_50_fits() -> None:
    frame = pd.read_csv(
        FOLD_PATH
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
        counts == 50
    ).all()


def test_metric_ranges() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    columns = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "recall_top_1pct",
        "recall_top_5pct",
        "recall_top_10pct",
    ]

    for column in columns:
        assert frame[column].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_no_zero_positive_test_fold() -> None:
    frame = pd.read_csv(
        FOLD_PATH
    )

    assert (
        frame[
            "test_positives"
        ]
        > 0
    ).all()


def test_manifest_semantics() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
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
            "oracle_used_for_background_sampling"
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
            "spatial_group_overlap_max"
        ]
        == 0
    )


def test_oracle_features_absent() -> None:
    module = load_module()

    manifest = module.load_json(
        module.MANIFEST_PATH
    )

    feature_sets = (
        module.feature_sets_from_manifest(
            manifest
        )
    )

    all_features = set().union(
        *(
            set(columns)
            for columns
            in feature_sets.values()
        )
    )

    assert not (
        all_features
        & module.FORBIDDEN_ORACLE
    )


def test_background_sampling_is_deterministic() -> None:
    module = load_module()

    evidence = pd.read_csv(
        module.EVIDENCE_PATH
    )

    first = module.sample_background(
        evidence,
        0,
    )

    second = module.sample_background(
        evidence,
        0,
    )

    assert first[
        "cell_id"
    ].tolist() == second[
        "cell_id"
    ].tolist()


def test_background_replicates_differ() -> None:
    module = load_module()

    evidence = pd.read_csv(
        module.EVIDENCE_PATH
    )

    first = module.sample_background(
        evidence,
        0,
    )

    second = module.sample_background(
        evidence,
        1,
    )

    first_background = set(
        first.loc[
            first[
                module.TARGET
            ]
            == 0,
            "cell_id",
        ]
    )

    second_background = set(
        second.loc[
            second[
                module.TARGET
            ]
            == 0,
            "cell_id",
        ]
    )

    assert (
        first_background
        != second_background
    )


def test_lithology_is_fold_local_encoded() -> None:
    module = load_module()

    manifest = module.load_json(
        module.MANIFEST_PATH
    )

    feature_sets = (
        module.feature_sets_from_manifest(
            manifest
        )
    )

    pipeline = (
        module.build_model_pipeline(
            "logistic_regression",
            feature_sets["A"],
            module.SEED,
        )
    )

    assert (
        "preprocess"
        in pipeline.named_steps
    )

    assert (
        "model"
        in pipeline.named_steps
    )


def test_success_curve_top_fractions() -> None:
    frame = pd.read_csv(
        CURVE_PATH
    )

    assert set(
        frame[
            "top_fraction"
        ].unique()
    ) == {
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    }
'''


RUNNER_CODE = r'''"""Run LAB05 Gate 4 and its QA non-interactively."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

PYTHON = (
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

BENCHMARK = (
    LAB_DIR
    / "05d_run_spatial_benchmark.py"
)

TEST_FILE = (
    ROOT
    / "tests"
    / "test_lab05_spatial_benchmark.py"
)

RESULT_FILES = (
    LAB_DIR
    / "results"
    / "benchmark_fold_metrics.csv",
    LAB_DIR
    / "results"
    / "benchmark_aggregate_metrics.csv",
    LAB_DIR
    / "results"
    / "spatial_random_gap.csv",
    LAB_DIR
    / "results"
    / "success_rate_curve.csv",
    LAB_DIR
    / "results"
    / "benchmark_manifest.json",
)


def environment() -> dict[str, str]:
    env = os.environ.copy()

    env["PYTHONPATH"] = str(
        ROOT / "src"
    )

    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["CI"] = "1"

    return env


def run(
    command: list[str],
) -> None:
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=environment(),
        stdin=subprocess.DEVNULL,
    )


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def snapshot() -> dict[str, str]:
    result: dict[str, str] = {}

    for path in RESULT_FILES:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        result[
            str(
                path.relative_to(ROOT)
            )
        ] = sha256(
            path
        )

    return result


def verify_sizes() -> None:
    limit = 5 * 1024 * 1024

    for path in RESULT_FILES:
        size = path.stat().st_size

        print(
            f"RESULT_BYTES {path.name} = "
            f"{size}"
        )

        if size > limit:
            raise RuntimeError(
                f"Result exceeds 5 MiB: {path}"
            )


def main() -> int:
    print(
        "LAB05_GATE04_RUNNER_START"
    )

    run(
        [
            PYTHON,
            "-m",
            "py_compile",
            str(BENCHMARK),
            str(TEST_FILE),
        ]
    )

    run(
        [
            PYTHON,
            str(BENCHMARK),
        ]
    )

    first = snapshot()

    run(
        [
            PYTHON,
            str(BENCHMARK),
        ]
    )

    second = snapshot()

    if first != second:
        changed = [
            key
            for key in first
            if first[key]
            != second[key]
        ]

        raise RuntimeError(
            "Gate 4 outputs are not "
            "byte-reproducible: "
            f"{changed}"
        )

    print(
        "BYTE_REPRODUCIBILITY = PASS"
    )

    verify_sizes()

    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/"
            "test_lab05_spatial_benchmark.py",
            "-q",
        ]
    )

    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
        ]
    )

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--check",
        ]
    )

    print()
    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "LAB05_GATE04_QA = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
'''


def environment() -> dict[str, str]:
    env = os.environ.copy()

    env["PYTHONPATH"] = str(
        ROOT / "src"
    )

    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["CI"] = "1"

    return env


def run(
    command: list[str],
) -> None:
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=environment(),
        stdin=subprocess.DEVNULL,
    )


def write_file(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )

    print(
        "WRITTEN = "
        f"{path.relative_to(ROOT)}"
    )


def verify_initial_repository_state() -> None:
    branch = subprocess.run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Unexpected branch: {branch}"
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.splitlines()

    allowed = {
        f"?? {SELF_RELATIVE}",
    }

    unexpected = [
        line
        for line in status
        if line not in allowed
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected repository changes "
            "before Gate 4 setup: "
            f"{unexpected}"
        )

    print(
        "INITIAL_REPOSITORY_STATE = PASS"
    )


def main() -> int:
    print(
        "LAB05_GATE04_AUTOMATION_START"
    )

    verify_initial_repository_state()

    write_file(
        BENCHMARK_PATH,
        BENCHMARK_CODE,
    )

    write_file(
        TEST_PATH,
        TEST_CODE,
    )

    write_file(
        RUNNER_PATH,
        RUNNER_CODE,
    )

    run(
        [
            PYTHON,
            "-m",
            "py_compile",
            str(BENCHMARK_PATH),
            str(TEST_PATH),
            str(RUNNER_PATH),
        ]
    )

    run(
        [
            PYTHON,
            str(RUNNER_PATH),
        ]
    )

    print()
    print(
        "LAB05_GATE04_AUTOMATION = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
