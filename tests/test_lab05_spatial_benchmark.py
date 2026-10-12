"""Tests for LAB05 Gate 4 spatial benchmarking."""

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
