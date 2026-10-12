"""LAB05 Gate 5 preflight: full-grid prospectivity scoring contract.

This preflight freezes the operational scoring design before any
full-grid model fitting or prospectivity ranking is generated.

Scientific rules:
- Gate 4 Spatial CV drives operational model selection.
- Random CV remains diagnostic only.
- Background cells are unlabeled, not confirmed negatives.
- Oracle values remain sealed.
- Coordinates and identifiers may be carried to outputs but never used as X.
- Prospectivity scores are ranking scores, not calibrated deposit probabilities.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

RESULTS_DIR = LAB_DIR / "results"

EVIDENCE_PATH = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

EVIDENCE_MANIFEST_PATH = (
    RESULTS_DIR
    / "evidence_manifest.json"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

BENCHMARK_AGGREGATE_PATH = (
    RESULTS_DIR
    / "benchmark_aggregate_metrics.csv"
)

BENCHMARK_FOLD_PATH = (
    RESULTS_DIR
    / "benchmark_fold_metrics.csv"
)

BENCHMARK_MANIFEST_PATH = (
    RESULTS_DIR
    / "benchmark_manifest.json"
)

EXPECTED_BRANCH = (
    "feat/lab05-mineral-prospectivity"
)

EXPECTED_HEAD = "0375e6c"

SELF_STATUS = (
    "?? scripts/lab05/preflight_gate05.py"
)

PRIMARY_FEATURE_SET = "B"

PRIMARY_MODEL = "random_forest"

BACKGROUND_REPLICATES = 10

EXPECTED_GRID_ROWS = 6400

EXPECTED_KNOWN_POSITIVES = 100

EXPECTED_BACKGROUND_ROWS = 6300

EXPECTED_FEATURE_B_COUNT = 19

EXPECTED_FORBIDDEN_COUNT = 10

EXPECTED_MODEL_FITS_GATE04 = 900

OUTPUT_IDENTIFIERS = (
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "known_positive",
)

SCORE_OUTPUT_COLUMNS = (
    "prospectivity_score_mean",
    "prospectivity_score_median",
    "prospectivity_score_std",
    "prospectivity_score_p25",
    "prospectivity_score_p75",
    "prospectivity_score_iqr",
    "prospectivity_rank",
    "prospectivity_percentile",
    "top_1pct",
    "top_5pct",
    "top_10pct",
)

RANK_TIE_BREAK = (
    "mean_desc,"
    "median_desc,"
    "std_asc,"
    "cell_id_asc"
)

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "deposit_id",
    "distance_to_known_deposit",
}


def git_output(
    *args: str,
) -> str:
    """Run non-interactive Git and return stdout."""
    result = subprocess.run(
        [
            "git",
            "--no-pager",
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    return result.stdout.strip()


def load_json(
    path: Path,
) -> dict:
    """Load UTF-8 JSON."""
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def validate_repository() -> None:
    """Require exact Gate 4 checkpoint and clean repository."""
    branch = git_output(
        "branch",
        "--show-current",
    )

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Unexpected branch: {branch}"
        )

    head = git_output(
        "rev-parse",
        "--short=7",
        "HEAD",
    )

    if head != EXPECTED_HEAD:
        raise RuntimeError(
            "Unexpected starting HEAD: "
            f"{head}"
        )

    porcelain = git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    status_lines = [
        line
        for line in porcelain.splitlines()
        if line.strip()
    ]

    unexpected = [
        line
        for line in status_lines
        if line != SELF_STATUS
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected repository changes: "
            f"{unexpected}"
        )

    print(
        f"BRANCH = {branch}"
    )

    print(
        f"STARTING_HEAD = {head}"
    )

    if status_lines:
        print(
            "WORKTREE_CLEAN_EXCEPT_PREFLIGHT = YES"
        )
    else:
        print(
            "WORKTREE_CLEAN = YES"
        )


def validate_required_files() -> None:
    """Require all Gate 3 and Gate 4 inputs."""
    paths = (
        EVIDENCE_PATH,
        EVIDENCE_MANIFEST_PATH,
        CONTRACT_PATH,
        BENCHMARK_AGGREGATE_PATH,
        BENCHMARK_FOLD_PATH,
        BENCHMARK_MANIFEST_PATH,
    )

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    print(
        "REQUIRED_GATE04_ARTIFACTS = PASS"
    )


def validate_gate04_contract(
    benchmark_manifest: dict,
) -> None:
    """Verify the completed Gate 4 scientific contract."""
    if (
        benchmark_manifest[
            "validation_primary"
        ]
        != "spatial"
    ):
        raise RuntimeError(
            "Spatial CV must remain primary."
        )

    if (
        benchmark_manifest[
            "validation_diagnostic"
        ]
        != "random"
    ):
        raise RuntimeError(
            "Random CV must remain diagnostic."
        )

    if (
        benchmark_manifest[
            "observed_model_fits"
        ]
        != EXPECTED_MODEL_FITS_GATE04
    ):
        raise RuntimeError(
            "Unexpected Gate 4 fit count."
        )

    if (
        benchmark_manifest[
            "spatial_group_overlap_max"
        ]
        != 0
    ):
        raise RuntimeError(
            "Gate 4 spatial leakage detected."
        )

    if benchmark_manifest[
        "oracle_used_for_training"
    ]:
        raise RuntimeError(
            "Gate 4 used oracle for training."
        )

    if benchmark_manifest[
        "oracle_used_for_background_sampling"
    ]:
        raise RuntimeError(
            "Gate 4 used oracle for background sampling."
        )

    if benchmark_manifest[
        "background_is_confirmed_negative"
    ]:
        raise RuntimeError(
            "Background cannot be treated "
            "as confirmed negative."
        )

    print(
        "GATE04_CONTRACT = PASS"
    )


def select_primary_model(
    aggregate: pd.DataFrame,
) -> tuple[str, str, float]:
    """Reproduce the frozen operational selection rule."""
    required = {
        "validation_strategy",
        "feature_set",
        "model",
        "average_precision_mean",
    }

    missing = required - set(
        aggregate.columns
    )

    if missing:
        raise RuntimeError(
            "Aggregate metrics missing columns: "
            f"{sorted(missing)}"
        )

    spatial = aggregate.loc[
        (
            aggregate[
                "validation_strategy"
            ]
            == "spatial"
        )
        & (
            aggregate[
                "model"
            ]
            != "dummy"
        )
    ].copy()

    if spatial.empty:
        raise RuntimeError(
            "No non-dummy Spatial-CV results."
        )

    spatial = spatial.sort_values(
        [
            "average_precision_mean",
            "feature_set",
            "model",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    winner = spatial.iloc[0]

    feature_set = str(
        winner[
            "feature_set"
        ]
    )

    model = str(
        winner[
            "model"
        ]
    )

    ap = float(
        winner[
            "average_precision_mean"
        ]
    )

    if feature_set != PRIMARY_FEATURE_SET:
        raise RuntimeError(
            "Unexpected primary feature set: "
            f"{feature_set}"
        )

    if model != PRIMARY_MODEL:
        raise RuntimeError(
            "Unexpected primary model: "
            f"{model}"
        )

    print(
        f"PRIMARY_FEATURE_SET = {feature_set}"
    )

    print(
        f"PRIMARY_MODEL = {model}"
    )

    print(
        "PRIMARY_SPATIAL_AP = "
        f"{ap:.12f}"
    )

    print(
        "PRIMARY_SELECTION_RULE = "
        "highest_mean_spatial_average_precision_non_dummy"
    )

    return (
        feature_set,
        model,
        ap,
    )


def validate_replica_diagnostics(
    fold_metrics: pd.DataFrame,
) -> None:
    """Record robustness evidence used to contextualize selection."""
    required = {
        "replicate",
        "validation_strategy",
        "feature_set",
        "model",
        "average_precision",
    }

    missing = required - set(
        fold_metrics.columns
    )

    if missing:
        raise RuntimeError(
            "Fold metrics missing columns: "
            f"{sorted(missing)}"
        )

    rf = (
        fold_metrics.loc[
            (
                fold_metrics[
                    "validation_strategy"
                ]
                == "spatial"
            )
            & (
                fold_metrics[
                    "model"
                ]
                == "random_forest"
            )
        ]
        .groupby(
            [
                "replicate",
                "feature_set",
            ],
            as_index=False,
        )[
            "average_precision"
        ]
        .mean()
        .pivot(
            index="replicate",
            columns="feature_set",
            values="average_precision",
        )
    )

    if len(rf) != BACKGROUND_REPLICATES:
        raise RuntimeError(
            "Unexpected RF replicate count."
        )

    b_gt_a = int(
        (
            rf["B"]
            > rf["A"]
        ).sum()
    )

    b_gt_c = int(
        (
            rf["B"]
            > rf["C"]
        ).sum()
    )

    print(
        "RF_B_GT_A_REPLICATES = "
        f"{b_gt_a}/10"
    )

    print(
        "RF_B_GT_C_REPLICATES = "
        f"{b_gt_c}/10"
    )

    lr = (
        fold_metrics.loc[
            (
                fold_metrics[
                    "validation_strategy"
                ]
                == "spatial"
            )
            & (
                fold_metrics[
                    "model"
                ]
                == "logistic_regression"
            )
        ]
        .groupby(
            [
                "replicate",
                "feature_set",
            ],
            as_index=False,
        )[
            "average_precision"
        ]
        .mean()
        .pivot(
            index="replicate",
            columns="feature_set",
            values="average_precision",
        )
    )

    c_gt_b = int(
        (
            lr["C"]
            > lr["B"]
        ).sum()
    )

    print(
        "LR_C_GT_B_REPLICATES = "
        f"{c_gt_b}/10"
    )

    print(
        "MODEL_SELECTION_INTERPRETATION = "
        "operational_not_confirmatory"
    )


def validate_feature_contract(
    evidence_manifest: dict,
    contract: dict,
) -> list[str]:
    """Freeze feature set B and leakage restrictions."""
    feature_sets = evidence_manifest.get(
        "feature_sets"
    )

    if not isinstance(
        feature_sets,
        dict,
    ):
        raise RuntimeError(
            "Missing feature_sets dictionary."
        )

    feature_b = feature_sets.get(
        "B_raw_geochemistry"
    )

    if not isinstance(
        feature_b,
        list,
    ):
        raise RuntimeError(
            "Missing B_raw_geochemistry."
        )

    if len(feature_b) != (
        EXPECTED_FEATURE_B_COUNT
    ):
        raise RuntimeError(
            "Unexpected B feature count: "
            f"{len(feature_b)}"
        )

    forbidden = contract.get(
        "forbidden_model_features"
    )

    if not isinstance(
        forbidden,
        list,
    ):
        raise RuntimeError(
            "Missing forbidden feature contract."
        )

    if len(forbidden) != (
        EXPECTED_FORBIDDEN_COUNT
    ):
        raise RuntimeError(
            "Unexpected forbidden feature count: "
            f"{len(forbidden)}"
        )

    leakage = (
        set(feature_b)
        & set(forbidden)
    )

    if leakage:
        raise RuntimeError(
            "Primary model feature leakage: "
            f"{sorted(leakage)}"
        )

    oracle_overlap = (
        set(feature_b)
        & ORACLE_FIELDS
    )

    if oracle_overlap:
        raise RuntimeError(
            "Oracle present in feature B: "
            f"{sorted(oracle_overlap)}"
        )

    print(
        f"PRIMARY_FEATURE_COUNT = "
        f"{len(feature_b)}"
    )

    print(
        "PRIMARY_FEATURES = "
        + ",".join(feature_b)
    )

    print(
        f"FORBIDDEN_MODEL_FEATURES = "
        f"{len(forbidden)}"
    )

    print(
        "PRIMARY_FEATURE_LEAKAGE = 0"
    )

    print(
        "ORACLE_FEATURE_LEAKAGE = 0"
    )

    return list(
        feature_b
    )


def validate_evidence_without_oracle(
    feature_b: list[str],
) -> None:
    """Read only safe evidence columns; oracle values stay unopened."""
    header = pd.read_csv(
        EVIDENCE_PATH,
        nrows=0,
    )

    required_safe = list(
        dict.fromkeys(
            [
                *feature_b,
                *OUTPUT_IDENTIFIERS,
            ]
        )
    )

    missing = (
        set(required_safe)
        - set(header.columns)
    )

    if missing:
        raise RuntimeError(
            "Evidence missing safe columns: "
            f"{sorted(missing)}"
        )

    safe = pd.read_csv(
        EVIDENCE_PATH,
        usecols=required_safe,
    )

    if len(safe) != (
        EXPECTED_GRID_ROWS
    ):
        raise RuntimeError(
            "Unexpected full-grid row count: "
            f"{len(safe)}"
        )

    known = int(
        safe[
            "known_positive"
        ].sum()
    )

    background = int(
        len(safe) - known
    )

    if known != (
        EXPECTED_KNOWN_POSITIVES
    ):
        raise RuntimeError(
            "Unexpected known-positive count."
        )

    if background != (
        EXPECTED_BACKGROUND_ROWS
    ):
        raise RuntimeError(
            "Unexpected unlabeled/background count."
        )

    if safe[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "cell_id must be unique."
        )

    print(
        f"FULL_GRID_ROWS = {len(safe)}"
    )

    print(
        f"KNOWN_POSITIVES = {known}"
    )

    print(
        f"UNLABELED_BACKGROUND = {background}"
    )

    print(
        "ORACLE_VALUES_LOADED = NO"
    )


def validate_scoring_plan(
    benchmark_manifest: dict,
) -> None:
    """Freeze ensemble and ranking semantics."""
    gate04_replicates = int(
        benchmark_manifest[
            "background_replicates"
        ]
    )

    if gate04_replicates != (
        BACKGROUND_REPLICATES
    ):
        raise RuntimeError(
            "Gate 5 must reuse the 10 "
            "Gate 4 background replicates."
        )

    rf_config = benchmark_manifest.get(
        "random_forest"
    )

    if not isinstance(
        rf_config,
        dict,
    ):
        raise RuntimeError(
            "Missing frozen Random Forest config."
        )

    expected_predictions = (
        EXPECTED_GRID_ROWS
        * BACKGROUND_REPLICATES
    )

    top_1 = int(
        math.ceil(
            EXPECTED_GRID_ROWS
            * 0.01
        )
    )

    top_5 = int(
        math.ceil(
            EXPECTED_GRID_ROWS
            * 0.05
        )
    )

    top_10 = int(
        math.ceil(
            EXPECTED_GRID_ROWS
            * 0.10
        )
    )

    print(
        "ENSEMBLE_MODELS = 10"
    )

    print(
        "TRAINING_ROWS_PER_MODEL = 600"
    )

    print(
        "KNOWN_POSITIVES_PER_MODEL = 100"
    )

    print(
        "BACKGROUND_PER_MODEL = 500"
    )

    print(
        "FULL_GRID_PREDICTIONS = "
        f"{expected_predictions}"
    )

    print(
        "ENSEMBLE_AGGREGATION = "
        "mean,median,std,p25,p75,iqr"
    )

    print(
        "PRIMARY_RANK_SCORE = "
        "prospectivity_score_mean"
    )

    print(
        "RANK_TIE_BREAK = "
        f"{RANK_TIE_BREAK}"
    )

    print(
        f"TOP_1PCT_CELLS = {top_1}"
    )

    print(
        f"TOP_5PCT_CELLS = {top_5}"
    )

    print(
        f"TOP_10PCT_CELLS = {top_10}"
    )

    print(
        "SCORE_SEMANTICS = "
        "prospectivity_score_not_calibrated_probability"
    )

    print(
        "RANDOM_FOREST_CONFIG = "
        f"{json.dumps(rf_config, sort_keys=True)}"
    )


def validate_output_contract() -> None:
    """Declare identifiers, score fields and future oracle boundary."""
    print(
        "OUTPUT_IDENTIFIERS = "
        + ",".join(
            OUTPUT_IDENTIFIERS
        )
    )

    print(
        "OUTPUT_SCORE_COLUMNS = "
        + ",".join(
            SCORE_OUTPUT_COLUMNS
        )
    )

    print(
        "COORDINATES_USED_AS_MODEL_FEATURES = NO"
    )

    print(
        "KNOWN_POSITIVE_USED_AS_MODEL_FEATURE = NO"
    )

    print(
        "GATE05_ORACLE_ACCESS = PROHIBITED"
    )

    print(
        "GATE06_ORACLE_ROLE = "
        "post_freeze_hidden_deposit_diagnostic_only"
    )


def main() -> int:
    """Run Gate 5 design preflight."""
    print(
        "LAB05_GATE05_PREFLIGHT_START"
    )

    validate_repository()

    validate_required_files()

    benchmark_manifest = load_json(
        BENCHMARK_MANIFEST_PATH
    )

    evidence_manifest = load_json(
        EVIDENCE_MANIFEST_PATH
    )

    contract = load_json(
        CONTRACT_PATH
    )

    validate_gate04_contract(
        benchmark_manifest
    )

    aggregate = pd.read_csv(
        BENCHMARK_AGGREGATE_PATH
    )

    fold_metrics = pd.read_csv(
        BENCHMARK_FOLD_PATH
    )

    select_primary_model(
        aggregate
    )

    validate_replica_diagnostics(
        fold_metrics
    )

    feature_b = (
        validate_feature_contract(
            evidence_manifest,
            contract,
        )
    )

    validate_evidence_without_oracle(
        feature_b
    )

    validate_scoring_plan(
        benchmark_manifest
    )

    validate_output_contract()

    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "GATE05_DESIGN_CONTRACT = FROZEN"
    )

    print(
        "LAB05_GATE05_PREFLIGHT = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
