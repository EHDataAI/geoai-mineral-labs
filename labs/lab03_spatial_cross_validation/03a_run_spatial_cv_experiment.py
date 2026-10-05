from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor

from geoai.spatial import (
    DEFAULT_BOUNDS,
    DEFAULT_CRS,
    generate_synthetic_sampling,
)
from geoai.validation import (
    FoldSplit,
    assign_spatial_blocks,
    random_kfold_indices,
    spatial_group_kfold_indices,
    summarize_fold_geometry,
)


SEED = 12345
N_SPLITS = 5
N_NEIGHBORS = 15
BLOCK_SIZES_M = (5000.0, 10000.0, 20000.0)

LAB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LAB_DIR / "results"


def build_features(
    frame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract coordinate features and observed/reference targets."""
    x = frame.geometry.x.to_numpy(dtype=float)
    y = frame.geometry.y.to_numpy(dtype=float)

    features = np.column_stack(
        [x, y]
    )

    target = frame["signal"].to_numpy(
        dtype=float
    )

    target_true = frame["signal_true"].to_numpy(
        dtype=float
    )

    return features, target, target_true


def make_model() -> KNeighborsRegressor:
    """
    Return the fixed local baseline used for every validation strategy.

    The model is intentionally simple. LAB03 isolates validation design,
    rather than performing model selection.
    """
    return KNeighborsRegressor(
        n_neighbors=N_NEIGHBORS,
        weights="distance",
        metric="euclidean",
    )


def evaluate_splits(
    *,
    features: np.ndarray,
    target: np.ndarray,
    splits: tuple[FoldSplit, ...],
    strategy: str,
    block_size_m: float,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Evaluate one fixed model under a supplied CV design."""
    geometry_rows = {
        int(row["fold"]): row
        for row in summarize_fold_geometry(
            x,
            y,
            splits,
            groups=groups,
        )
    }

    rows: list[dict[str, float | int | str]] = []

    for split in splits:
        model = make_model()

        model.fit(
            features[split.train_idx],
            target[split.train_idx],
        )

        prediction = model.predict(
            features[split.test_idx]
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    target[split.test_idx],
                    prediction,
                )
            )
        )

        mae = float(
            mean_absolute_error(
                target[split.test_idx],
                prediction,
            )
        )

        r2 = float(
            r2_score(
                target[split.test_idx],
                prediction,
            )
        )

        geometry = geometry_rows[int(split.fold)]

        rows.append(
            {
                "strategy": strategy,
                "block_size_m": float(block_size_m),
                "fold": int(split.fold),
                "train_n": int(split.train_idx.size),
                "test_n": int(split.test_idx.size),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "min_test_to_train_distance_m": float(
                    geometry["min_test_to_train_distance"]
                ),
                "median_test_to_train_distance_m": float(
                    geometry["median_test_to_train_distance"]
                ),
                "mean_test_to_train_distance_m": float(
                    geometry["mean_test_to_train_distance"]
                ),
                "group_overlap": int(
                    geometry["group_overlap"]
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold-level performance by validation strategy."""
    summary = (
        fold_results
        .groupby(
            ["block_size_m", "strategy"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            median_separation_m=(
                "median_test_to_train_distance_m",
                "median",
            ),
            mean_group_overlap=(
                "group_overlap",
                "mean",
            ),
        )
    )

    return summary


def build_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare random and spatial validation at each block resolution.

    Positive RMSE optimism means random CV reports a smaller error than
    spatial CV.
    """
    random_rows = (
        summary[
            summary["strategy"] == "random_kfold"
        ]
        .set_index("block_size_m")
    )

    spatial_rows = (
        summary[
            summary["strategy"] == "spatial_block"
        ]
        .set_index("block_size_m")
    )

    rows: list[dict[str, float]] = []

    for block_size in BLOCK_SIZES_M:
        random_row = random_rows.loc[block_size]
        spatial_row = spatial_rows.loc[block_size]

        random_rmse = float(
            random_row["rmse_mean"]
        )
        spatial_rmse = float(
            spatial_row["rmse_mean"]
        )

        random_mae = float(
            random_row["mae_mean"]
        )
        spatial_mae = float(
            spatial_row["mae_mean"]
        )

        random_r2 = float(
            random_row["r2_mean"]
        )
        spatial_r2 = float(
            spatial_row["r2_mean"]
        )

        relative_rmse_optimism_pct = (
            np.nan
            if spatial_rmse == 0
            else
            100.0
            * (spatial_rmse - random_rmse)
            / spatial_rmse
        )

        rows.append(
            {
                "block_size_m": float(block_size),
                "random_rmse": random_rmse,
                "spatial_rmse": spatial_rmse,
                "rmse_gap_spatial_minus_random": (
                    spatial_rmse - random_rmse
                ),
                "relative_rmse_optimism_pct": float(
                    relative_rmse_optimism_pct
                ),
                "random_mae": random_mae,
                "spatial_mae": spatial_mae,
                "mae_gap_spatial_minus_random": (
                    spatial_mae - random_mae
                ),
                "random_r2": random_r2,
                "spatial_r2": spatial_r2,
                "r2_gap_random_minus_spatial": (
                    random_r2 - spatial_r2
                ),
                "random_median_separation_m": float(
                    random_row[
                        "median_separation_m"
                    ]
                ),
                "spatial_median_separation_m": float(
                    spatial_row[
                        "median_separation_m"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_experiment() -> None:
    """Run the controlled LAB03 validation experiment."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = generate_synthetic_sampling(
        seed=SEED
    )

    features, target, target_true = (
        build_features(frame)
    )

    x = features[:, 0]
    y = features[:, 1]

    random_splits = random_kfold_indices(
        n_samples=len(frame),
        n_splits=N_SPLITS,
        random_state=SEED,
    )

    result_frames: list[pd.DataFrame] = []

    for block_size in BLOCK_SIZES_M:
        groups = assign_spatial_blocks(
            x,
            y,
            block_size=block_size,
            origin_x=DEFAULT_BOUNDS[0],
            origin_y=DEFAULT_BOUNDS[2],
        )

        spatial_splits = (
            spatial_group_kfold_indices(
                groups,
                n_splits=N_SPLITS,
            )
        )

        random_result = evaluate_splits(
            features=features,
            target=target,
            splits=random_splits,
            strategy="random_kfold",
            block_size_m=block_size,
            x=x,
            y=y,
            groups=groups,
        )

        spatial_result = evaluate_splits(
            features=features,
            target=target,
            splits=spatial_splits,
            strategy="spatial_block",
            block_size_m=block_size,
            x=x,
            y=y,
            groups=groups,
        )

        result_frames.extend(
            [
                random_result,
                spatial_result,
            ]
        )

    fold_results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    summary = summarize_results(
        fold_results
    )

    comparison = build_comparison(
        summary
    )

    fold_results.to_csv(
        RESULTS_DIR / "fold_results.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "strategy_summary.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR / "validation_comparison.csv",
        index=False,
    )

    metadata = {
        "seed": SEED,
        "n_samples": int(len(frame)),
        "n_regional": int(
            (frame["sampling_component"] == "regional")
            .sum()
        ),
        "n_campaign": int(
            (frame["sampling_component"] == "campaign")
            .sum()
        ),
        "crs": DEFAULT_CRS,
        "bounds": list(DEFAULT_BOUNDS),
        "target": "signal",
        "reference_target": "signal_true",
        "signal_true_min": float(
            target_true.min()
        ),
        "signal_true_max": float(
            target_true.max()
        ),
        "n_splits": N_SPLITS,
        "model": "KNeighborsRegressor",
        "n_neighbors": N_NEIGHBORS,
        "weights": "distance",
        "features": [
            "easting",
            "northing",
        ],
        "block_sizes_m": list(
            BLOCK_SIZES_M
        ),
        "interpretation_rule": (
            "Positive RMSE gap (spatial minus random) indicates "
            "that random CV reports a more optimistic error estimate."
        ),
    }

    (
        RESULTS_DIR
        / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline=chr(10),
    )

    print("LAB03_EXPERIMENT = PASS")
    print()
    print("VALIDATION COMPARISON")
    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run_experiment()
