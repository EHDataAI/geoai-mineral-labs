from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geoai.spatial import (
    DEFAULT_BOUNDS,
    generate_synthetic_sampling,
)
from geoai.validation import (
    assign_spatial_blocks,
    spatial_group_kfold_indices,
)


LAB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LAB_DIR / "results"

SEED = 12345
N_SPLITS = 5
BLOCK_SIZES_M = (5000.0, 10000.0, 20000.0)


def aggregate_fold_metrics(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (block_size, strategy), group in fold_results.groupby(
        ["block_size_m", "strategy"],
        sort=True,
    ):
        n = group["test_n"].to_numpy(dtype=float)
        rmse = group["rmse"].to_numpy(dtype=float)
        mae = group["mae"].to_numpy(dtype=float)

        total_n = float(n.sum())

        pooled_rmse = float(
            np.sqrt(
                np.sum(n * rmse**2)
                / total_n
            )
        )

        pooled_mae = float(
            np.sum(n * mae)
            / total_n
        )

        rows.append(
            {
                "block_size_m": float(block_size),
                "strategy": strategy,
                "test_n_total": int(total_n),
                "pooled_rmse": pooled_rmse,
                "pooled_mae": pooled_mae,
                "fold_r2_mean": float(
                    group["r2"].mean()
                ),
                "fold_r2_sd": float(
                    group["r2"].std(ddof=1)
                ),
                "fold_rmse_sd": float(
                    group["rmse"].std(ddof=1)
                ),
                "median_separation_m": float(
                    group[
                        "median_test_to_train_distance_m"
                    ].median()
                ),
                "mean_group_overlap": float(
                    group["group_overlap"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_comparison(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for block_size in BLOCK_SIZES_M:
        subset = aggregate[
            aggregate["block_size_m"] == block_size
        ]

        random_row = subset[
            subset["strategy"] == "random_kfold"
        ].iloc[0]

        spatial_row = subset[
            subset["strategy"] == "spatial_block"
        ].iloc[0]

        random_rmse = float(
            random_row["pooled_rmse"]
        )
        spatial_rmse = float(
            spatial_row["pooled_rmse"]
        )

        rows.append(
            {
                "block_size_m": block_size,
                "random_pooled_rmse": random_rmse,
                "spatial_pooled_rmse": spatial_rmse,
                "rmse_gap_spatial_minus_random": (
                    spatial_rmse - random_rmse
                ),
                "relative_rmse_optimism_pct": (
                    100.0
                    * (spatial_rmse - random_rmse)
                    / spatial_rmse
                ),
                "random_pooled_mae": float(
                    random_row["pooled_mae"]
                ),
                "spatial_pooled_mae": float(
                    spatial_row["pooled_mae"]
                ),
                "random_median_separation_m": float(
                    random_row["median_separation_m"]
                ),
                "spatial_median_separation_m": float(
                    spatial_row["median_separation_m"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_scenario_diagnostics() -> pd.DataFrame:
    frame = generate_synthetic_sampling(
        seed=SEED
    )

    x = frame.geometry.x.to_numpy(dtype=float)
    y = frame.geometry.y.to_numpy(dtype=float)

    target_true = frame[
        "signal_true"
    ].to_numpy(dtype=float)

    rows = []

    for block_size in BLOCK_SIZES_M:
        groups = assign_spatial_blocks(
            x,
            y,
            block_size=block_size,
            origin_x=DEFAULT_BOUNDS[0],
            origin_y=DEFAULT_BOUNDS[2],
        )

        splits = spatial_group_kfold_indices(
            groups,
            n_splits=N_SPLITS,
        )

        block_counts = []
        test_sizes = []
        true_sds = []

        for split in splits:
            block_counts.append(
                int(
                    np.unique(
                        groups[split.test_idx]
                    ).size
                )
            )

            test_sizes.append(
                int(split.test_idx.size)
            )

            true_sds.append(
                float(
                    np.std(
                        target_true[split.test_idx],
                        ddof=1,
                    )
                )
            )

        rows.append(
            {
                "block_size_m": block_size,
                "populated_blocks": int(
                    np.unique(groups).size
                ),
                "min_test_blocks": min(block_counts),
                "max_test_blocks": max(block_counts),
                "median_test_blocks": float(
                    np.median(block_counts)
                ),
                "min_test_n": min(test_sizes),
                "max_test_n": max(test_sizes),
                "min_signal_true_sd": min(true_sds),
                "max_signal_true_sd": max(true_sds),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    fold_results = pd.read_csv(
        RESULTS_DIR / "fold_results.csv"
    )

    aggregate = aggregate_fold_metrics(
        fold_results
    )

    comparison = build_comparison(
        aggregate
    )

    diagnostics = build_scenario_diagnostics()

    aggregate.to_csv(
        RESULTS_DIR / "aggregate_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR / "aggregate_comparison.csv",
        index=False,
    )

    diagnostics.to_csv(
        RESULTS_DIR / "scenario_diagnostics.csv",
        index=False,
    )

    print("AGGREGATE METRICS")
    print(aggregate.to_string(index=False))

    print()
    print("AGGREGATE COMPARISON")
    print(comparison.to_string(index=False))

    print()
    print("SCENARIO DIAGNOSTICS")
    print(diagnostics.to_string(index=False))

    print()
    print("LAB03_AGGREGATION = PASS")


if __name__ == "__main__":
    main()
