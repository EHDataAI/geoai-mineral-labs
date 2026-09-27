from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from geoai.spatial import (
    assign_regular_grid,
    generate_synthetic_sampling,
    nearest_neighbor_distances,
    summarize_grid,
)


GRID_SIZES_M = (
    5000.0,
    10000.0,
    20000.0,
)

BASE_ORIGIN = (
    350000.0,
    7100000.0,
)

OFFSET_FRACTIONS = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
)

FIRST_SEED = 12345
N_REALIZATIONS = 400

THEORETICAL_SIGNAL_MAX = 90.0

WILSON_Z_95 = 1.959963984540054

PRIMARY_CELL_SIZE_M = 10000.0

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

RESOLUTION_SUMMARY_PATH = (
    RESULTS_DIR
    / "resolution_summary.csv"
)

PER_REALIZATION_PATH = (
    RESULTS_DIR
    / "monte_carlo_by_realization.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "experiment_metadata.json"
)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")

    if successes < 0 or successes > trials:
        raise ValueError(
            "successes must be between 0 and trials"
        )

    proportion = successes / trials
    z_squared = z**2

    denominator = 1.0 + z_squared / trials

    centre = (
        proportion
        + z_squared / (2.0 * trials)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                proportion
                * (1.0 - proportion)
                / trials
            )
            + (
                z_squared
                / (4.0 * trials**2)
            )
        )
        / denominator
    )

    return (
        centre - margin,
        centre + margin,
    )


def run_experiment() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    grid_records: list[dict[str, object]] = []
    nn_records: list[dict[str, float | int]] = []

    seeds = range(
        FIRST_SEED,
        FIRST_SEED + N_REALIZATIONS,
    )

    for seed in seeds:
        samples = generate_synthetic_sampling(
            seed=seed,
        )

        regional = samples[
            samples["sampling_component"]
            == "regional"
        ]

        campaign = samples[
            samples["sampling_component"]
            == "campaign"
        ]

        regional_mean_nn = float(
            nearest_neighbor_distances(
                regional
            ).mean()
        )

        campaign_mean_nn = float(
            nearest_neighbor_distances(
                campaign
            ).mean()
        )

        nn_records.append(
            {
                "seed": seed,
                "regional_mean_nn_m": (
                    regional_mean_nn
                ),
                "campaign_mean_nn_m": (
                    campaign_mean_nn
                ),
                "nn_ratio": (
                    campaign_mean_nn
                    / regional_mean_nn
                ),
            }
        )

        for cell_size_m in GRID_SIZES_M:
            for (
                offset_x,
                offset_y,
            ) in OFFSET_FRACTIONS:
                origin = (
                    BASE_ORIGIN[0]
                    + offset_x * cell_size_m,
                    BASE_ORIGIN[1]
                    + offset_y * cell_size_m,
                )

                gridded = assign_regular_grid(
                    samples,
                    cell_size_m=cell_size_m,
                    origin=origin,
                )

                observed = summarize_grid(
                    gridded,
                    signal_column="signal",
                )

                true_signal = summarize_grid(
                    gridded,
                    signal_column="signal_true",
                )

                observed_hotspot = observed.loc[
                    observed[
                        "mean_signal"
                    ].idxmax()
                ]

                true_hotspot = true_signal.loc[
                    true_signal[
                        "mean_signal"
                    ].idxmax()
                ]

                maximum_cell_count = int(
                    observed[
                        "observation_count"
                    ].max()
                )

                true_hotspot_count = int(
                    true_hotspot[
                        "observation_count"
                    ]
                )

                distance_m = math.hypot(
                    float(
                        observed_hotspot[
                            "grid_center_x"
                        ]
                        - true_hotspot[
                            "grid_center_x"
                        ]
                    ),
                    float(
                        observed_hotspot[
                            "grid_center_y"
                        ]
                        - true_hotspot[
                            "grid_center_y"
                        ]
                    ),
                )

                grid_records.append(
                    {
                        "seed": seed,
                        "cell_size_m": (
                            cell_size_m
                        ),
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "observed_equals_true": (
                            observed_hotspot[
                                "grid_id"
                            ]
                            == true_hotspot[
                                "grid_id"
                            ]
                        ),
                        "true_hotspot_not_densest": (
                            true_hotspot_count
                            < maximum_cell_count
                        ),
                        "normalized_distance": (
                            distance_m
                            / cell_size_m
                        ),
                        "true_hotspot_n": (
                            true_hotspot_count
                        ),
                        "true_hotspot_mean": float(
                            true_hotspot[
                                "mean_signal"
                            ]
                        ),
                    }
                )

    return (
        pd.DataFrame(grid_records),
        pd.DataFrame(nn_records),
    )


def summarize_by_realization(
    results: pd.DataFrame,
) -> pd.DataFrame:
    return (
        results
        .groupby(
            [
                "seed",
                "cell_size_m",
            ],
            as_index=False,
        )
        .agg(
            offset_match_rate=(
                "observed_equals_true",
                "mean",
            ),
            all_offsets_match=(
                "observed_equals_true",
                "all",
            ),
            true_hotspot_not_densest_all_offsets=(
                "true_hotspot_not_densest",
                "all",
            ),
            max_normalized_distance=(
                "normalized_distance",
                "max",
            ),
            median_true_hotspot_n=(
                "true_hotspot_n",
                "median",
            ),
            median_true_hotspot_mean=(
                "true_hotspot_mean",
                "median",
            ),
        )
    )


def build_resolution_summary(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []

    for cell_size_m in GRID_SIZES_M:
        group = per_seed[
            per_seed["cell_size_m"]
            == cell_size_m
        ]

        n = len(group)

        if n != N_REALIZATIONS:
            raise ValueError(
                "Unexpected number of independent "
                "realizations for cell size "
                f"{cell_size_m}: {n}"
            )

        all_match_count = int(
            group["all_offsets_match"].sum()
        )

        all_match_ci = wilson_interval(
            all_match_count,
            n,
        )

        density_separated_count = int(
            group[
                "true_hotspot_not_densest_all_offsets"
            ].sum()
        )

        density_separated_ci = wilson_interval(
            density_separated_count,
            n,
        )

        median_signal = float(
            group[
                "median_true_hotspot_mean"
            ].median()
        )

        records.append(
            {
                "cell_size_m": cell_size_m,
                "grid_km": cell_size_m / 1000.0,
                "n_realizations": n,
                "mean_offset_match_rate": float(
                    group[
                        "offset_match_rate"
                    ].mean()
                ),
                "all_offsets_match_rate": (
                    all_match_count / n
                ),
                "all_offsets_match_ci_low": (
                    all_match_ci[0]
                ),
                "all_offsets_match_ci_high": (
                    all_match_ci[1]
                ),
                "density_separation_rate": (
                    density_separated_count / n
                ),
                "density_separation_ci_low": (
                    density_separated_ci[0]
                ),
                "density_separation_ci_high": (
                    density_separated_ci[1]
                ),
                "median_worst_normalized_distance": (
                    float(
                        group[
                            "max_normalized_distance"
                        ].median()
                    )
                ),
                "p95_worst_normalized_distance": (
                    float(
                        group[
                            "max_normalized_distance"
                        ].quantile(0.95)
                    )
                ),
                "median_true_hotspot_n": float(
                    group[
                        "median_true_hotspot_n"
                    ].median()
                ),
                "median_true_hotspot_signal": (
                    median_signal
                ),
                "peak_retention_rate": (
                    median_signal
                    / THEORETICAL_SIGNAL_MAX
                ),
            }
        )

    return pd.DataFrame(records)


def build_experiment_metadata(
    nn_results: pd.DataFrame,
) -> dict[str, object]:
    reference_samples = (
        generate_synthetic_sampling(
            seed=FIRST_SEED,
        )
    )

    sampling_counts = (
        reference_samples[
            "sampling_component"
        ]
        .value_counts()
        .sort_index()
    )

    campaign_denser = (
        nn_results["nn_ratio"] < 1.0
    )

    return {
        "schema_version": 1,
        "lab": "LAB02 Spatial EDA",
        "synthetic_data": True,
        "crs": str(reference_samples.crs),
        "observations_per_realization": int(
            len(reference_samples)
        ),
        "sampling_components": {
            str(component): int(count)
            for component, count
            in sampling_counts.items()
        },
        "monte_carlo": {
            "first_seed": FIRST_SEED,
            "last_seed": (
                FIRST_SEED
                + N_REALIZATIONS
                - 1
            ),
            "independent_realizations": (
                N_REALIZATIONS
            ),
            "grid_resolutions_m": [
                int(value)
                for value in GRID_SIZES_M
            ],
            "offset_fractions": [
                {
                    "x": offset_x,
                    "y": offset_y,
                }
                for offset_x, offset_y
                in OFFSET_FRACTIONS
            ],
            "offsets_per_resolution": len(
                OFFSET_FRACTIONS
            ),
            "total_grid_evaluations": (
                N_REALIZATIONS
                * len(GRID_SIZES_M)
                * len(OFFSET_FRACTIONS)
            ),
        },
        "primary_descriptive_resolution_m": int(
            PRIMARY_CELL_SIZE_M
        ),
        "theoretical_signal_max": (
            THEORETICAL_SIGNAL_MAX
        ),
        "nearest_neighbour_robustness": {
            "campaign_denser_rate": float(
                campaign_denser.mean()
            ),
            "median_campaign_regional_ratio": (
                float(
                    nn_results[
                        "nn_ratio"
                    ].median()
                )
            ),
            "p95_campaign_regional_ratio": (
                float(
                    nn_results[
                        "nn_ratio"
                    ].quantile(0.95)
                )
            ),
        },
    }


def write_results(
    per_seed: pd.DataFrame,
    resolution_summary: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        resolution_summary
        .sort_values("cell_size_m")
        .to_csv(
            RESOLUTION_SUMMARY_PATH,
            index=False,
            encoding="utf-8",
            float_format="%.6f",
            lineterminator="\n",
        )
    )

    (
        per_seed
        .sort_values(
            [
                "seed",
                "cell_size_m",
            ]
        )
        .to_csv(
            PER_REALIZATION_PATH,
            index=False,
            encoding="utf-8",
            float_format="%.6f",
            lineterminator="\n",
        )
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def print_results(
    results: pd.DataFrame,
    nn_results: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("LAB 02 - MONTE CARLO RESOLUTION SENSITIVITY")
    print("=" * 72)

    print("\nEXPERIMENTAL DESIGN")
    print(
        "Independent spatial realizations: "
        f"{N_REALIZATIONS}"
    )
    print(
        "Grid resolutions: "
        f"{len(GRID_SIZES_M)}"
    )
    print(
        "Offsets per resolution: "
        f"{len(OFFSET_FRACTIONS)}"
    )
    print(
        "Total grid evaluations: "
        f"{len(results)}"
    )

    print("\nNEAREST-NEIGHBOUR ROBUSTNESS")

    campaign_denser = (
        nn_results["nn_ratio"] < 1.0
    )

    print(
        "Campaign denser in realizations: "
        f"{campaign_denser.mean():.1%}"
    )

    print(
        "Median campaign/regional NN ratio: "
        f"{nn_results['nn_ratio'].median():.3f}"
    )

    print(
        "95th percentile NN ratio: "
        f"{nn_results['nn_ratio'].quantile(0.95):.3f}"
    )

    for cell_size_m in GRID_SIZES_M:
        group = per_seed[
            per_seed["cell_size_m"]
            == cell_size_m
        ]

        n = len(group)

        all_match_count = int(
            group["all_offsets_match"].sum()
        )

        all_match_ci = wilson_interval(
            all_match_count,
            n,
        )

        density_separated_count = int(
            group[
                "true_hotspot_not_densest_all_offsets"
            ].sum()
        )

        density_separated_ci = wilson_interval(
            density_separated_count,
            n,
        )

        median_signal = float(
            group[
                "median_true_hotspot_mean"
            ].median()
        )

        print("\n" + "=" * 72)
        print(
            "GRID: "
            f"{cell_size_m / 1000:.0f} km"
        )
        print("=" * 72)

        print(
            "Independent realizations: "
            f"{n}"
        )

        print(
            "Mean offset match rate: "
            f"{group['offset_match_rate'].mean():.1%}"
        )

        print(
            "Seeds with all 4 offsets matching: "
            f"{all_match_count / n:.1%}"
        )

        print(
            "Wilson 95% CI for all-offset match: "
            f"{all_match_ci[0]:.1%} to "
            f"{all_match_ci[1]:.1%}"
        )

        print(
            "True hotspot separated from densest "
            "sampling in all offsets: "
            f"{density_separated_count / n:.1%}"
        )

        print(
            "Wilson 95% CI for density separation: "
            f"{density_separated_ci[0]:.1%} to "
            f"{density_separated_ci[1]:.1%}"
        )

        print(
            "Median worst-case normalized distance: "
            f"{group['max_normalized_distance'].median():.3f}"
        )

        print(
            "95th percentile worst-case "
            "normalized distance: "
            f"{group['max_normalized_distance'].quantile(0.95):.3f}"
        )

        print(
            "Median n at true hotspot: "
            f"{group['median_true_hotspot_n'].median():.1f}"
        )

        print(
            "Median true hotspot signal: "
            f"{median_signal:.2f}"
        )

        print(
            "Peak retention vs theoretical maximum: "
            f"{median_signal / THEORETICAL_SIGNAL_MAX:.1%}"
        )


def main() -> None:
    results, nn_results = run_experiment()

    per_seed = summarize_by_realization(
        results
    )

    resolution_summary = (
        build_resolution_summary(
            per_seed
        )
    )

    metadata = build_experiment_metadata(
        nn_results
    )

    write_results(
        per_seed,
        resolution_summary,
        metadata,
    )

    print_results(
        results,
        nn_results,
        per_seed,
    )


if __name__ == "__main__":
    main()
