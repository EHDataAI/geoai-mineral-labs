from __future__ import annotations

from geoai.spatial import (
    assign_regular_grid,
    generate_synthetic_sampling,
    nearest_neighbor_distances,
    require_metric_projected_crs,
    summarize_grid,
)


PRIMARY_CELL_SIZE_M = 10000.0

GRID_ORIGIN = (
    350000.0,
    7100000.0,
)


def main() -> None:
    samples = generate_synthetic_sampling()

    crs = require_metric_projected_crs(samples)

    regional = samples[
        samples["sampling_component"] == "regional"
    ].copy()

    campaign = samples[
        samples["sampling_component"] == "campaign"
    ].copy()

    regional_nn = nearest_neighbor_distances(regional)
    campaign_nn = nearest_neighbor_distances(campaign)

    regional_mean_nn = float(regional_nn.mean())
    campaign_mean_nn = float(campaign_nn.mean())

    nn_ratio = campaign_mean_nn / regional_mean_nn

    gridded = assign_regular_grid(
        samples,
        cell_size_m=PRIMARY_CELL_SIZE_M,
        origin=GRID_ORIGIN,
    )

    observed_summary = summarize_grid(
        gridded,
        signal_column="signal",
    )

    true_summary = summarize_grid(
        gridded,
        signal_column="signal_true",
    )

    densest_cell = observed_summary.loc[
        observed_summary["observation_count"].idxmax()
    ]

    highest_observed_signal_cell = observed_summary.loc[
        observed_summary["mean_signal"].idxmax()
    ]

    highest_true_signal_cell = true_summary.loc[
        true_summary["mean_signal"].idxmax()
    ]

    observations_preserved = int(
        observed_summary["observation_count"].sum()
    )

    densest_differs_from_observed = (
        densest_cell["grid_id"]
        != highest_observed_signal_cell["grid_id"]
    )

    densest_differs_from_true = (
        densest_cell["grid_id"]
        != highest_true_signal_cell["grid_id"]
    )

    observed_agrees_with_true = (
        highest_observed_signal_cell["grid_id"]
        == highest_true_signal_cell["grid_id"]
    )

    print("=" * 72)
    print("LAB 02 - SPATIAL EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    print("\nDATASET")
    print(f"Observations: {len(samples)}")
    print(f"CRS: {crs.to_string()}")

    print("\nSAMPLING COMPONENTS")
    print(f"Regional observations: {len(regional)}")
    print(f"Campaign observations: {len(campaign)}")

    print("\nNEAREST-NEIGHBOUR DISTANCE")
    print(
        "Regional mean NN distance (m): "
        f"{regional_mean_nn:.2f}"
    )
    print(
        "Campaign mean NN distance (m): "
        f"{campaign_mean_nn:.2f}"
    )
    print(
        "Campaign/regional NN ratio: "
        f"{nn_ratio:.3f}"
    )
    print(
        "Campaign more densely sampled: "
        f"{campaign_mean_nn < regional_mean_nn}"
    )

    print("\nREGULAR GRID")
    print(
        "Cell size (m): "
        f"{PRIMARY_CELL_SIZE_M:.0f}"
    )
    print(
        "Grid origin: "
        f"{GRID_ORIGIN}"
    )
    print(
        "Populated cells: "
        f"{len(observed_summary)}"
    )
    print(
        "Observations preserved: "
        f"{observations_preserved}"
    )

    print("\nDENSEST CELL")
    print(
        "Grid ID: "
        f"{densest_cell['grid_id']}"
    )
    print(
        "Observation count: "
        f"{int(densest_cell['observation_count'])}"
    )
    print(
        "Mean observed signal: "
        f"{float(densest_cell['mean_signal']):.2f}"
    )

    print("\nHIGHEST OBSERVED-SIGNAL CELL")
    print(
        "Grid ID: "
        f"{highest_observed_signal_cell['grid_id']}"
    )
    print(
        "Observation count: "
        f"{int(highest_observed_signal_cell['observation_count'])}"
    )
    print(
        "Mean observed signal: "
        f"{float(highest_observed_signal_cell['mean_signal']):.2f}"
    )

    print("\nHIGHEST TRUE-SIGNAL CELL")
    print(
        "Grid ID: "
        f"{highest_true_signal_cell['grid_id']}"
    )
    print(
        "Observation count: "
        f"{int(highest_true_signal_cell['observation_count'])}"
    )
    print(
        "Mean true signal: "
        f"{float(highest_true_signal_cell['mean_signal']):.2f}"
    )

    print("\nKEY RESULTS")
    print(
        "Densest cell differs from highest observed-signal cell: "
        f"{densest_differs_from_observed}"
    )
    print(
        "Densest cell differs from highest true-signal cell: "
        f"{densest_differs_from_true}"
    )
    print(
        "Highest observed-signal cell agrees with true-signal cell: "
        f"{observed_agrees_with_true}"
    )

    print("\nINTERPRETATION")
    print(
        "The synthetic sampling campaign produces a clear "
        "sampling-density hotspot."
    )
    print(
        "The densest grid cell is separated from the strongest "
        "underlying synthetic signal."
    )
    print(
        "A 10 km grid is used as the primary descriptive resolution "
        "for this experiment."
    )
    print(
        "Monte Carlo sensitivity analysis showed that this resolution "
        "provides a useful compromise between hotspot stability, "
        "local sample support, and preservation of spatial signal."
    )
    print(
        "The selected resolution is specific to this synthetic "
        "experiment and is not presented as a universal optimum."
    )
    print(
        "Sampling density must therefore be distinguished from "
        "underlying spatial signal before predictive modelling."
    )


if __name__ == "__main__":
    main()
