import geopandas as gpd
import pytest
from shapely.geometry import Point

from geoai.spatial import (
    assign_regular_grid,
    generate_synthetic_sampling,
    nearest_neighbor_distances,
    require_metric_projected_crs,
    summarize_grid,
)


@pytest.fixture(scope="module")
def synthetic_samples() -> gpd.GeoDataFrame:
    return generate_synthetic_sampling()


def test_generate_synthetic_sampling_has_expected_structure(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    assert len(synthetic_samples) == 800
    assert synthetic_samples.crs.to_epsg() == 32719

    counts = synthetic_samples["sampling_component"].value_counts()

    assert counts["regional"] == 500
    assert counts["campaign"] == 300

    expected_columns = {
        "sample_id",
        "sampling_component",
        "signal_true",
        "signal",
        "geometry",
    }

    assert expected_columns.issubset(synthetic_samples.columns)


def test_generate_synthetic_sampling_is_reproducible() -> None:
    first = generate_synthetic_sampling(seed=12345)
    second = generate_synthetic_sampling(seed=12345)

    assert first.equals(second)


def test_require_metric_projected_crs_accepts_utm(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    crs = require_metric_projected_crs(synthetic_samples)

    assert crs.is_projected
    assert crs.to_epsg() == 32719


def test_require_metric_projected_crs_rejects_geographic() -> None:
    geographic = gpd.GeoDataFrame(
        {"sample_id": ["A", "B"]},
        geometry=[
            Point(-70.0, -30.0),
            Point(-70.1, -30.1),
        ],
        crs="EPSG:4326",
    )

    with pytest.raises(
        ValueError,
        match="projected CRS",
    ):
        require_metric_projected_crs(geographic)


def test_nearest_neighbor_distances_are_positive(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    distances = nearest_neighbor_distances(synthetic_samples)

    assert len(distances) == len(synthetic_samples)
    assert distances.notna().all()
    assert (distances > 0).all()


def test_campaign_is_more_dense_than_regional(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    regional = synthetic_samples[
        synthetic_samples["sampling_component"] == "regional"
    ]

    campaign = synthetic_samples[
        synthetic_samples["sampling_component"] == "campaign"
    ]

    regional_nn = nearest_neighbor_distances(regional)
    campaign_nn = nearest_neighbor_distances(campaign)

    assert campaign_nn.mean() < regional_nn.mean()


def test_regular_grid_preserves_all_observations(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    gridded = assign_regular_grid(
        synthetic_samples,
        cell_size_m=5000.0,
    )

    summary = summarize_grid(gridded)

    assert len(gridded) == len(synthetic_samples)
    assert gridded["grid_id"].notna().all()
    assert summary["observation_count"].sum() == len(
        synthetic_samples
    )


def test_densest_cell_differs_from_highest_signal_cell(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    gridded = assign_regular_grid(
        synthetic_samples,
        cell_size_m=5000.0,
    )

    summary = summarize_grid(gridded)

    densest = summary.loc[
        summary["observation_count"].idxmax()
    ]

    highest_signal = summary.loc[
        summary["mean_signal"].idxmax()
    ]

    assert densest["grid_id"] != highest_signal["grid_id"]


def test_assign_regular_grid_rejects_non_positive_cell_size(
    synthetic_samples: gpd.GeoDataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="cell_size_m must be positive",
    ):
        assign_regular_grid(
            synthetic_samples,
            cell_size_m=0.0,
        )
