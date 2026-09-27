from __future__ import annotations

from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS


DEFAULT_CRS: Final[str] = "EPSG:32719"

DEFAULT_BOUNDS: Final[tuple[float, float, float, float]] = (
    350000.0,
    430000.0,
    7100000.0,
    7180000.0,
)

DEFAULT_CAMPAIGN_CENTER: Final[tuple[float, float]] = (
    376500.0,
    7138500.0,
)

DEFAULT_SIGNAL_CENTER: Final[tuple[float, float]] = (
    411500.0,
    7162500.0,
)


def require_metric_projected_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Validate that a GeoDataFrame uses projected metre-based coordinates."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must define a CRS.")

    crs = CRS.from_user_input(gdf.crs)

    if not crs.is_projected:
        raise ValueError("Metric spatial analysis requires a projected CRS.")

    axis_info = crs.axis_info

    if len(axis_info) < 2:
        raise ValueError("Projected CRS must expose two coordinate axes.")

    for axis in axis_info[:2]:
        factor = axis.unit_conversion_factor

        if factor is None or not np.isclose(float(factor), 1.0):
            raise ValueError(
                "Projected CRS axes must use metre-based units."
            )

    return crs


def _validate_point_geometries(gdf: gpd.GeoDataFrame) -> None:
    if gdf.empty:
        raise ValueError(
            "GeoDataFrame must contain at least one observation."
        )

    if gdf.geometry.isna().any():
        raise ValueError("Geometry column contains missing values.")

    if gdf.geometry.is_empty.any():
        raise ValueError("Geometry column contains empty geometries.")

    if not gdf.geometry.geom_type.eq("Point").all():
        raise ValueError("This operation requires Point geometries.")


def generate_synthetic_sampling(
    *,
    seed: int = 12345,
    n_regional: int = 500,
    n_campaign: int = 300,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
    campaign_center: tuple[float, float] = DEFAULT_CAMPAIGN_CENTER,
    campaign_sigma_m: float = 3500.0,
    signal_center: tuple[float, float] = DEFAULT_SIGNAL_CENTER,
    signal_sigma_m: float = 7000.0,
    signal_baseline: float = 10.0,
    signal_amplitude: float = 80.0,
    noise_sd: float = 3.0,
    crs: str = DEFAULT_CRS,
) -> gpd.GeoDataFrame:
    """Generate a reproducible synthetic mineral exploration dataset."""
    if n_regional <= 0 or n_campaign <= 0:
        raise ValueError("Sample counts must be positive.")

    if campaign_sigma_m <= 0 or signal_sigma_m <= 0:
        raise ValueError("Spatial scales must be positive.")

    if noise_sd < 0:
        raise ValueError("noise_sd must be non-negative.")

    xmin, xmax, ymin, ymax = bounds

    if not (xmin < xmax and ymin < ymax):
        raise ValueError(
            "Bounds must satisfy xmin < xmax and ymin < ymax."
        )

    rng = np.random.default_rng(seed)

    regional_x = rng.uniform(
        xmin,
        xmax,
        n_regional,
    )
    regional_y = rng.uniform(
        ymin,
        ymax,
        n_regional,
    )

    campaign_x = np.clip(
        rng.normal(
            campaign_center[0],
            campaign_sigma_m,
            n_campaign,
        ),
        xmin,
        xmax,
    )
    campaign_y = np.clip(
        rng.normal(
            campaign_center[1],
            campaign_sigma_m,
            n_campaign,
        ),
        ymin,
        ymax,
    )

    x = np.concatenate(
        (
            regional_x,
            campaign_x,
        )
    )
    y = np.concatenate(
        (
            regional_y,
            campaign_y,
        )
    )

    component = np.concatenate(
        (
            np.repeat("regional", n_regional),
            np.repeat("campaign", n_campaign),
        )
    )

    scaled_x = (
        x - signal_center[0]
    ) / signal_sigma_m

    scaled_y = (
        y - signal_center[1]
    ) / signal_sigma_m

    signal_true = (
        signal_baseline
        + signal_amplitude
        * np.exp(
            -0.5
            * (
                scaled_x**2
                + scaled_y**2
            )
        )
    )

    signal = (
        signal_true
        + rng.normal(
            0.0,
            noise_sd,
            x.size,
        )
    )

    data = pd.DataFrame(
        {
            "sample_id": [
                f"S{i:04d}"
                for i in range(
                    1,
                    x.size + 1,
                )
            ],
            "sampling_component": component,
            "signal_true": signal_true,
            "signal": signal,
        }
    )

    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            x,
            y,
        ),
        crs=crs,
    )

    require_metric_projected_crs(gdf)

    return gdf


def nearest_neighbor_distances(
    gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """Calculate Euclidean nearest-neighbour distances in metres."""
    require_metric_projected_crs(gdf)
    _validate_point_geometries(gdf)

    if len(gdf) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    coordinates = np.column_stack(
        (
            gdf.geometry.x.to_numpy(
                dtype=float
            ),
            gdf.geometry.y.to_numpy(
                dtype=float
            ),
        )
    )

    deltas = (
        coordinates[:, np.newaxis, :]
        - coordinates[np.newaxis, :, :]
    )

    squared_distances = np.einsum(
        "ijk,ijk->ij",
        deltas,
        deltas,
    )

    np.fill_diagonal(
        squared_distances,
        np.inf,
    )

    distances = np.sqrt(
        np.min(
            squared_distances,
            axis=1,
        )
    )

    return pd.Series(
        distances,
        index=gdf.index,
        name="nearest_neighbor_m",
        dtype=float,
    )


def assign_regular_grid(
    gdf: gpd.GeoDataFrame,
    *,
    cell_size_m: float = 5000.0,
    origin: tuple[float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Assign point observations to a regular metric grid."""
    require_metric_projected_crs(gdf)
    _validate_point_geometries(gdf)

    if cell_size_m <= 0:
        raise ValueError(
            "cell_size_m must be positive."
        )

    if origin is None:
        minx, miny, _, _ = gdf.total_bounds

        origin_x = (
            np.floor(
                minx / cell_size_m
            )
            * cell_size_m
        )

        origin_y = (
            np.floor(
                miny / cell_size_m
            )
            * cell_size_m
        )
    else:
        origin_x, origin_y = origin

    result = gdf.copy()

    result["grid_col"] = np.floor(
        (
            result.geometry.x.to_numpy(
                dtype=float
            )
            - origin_x
        )
        / cell_size_m
    ).astype(int)

    result["grid_row"] = np.floor(
        (
            result.geometry.y.to_numpy(
                dtype=float
            )
            - origin_y
        )
        / cell_size_m
    ).astype(int)

    result["grid_id"] = (
        "c"
        + result["grid_col"].astype(str)
        + "_r"
        + result["grid_row"].astype(str)
    )

    result["grid_center_x"] = (
        origin_x
        + (
            result["grid_col"]
            + 0.5
        )
        * cell_size_m
    )

    result["grid_center_y"] = (
        origin_y
        + (
            result["grid_row"]
            + 0.5
        )
        * cell_size_m
    )

    return result


def summarize_grid(
    gridded: gpd.GeoDataFrame,
    *,
    signal_column: str = "signal",
) -> pd.DataFrame:
    """Summarize sampling intensity and mean signal by grid cell."""
    required = {
        "grid_id",
        "grid_col",
        "grid_row",
        "grid_center_x",
        "grid_center_y",
        signal_column,
    }

    missing = required.difference(
        gridded.columns
    )

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            f"Missing required grid columns: {missing_text}"
        )

    summary = (
        gridded.groupby(
            [
                "grid_id",
                "grid_col",
                "grid_row",
                "grid_center_x",
                "grid_center_y",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            observation_count=(
                signal_column,
                "size",
            ),
            mean_signal=(
                signal_column,
                "mean",
            ),
        )
        .sort_values(
            [
                "grid_row",
                "grid_col",
            ],
            ignore_index=True,
        )
    )

    return summary
