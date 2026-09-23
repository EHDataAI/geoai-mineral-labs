"""Tests for cross-zone CRS behaviour."""

from math import hypot

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point


def _locations() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"site": ["SITE_A", "SITE_B"]},
        geometry=[
            Point(-72.2, -30.0),
            Point(-71.8, -30.0),
        ],
        crs="EPSG:4326",
    )


def test_common_crs_matches_geodesic_reference() -> None:
    locations = _locations()

    geod = Geod(ellps="WGS84")

    point_a = locations.geometry.iloc[0]
    point_b = locations.geometry.iloc[1]

    _, _, geodesic_distance_m = geod.inv(
        point_a.x,
        point_a.y,
        point_b.x,
        point_b.y,
    )

    locations_common = locations.to_crs(epsg=32719)

    projected_distance_m = (
        locations_common.geometry.iloc[0].distance(
            locations_common.geometry.iloc[1]
        )
    )

    relative_difference_pct = (
        abs(projected_distance_m - geodesic_distance_m)
        / geodesic_distance_m
        * 100
    )

    assert relative_difference_pct < 0.1


def test_mixed_utm_coordinates_produce_large_error() -> None:
    locations = _locations()

    geod = Geod(ellps="WGS84")

    point_a = locations.geometry.iloc[0]
    point_b = locations.geometry.iloc[1]

    _, _, geodesic_distance_m = geod.inv(
        point_a.x,
        point_a.y,
        point_b.x,
        point_b.y,
    )

    site_a_utm18 = locations.iloc[[0]].to_crs(epsg=32718)
    site_b_utm19 = locations.iloc[[1]].to_crs(epsg=32719)

    projected_a = site_a_utm18.geometry.iloc[0]
    projected_b = site_b_utm19.geometry.iloc[0]

    invalid_distance_m = hypot(
        projected_b.x - projected_a.x,
        projected_b.y - projected_a.y,
    )

    relative_difference_pct = (
        abs(invalid_distance_m - geodesic_distance_m)
        / geodesic_distance_m
        * 100
    )

    assert relative_difference_pct > 1000
