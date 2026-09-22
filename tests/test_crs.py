"""Tests for coordinate reference system utilities."""

import pytest

from geoai.crs import (
    utm_zone_from_longitude,
    wgs84_utm_epsg,
)


def test_utm_zone_northern_chile() -> None:
    assert utm_zone_from_longitude(-70.4) == 19


def test_utm_zone_southern_chile_zone_18() -> None:
    assert utm_zone_from_longitude(-73.0) == 18


def test_wgs84_utm_southern_hemisphere() -> None:
    assert wgs84_utm_epsg(-70.4, -23.6) == 32719


def test_wgs84_utm_northern_hemisphere() -> None:
    assert wgs84_utm_epsg(-70.4, 23.6) == 32619


def test_longitude_180_maps_to_zone_60() -> None:
    assert utm_zone_from_longitude(180.0) == 60


@pytest.mark.parametrize(
    "longitude",
    [-180.1, 180.1],
)
def test_invalid_longitude_raises(longitude: float) -> None:
    with pytest.raises(ValueError):
        utm_zone_from_longitude(longitude)


@pytest.mark.parametrize(
    "latitude",
    [-80.1, 84.1],
)
def test_latitude_outside_utm_domain_raises(
    latitude: float,
) -> None:
    with pytest.raises(ValueError):
        wgs84_utm_epsg(-70.0, latitude)


def test_longitude_minus_180_maps_to_zone_1() -> None:
    assert utm_zone_from_longitude(-180.0) == 1


@pytest.mark.parametrize(
    ("longitude", "expected_zone"),
    [
        (-72.0001, 18),
        (-72.0000, 19),
        (-71.9999, 19),
    ],
)
def test_utm_zone_boundary_18_19(
    longitude: float,
    expected_zone: int,
) -> None:
    assert utm_zone_from_longitude(longitude) == expected_zone
