"""
LAB 01A — Automatic UTM zone selection

Objective
---------
Derive an appropriate WGS84 UTM EPSG code from longitude
and latitude instead of manually hard-coding the CRS.

The implementation is intended for ordinary Chilean coordinates.
"""

from __future__ import annotations

from math import floor

from pyproj import CRS


def utm_zone_from_longitude(longitude: float) -> int:
    """Return the standard UTM zone for a longitude."""

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180 degrees."
        )

    # Longitude 180 requires a boundary correction because the
    # standard formula would otherwise return zone 61.
    if longitude == 180:
        return 60

    return floor((longitude + 180) / 6) + 1


def utm_epsg_from_lon_lat(
    longitude: float,
    latitude: float,
) -> int:
    """
    Return a WGS84 UTM EPSG code from longitude and latitude.

    Northern hemisphere:
        EPSG 32601 ... 32660

    Southern hemisphere:
        EPSG 32701 ... 32760
    """

    if not -90 <= latitude <= 90:
        raise ValueError(
            "Latitude must be between -90 and 90 degrees."
        )

    zone = utm_zone_from_longitude(longitude)

    if latitude >= 0:
        return 32600 + zone

    return 32700 + zone


# ---------------------------------------------------------------------
# Synthetic Chile examples
# ---------------------------------------------------------------------

sites = [
    {
        "name": "NORTHERN_CHILE_SITE",
        "longitude": -70.4000,
        "latitude": -23.6000,
    },
    {
        "name": "ZONE_18_SITE",
        "longitude": -73.0000,
        "latitude": -40.0000,
    },
    {
        "name": "ZONE_19_SITE",
        "longitude": -71.0000,
        "latitude": -33.0000,
    },
]


print("=== AUTOMATIC UTM SELECTION ===")

for site in sites:
    lon = site["longitude"]
    lat = site["latitude"]

    zone = utm_zone_from_longitude(lon)
    epsg = utm_epsg_from_lon_lat(lon, lat)
    crs = CRS.from_epsg(epsg)

    print()
    print("Site      :", site["name"])
    print("Longitude :", lon)
    print("Latitude  :", lat)
    print("UTM zone  :", zone)
    print("EPSG      :", epsg)
    print("CRS       :", crs.name)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

assert utm_zone_from_longitude(-70.4) == 19
assert utm_epsg_from_lon_lat(-70.4, -23.6) == 32719

assert utm_zone_from_longitude(-73.0) == 18
assert utm_epsg_from_lon_lat(-73.0, -40.0) == 32718

assert utm_zone_from_longitude(-71.0) == 19
assert utm_epsg_from_lon_lat(-71.0, -33.0) == 32719

# Northern hemisphere sanity check.
assert utm_epsg_from_lon_lat(-70.4, 23.6) == 32619


print()
print("================================")
print("LAB 01A UTM SELECTION: PASS")
print("================================")
