"""Coordinate reference system utilities for GeoAI workflows."""

from __future__ import annotations

from math import floor


def utm_zone_from_longitude(longitude: float) -> int:
    """Return the standard UTM zone number for a longitude.

    Parameters
    ----------
    longitude:
        Longitude in decimal degrees within [-180, 180].

    Returns
    -------
    int
        UTM zone in the range 1..60.
    """

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180 degrees."
        )

    if longitude == 180:
        return 60

    return floor((longitude + 180) / 6) + 1


def wgs84_utm_epsg(
    longitude: float,
    latitude: float,
) -> int:
    """Return a WGS84 UTM EPSG code for a coordinate.

    Parameters
    ----------
    longitude:
        Longitude in decimal degrees within [-180, 180].

    latitude:
        Latitude in decimal degrees within the practical
        UTM coverage of [-80, 84].

    Returns
    -------
    int
        EPSG code in the WGS84 UTM north or south family.
    """

    if not -80 <= latitude <= 84:
        raise ValueError(
            "UTM is defined here only for latitudes "
            "between -80 and 84 degrees."
        )

    zone = utm_zone_from_longitude(longitude)

    if latitude >= 0:
        return 32600 + zone

    return 32700 + zone
