"""
LAB 01A — Coordinate Foundations

Objective
---------
Demonstrate the difference between:

1. Geographic coordinates in EPSG:4326
2. Planar distance calculated directly in degrees
3. Projected distance using UTM
4. Geodesic distance over the reference ellipsoid

The coordinates are synthetic and represent two generic locations
in northern Chile.
"""

from __future__ import annotations

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point


# ---------------------------------------------------------------------
# 1. Synthetic locations in northern Chile
# ---------------------------------------------------------------------

locations = gpd.GeoDataFrame(
    {
        "site": ["SITE_A", "SITE_B"],
    },
    geometry=[
        Point(-70.4000, -23.6000),
        Point(-70.3900, -23.5900),
    ],
    crs="EPSG:4326",
)


print("=== ORIGINAL DATA ===")
print(locations)
print()
print("CRS:", locations.crs)


# ---------------------------------------------------------------------
# 2. INCORRECT interpretation:
#    planar distance directly in geographic coordinates
# ---------------------------------------------------------------------

degree_distance = locations.geometry.iloc[0].distance(
    locations.geometry.iloc[1]
)

print()
print("=== DISTANCE IN EPSG:4326 ===")
print(f"Raw Shapely distance: {degree_distance:.8f}")
print("Unit: degrees")
print(
    "Warning: this value must NOT be interpreted as metres "
    "or kilometres."
)


# ---------------------------------------------------------------------
# 3. Project to UTM zone 19 South
#
# EPSG:32719
# WGS 84 / UTM zone 19S
#
# Northern Chile near longitude -70° lies within UTM zone 19.
# ---------------------------------------------------------------------

utm = locations.to_crs("EPSG:32719")

utm_distance_m = utm.geometry.iloc[0].distance(
    utm.geometry.iloc[1]
)

print()
print("=== PROJECTED COORDINATES ===")
print(utm)
print()
print("Projected CRS:", utm.crs)
print(f"UTM distance: {utm_distance_m:.2f} m")
print(f"UTM distance: {utm_distance_m / 1000:.4f} km")


# ---------------------------------------------------------------------
# 4. Geodesic reference distance
#
# This calculates the shortest path on the WGS84 ellipsoid rather
# than assuming a flat projected plane.
# ---------------------------------------------------------------------

geod = Geod(ellps="WGS84")

lon1 = locations.geometry.iloc[0].x
lat1 = locations.geometry.iloc[0].y
lon2 = locations.geometry.iloc[1].x
lat2 = locations.geometry.iloc[1].y

azimuth_forward, azimuth_back, geodesic_distance_m = geod.inv(
    lon1,
    lat1,
    lon2,
    lat2,
)

print()
print("=== GEODESIC REFERENCE ===")
print(f"Geodesic distance: {geodesic_distance_m:.2f} m")
print(f"Forward azimuth: {azimuth_forward:.2f} degrees")


# ---------------------------------------------------------------------
# 5. Compare UTM approximation against geodesic distance
# ---------------------------------------------------------------------

absolute_difference_m = abs(
    utm_distance_m - geodesic_distance_m
)

relative_difference_pct = (
    absolute_difference_m / geodesic_distance_m
) * 100

print()
print("=== UTM vs GEODESIC ===")
print(f"Absolute difference: {absolute_difference_m:.4f} m")
print(f"Relative difference: {relative_difference_pct:.6f} %")


# ---------------------------------------------------------------------
# 6. Basic validation
# ---------------------------------------------------------------------

assert locations.crs.to_epsg() == 4326
assert utm.crs.to_epsg() == 32719

assert degree_distance > 0
assert utm_distance_m > 0
assert geodesic_distance_m > 0

# For these nearby synthetic points, UTM and geodesic distance
# should be very similar.
assert relative_difference_pct < 1.0


print()
print("====================================")
print("LAB 01A COORDINATE FOUNDATIONS: PASS")
print("====================================")
