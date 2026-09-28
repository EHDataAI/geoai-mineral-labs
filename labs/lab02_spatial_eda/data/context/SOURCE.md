# Context data provenance

## Purpose

This directory contains a lightweight geographic context layer used only to
locate the synthetic LAB02 study domain within Chile.

The contextual country geometry is real cartographic reference data. The LAB02
study area, sampling locations, signal, campaign design, and analytical results
are entirely synthetic.

## Source

- Provider: Natural Earth
- Dataset: Admin 0 - Countries
- Scale: 1:110m
- Version: 5.1.1
- Source archive: `ne_110m_admin_0_countries.zip`
- Source CRS: EPSG:4326
- Distribution: Natural Earth public-domain vector data

## Derived artifact

File:

`chile_context.geojson`

The derived layer was created from the Natural Earth country dataset using the
following procedure:

1. Read the original archive with GeoPandas using the Pyogrio engine.
2. Select exactly one feature using `ISO_A3 == "CHL"`.
3. Retain only the country name, ISO alpha-2 code, ISO alpha-3 code, and geometry.
4. Normalize the output CRS to EPSG:4326.
5. Validate the resulting geometry.
6. Add minimal provenance metadata.
7. Write the derived feature as GeoJSON using Pyogrio.

The resulting geometry is a valid MultiPolygon containing two polygon parts.

## Integrity

Original Natural Earth archive SHA-256:

`0F243AEAC8AC6CF26F0417285B0BD33AC47F1B5BDB719FD3E0DF37D03EA37110`

Derived `chile_context.geojson` SHA-256:

`4B36F55393CB3DED799C6AF73664A7531A90B97580D2B55250C2134F4EA61BE7`

These hashes identify the exact files used during preparation of this LAB02
context layer.

## Analytical scope

The country geometry is used only as a geographic locator.

It is not used to derive the synthetic signal, sampling density, Monte Carlo
experiment, spatial statistics, or any mineral prospectivity result.

The synthetic LAB02 domain is fully contained within the Chile context geometry,
but its location does not represent an actual exploration project, mining
property, prospect, concession, deposit, mineral occurrence, or field campaign.

## Reproducibility note

The contextual layer is stored locally so the LAB02 notebook can be executed
without requiring a web-map service, tile provider, or Internet connection.

No Cartopy, Contextily, Fiona, or external basemap service is required.
