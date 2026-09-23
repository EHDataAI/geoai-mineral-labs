# GeoAI Mineral Labs

A reproducible laboratory and technical portfolio focused on the application of geospatial data science, machine learning, deep learning, and GeoAI to mineral exploration.

The repository is developed progressively, from engineering foundations and spatial data analysis to mineral prospectivity modelling, explainability, uncertainty quantification, remote sensing, and advanced GeoAI methods.

## Purpose

The objective is to build a structured collection of reproducible experiments connecting geospatial data science, spatial statistics, geostatistics, geochemistry, mineral prospectivity mapping, machine learning, deep learning, remote sensing, explainable AI, uncertainty quantification, and reproducible software engineering practices.

Each laboratory is intended to produce not only a model or notebook, but also evidence about data quality, validation strategy, limitations, interpretation, and reproducibility.

## Technical principles

- Reproducibility first.
- Spatially aware validation.
- Explicit control of spatial leakage.
- Simple baselines before complex models.
- Geological interpretation alongside predictive performance.
- Clear separation between training, validation, and evaluation data.
- Traceable experiments and configuration.
- Uncertainty considered alongside predictions.
- No credentials or private data committed to Git.
- Incremental development with testing and version control.

Predictive performance alone is not considered sufficient evidence of a useful geospatial model.

## Laboratory roadmap

| Lab | Topic | Status |
| --- | --- | --- |
| LAB 00 | [Engineering & GeoAI Environment](labs/lab00_engineering_setup/README.md) | Completed |
| LAB 01 | [Geospatial Python Foundations](labs/lab01_geospatial_foundations/README.md) | In progress |
| LAB 02 | Spatial Exploratory Data Analysis | Planned |
| LAB 03 | Spatial Cross-Validation | Planned |
| LAB 04 | Geochemical CoDA | Planned |
| LAB 05 | Mineral Prospectivity Mapping | Planned |
| LAB 06 | Explainable GeoAI | Planned |
| LAB 07 | Uncertainty Quantification | Planned |
| LAB 08 | Remote Sensing | Planned |
| LAB 09 | Deep Learning for GeoAI | Planned |
| LAB 10 | Graph & Multimodal GeoAI | Planned |

The roadmap may evolve as the laboratories incorporate new datasets, methods, and evidence.

## Featured portfolio walkthroughs

### LAB 01A — Coordinate Systems for Mineral Exploration

[Open LAB 01A](labs/lab01_geospatial_foundations/01a_coordinate_foundations.ipynb)

Demonstrates why geographic coordinates expressed in degrees must not be interpreted directly as metric distances.

Using synthetic locations in northern Chile, the walkthrough compares geographic, projected UTM, and geodesic distance calculations and shows why a numerically valid spatial calculation may still be geoscientifically incorrect.

Key evidence includes:

- WGS84 / EPSG:4326 interpretation;
- automatic UTM zone selection;
- projected versus geodesic distance comparison;
- validation against GeoPandas `estimate_utm_crs()`;
- reusable CRS utilities;
- automated testing.

### LAB 01B — Crossing UTM Zones in Spatial Analysis

[Open LAB 01B](labs/lab01_geospatial_foundations/01b_cross_zone_crs.ipynb)

Demonstrates why projected coordinates expressed in metres are still incompatible when they belong to different Coordinate Reference Systems.

The experiment compares:

- an independent WGS84 geodesic reference;
- a valid calculation using one common projected CRS;
- an intentionally invalid calculation that mixes UTM Zones 18S and 19S.

Key results:

| Method | Distance | Relative difference vs geodesic | Spatially valid |
| --- | ---: | ---: | --- |
| WGS84 geodesic reference | ~38.594 km | 0.0000% | Yes |
| Common CRS — EPSG:32719 | ~38.619 km | ~0.0635% | Yes |
| Mixed UTM 18S / 19S | ~540.215 km | ~1299.72% | No |

The experiment demonstrates that a calculation may return a valid numerical result while still being geographically invalid.

### Combined GeoAI lesson

Together, both laboratories establish two foundational spatial controls:

```text
LAB 01A
degrees are not metres
        ↓
CRS and units must be interpreted correctly
        ↓
LAB 01B
metres are not automatically comparable
        ↓
joint spatial operations require a compatible common CRS
