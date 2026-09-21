# LAB 01 — Geospatial Python Foundations

## Overview

LAB 01 introduces the geospatial foundations required for the GeoAI Mineral Labs project.

The objective is to establish a correct and reproducible understanding of vector and raster spatial data before introducing spatial machine learning, prospectivity modelling, remote sensing, or advanced GeoAI techniques.

This laboratory prioritizes spatial correctness over model complexity.

## Objectives

LAB 01 aims to develop and validate the ability to:

* understand fundamental spatial data structures;
* work with vector and raster geospatial data;
* identify and manage Coordinate Reference Systems (CRS);
* reproject spatial datasets correctly;
* perform geometric and spatial operations;
* calculate spatial distances using appropriate projected coordinate systems;
* perform spatial joins and overlays;
* inspect and validate geometry quality;
* sample raster information using vector geometries;
* create reproducible geospatial visualizations;
* build reusable Python components for later GeoAI laboratories.

## Scientific principles

The laboratory follows several principles that will remain important throughout the project.

### Spatial location matters

Geospatial observations are not independent in the same way as ordinary tabular observations.

Spatial relationships, proximity, scale, projection, and autocorrelation must be considered explicitly.

### CRS is part of the data

A coordinate pair without its Coordinate Reference System is incomplete.

CRS validation must therefore occur before:

* distance calculations;
* buffering;
* overlay;
* area calculations;
* raster/vector alignment;
* model feature extraction.

### Visualization is not validation

A map that looks correct may still contain:

* incorrect CRS definitions;
* invalid geometries;
* spatial misalignment;
* unit inconsistencies;
* duplicated features;
* incorrect raster transforms.

Geospatial workflows require programmatic validation in addition to visual inspection.

## Scope

LAB 01 will cover the following foundations.

### Vector data

Topics include:

* points;
* lines;
* polygons;
* GeoDataFrames;
* geometry columns;
* geometry validation;
* spatial relationships;
* spatial indexing.

### Coordinate Reference Systems

Topics include:

* geographic CRS;
* projected CRS;
* EPSG identifiers;
* transformations;
* coordinate units;
* appropriate CRS selection for spatial measurements.

### Core spatial operations

The laboratory will introduce:

* spatial joins;
* clipping;
* buffering;
* intersections;
* overlays;
* centroid operations;
* bounding boxes;
* distance calculations.

### Raster foundations

Topics include:

* raster dimensions;
* bands;
* cells;
* resolution;
* transform;
* extent;
* CRS;
* NoData;
* raster sampling;
* raster/vector relationships.

### Visualization

The laboratory will produce basic geospatial visualizations suitable for:

* exploratory inspection;
* QA;
* communication;
* future portfolio demonstrations.

## Initial dependency assessment

The candidate scientific stack will be evaluated before installation.

Potential dependencies include:

```text
NumPy
Pandas
GeoPandas
Shapely
PyProj
Rasterio
Matplotlib
Jupyter / IPython
```

Dependencies will not be added solely because they are commonly used.

Each package must have a clear role in the laboratory and be compatible with the validated project environment.

## Environment strategy

The project continues to use the dedicated Conda environment:

```text
geoai-mineral
```

Project-level dependencies are controlled through:

```text
environment.yml
```

The environment currently targets Python 3.12.

The geospatial stack will be installed using the minimum set of explicitly required dependencies.

`conda-forge` remains the preferred dependency source for the project environment.

## Data strategy

LAB 01 should preferentially use:

* public geospatial data;
* redistribution-safe data;
* small synthetic spatial examples.

Private exploration datasets are not required for this laboratory.

Any external dataset must be reviewed for:

* source;
* license;
* attribution requirements;
* redistribution permission;
* CRS information;
* update/version information.

## Repository strategy

Scientific exploration may initially occur in notebooks.

Reusable logic should progressively move into:

```text
src/geoai/
```

where it can be tested and reused by later laboratories.

The intended separation is:

```text
notebook
    ↓
exploration and explanation
    ↓
reusable implementation
    ↓
src/geoai/
    ↓
tests
```

Notebooks should not become the only implementation layer of the project.

## Planned learning sequence

The initial sequence is:

```text
1. Geospatial data concepts
        ↓
2. Vector geometries
        ↓
3. CRS and projections
        ↓
4. Geometry validation
        ↓
5. Spatial operations
        ↓
6. Distance and area calculations
        ↓
7. Raster foundations
        ↓
8. Raster/vector integration
        ↓
9. Geospatial visualization
        ↓
10. Reusable utilities and tests
```

## Quality controls

Applicable LAB 01 work should include:

* reproducible environment;
* CRS validation;
* geometry validation;
* explicit units;
* deterministic examples where appropriate;
* assertions or tests;
* source attribution;
* pre-commit validation;
* Gitleaks scanning;
* signed Git commits;
* Pull Request integration.

## Publication classification

LAB 01 is expected to contain a high proportion of material suitable for future public portfolio use because its concepts are primarily foundational and generic.

Typical candidates for `PUBLIC` classification include:

* CRS demonstrations;
* vector operations;
* raster fundamentals;
* generic geospatial utilities;
* spatial QA examples;
* simple public or synthetic datasets;
* reproducible maps and figures.

However, all new material remains `INTERNAL` by default until explicitly reviewed through the project Publication Gate.

Advanced or differentiating extensions remain `RESERVED`.

## Definition of Done

LAB 01 will be considered technically complete when:

* [ ] dependency compatibility has been assessed;
* [ ] the minimal geospatial environment has been installed;
* [ ] environment recreation has been validated;
* [ ] GeoPandas can read and manipulate vector geometries;
* [ ] CRS inspection and reprojection have been demonstrated;
* [ ] geometry validation has been demonstrated;
* [ ] spatial joins have been demonstrated;
* [ ] buffering and overlay operations have been demonstrated;
* [ ] spatial distance calculations use appropriate units and CRS;
* [ ] Rasterio can read and inspect raster metadata;
* [ ] raster sampling has been demonstrated;
* [ ] raster and vector CRS alignment has been demonstrated;
* [ ] basic geospatial visualizations have been generated;
* [ ] reusable utilities exist under `src/geoai/` where appropriate;
* [ ] tests cover reusable functionality;
* [ ] documentation explains assumptions and limitations;
* [ ] security and repository QA checks pass;
* [ ] Publication Gate classification has been completed.

## Relationship with later laboratories

LAB 01 provides the technical foundation for:

```text
LAB 01 — Geospatial Python Foundations
        ↓
LAB 02 — Spatial EDA
        ↓
LAB 03 — Spatial Cross-Validation
        ↓
LAB 04 — Geochemical CoDA
        ↓
LAB 05 — Mineral Prospectivity Mapping
        ↓
LAB 06+ — Advanced GeoAI
```

Incorrect spatial foundations would propagate errors into every later stage.

For that reason, LAB 01 emphasizes spatial correctness, validation, and reproducibility before predictive modelling.

## Current status

**Status: Design and dependency assessment**

No geospatial scientific dependencies should be considered validated until compatibility testing has been completed.
