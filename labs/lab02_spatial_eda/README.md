# LAB 02 - Spatial Exploratory Data Analysis

## Purpose

This laboratory examines how uneven spatial sampling can distort the interpretation of exploratory geospatial data.

Central question:

> Can an apparent spatial hotspot be explained by sampling intensity rather than by the underlying spatial signal?

A controlled synthetic experiment is used so that the underlying spatial signal is known. The goal is to compare sampling density, observed signal, and synthetic ground truth before predictive modelling.

## Experimental setting

All observations are generated in **EPSG:32719 (WGS 84 / UTM zone 19S)** over an approximately 80 km x 80 km synthetic domain.

Each independent spatial realization contains:

- 500 regional observations;
- 300 intensive-campaign observations;
- 800 observations in total.

The intensive campaign is deliberately concentrated in a different area from the strongest underlying synthetic signal.

Campaign centre:

- Easting: 376,500 m
- Northing: 7,138,500 m

Signal centre:

- Easting: 411,500 m
- Northing: 7,162,500 m

The baseline demonstration uses seed `12345`.

## Main variables

The main variables are:

- `sample_id`: observation identifier;
- `sampling_component`: regional or intensive campaign;
- `geometry`: projected point coordinates;
- `signal_true`: known underlying synthetic spatial signal;
- `signal`: observed signal after random noise is added.

Derived variables include nearest-neighbour distance, grid identifiers, grid centres, observation counts, and mean signal by grid cell.

The synthetic signal is a teaching variable. It is not a real ore grade, geochemical assay, resource estimate, or prospectivity score.

## Geographic context and provenance

The notebook includes a lightweight geographic locator so that the synthetic
study domain can be understood in a real coordinate reference context.

The country geometry is derived from **Natural Earth Admin 0 - Countries,
1:110m, version 5.1.1** and is stored locally as
`data/context/chile_context.geojson` in **EPSG:4326**.

The highlighted LAB 02 domain is not real exploration data. It is generated
from `DEFAULT_BOUNDS` in `DEFAULT_CRS` (**EPSG:32719**) and is reprojected to
EPSG:4326 only for the contextual figure.

The location does not represent an actual exploration project, mining property,
concession, prospect, deposit, mineral occurrence, or field campaign.

`data/context/SOURCE.md` documents the source, derivation procedure,
public-domain distribution status, and SHA-256 integrity hashes for both the
downloaded Natural Earth archive and the derived GeoJSON artifact.

The context layer is an auxiliary cartographic input only. It does not
contribute to the synthetic signal, sampling process, Monte Carlo experiment,
grid selection, or analytical results.

## Reusable code

Reusable spatial functions are implemented in `src/geoai/spatial.py`:

- `require_metric_projected_crs`
- `generate_synthetic_sampling`
- `nearest_neighbor_distances`
- `assign_regular_grid`
- `summarize_grid`

The baseline exploratory script is `labs/lab02_spatial_eda/02_spatial_eda.py`.

The sensitivity experiment is `labs/lab02_spatial_eda/02b_resolution_sensitivity.py`.

## Reproducible results pipeline

The confirmatory Monte Carlo experiment persists its outputs under
`labs/lab02_spatial_eda/results/`:

- `resolution_summary.csv`: summary metrics by grid resolution;
- `monte_carlo_by_realization.csv`: one record per independent
  realization and grid resolution;
- `experiment_metadata.json`: experiment design, CRS, sampling counts,
  Monte Carlo configuration, and nearest-neighbour robustness metadata.

The visual notebook does not contain manually transcribed Monte Carlo
results. `02c_build_notebook.py` generates the notebook so that it reads
the persisted CSV and JSON artifacts directly.

`02d_execute_notebook.py` then executes and validates the notebook in a
reproducible local kernel session. On Windows, the executor uses the
selector event-loop policy required by ZeroMQ and requires CurveZMQ
transport encryption over the local loopback interface.

The resulting data lineage is:

`simulation -> persisted results -> notebook -> figures and interpretation`

The geographic context layer follows a separate provenance path and is consumed
only by the notebook visualization stage.

## Baseline result

For seed `12345`:

- observations: 800;
- regional observations: 500;
- campaign observations: 300;
- regional mean nearest-neighbour distance: 1807.35 m;
- campaign mean nearest-neighbour distance: 502.73 m;
- campaign/regional nearest-neighbour ratio: 0.278.

The intensive campaign is therefore substantially more spatially concentrated than the regional component.

Using the selected 10 km grid, the densest sampling cell differs from the cell containing the strongest underlying synthetic signal.

## Why grid resolution matters

Spatial aggregation creates a scale trade-off.

A small grid preserves local detail but may contain very few observations per cell. A large grid increases local sample support but may smooth or attenuate the underlying spatial signal. Results can also change when the grid origin is shifted.

For this reason, grid resolution is evaluated with repeated simulations and multiple grid origins rather than with a single map or random seed.

## Monte Carlo sensitivity design

The confirmatory design uses:

- **400 independent spatial realizations**;
- 3 grid resolutions: 5 km, 10 km, and 20 km;
- 4 grid-origin configurations per resolution.

This produces **4,800 grid evaluations**.

The statistical replication unit is the spatial realization. Grid resolution and grid origin are repeated evaluations within each realization and are not treated as independent observations.

The four offsets are:

- `(0, 0)`
- `(0.5 cell, 0)`
- `(0, 0.5 cell)`
- `(0.5 cell, 0.5 cell)`

## Nearest-neighbour robustness

Across the 400 independent spatial realizations:

- campaign denser than regional sampling: 100.0%;
- median campaign/regional NN ratio: 0.261;
- 95th percentile NN ratio: 0.284.

This confirms that the intensive campaign is consistently more spatially concentrated than the regional sampling component.

## Resolution sensitivity

### 5 km

- mean offset match rate: 87.4%;
- all four offsets matching: 63.0%;
- Wilson 95% CI: 58.2% to 67.6%;
- median observations at true hotspot: 2.0;
- median true hotspot signal: 84.89;
- peak retention: 94.3%;
- 95th percentile worst-case normalized displacement: 1.414 cell widths.

The 5 km grid preserves local signal well but has low local sample support and is sensitive to grid origin.

### 10 km

- mean offset match rate: 98.8%;
- all four offsets matching: 95.8%;
- Wilson 95% CI: 93.3% to 97.3%;
- median observations at true hotspot: 7.5;
- median true hotspot signal: 70.93;
- peak retention: 78.8%;
- 95th percentile worst-case normalized displacement: 0.000 cell widths.

The **10 km x 10 km** grid is selected as the primary descriptive resolution for this synthetic experiment because it provides high hotspot stability while retaining substantially more spatial signal than the 20 km grid.

### 20 km

- mean offset match rate: 98.9%;
- all four offsets matching: 95.8%;
- Wilson 95% CI: 93.3% to 97.3%;
- median observations at true hotspot: 31.5;
- median true hotspot signal: 41.12;
- peak retention: 45.7%;
- 95th percentile worst-case normalized displacement: 0.000 cell widths.

The 20 km grid provides strong local support and stability but substantially attenuates the spatial peak, illustrating over-smoothing.

## Interpretation

The 10 km resolution is not a universal recommendation for mineral exploration. It is specific to this synthetic design.

The central result is:

> Sampling density and spatial signal are different properties of a geospatial dataset and should be evaluated separately before predictive modelling.

Without spatial validation, a predictive model may partly learn where data were sampled most intensively instead of where the underlying geological signal is strongest.

This laboratory therefore prepares the foundation for spatial validation and spatial cross-validation before predictive mineral-exploration modelling.

## Limitations

This laboratory deliberately uses a simplified synthetic process. It does not yet include real geological units, structural geology, real geochemical assays, compositional-data analysis, anisotropy, a fitted spatial autocorrelation model, measurement error, detection limits, preferential-sampling correction, predictive modelling, or mineral-prospectivity ranking.

These elements are deferred so that sampling-density and aggregation effects can first be studied in isolation.

## Current validation status

- LAB 02 automated tests: 9 passed
- Full project test suite: **24 passed**
- Monte Carlo realizations: 400
- Grid evaluations: 4,800
- Persisted realization-resolution records: 1,200
- Resolution-summary records: 3
- Primary descriptive resolution: 10 km x 10 km
- Notebook cells: 31
- Notebook code cells executed: 13 / 13
- Notebook error outputs: 0
- Embedded PNG figures: 6
- Notebook schema validation: passed
- Geographic context feature: 1 valid Chile MultiPolygon
- Geographic context CRS: EPSG:4326
- Geographic context provenance: documented in `data/context/SOURCE.md`
- Notebook encoding: UTF-8 without BOM
- Notebook line endings: LF
- Pre-commit validation: passed
- Gitleaks secret scan: passed

The computational core, Monte Carlo experiment, persisted result
artifacts, notebook generation, notebook execution, and automated
validation form a reproducible end-to-end analytical workflow.
