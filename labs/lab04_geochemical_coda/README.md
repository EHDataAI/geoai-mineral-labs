# LAB04 - Geochemical Compositional Data Analysis

**Status: Completed**

## Overview

LAB04 demonstrates a reproducible compositional-data workflow applied to a controlled synthetic geochemical experiment.

The laboratory focuses on ten selected elements: Cu, Mo, Au, As, Sb, Fe, Mg, Ca, Al, and K.

The data are fully synthetic and are intended for methodological experimentation, not geological inference about a real deposit or exploration property.

## Scientific objective

The central objective is to evaluate how conventional analysis of geochemical concentrations differs from analysis performed in log-ratio geometry.

The workflow compares:

- raw concentrations;
- closed compositions;
- centered log-ratio (CLR) coordinates;
- isometric log-ratio (ILR) coordinates;
- PCA under raw, closed, CLR, and ILR representations;
- sensitivity to two simple below-detection-limit replacement rules.

## Synthetic experiment

The synthetic dataset contains 1,200 spatial samples in EPSG:32719.

The analytical workflow contains 1,092 complete rows and excludes 108 rows with missing observations.

The simulation includes synthetic UTM coordinates, lithological regimes, mineralization gradients, geochemical pathfinder behavior, analytical censoring, missing observations, and latent ground-truth variables used only for diagnostic validation.

## Geochemical data contract

`geochemistry_contract.json` is the authoritative LAB04 data contract.

It defines element ordering, synthetic concentration units, detection limits, CRS, censoring semantics, missing-value semantics, closure interpretation, and replacement rules.

The synthetic concentration convention is ppm. These values are modelling constructs and are not certified laboratory measurements.

The ten selected elements form a geochemical subcomposition. Closure to 100 is used only as relative normalization and must not be interpreted as a complete whole-rock mass balance.

## Detection limits and missingness

Observed values below their synthetic detection limit are encoded as zero.

Missing observations are stored separately as NaN.

The primary replacement rule is:

```text
LOD / 2
```

The sensitivity rule is:

```text
LOD / sqrt(2)
```

The contract audit reports:

- 1,200 samples;
- 10 elements;
- 111 missing cells across 108 rows;
- 1,169 censored cells;
- synthetic-only data sources;
- latent columns restricted to diagnostic ground truth.

## Compositional transformations

The closed composition contains 10 parts.

CLR produces 10 coordinates constrained to rank 9.

ILR produces 9 orthonormal coordinates.

The analysis confirms that CLR and ILR encode the same compositional information under their respective geometries.

LAB04 explicitly tests closure, scale invariance, CLR inversion, ILR inversion, Helmert-basis orthonormality, Aitchison-distance properties, and rejection of invalid compositions.

## PCA comparison

The first principal component explains:

| Representation | PC1 explained variance |
| --- | ---: |
| Raw | 0.304757 |
| Closed | 0.310453 |
| CLR | 0.225598 |
| ILR | 0.225598 |

CLR and ILR produce matching compositional eigenstructure to numerical precision.

The laboratory does not interpret raw or closed PCA as intrinsically invalid. It demonstrates that Euclidean analysis of constrained compositions answers a different geometric question from log-ratio analysis.

## LOD sensitivity

The LOD sensitivity experiment compares `LOD/2` with `LOD/sqrt(2)` on the same 1,092 complete samples.

PCA score correlations between replacement rules are:

| Component | Correlation |
| --- | ---: |
| PC1 | 0.983686 |
| PC2 | 0.981298 |
| PC3 | 0.954433 |

PC1 explained variance changes from `0.225598` under `LOD/2` to `0.246496` under `LOD/sqrt(2)`, an absolute difference of approximately `0.020898`.

The main low-dimensional structure remains highly correlated across the two simple replacement rules, while below-detection-limit treatment is not analytically neutral.

## Reproducibility

The LAB04 workflow includes synthetic-data generation, CoDA analysis, LOD sensitivity analysis, data-contract audit, reproducibility verification, notebook generation, notebook execution, persisted result artifacts, and automated tests.

The reproducibility verifier reports:

- numerical regression: PASS;
- analysis-summary numerical consistency: PASS;
- LOD-sensitivity numerical consistency: PASS.

Byte-for-byte hashes are retained as a historical pre-refactoring baseline. Some current serialized artifacts differ bytewise from that historical baseline, so LAB04 reproducibility claims are based on numerical regression and validated scientific outputs rather than universal cross-platform byte identity.

## Validation status

LAB04 currently has 28 dedicated tests covering compositional mathematics, below-detection-limit replacement, the data contract, UTF-8 integrity, and notebook execution/structure.

At the v0.2.0 documentation checkpoint:

```text
28 passed
```

## Versioned result artifacts

Versioned LAB04 evidence includes:

- `analysis_summary.json`;
- `data_contract_audit.json`;
- `lod_sensitivity_summary.json`;
- `lod_sensitivity_variance.csv`;
- raw, closed, and CLR correlation matrices;
- PCA explained-variance results;
- PCA loading matrices for raw, closed, CLR, and ILR representations.

## Limitations

LAB04 is intentionally methodological.

Important limitations include:

1. all observations are synthetic;
2. the concentration values are not laboratory measurements;
3. detection limits are experimental assumptions;
4. the ten elements form only a selected subcomposition;
5. simple replacement rules are methodological baselines rather than advanced censoring models;
6. closure to 100 is relative normalization, not whole-rock mass balance;
7. the experiment does not establish geological conclusions about a real mineral system.

## Key takeaway

Geochemical concentrations are constrained multivariate data.

LAB04 demonstrates that closure, log-ratio geometry, censoring treatment, and representation choice can materially change the geometry of the analytical problem before those variables are introduced into downstream GeoAI models.
