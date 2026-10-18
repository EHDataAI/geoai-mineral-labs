# GeoAI Mineral Labs

A reproducible technical portfolio focused on geospatial data science, spatial validation, compositional geochemistry, machine learning, and GeoAI for mineral exploration.

**Current milestone: v0.2.0**

The repository is developed progressively, from engineering and geospatial foundations to spatial exploratory analysis, leakage-aware validation, compositional data analysis, mineral prospectivity modelling, explainability, uncertainty quantification, remote sensing, and advanced GeoAI methods.

## Purpose

The objective is to build a structured collection of reproducible experiments connecting:

- geospatial data science;
- spatial statistics;
- geostatistics;
- geochemistry;
- compositional data analysis;
- mineral prospectivity mapping;
- machine learning;
- remote sensing;
- explainable AI;
- uncertainty quantification;
- reproducible software engineering.

Each laboratory is designed to produce not only a model or notebook, but also evidence about data quality, validation strategy, limitations, interpretation, and reproducibility.

Predictive performance alone is not considered sufficient evidence of a useful geospatial model.

## Technical principles

- Reproducibility first.
- Spatial correctness before model complexity.
- Spatially aware validation.
- Explicit control of spatial leakage.
- Simple baselines before complex models.
- Geological interpretation alongside predictive performance.
- Clear separation between development, validation, and final evaluation.
- Fold-local preprocessing where applicable.
- Traceable experiments and configuration.
- Reproducible synthetic experiments when controlled ground truth is required.
- Uncertainty considered alongside predictions.
- No credentials or private data committed to Git.
- Incremental development with automated testing, pre-commit validation, and secret scanning.

## Laboratory roadmap

| Laboratory | Topic | Status |
| --- | --- | --- |
| LABORATORY 00 | [Engineering & GeoAI Environment](labs/lab00_engineering_setup/README.md) | Completed |
| LABORATORY 01 | [Geospatial Python Foundations](labs/lab01_geospatial_foundations/README.md) | In progress |
| LABORATORY 02 | [Spatial Exploratory Data Analysis](labs/lab02_spatial_eda/README.md) | Completed |
| LABORATORY 03 | [Spatial Cross-Validation](labs/lab03_spatial_cross_validation/README.md) | Completed |
| LABORATORY 04 | [Geochemical Compositional Data Analysis](labs/lab04_geochemical_coda/README.md) | Completed |
| LABORATORY 05 | [Mineral Prospectivity Mapping](labs/lab05_mineral_prospectivity/README.md) | Completed |
| LABORATORY 06 | [Explainable GeoAI](labs/lab06_explainable_geoai/README.md) | Completed |
| LABORATORY 07 | Uncertainty Quantification | Planned |
| LABORATORY 08 | Remote Sensing | Planned |
| LABORATORY 09 | Deep Learning for GeoAI | Planned |
| LABORATORY 10 | Graph & Multimodal GeoAI | Planned |

The roadmap may evolve as new datasets, validation requirements, and modelling questions are introduced.

## Laboratory highlights

### LABORATORY 01 - Geospatial Python Foundations

LABORATORY 01 remains in progress, while its CRS-focused portfolio experiments are already completed and validated.

The current work demonstrates:

- geographic versus projected coordinate interpretation;
- automatic WGS84 UTM zone selection;
- projected versus geodesic distance validation;
- cross-zone CRS failure modes;
- reusable CRS utilities;
- automated spatial validation.

A central lesson is that a numerically valid spatial calculation may still be geographically invalid when CRS semantics are ignored.

### LABORATORY 02 - Spatial Exploratory Data Analysis

LABORATORY 02 studies how uneven spatial sampling can distort exploratory interpretation.

A controlled synthetic experiment separates:

- sampling intensity;
- observed spatial signal;
- underlying synthetic ground truth.

The laboratory demonstrates why apparent hotspots can reflect sampling design rather than the underlying process and establishes the analytical motivation for spatially aware validation.

### LABORATORY 03 - Spatial Cross-Validation

LABORATORY 03 evaluates conventional Random K-Fold against Spatial Block Cross-Validation under controlled spatial autocorrelation.

The dataset, predictors, target, model, and model hyperparameters remain fixed so that the experiment isolates the effect of validation design.

The laboratory demonstrates why geographically close observations in training and validation sets can produce optimistic error estimates and why validation design must reflect the intended spatial generalization problem.

### LABORATORY 04 - Geochemical Compositional Data Analysis

LABORATORY 04 implements a reproducible compositional-data workflow for synthetic geochemical observations.

The analysis compares:

- raw concentrations;
- closed compositions;
- centered log-ratio coordinates;
- isometric log-ratio coordinates;
- PCA under conventional and compositional representations.

The laboratory emphasizes the geometry of compositional data and the risks of interpreting constrained geochemical measurements using conventional Euclidean methods alone.

### LABORATORY 05 - Mineral Prospectivity Mapping

LABORATORY 05 integrates geological, structural, geophysical, alteration, and geochemical evidence into an end-to-end synthetic mineral prospectivity experiment.

The synthetic study area contains 6,400 one-square-kilometre cells in EPSG:32719.

The frozen synthetic mineral system contains:

- 215 true synthetic deposits;
- 100 known positives available during model development;
- 115 hidden deposits reserved for final post-freeze evaluation.

Unlabelled cells are treated as background rather than confirmed negatives.

Three evidence configurations are evaluated:

- Feature Set A: geology, structure, geophysics, and alteration;
- Feature Set B: Feature Set A plus raw geochemistry;
- Feature Set C: Feature Set A plus ILR-transformed compositional geochemistry.

Dummy Classifier, Logistic Regression, and Random Forest models are compared across ten background-sampling replicates.

The primary validation strategy uses five-fold `StratifiedGroupKFold` with spatial block grouping and zero train/test group overlap.

Random cross-validation is retained only as a diagnostic comparison and is not used for model selection.

#### Frozen operational model

The predeclared primary model-selection criterion chooses:

- Feature Set A;
- Logistic Regression;
- mean spatial pooled-OOF Average Precision: `0.393412`;
- median spatial pooled-OOF Average Precision: `0.380985`;
- mean spatial ROC-AUC: `0.734202`.

Alternative configurations remain competitive, but model selection follows the predeclared primary metric rather than post-hoc selection using secondary metrics.

The result also illustrates an important modelling principle: additional predictors or model complexity do not automatically produce a better spatially validated model.

#### Out-of-sample candidate ranking

Full-grid ranking is performed using only model predictions generated for background cells when those cells were not included in the corresponding replicate's sampled training background.

The resulting `prospectivity_score` is an ensemble ranking score.

It is not a calibrated probability of mineral occurrence.

#### Hidden-oracle evaluation

The synthetic oracle remains sealed until the model configuration and candidate ranking procedure are frozen.

Final hidden-deposit evaluation:

| Prioritized candidate area | Hidden deposits captured | Hidden capture rate | Lift over random |
| --- | ---: | ---: | ---: |
| Top 1% | 6 / 115 | 5.22% | 5.22x |
| Top 5% | 25 / 115 | 21.74% | 4.35x |
| Top 10% | 44 / 115 | 38.26% | 3.83x |

Additional final metrics:

- hidden-oracle Average Precision: `0.056650`;
- hidden-oracle ROC-AUC: `0.709070`;
- median hidden-deposit rank: `1,117 / 6,300`.

The absolute Average Precision must be interpreted in the context of a low-prevalence ranking problem.

The operational objective is spatial prioritization and search-space reduction rather than calibrated deposit detection.

The LABORATORY 05 experiment therefore emphasizes validation design, leakage prevention, reproducibility, and prospectivity ranking rather than maximizing a single headline metric.

## Leakage and validation controls

The project progressively introduces controls against common sources of misleading geospatial model performance, including:

- incorrect CRS or coordinate-system interpretation;
- spatial train/test overlap;
- target and oracle leakage;
- identifier and coordinate leakage;
- preprocessing leakage;
- evaluation contamination;
- in-sample candidate scoring;
- post-evaluation model reselection;
- source-code drift in generated experiment components.

LABORATORY 05 additionally applies:

- forbidden-feature contracts;
- spatial group validation;
- fold-local preprocessing;
- pooled out-of-fold predictions;
- out-of-sample-only full-grid ranking;
- model freeze before hidden-oracle access;
- source synchronization tests.

## Reproducibility and engineering

The repository uses:

- Python 3.12;
- `pytest`;
- `pre-commit`;
- Gitleaks;
- deterministic random seeds;
- explicit experiment contracts;
- version-controlled result artifacts;
- signed milestone commits;
- feature branches;
- pull-request integration.

Generated or intermediate artifacts that can be reconstructed from committed inputs and scripts may be excluded from Git when appropriate.

## Repository structure

```text
geoai-mineral-labs/
|-- labs/
|   |-- lab00_engineering_setup/
|   |-- lab01_geospatial_foundations/
|   |-- lab02_spatial_eda/
|   |-- lab03_spatial_cross_validation/
|   |-- lab04_geochemical_coda/
|   `-- lab05_mineral_prospectivity/
|-- scripts/
|-- src/
|   `-- geoai/
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Current milestone

Version `0.2.0` marks the transition from the initial engineering and geospatial foundations to a reproducible multi-laboratory workflow that now includes:

- spatial exploratory analysis;
- spatial cross-validation;
- compositional geochemistry;
- mineral prospectivity benchmarking;
- pooled out-of-fold model evaluation;
- out-of-sample candidate ranking;
- final hidden-oracle evaluation.

The project remains pre-1.0 while advanced GeoAI capabilities such as explainability, uncertainty quantification, remote sensing, deep learning, and multimodal methods remain on the development roadmap.

## Next steps

Planned development includes:

1. Explainable GeoAI.
2. Spatial uncertainty quantification.
3. Remote sensing integration.
4. Deep learning for geospatial mineral exploration.
5. Graph and multimodal GeoAI.
6. Evaluation of geographic distribution shift and model robustness across spatial domains.

## Scope and limitations

The current laboratories are methodological experiments and technical demonstrations.

Synthetic datasets are intentionally used where controlled ground truth is necessary to study:

- validation design;
- spatial leakage;
- compositional transformations;
- ranking performance;
- model behaviour;
- evaluation contamination.

Results must not be interpreted as geological conclusions about a real mineral deposit or exploration property.
