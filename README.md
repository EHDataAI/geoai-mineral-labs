# GeoAI Mineral Labs

A reproducible laboratory and technical portfolio focused on the application of geospatial data science, machine learning, deep learning, and GeoAI to mineral exploration.

The repository is developed progressively, from engineering foundations and spatial data analysis to mineral prospectivity modelling, explainability, uncertainty quantification, remote sensing, and advanced GeoAI methods.

## Purpose

The objective is to build a structured collection of reproducible experiments connecting geospatial data science, spatial statistics, geostatistics, geochemistry, mineral prospectivity mapping, machine learning, deep learning, remote sensing, explainable AI, uncertainty quantification, and reproducible software engineering practices.

Each laboratory is intended to produce not only a model or notebook, but also evidence about data quality, validation strategy, limitations, interpretation, and reproducibility.

## Technical principles

* Reproducibility first.
* Spatially aware validation.
* Explicit control of spatial leakage.
* Simple baselines before complex models.
* Geological interpretation alongside predictive performance.
* Clear separation between training, validation, and evaluation data.
* Traceable experiments and configuration.
* Uncertainty considered alongside predictions.
* No credentials or private data committed to Git.
* Incremental development with testing and version control.

Predictive performance alone is not considered sufficient evidence of a useful geospatial model.

## Laboratory roadmap

| Lab    | Topic                             | Status      |
| ------ | --------------------------------- | ----------- |
| LAB 00 | Engineering & GeoAI Environment   | In progress |
| LAB 01 | Geospatial Python Foundations     | Planned     |
| LAB 02 | Spatial Exploratory Data Analysis | Planned     |
| LAB 03 | Spatial Cross-Validation          | Planned     |
| LAB 04 | Geochemical CoDA                  | Planned     |
| LAB 05 | Mineral Prospectivity Mapping     | Planned     |
| LAB 06 | Explainable GeoAI                 | Planned     |
| LAB 07 | Uncertainty Quantification        | Planned     |
| LAB 08 | Remote Sensing                    | Planned     |
| LAB 09 | Deep Learning for GeoAI           | Planned     |
| LAB 10 | Graph & Multimodal GeoAI          | Planned     |

The roadmap may evolve as the laboratories incorporate new datasets, methods, and evidence.

## Repository structure

```text
geoai-mineral-labs/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
├── labs/
├── notebooks/
├── references/
├── reports/
├── src/geoai/
└── tests/
```

Large, private, licensed, or sensitive datasets are intentionally excluded from version control.

## Validation philosophy

Geospatial machine-learning problems require validation strategies that account for spatial dependence.

Where applicable, the laboratories will compare conventional validation approaches with spatially structured alternatives and explicitly examine risks such as:

* spatial autocorrelation;
* spatial leakage;
* class imbalance;
* extrapolation outside the training domain;
* unreliable or incomplete negative labels;
* distribution shift;
* uncertainty in predictions.

Model evaluation will therefore extend beyond a single accuracy metric.

## Engineering and security

The repository is developed using a controlled Git workflow with:

* isolated Conda environments;
* Git version control;
* pre-commit validation;
* automated secret detection;
* signed commits;
* explicit `.gitignore` policies;
* reproducible configuration;
* progressive testing and quality controls.

Credentials, API keys, private keys, and private datasets must never be committed to the repository.

## Current development environment

Initial LAB 00 environment:

* Windows 11 x64
* Git
* GitHub CLI
* Conda
* Python 3.12
* VS Code
* pre-commit
* Gitleaks

Scientific, geospatial, machine-learning, and deep-learning dependencies will be introduced incrementally rather than installing the complete stack at once.

This allows each dependency to be associated with a concrete technical requirement.

## Project status

The repository is currently under active development.

LAB 00 establishes the engineering, reproducibility, security, and repository foundations required by the subsequent GeoAI laboratories.

The project should be interpreted as an evolving technical laboratory and portfolio rather than a production mineral exploration platform.
