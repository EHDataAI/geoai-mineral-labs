# LAB04 - Geochemical Compositional Data Analysis

## Overview

This laboratory demonstrates a reproducible workflow for compositional
data analysis applied to synthetic geochemical observations.

The experiment focuses on ten selected elements:

- Cu
- Mo
- Au
- As
- Sb
- Fe
- Mg
- Ca
- Al
- K

The dataset is fully synthetic and is designed for methodological
experimentation rather than geological inference about a real deposit.

## Scientific objective

The laboratory evaluates how conventional analysis of geochemical
concentrations differs from analysis performed in log-ratio geometry.

The workflow compares:

- raw concentrations;
- closed concentrations;
- centered log-ratio coordinates;
- isometric log-ratio coordinates;
- PCA under conventional and compositional representations;
- sensitivity to two simple below-detection-limit replacement rules.

## Synthetic experiment

The synthetic dataset contains 1,200 spatial samples.

The simulation includes:

- synthetic UTM coordinates;
- lithological regimes;
- mineralization gradients;
- geochemical pathfinder behavior;
- analytical censoring;
- missing observations;
- latent ground-truth variables for diagnostic validation.

The latent variables are used only to validate the simulation and must
not be interpreted as variables available in a real exploration survey.

## Geochemical data contract

The file `geochemistry_contract.json` is the authoritative LAB04 data
contract.

It defines:

- element ordering;
- synthetic concentration unit convention;
- detection limits;
- coordinate reference system;
- censoring semantics;
- missing-value semantics;
- closure interpretation;
- replacement strategies.

The synthetic concentration convention is ppm.

This is a modelling convention for the experiment and does not represent
certified laboratory measurements.

## Compositional scope

The ten selected elements are treated as a geochemical subcomposition.

Closure to 100 is used as relative normalization only.

It must not be interpreted as a complete whole-rock mass balance or as
a statement that the ten measured elements constitute 100 percent of a
rock sample.

## Detection limits

Observed values below their synthetic detection limit are encoded as
zero.

Missing observations are stored separately as NaN.

The primary analytical baseline replaces censored observations using:

```text
LOD / 2
