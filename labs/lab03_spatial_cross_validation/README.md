# LAB03 — Spatial Cross-Validation

## Research question

How much optimism can conventional random cross-validation introduce when training and validation observations remain spatially close?

## Objective

This laboratory evaluates how the estimated predictive error changes when a spatially autocorrelated synthetic dataset is validated using two different cross-validation strategies:

- conventional Random K-Fold cross-validation;
- Spatial Block Cross-Validation.

The experiment intentionally keeps the dataset, features, target, model, and model hyperparameters fixed. The purpose is not to maximize predictive performance, but to isolate the effect of the validation design.

## Hypothesis

When nearby observations share spatial structure, conventional random cross-validation may place geographically close samples in both training and validation sets.

For a local prediction model, this can produce a more optimistic estimate of out-of-sample error than a validation strategy that enforces stronger spatial separation between training and validation observations.

This laboratory tests that hypothesis under a controlled synthetic setting. It does not assume that the magnitude of the effect observed here generalizes to other datasets, models, spatial processes, or block definitions.

## Dataset

LAB03 reuses the synthetic spatial sampling generator developed in the project:

`generate_synthetic_sampling(seed=12345)`

Source:

`src/geoai/spatial.py`

The dataset contains 800 observations:

- 500 regional samples;
- 300 campaign samples.

Coordinate reference system:

`EPSG:32719`

Approximate spatial extent:

- Easting: 350000–430000 m;
- Northing: 7100000–7180000 m.

### Prediction target

The observed prediction target is:

`signal`

The synthetic latent field:

`signal_true`

is retained only as a diagnostic reference and is not supplied to the predictive model.

### Model features

The model uses only:

- `Easting`;
- `Northing`.

This simple feature definition is deliberate. LAB03 studies validation design rather than feature engineering.

## Fixed predictive model

The experiment uses a fixed `KNeighborsRegressor` configuration:

- `n_neighbors = 15`;
- `weights = "distance"`;
- `metric = "euclidean"`.

KNN was selected as a deliberately local baseline because its predictions depend strongly on nearby observations.

No hyperparameter tuning or model selection is performed inside this laboratory.

## Validation strategies

### Random K-Fold

Configuration:

- 5 folds;
- random seed: `12345`.

Random K-Fold does not explicitly enforce spatial separation between training and validation observations.

### Spatial Block Cross-Validation

Three spatial block sizes are evaluated:

- 5 km;
- 10 km;
- 20 km.

Entire spatial blocks are assigned to folds, so observations belonging to the same block are not split between training and validation within a spatial CV scenario.

## Primary scenario: 5 km spatial blocks

The 5 km configuration is the primary inferential scenario.

It provides a useful balance between increased train-validation spatial separation and sufficient spatial support within each validation fold.

Diagnostics showed:

- 224 populated spatial blocks;
- approximately 37–48 validation blocks per fold;
- exactly 160 validation observations per fold;
- adequate variation of the latent synthetic signal across folds;
- positive spatial-fold R² values.

For these reasons, the 5 km result is used for the principal comparison with Random K-Fold.

## Aggregate results

| Validation design | Pooled RMSE | Pooled MAE | Median test-to-train separation |
| --- | ---: | ---: | ---: |
| Random K-Fold | 3.9676 | 2.8067 | 1.11 km |
| Spatial Block CV — 5 km | 4.2610 | 2.9618 | 2.39 km |

For the primary 5 km scenario:

- absolute RMSE gap: `0.2934`;
- relative RMSE optimism: `6.89%`;
- mean 5 km group overlap under Random K-Fold: `73.6`;
- mean 5 km group overlap under Spatial Block CV: `0.0`.

The term relative RMSE optimism describes the difference observed in this controlled experiment between the lower Random K-Fold error estimate and the corresponding spatial validation error estimate.

It must not be interpreted as a universal correction factor.

## Sensitivity analysis

Larger blocks increase spatial separation, but they also reduce the number of independent spatial groups available for constructing balanced folds.

| Scenario | Random pooled RMSE | Spatial pooled RMSE | Relative RMSE optimism | Spatial median separation |
| --- | ---: | ---: | ---: | ---: |
| 5 km — primary | 3.9676 | 4.2610 | 6.89% | 2.39 km |
| 10 km — sensitivity | 3.9676 | 5.0866 | 22.00% | 3.19 km |
| 20 km — stress test | 3.9676 | 5.2320 | 24.17% | 5.41 km |

### 10 km blocks

The 10 km scenario contains 64 populated blocks.

Some validation folds have limited spatial support, including a fold with only one validation block and very low within-fold variation in the latent synthetic signal.

Consequently, this configuration is treated as a sensitivity analysis rather than the primary estimate.

### 20 km blocks

The 20 km configuration contains only 16 populated blocks.

At this scale, some folds are effectively dominated by very few spatial regions and validation sample sizes become substantially less balanced.

The 20 km scenario therefore acts as a stress test of stronger spatial separation rather than as the principal validation design.

## Interpretation of R²

RMSE and MAE are the primary evaluation metrics in LAB03.

R² is retained as a complementary diagnostic.

For spatial folds with very low within-fold target variance, R² can become unstable or negative even when absolute prediction errors remain interpretable.

A negative spatial-fold R² should therefore not automatically be interpreted as evidence that spatial validation itself has failed.

Instead, it must be assessed together with:

- validation-fold sample size;
- target variance;
- number of spatial groups;
- train-validation separation;
- RMSE;
- MAE.

This is one reason why the 5 km scenario is preferred for the primary comparison.

## Scientific interpretation

In this controlled synthetic experiment with a fixed local KNN model, Random K-Fold produced a more optimistic estimate of predictive error than Spatial Block Cross-Validation.

Increasing spatial separation between training and validation observations increased the estimated prediction error.

However, larger spatial blocks also reduced the spatial support available within individual folds and increased metric instability.

The result therefore does not imply that:

- Random K-Fold always overestimates performance by 6.89%;
- 5 km is an optimal spatial block size for other datasets;
- Spatial Block CV is universally superior to every other validation strategy;
- the 10 km and 20 km differences represent unbiased estimates of real-world deployment error.

Instead, LAB03 demonstrates that validation geometry can materially affect estimated model performance when observations and predictions are spatially structured.

## Limitations

This laboratory has several intentional limitations.

1. The dataset is synthetic.
2. Only one synthetic realization and fixed random seed are used.
3. The predictive model is a single local KNN baseline.
4. Only coordinate features are supplied to the model.
5. Spatial block size is evaluated at three predefined scales.
6. Larger block sizes reduce the number of populated spatial groups and can produce unstable folds.
7. The experiment does not reproduce the geological, geochemical, geophysical, sampling, censoring, and measurement complexities of a real mineral exploration program.
8. The observed error differences are experiment-specific and must not be generalized as universal spatial-CV correction factors.

These limitations are deliberate because the objective is methodological isolation rather than production-grade mineral prospectivity modelling.

## Reproducibility

The experiment uses deterministic seeds where randomness is involved.

Core implementation:

`src/geoai/validation.py`

Validation tests:

`tests/test_validation.py`

LAB03 workflow:

`labs/lab03_spatial_cross_validation/03a_run_spatial_cv_experiment.py`

`labs/lab03_spatial_cross_validation/03b_summarize_cv_results.py`

`labs/lab03_spatial_cross_validation/03c_build_notebook.py`

`labs/lab03_spatial_cross_validation/03d_execute_notebook.py`

Executed notebook:

`labs/lab03_spatial_cross_validation/03_spatial_cross_validation.ipynb`

### Persisted results

The laboratory persists its analytical outputs under:

`labs/lab03_spatial_cross_validation/results/`

Generated files include:

- `experiment_metadata.json`;
- `fold_results.csv`;
- `strategy_summary.csv`;
- `validation_comparison.csv`;
- `aggregate_metrics.csv`;
- `aggregate_comparison.csv`;
- `scenario_diagnostics.csv`.

The experiment, aggregation pipeline, unit tests, and executed notebook were validated before finalizing this README.

## Validation status

Validated LAB03 checkpoints include:

- LAB03 experiment: PASS;
- LAB03 aggregation: PASS;
- LAB03 notebook build: PASS;
- LAB03 notebook execution: PASS;
- executed notebook code cells: 14/14;
- notebook error outputs: 0;
- notebook PNG outputs: 4;
- LAB03 notebook QA: PASS;
- LAB03-specific tests: 7 passed;
- complete project test suite at the validated checkpoint: 31 passed.

The notebook is already a validated artifact and should not be rebuilt solely because of README changes.

## Publication boundary

The private repository remains the canonical scientific and engineering implementation.

A curated public version of LAB03 may subsequently be published because this laboratory:

- uses synthetic data;
- demonstrates generic spatial-validation principles;
- contains no real mineral exploration data;
- does not expose competitive mineral prospectivity ranking logic;
- does not contain proprietary feature engineering.

Any public release must still undergo a separate publication review before transfer to the public `Mining Geospatial Analytics` repository.

## Key takeaway

Spatial validation is not only a change in cross-validation syntax.

It changes the geometry of the prediction problem being evaluated.

In LAB03, enforcing spatial separation exposed a larger estimated prediction error than conventional Random K-Fold, while excessively large blocks reduced validation support and increased instability.

The central methodological lesson is therefore to choose spatial validation structures that represent the intended prediction task while retaining sufficient spatial support for meaningful evaluation.
