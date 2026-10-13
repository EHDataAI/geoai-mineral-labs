# LAB05 - Mineral Prospectivity Mapping

**Status: Completed**

## Overview

LAB05 implements an end-to-end synthetic mineral prospectivity experiment with explicit controls against spatial leakage, oracle leakage, evaluation contamination, and source drift.

The laboratory integrates geological, structural, geophysical, alteration, and geochemical evidence into a leakage-aware spatial machine-learning workflow.

The experiment is methodological and uses synthetic data only.

## Research question

Can spatially validated machine learning integrate multiple synthetic mineral-system evidence layers to rank mineral prospectivity while minimizing spatial leakage and reducing the exploration search area?

## Experimental design

The synthetic study area contains 6,400 one-square-kilometre cells in EPSG:32719.

The frozen synthetic mineral system contains:

- 215 true synthetic deposits;
- 100 known positives available during model development;
- 115 hidden deposits reserved for final post-freeze evaluation.

Unlabelled cells are treated as background rather than confirmed negatives.

## Evidence configurations

Three frozen feature configurations are evaluated.

### Feature Set A

Geology, structure, geophysics, and alteration:

- lithology;
- distance to intrusive;
- intrusive affinity;
- distance to fault;
- distance to fault intersection;
- distance to contact;
- magnetic anomaly;
- gravity anomaly;
- alteration index.

### Feature Set B

Feature Set A plus ten raw geochemical variables: Cu, Mo, Au, As, Sb, Fe, Mg, Ca, Al, and K.

### Feature Set C

Feature Set A plus nine ILR-transformed compositional geochemical coordinates.

## Candidate models

The benchmark compares:

- Dummy Classifier;
- Logistic Regression;
- Random Forest Classifier.

Ten background-sampling replicates are used.

Each replicate combines 100 known positives and 500 sampled unlabelled background cells.

## Spatial validation

The primary validation design uses five-fold `StratifiedGroupKFold` with `spatial_block_id` as the grouping variable.

Train/test spatial-group overlap is zero.

Random stratified cross-validation is retained only as a diagnostic comparison and is not used for model selection.

The primary selection metric is mean pooled out-of-fold Average Precision under spatial validation.

ROC-AUC and top-fraction success metrics are secondary diagnostics.

## Leakage controls

LAB05 explicitly excludes the following fields from model features:

- `Easting`;
- `Northing`;
- `spatial_block_id`;
- `known_positive`;
- `true_deposit`;
- `latent_favourability`;
- `deposit_id`;
- `distance_to_known_deposit`;
- `true_deposit_probability`;
- `cell_id`.

Additional controls include:

- fold-local preprocessing through scikit-learn pipelines;
- zero train/test group overlap in primary spatial CV;
- exactly one pooled OOF prediction per evaluated observation;
- out-of-sample-only candidate scoring;
- oracle sealing until model and ranking freeze;
- no post-oracle model, feature, or threshold reselection;
- source-synchronization tests for generated evidence components.

## Spatial model selection

The frozen operational model is:

- Feature Set A;
- Logistic Regression.

Primary spatial pooled-OOF performance:

| Metric | Value |
| --- | ---: |
| Mean Average Precision | 0.393412 |
| Median Average Precision | 0.380985 |
| Mean ROC-AUC | 0.734202 |

Alternative configurations remain competitive.

The winning configuration is intentionally selected using the predeclared primary metric rather than post-hoc selection using median AP, ROC-AUC, model complexity, or another secondary criterion.

## Out-of-sample prospectivity ranking

After model freeze, the 6,300 unlabelled candidate cells are ranked using out-of-sample-only scores.

A candidate receives a score from a replicate only when that cell was not included in that replicate's sampled training background.

The resulting `prospectivity_score` is an ensemble ranking score, not a calibrated probability of mineral occurrence.

## Hidden-oracle evaluation

The synthetic oracle is opened only after model selection and candidate-ranking procedures are frozen.

Final hidden-oracle evaluation contains 215 true deposits, 100 known positives, and 115 hidden deposits.

Performance against the hidden deposits:

| Prioritized candidate area | Candidate cells | Hidden deposits captured | Hidden capture rate | Lift over random |
| --- | ---: | ---: | ---: | ---: |
| Top 1% | 63 | 6 / 115 | 5.22% | 5.22x |
| Top 5% | 315 | 25 / 115 | 21.74% | 4.35x |
| Top 10% | 630 | 44 / 115 | 38.26% | 3.83x |

Additional final metrics:

- hidden-oracle Average Precision: `0.056650`;
- hidden-oracle ROC-AUC: `0.709070`;
- median hidden-deposit rank: `1,117 / 6,300`.

The absolute Average Precision must be interpreted in the context of the low-prevalence hidden-deposit ranking problem.

The operational objective is spatial prioritization and search-space reduction rather than calibrated deposit detection.

## Reproducibility and evidence

The LAB05 workflow persists evidence for synthetic-system generation, plausibility audit, label and spatial-CV audit, evidence-layer construction, benchmark metrics, pooled OOF metrics, OOS full-grid scoring, and hidden-oracle evaluation.

Important versioned artifacts include:

- `prospectivity_contract.json`;
- `synthetic_mineral_system.csv`;
- `synthetic_system_summary.json`;
- `synthetic_plausibility_manifest.json`;
- `spatial_cv_audit_summary.json`;
- `evidence_manifest.json`;
- `benchmark_oof_manifest.json`;
- `benchmark_oof_metrics.csv`;
- `benchmark_oof_aggregate_metrics.csv`;
- `prospectivity_model_manifest_oos.json`;
- `prospectivity_ranking_summary_oos.csv`;
- `prospectivity_scores_oos.csv`;
- `oracle_evaluation_manifest.json`;
- `oracle_evaluation_summary.csv`;
- `oracle_hidden_deposit_ranks.csv`.

The complete project test suite at the v0.2.0 documentation checkpoint contains:

```text
163 passed
```

## Interpretation

The most complex model did not win the predeclared selection criterion.

Feature Set A with Logistic Regression slightly outperformed the alternatives in mean spatial pooled-OOF Average Precision. That frozen model subsequently produced useful enrichment of hidden deposits while examining only a small fraction of the candidate area.

The experiment therefore emphasizes validation design, leakage prevention, reproducibility, and prospectivity ranking rather than maximizing a single headline metric.

## Limitations

Important limitations include:

1. the mineral system is synthetic;
2. background cells are unlabelled, not confirmed negatives;
3. the study area and occurrence process are controlled simulation constructs;
4. model scores are ranking scores rather than calibrated deposit probabilities;
5. hidden-oracle results quantify performance only under this frozen synthetic DGP;
6. the experiment does not establish geological conclusions about a real exploration property;
7. random cross-validation is only a diagnostic comparator;
8. production data drift and geographic distribution shift are not evaluated in this laboratory.

## Key takeaway

Mineral prospectivity modelling is not only a model-selection problem.

LAB05 demonstrates an end-to-end workflow in which spatial validation, leakage control, compositional evidence, out-of-fold evaluation, out-of-sample ranking, model freeze, and hidden-oracle evaluation are treated as part of the scientific design.

Under the frozen synthetic experiment, the selected model achieved useful hidden-deposit enrichment while prioritizing substantially less than the full candidate area.
