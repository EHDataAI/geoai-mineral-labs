## Final validated workflow

LAB05 implements an end-to-end synthetic mineral prospectivity experiment with explicit controls against spatial leakage, oracle leakage, evaluation contamination, and source drift.

### Experimental design

The synthetic study area contains 6,400 one-kilometre cells in EPSG:32719. The frozen DGP contains 215 true synthetic deposits. Exactly 100 deposits are exposed as known positives for model development, while 115 remain hidden until final post-freeze evaluation.

Unlabelled cells are treated as background, not as confirmed negatives.

Three evidence configurations are evaluated:

- A: geology, structure, geophysics, and alteration.
- B: Feature Set A plus raw geochemistry.
- C: Feature Set A plus ILR-transformed compositional geochemistry.

Dummy, Logistic Regression, and Random Forest models are compared using ten background-sampling replicates.

### Spatial model selection

The primary validation design uses five-fold StratifiedGroupKFold with spatial block grouping and zero train/test group overlap.

Random cross-validation is retained only as a diagnostic comparison and is not used for model selection.

Model selection is based on mean pooled out-of-fold Average Precision under spatial validation.

The frozen operational model is:

- Feature set: A.
- Model: Logistic Regression.
- Mean spatial pooled-OOF Average Precision: 0.393412.
- Median spatial pooled-OOF Average Precision: 0.380985.
- Mean spatial ROC-AUC: 0.734202.

The winning configuration is intentionally selected using the predeclared primary metric even though alternative configurations are close and some show higher median AP or ROC-AUC.

### OOS prospectivity ranking

After model freeze, the 6,300 unlabelled candidate cells are ranked using out-of-sample-only scores.

A candidate receives a score from a replicate only when that cell was not part of the replicate's sampled training background.

OOS model support ranges from 4 to 10 models per candidate, with a mean of 9.206349.

The prospectivity score is a ranking score and must not be interpreted as a calibrated deposit probability.

### Post-freeze hidden-oracle evaluation

The synthetic oracle is opened only after model and ranking freeze.

Final evaluation contains:

- 215 true deposits.
- 100 known positives.
- 115 hidden deposits.

Performance against the hidden deposits:

| Prioritized area | Candidate cells | Hidden deposits captured | Hidden capture rate | Lift over random |
| --- | ---: | ---: | ---: | ---: |
| Top 1% | 63 | 6 / 115 | 5.22% | 5.22x |
| Top 5% | 315 | 25 / 115 | 21.74% | 4.35x |
| Top 10% | 630 | 44 / 115 | 38.26% | 3.83x |

Final hidden-oracle metrics:

- Average Precision: 0.056650.
- ROC-AUC: 0.709070.
- Median hidden-deposit rank: 1,117 of 6,300 candidates.

These results support interpretation of the model as a spatial prioritization and search-space reduction tool rather than a calibrated deposit detector.

### Leakage and drift controls

LAB05 includes explicit controls for:

- forbidden oracle and identifier features;
- zero spatial group overlap during primary cross-validation;
- fold-local preprocessing through scikit-learn pipelines;
- exactly one pooled OOF prediction per evaluated observation;
- OOS-only candidate scoring;
- model freeze before opening the hidden oracle;
- no model, feature, or threshold selection after oracle access;
- source synchronization between the Gate 3 builder and canonical files.

Data drift is not evaluated as a production phenomenon in this synthetic laboratory. A future extension could evaluate geographic or temporal distribution shift between development and deployment regions.

### Interpretation

The most complex model did not win the predeclared selection criterion.

Feature Set A with Logistic Regression slightly outperformed the alternatives in mean spatial pooled-OOF Average Precision. This simpler frozen model subsequently achieved useful enrichment of hidden deposits while examining only a small fraction of the candidate area.

The experiment therefore emphasizes validation design, leakage prevention, reproducibility, and prospectivity ranking rather than maximizing a single headline metric.
