# LABORATORY 06 - Explainable GeoAI

**Status: Completed - Phase F executed notebook and publication QA**

## Purpose

LABORATORY 06 extends the frozen mineral-prospectivity workflow from
LABORATORY 05 into explainable GeoAI.

The laboratory does not reselect the predictive model. It inherits the
frozen operational configuration from LABORATORY 05:

- Feature Set A;
- Logistic Regression;
- spatially validated model-selection evidence;
- out-of-sample candidate-ranking semantics.

The first objective is to prove that the explanatory workflow can be
constructed without changing the frozen model contract or crossing the
sealed-oracle boundary.

## Research questions

1. Which geological, structural, geophysical, and alteration variables
   drive the frozen prospectivity ranking?
2. Are global explanations consistent with the fitted linear model?
3. How do local feature contributions vary across candidate cells?
4. Are explanation patterns spatially coherent?
5. Are feature-importance rankings stable across spatially valid
   resamples and background-sampling replicates?

## Planned explanation layers

The planned workflow is deliberately layered.

1. Baseline interpretability
   - transformed Logistic Regression coefficients;
   - coefficient sign and magnitude;
   - feature-name reconstruction after preprocessing.

2. Model-agnostic importance
   - permutation importance on held-out / out-of-fold evidence;
   - repeated estimates and uncertainty summaries.

3. SHAP
   - global mean absolute SHAP;
   - beeswarm summaries;
   - selected local explanations;
   - spatial maps of feature contributions.

4. Response analysis
   - partial dependence where assumptions are defensible;
   - ICE or ALE-style diagnostics when needed;
   - explicit correlation checks before interpretation.

5. Spatial explainability
   - dominant explanatory driver per cell;
   - positive and negative contribution maps;
   - comparison of explanation regimes across candidate tiers.

6. Explanation stability
   - ranking stability across spatial folds / replicates;
   - sign stability;
   - top-k overlap;
   - sensitivity to correlated predictors.

## Methodological constraints

LABORATORY 06 must not:

- reselect the LABORATORY 05 operational model using explanation output;
- use hidden-oracle variables as explanatory model inputs;
- use the hidden-oracle evaluation to tune explanation settings;
- interpret SHAP, coefficients, or permutation importance as causal effects;
- claim that prospectivity scores are calibrated deposit probabilities;
- explain in-sample predictions as if they were independent validation evidence.

The hidden oracle remains evaluation-only evidence from LABORATORY 05.

## Phase A

Phase A is a contract and feasibility audit.

It validates:

- the frozen Feature Set A definition;
- the frozen Logistic Regression model identity;
- the deterministic LABORATORY 05 seed and model construction;
- the nine safe Feature Set A columns;
- exclusion of protected and oracle variables;
- 6,400 evidence rows and 100 known positives;
- availability of the LABORATORY 05 OOS model manifest;
- availability of the current Python / scikit-learn runtime;
- presence or absence of SHAP in the QA environment.

SHAP availability is recorded, not silently installed. Dependency changes
will be made only after the audit establishes the current environment.

## Expected Phase A outputs

- `06a_audit_lab05_contract.py`;
- `results/phase_a_audit.json`;
- `tests/test_lab06_contract.py`;
- this README.

## Planned implementation sequence

- Phase A: contract and feasibility audit.
- Phase B: deterministic frozen-model reconstruction and baseline
  interpretability.
- Phase C: global SHAP and permutation importance.
- Phase D: local and spatial explanations.
- Phase E: explanation stability and adversarial interpretation checks.
- Phase F: executed public notebook and documentation integration.

## Status gate

Phase A is ready only when the audit reports:

```text
LAB05_CONTRACT_AUDIT = PASS
FROZEN_FEATURE_SET_A = PASS
MODEL_RECONSTRUCTION_CONTRACT = PASS
ORACLE_ISOLATION = PASS
SPATIAL_GROUP_CONTRACT = PASS
SHAP_ENVIRONMENT_AUDIT = PASS
LAB06_PHASE_A = READY
```

`SHAP_ENVIRONMENT_AUDIT = PASS` means the environment was inspected
successfully. The manifest records whether SHAP is currently installed.
## Phase B - Frozen model reconstruction and baseline interpretability

Phase B reconstructs the frozen LABORATORY 05 operational ensemble without
changing model selection, feature selection, or oracle boundaries.

The phase verifies that the ten deterministic background-replicate Logistic
Regression models reproduce the persisted out-of-sample ranking evidence
within a strict numerical tolerance.

It also extracts the transformed feature space after preprocessing and
summarizes Logistic Regression coefficients across the ten deterministic
background replicates.

Phase B artifacts:

- `results/model_reconstruction_manifest.json`;
- `results/reconstructed_oos_scores.csv`;
- `results/transformed_coefficients.csv`;
- `results/coefficient_summary.csv`;
- `results/baseline_interpretability_manifest.json`.

The coefficient analysis is an interpretability baseline, not a causal
analysis. Numeric predictors are standardized inside the fitted pipeline and
lithology is one-hot encoded.

SHAP is intentionally deferred to Phase C so dependency changes do not occur
before deterministic model reconstruction has been demonstrated.

## Phase C - Global explainability

Phase C adds spatial held-out permutation importance and OOS-aware global SHAP
without changing the frozen LABORATORY 05 model or opening the hidden oracle.

Permutation importance uses the original nine Feature Set A variables on the
same spatial validation design. SHAP explains only candidate cells that were
out of sample for each operational background-replicate model.

SHAP values are evaluated in Logistic Regression log-odds space and must pass
a numerical additivity check. Coefficients, permutation importance, and SHAP
are treated as complementary evidence because correlated predictors can share
or redistribute attribution. None of these methods is interpreted causally.
## Phase D - Local and spatial explainability

Phase D extends the frozen OOS explanation contract from global attribution to
candidate-specific and spatial interpretation.

Five representative local cases are selected only by predeclared candidate
rank rules: rank 1, the top-1% boundary, the top-5% boundary, the median
candidate, and rank 6300. No local example is selected because it produces a
visually convenient explanation.

For every candidate cell, SHAP contributions are aggregated only across the
background-replicate models for which that cell was out of sample. This keeps
the explanation population aligned with the operational LABORATORY 05
prospectivity score.

The spatial artifacts expose both the dominant raw feature and a higher-level
driver family: lithology, intrusive context, structure, geophysics, or
alteration. Dominance is descriptive rather than causal and is based on mean
absolute OOS SHAP contribution.
## Phase E - Explanation stability and method agreement

Phase E evaluates whether the frozen LABORATORY 06 explanations are robust
across the ten background-replicate models and whether the principal
importance rankings agree across complementary interpretation methods.

The stability analysis recomputes OOS SHAP attribution for every replicate
using the same 600-row replicate-specific training background and the same
5,800 eligible candidate cells used by the frozen LABORATORY 05 scoring
contract.

The analysis reports:

- feature-level SHAP rank dispersion across replicates;
- feature sign stability across replicates;
- pairwise top-3 and top-5 SHAP overlap;
- driver-family stability;
- Spearman rank agreement among grouped logistic coefficients, spatial
  held-out permutation importance, and global SHAP;
- top-k overlap among interpretation methods.

Agreement is treated as supporting evidence of interpretive robustness, not as
proof of causality. Disagreement is retained as scientific information rather
than normalized away.
## Phase F - Executed public notebook candidate

Phase F assembles the validated Phase A-E evidence into the executed public
`06_explainable_geoai.ipynb` notebook.

The notebook is intentionally presentation-grade rather than a raw collection
of diagnostic plots. It contains ten coordinated figures and four compact
publication-style tables spanning:

- the frozen explanation contract;
- coefficient, permutation, and SHAP global evidence;
- cross-method rank agreement;
- deterministic local explanations;
- spatial dominant-driver regimes;
- feature-specific spatial attribution;
- explanation stability across background replicates;
- scientific interpretation and publication boundaries.

The executed notebook must pass structural QA, embedded-output QA, Ruff on new
code, the full regression suite, pre-commit, Gitleaks, and the repository
large-file policy.

A passing automated QA result does not authorize immediate merge. Phase F also
requires a manual visual review of figure composition, text clipping, legends,
table rendering, notebook narrative flow, and GitHub rendering before the
laboratory is marked complete.

## Phase F completion and publication QA

Phase F completes the public explainability workflow by assembling the
validated Phase A-E evidence into the executed
`06_explainable_geoai.ipynb` notebook.

Publication QA for the executed notebook records:

- 16 code cells;
- 14 Markdown cells;
- 10 embedded PNG outputs;
- zero notebook error outputs;
- 23 targeted LABORATORY 06 tests passing;
- 186 repository regression tests passing;
- pre-commit validation passing;
- Gitleaks secret scanning passing;
- no model reselection;
- no hidden-oracle loading during explanation construction.

The final notebook preserves the frozen LABORATORY 05 operational contract.
Global, local, spatial, and stability explanations are descriptive
interpretation evidence rather than causal claims. Prospectivity scores remain
ranking scores and are not interpreted as calibrated deposit probabilities.

### Viewer note

The embedded PNG artifacts were independently extracted and verified as
correct. In the local VS Code environment, the
`ms-toolsai.jupyter-renderers@1.3.0` extension produced a display-only
rendering defect affecting figure margins and legend placement. Disabling that
extension restored the correct notebook presentation. This is an editor/viewer
issue and does not alter the notebook outputs, scientific results, or committed
artifacts.
