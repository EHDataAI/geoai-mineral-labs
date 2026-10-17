# LABORATORY 06 - Explainable GeoAI

**Status: Phase A - Contract and feasibility audit**

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
