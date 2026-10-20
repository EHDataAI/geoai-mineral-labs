# LABORATORY 07 - Spatial Uncertainty Quantification

**Status: Phase A - Frozen upstream uncertainty contract**

## Purpose

LABORATORY 07 extends the frozen LABORATORY 05 prospectivity workflow and
LABORATORY 06 explainability workflow into spatial uncertainty quantification.

```text
LAB05 -> Predict
LAB06 -> Explain
LAB07 -> Quantify uncertainty
```

The laboratory does not reselect the predictive model. Prospectivity scores
remain ranking scores, not calibrated probabilities of mineral occurrence.

## Phase A

Phase A is a fail-closed upstream contract audit. It validates the frozen
Feature Set A plus Logistic Regression model, 6,300 candidate cells, ten
background replicates, 5,800 OOS candidates per replicate, cell-specific OOS
support from four to ten models, upstream uncertainty fields, LAB06
reconstruction tolerance, and oracle isolation.

LAB05 is the canonical prediction evidence. LAB06 provides an independent
deterministic reconstruction cross-check.

## Oracle boundary

Phase A does not read oracle-evaluation artifacts. Oracle variables remain
unavailable to uncertainty construction.

## Planned phases

- Phase A: frozen upstream uncertainty contract.
- Phase B: candidate-level OOS score uncertainty.
- Phase C: rank and top-k stability.
- Phase D: spatial uncertainty regimes.
- Phase E: decision stability.
- Phase F: prediction uncertainty versus explanation stability.
- Phase G: sensitivity and adversarial uncertainty audit.
- Phase H: executed public notebook and publication QA.

## Phase A gate

```text
LAB05_FROZEN_MODEL = PASS
FEATURE_SET_A_CONTRACT = PASS
CANDIDATE_POPULATION = PASS
OOS_ENSEMBLE_CONTRACT = PASS
LAB06_RECONSTRUCTION = PASS
UNCERTAINTY_INPUT_CONTRACT = PASS
ORACLE_ISOLATION = PASS
SCORE_SEMANTICS = PASS
LAB07_PHASE_A = READY
```

No model reselection, probability calibration, threshold optimization, or
hidden-oracle tuning is permitted in Phase A.
