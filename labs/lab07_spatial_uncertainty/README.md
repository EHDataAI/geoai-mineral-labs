# LABORATORY 07 - Spatial Uncertainty Quantification

**Status: Phases A-C finalized and published; Phase D repository QA passed; Phase E not authorized**

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

## Published evidence: Phases B and C

The [executed illustrated notebook](07_spatial_uncertainty_illustrated.ipynb)
presents the validated candidate-level uncertainty and ranking-stability
diagnostics.

- Phase B produces `results/score_uncertainty.csv`, describing conditional
  out-of-sample (OOS) score dispersion for candidate cells.
- Phase C produces `results/rank_stability.csv`, summarizing replicate-level
  rank stability and top-k inclusion frequencies.
- The analysis retains the frozen upstream model, feature contract, and
  oracle-exclusion boundary.

The frozen population contains 6,300 candidate cells. Each background
replicate contributes 5,800 eligible OOS candidates, while individual
candidate support ranges from four to ten replicates.

Because training sets overlap, replicate-level results must not be treated
as independent geological observations or independent inferential trials.
Top-5% inclusion frequencies and normalized ranking priorities are
relative diagnostics, not calibrated probabilities of mineral occurrence.

### Release boundary

Phases A-C are finalized. Phase D has passed the repository QA and
security review recorded in its release manifest. The Phase E
decision-stability gate remains closed. Later phases retain their
independent validation and publication requirements.

## Phase D: spatial uncertainty regimes

`07d_build_spatial_regimes.py` joins the frozen B/C tables by `cell_id`,
checks hashes and oracle exclusion, and constructs the observed 80 x 80
regular-grid topology. The 100 non-candidate positions remain masked.
The 6,300 candidate cells form one rook-connected component.

The primary spatial neighborhood shares a cell edge (rook); diagonal
neighbors (queen) provide a sensitivity comparison. Local mean score
dispersion and local mean priority dispersion exclude the focal cell.
Neighbor counts are exported so that boundary effects remain visible.
Global Moran I is reported descriptively for both weight definitions;
there are no permutation p-values or inferential confidence claims.

The categorical regime is a *descriptive, relative partition*: high score
and/or priority variability is defined by within-candidate percentiles.
The primary cutoff is q75 and q70/q80 form an explicit sensitivity analysis.

For reproducible numeric reconstruction, CSV export uses
`float_format="%.17g"`. Downstream validation should use
`pandas.read_csv(..., float_precision="round_trip")`: default parsing
can shift a near-boundary percentile by one floating-point ULP
(observed for cell 638 at q70), changing its thresholded regime label.
Labels are computed from full-precision values before serialization.
No weights combine score variability, priority variability and OOS support.
Support stays a separate reliability descriptor. Rook-connected regime
components are all retained, including single-cell components.

Phase D initially creates `results/spatial_uncertainty_regimes.csv` and
`results/phase_d_manifest.json` in `QA_PENDING` state. After the
independent release checks, the audited manifest records `PASS`;
`ready_for_phase_e` remains `false` and Phase E is not authorized.
The builder requires identical CSV bytes and verifies scientific
manifest fields and evidence hashes against existing artifacts.

## Phase status and roadmap

- Phase A: frozen upstream uncertainty contract (finalized).
- Phase B: candidate-level OOS score uncertainty (finalized).
- Phase C: rank and top-k stability (finalized).
- Phase D: spatial uncertainty regimes (QA PASS; Phase E not authorized).
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
