"""Phase E tests for LABORATORY 06."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "labs" / "lab06_explainable_geoai" / "results"


def test_phase_e_replicate_stability_contract():
    frame = pd.read_csv(RESULTS / "shap_replicate_stability.csv")

    assert len(frame) == 90
    assert frame["replicate"].nunique() == 10
    assert frame["feature"].nunique() == 9
    assert frame.groupby("replicate")["feature"].nunique().eq(9).all()
    assert frame["oos_candidate_count"].eq(5800).all()
    assert frame["rank"].between(1, 9).all()


def test_phase_e_summary_and_family_contract():
    stability = pd.read_csv(
        RESULTS / "explanation_stability_summary.csv"
    )
    family = pd.read_csv(
        RESULTS / "driver_family_stability.csv"
    )

    assert len(stability) == 9
    assert stability["feature"].nunique() == 9
    assert stability["sign_stability"].between(0.5, 1.0).all()
    assert stability["stable_top5_fraction"].between(0.0, 1.0).all()

    assert len(family) == 5
    assert family["driver_family"].nunique() == 5
    assert family["family_rank_mean"].between(1.0, 5.0).all()
    assert family["family_sign_stability"].between(0.5, 1.0).all()


def test_phase_e_method_agreement_contract():
    ranks = pd.read_csv(
        RESULTS / "explanation_rank_agreement.csv"
    )
    methods = pd.read_csv(
        RESULTS / "explanation_method_agreement.csv"
    )

    assert len(ranks) == 9
    assert ranks["feature"].nunique() == 9

    for column in (
        "coefficient_rank",
        "permutation_rank",
        "shap_rank",
    ):
        assert ranks[column].between(1, 9).all()

    assert len(methods) == 3
    assert methods["spearman_rank_correlation"].between(
        -1.0,
        1.0,
    ).all()
    assert methods["top3_jaccard"].between(0.0, 1.0).all()
    assert methods["top5_jaccard"].between(0.0, 1.0).all()


def test_phase_e_manifest_contract():
    manifest = json.loads(
        (
            RESULTS / "explanation_stability_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "PASS"
    assert manifest["phase"] == "E"
    assert manifest["background_replicates"] == 10
    assert manifest["raw_feature_count"] == 9
    assert manifest["total_oos_explanations"] == 58000
    assert (
        manifest["maximum_additivity_error"]
        <= manifest["additivity_tolerance"]
    )
    assert manifest["method_agreement_pairs"] == 3
    assert manifest["driver_family_count"] == 5
    assert manifest["oracle_loaded"] is False
    assert manifest["model_reselected"] is False
    assert manifest["causal_interpretation_allowed"] is False
