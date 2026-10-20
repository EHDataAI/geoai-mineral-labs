from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "labs"
    / "lab07_spatial_uncertainty"
    / "07a_audit_uncertainty_contract.py"
)

MANIFEST = (
    ROOT
    / "labs"
    / "lab07_spatial_uncertainty"
    / "results"
    / "phase_a_audit.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "lab07_phase_a",
        SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load LAB07 Phase A"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def fresh():
    return load_module().audit(ROOT)


def test_manifest_reproducible(fresh):
    persisted = json.loads(
        MANIFEST.read_text(
            encoding="utf-8",
        )
    )

    assert persisted == fresh


def test_ready(fresh):
    assert fresh["status"] == "PASS"
    assert fresh["ready_for_phase_b"] is True
    assert fresh["gates"]["LAB07_PHASE_A"] == "READY"


def test_frozen_model(fresh):
    item = fresh["frozen_model"]

    assert item["feature_set"] == "A"
    assert item["model"] == "logistic_regression"
    assert item["model_reselected"] is False
    assert len(item["feature_columns"]) == 9


def test_oos_contract(fresh):
    item = fresh["uncertainty_input"]

    assert item["candidate_rows"] == 6300
    assert item["background_replicates"] == 10
    assert item["oos_candidates_per_replicate"] == 5800
    assert item["oos_model_count_min"] == 4
    assert item["oos_model_count_max"] == 10


def test_score_semantics(fresh):
    value = fresh["uncertainty_input"]["score_semantics"]

    assert value == (
        "prospectivity_score_not_calibrated_"
        "deposit_probability"
    )


def test_oracle_boundary(fresh):
    item = fresh["oracle_boundary"]

    assert item["oracle_loaded"] is False
    assert item["oracle_used"] is False
    assert item["oracle_artifacts_read"] == []
    assert item["forbidden_columns_present"] == []


def test_lab06_crosscheck(fresh):
    item = fresh["lab06_crosscheck"]

    assert item["reconstruction_status"] == "PASS"

    assert (
        item["maximum_observed_difference"]
        <= item["absolute_tolerance"]
    )

    assert (
        item["explanation_stability_status"]
        == "PASS"
    )

    assert (
        item["spatial_explainability_status"]
        == "PASS"
    )
