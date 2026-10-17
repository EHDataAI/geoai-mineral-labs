"""Contract tests for LABORATORY 06 Phase A."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB05 = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB06 = ROOT / "labs" / "lab06_explainable_geoai"

AUDIT_SCRIPT = LAB06 / "06a_audit_lab05_contract.py"
AUDIT_RESULT = LAB06 / "results" / "phase_a_audit.json"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "lab06_phase_a_audit",
        AUDIT_SCRIPT,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lab06_phase_a_feature_contract_is_frozen():
    module = load_audit_module()

    manifest = json.loads(
        (LAB05 / "results" / "evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    observed = manifest["feature_sets"]["A_non_geochemical"]

    assert observed == module.EXPECTED_FEATURE_SET_A
    assert len(observed) == 9
    assert not (set(observed) & module.PROTECTED_FIELDS)


def test_lab06_phase_a_model_manifest_preserves_oracle_boundary():
    manifest = json.loads(
        (
            LAB05
            / "results"
            / "prospectivity_model_manifest_oos.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["gate"] == "5.1"
    assert manifest["model_frozen"] is True
    assert manifest["primary_feature_set"] == "A"
    assert manifest["primary_model"] == "logistic_regression"
    assert manifest["oracle_values_loaded"] is False
    assert manifest["oracle_used_for_training"] is False
    assert manifest["oracle_used_for_scoring"] is False


def test_lab06_phase_a_audit_result_is_ready():
    assert AUDIT_RESULT.exists()

    audit = json.loads(AUDIT_RESULT.read_text(encoding="utf-8"))

    assert audit["laboratory"] == "LABORATORY 06"
    assert audit["phase"] == "A"
    assert audit["status"] == "READY"
    assert audit["lab05"]["primary_feature_set"] == "A"
    assert audit["lab05"]["primary_model"] == "logistic_regression"
    assert audit["lab05"]["oracle_loaded_for_phase_a"] is False

    required = {
        "LAB05_CONTRACT_AUDIT",
        "FROZEN_FEATURE_SET_A",
        "MODEL_RECONSTRUCTION_CONTRACT",
        "ORACLE_ISOLATION",
        "SPATIAL_GROUP_CONTRACT",
        "SHAP_ENVIRONMENT_AUDIT",
    }

    assert required.issubset(audit["checks"])
    assert all(audit["checks"][name] == "PASS" for name in required)
