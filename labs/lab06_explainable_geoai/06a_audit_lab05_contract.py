"""LABORATORY 06 Phase A: audit the frozen LABORATORY 05 contract."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[2]
LAB05_DIR = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB06_DIR = ROOT / "labs" / "lab06_explainable_geoai"
RESULTS_DIR = LAB06_DIR / "results"

CONTRACT_PATH = LAB05_DIR / "prospectivity_contract.json"
EVIDENCE_PATH = LAB05_DIR / "data" / "prospectivity_evidence.csv"
EVIDENCE_MANIFEST_PATH = LAB05_DIR / "results" / "evidence_manifest.json"
MODEL_MANIFEST_PATH = (
    LAB05_DIR / "results" / "prospectivity_model_manifest_oos.json"
)
OOF_MANIFEST_PATH = LAB05_DIR / "results" / "benchmark_oof_manifest.json"
GATE04_PATH = LAB05_DIR / "05d_run_spatial_benchmark.py"
OUTPUT_PATH = RESULTS_DIR / "phase_a_audit.json"

EXPECTED_FEATURE_SET_A = [
    "lithology",
    "distance_to_intrusive_m",
    "intrusive_affinity",
    "distance_to_fault_m",
    "distance_to_fault_intersection_m",
    "distance_to_contact_m",
    "magnetic_anomaly",
    "gravity_anomaly",
    "alteration_index",
]

EXPECTED_SEED = 12345
EXPECTED_ROWS = 6400
EXPECTED_KNOWN_POSITIVES = 100
EXPECTED_CRS = "EPSG:32719"
EXPECTED_PRIMARY_MODEL = "logistic_regression"
EXPECTED_FEATURE_SET = "A"
EXPECTED_SELECTION_METRIC = "mean_spatial_oof_average_precision"

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "deposit_id",
    "distance_to_known_deposit",
}

PROTECTED_FIELDS = {
    "Easting",
    "Northing",
    "spatial_block_id",
    "known_positive",
    "cell_id",
    *ORACLE_FIELDS,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate04_module():
    if str(LAB05_DIR) not in sys.path:
        sys.path.insert(0, str(LAB05_DIR))

    spec = importlib.util.spec_from_file_location(
        "lab05_gate04_for_lab06_audit",
        GATE04_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load LABORATORY 05 Gate 4 module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_contract(
    contract: dict[str, Any],
    evidence_manifest: dict[str, Any],
    model_manifest: dict[str, Any],
    oof_manifest: dict[str, Any],
) -> list[str]:
    checks: list[str] = []

    if contract["random_seed"] != EXPECTED_SEED:
        raise RuntimeError("Unexpected LABORATORY 05 random seed.")

    if contract["study_area"]["crs"] != EXPECTED_CRS:
        raise RuntimeError("Unexpected LABORATORY 05 CRS.")

    if contract["study_area"]["expected_grid_cells"] != EXPECTED_ROWS:
        raise RuntimeError("Unexpected LABORATORY 05 grid-cell count.")

    if contract["validation"]["primary_splitter"] != "StratifiedGroupKFold":
        raise RuntimeError("Unexpected spatial validation splitter.")

    if contract["validation"]["group_column"] != "spatial_block_id":
        raise RuntimeError("Unexpected spatial grouping column.")

    if contract["validation"]["require_zero_group_overlap"] is not True:
        raise RuntimeError("Spatial overlap protection is not enabled.")

    feature_set_a = evidence_manifest["feature_sets"]["A_non_geochemical"]

    if feature_set_a != EXPECTED_FEATURE_SET_A:
        raise RuntimeError("Frozen Feature Set A has changed.")

    forbidden = set(evidence_manifest["forbidden_model_features"])

    if not PROTECTED_FIELDS.issubset(forbidden):
        missing = sorted(PROTECTED_FIELDS - forbidden)
        raise RuntimeError(
            f"Protected fields are missing from the forbidden set: {missing}"
        )

    leakage = sorted(set(feature_set_a) & forbidden)

    if leakage:
        raise RuntimeError(f"Feature Set A contains forbidden columns: {leakage}")

    if set(feature_set_a) & ORACLE_FIELDS:
        raise RuntimeError("Oracle leakage detected in Feature Set A.")

    if model_manifest.get("gate") != "5.1":
        raise RuntimeError("Expected LABORATORY 05 Gate 5.1 model manifest.")

    if model_manifest.get("model_frozen") is not True:
        raise RuntimeError("LABORATORY 05 operational model is not frozen.")

    if model_manifest.get("primary_feature_set") != EXPECTED_FEATURE_SET:
        raise RuntimeError("Unexpected frozen feature set.")

    if model_manifest.get("primary_model") != EXPECTED_PRIMARY_MODEL:
        raise RuntimeError("Unexpected frozen primary model.")

    if model_manifest.get("selection_metric") != EXPECTED_SELECTION_METRIC:
        raise RuntimeError("Unexpected frozen selection metric.")

    if model_manifest.get("oracle_values_loaded") is not False:
        raise RuntimeError("Gate 5.1 oracle boundary was not preserved.")

    if model_manifest.get("oracle_used_for_training") is not False:
        raise RuntimeError("Oracle was used for training.")

    if model_manifest.get("oracle_used_for_scoring") is not False:
        raise RuntimeError("Oracle was used for candidate scoring.")

    if oof_manifest.get("selection_metric") != EXPECTED_SELECTION_METRIC:
        raise RuntimeError("OOF model-selection metric drift detected.")

    checks.extend(
        [
            "LAB05_CONTRACT_AUDIT",
            "FROZEN_FEATURE_SET_A",
            "ORACLE_ISOLATION",
            "SPATIAL_GROUP_CONTRACT",
        ]
    )

    return checks


def validate_evidence(feature_columns: list[str]) -> dict[str, Any]:
    usecols = [
        "cell_id",
        "spatial_block_id",
        "known_positive",
        *feature_columns,
    ]

    frame = pd.read_csv(EVIDENCE_PATH, usecols=usecols)

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError("Unexpected evidence row count.")

    if frame["cell_id"].duplicated().any():
        raise RuntimeError("cell_id must be unique.")

    if int(frame["known_positive"].sum()) != EXPECTED_KNOWN_POSITIVES:
        raise RuntimeError("Unexpected known-positive count.")

    if frame[feature_columns].isna().any().any():
        missing = frame[feature_columns].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise RuntimeError(f"Missing values in Feature Set A: {missing}")

    return {
        "rows": int(len(frame)),
        "known_positives": int(frame["known_positive"].sum()),
        "spatial_groups": int(frame["spatial_block_id"].nunique()),
        "feature_columns": feature_columns,
    }


def validate_model_builder(gate04, feature_columns: list[str]) -> dict[str, Any]:
    pipeline = gate04.build_model_pipeline(
        EXPECTED_PRIMARY_MODEL,
        feature_columns,
        EXPECTED_SEED,
    )

    if list(pipeline.named_steps) != ["preprocess", "model"]:
        raise RuntimeError("Unexpected LABORATORY 05 pipeline structure.")

    estimator = pipeline.named_steps["model"]

    if estimator.__class__.__name__ != "LogisticRegression":
        raise RuntimeError("Frozen estimator is not LogisticRegression.")

    if estimator.solver != "liblinear":
        raise RuntimeError("Unexpected LogisticRegression solver.")

    if estimator.max_iter != 2000:
        raise RuntimeError("Unexpected LogisticRegression max_iter.")

    if estimator.random_state != EXPECTED_SEED:
        raise RuntimeError("Unexpected LogisticRegression random_state.")

    preprocessor = pipeline.named_steps["preprocess"]

    names = [name for name, _, _ in preprocessor.transformers]

    if names != ["categorical", "numerical"]:
        raise RuntimeError("Unexpected preprocessing transformer structure.")

    return {
        "estimator": estimator.__class__.__name__,
        "solver": estimator.solver,
        "max_iter": int(estimator.max_iter),
        "random_state": int(estimator.random_state),
        "pipeline_steps": list(pipeline.named_steps),
        "preprocessor_transformers": names,
    }


def main() -> int:
    required_paths = [
        CONTRACT_PATH,
        EVIDENCE_PATH,
        EVIDENCE_MANIFEST_PATH,
        MODEL_MANIFEST_PATH,
        OOF_MANIFEST_PATH,
        GATE04_PATH,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise RuntimeError(f"Required LABORATORY 05 inputs missing: {missing}")

    contract = load_json(CONTRACT_PATH)
    evidence_manifest = load_json(EVIDENCE_MANIFEST_PATH)
    model_manifest = load_json(MODEL_MANIFEST_PATH)
    oof_manifest = load_json(OOF_MANIFEST_PATH)

    checks = validate_contract(
        contract,
        evidence_manifest,
        model_manifest,
        oof_manifest,
    )

    feature_columns = list(
        evidence_manifest["feature_sets"]["A_non_geochemical"]
    )

    evidence_summary = validate_evidence(feature_columns)

    gate04 = load_gate04_module()
    model_summary = validate_model_builder(gate04, feature_columns)

    checks.append("MODEL_RECONSTRUCTION_CONTRACT")

    shap_version = package_version("shap")
    shap_installed = shap_version is not None

    runtime = {
        "python": sys.version.split()[0],
        "pandas": package_version("pandas"),
        "scikit_learn": sklearn.__version__,
        "shap_installed": shap_installed,
        "shap_version": shap_version,
    }

    checks.append("SHAP_ENVIRONMENT_AUDIT")

    audit = {
        "laboratory": "LABORATORY 06",
        "phase": "A",
        "purpose": "lab05_contract_and_xai_feasibility_audit",
        "status": "READY",
        "lab05": {
            "random_seed": EXPECTED_SEED,
            "crs": EXPECTED_CRS,
            "grid_cells": EXPECTED_ROWS,
            "primary_feature_set": EXPECTED_FEATURE_SET,
            "primary_model": EXPECTED_PRIMARY_MODEL,
            "selection_metric": EXPECTED_SELECTION_METRIC,
            "oracle_loaded_for_phase_a": False,
        },
        "evidence": evidence_summary,
        "model_contract": model_summary,
        "runtime": runtime,
        "planned_xai_methods": [
            "standardized_logistic_regression_coefficients",
            "permutation_importance",
            "shap_linear_explainer",
            "local_shap_explanations",
            "spatial_shap_maps",
            "explanation_stability",
        ],
        "interpretation_constraints": [
            "association_not_causation",
            "no_model_reselection_from_explanations",
            "no_hidden_oracle_tuning",
            "prospectivity_score_not_calibrated_probability",
            "correlated_predictor_attribution_requires_caution",
        ],
        "checks": {name: "PASS" for name in checks},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for name in checks:
        print(f"{name} = PASS")

    print(f"SHAP_INSTALLED = {'YES' if shap_installed else 'NO'}")

    if shap_version is not None:
        print(f"SHAP_VERSION = {shap_version}")

    print("LAB06_PHASE_A = READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
