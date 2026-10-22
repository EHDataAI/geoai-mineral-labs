from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = ROOT / "labs" / "lab07_spatial_uncertainty" / "07b_build_score_uncertainty.py"

RESULT = (
    ROOT / "labs" / "lab07_spatial_uncertainty" / "results" / "score_uncertainty.csv"
)

MANIFEST = (
    ROOT / "labs" / "lab07_spatial_uncertainty" / "results" / "phase_b_manifest.json"
)

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "distance_to_known_deposit",
    "deposit_id",
}


def load_module():
    spec = importlib.util.spec_from_file_location(
        "lab07_phase_b",
        SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load LAB07 Phase B")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(RESULT)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_candidate_population(frame):
    assert len(frame) == 6300
    assert frame["cell_id"].is_unique


def test_oracle_fields_absent(frame):
    assert not ORACLE_FIELDS.intersection(frame.columns)


def test_support_contract(frame):
    assert frame["oos_model_count"].between(4, 10).all()

    assert frame["oos_support_fraction"].between(0.4, 1.0).all()


def test_score_dispersion_nonnegative(frame):
    assert (frame["prospectivity_score_oos_std"] >= 0).all()

    assert (frame["prospectivity_score_oos_iqr"] >= 0).all()


def test_uncertainty_percentiles(frame):
    assert frame["score_std_percentile"].between(0.0, 1.0).all()

    assert frame["score_iqr_percentile"].between(0.0, 1.0).all()


def test_phase_b_manifest(manifest):
    assert manifest["status"] == "PASS"
    assert manifest["ready_for_phase_c"] is True

    assert manifest["candidate_rows"] == 6300

    assert manifest["oracle_loaded"] is False

    assert manifest["oracle_used"] is False

    assert manifest["rank_uncertainty_in_scope"] is False

    assert manifest["top_k_uncertainty_in_scope"] is False


def test_rebuild_is_deterministic():
    module = load_module()

    candidates = module.load_candidates()

    first = module.build_uncertainty(candidates)

    second = module.build_uncertainty(candidates)

    pd.testing.assert_frame_equal(
        first,
        second,
        check_exact=True,
    )
