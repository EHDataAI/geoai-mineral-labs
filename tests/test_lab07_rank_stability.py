"""Scientific and integrity gates for LAB07 Phase C."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "labs/lab07_spatial_uncertainty/07c_build_rank_stability.py"
CSV = ROOT / "labs/lab07_spatial_uncertainty/results/rank_stability.csv"
MANIFEST = ROOT / "labs/lab07_spatial_uncertainty/results/phase_c_manifest.json"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("lab07_phase_c_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing Phase C scientific module")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def output() -> pd.DataFrame:
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reconstructed(module):
    return module.reconstruct_ranks()


def test_candidate_population_and_schema(module, output):
    assert len(output) == 6300
    assert output["cell_id"].is_unique
    assert output.columns.to_list() == module.ALL_COLUMNS
    assert output["cell_id"].is_monotonic_increasing


def test_no_oracle_fields(module, output):
    assert not module.ORACLE_FIELDS.intersection(output.columns)


def test_candidate_rank_permutation(output):
    assert np.array_equal(
        np.sort(output["candidate_rank"].to_numpy(dtype=int)),
        np.arange(1, 6301),
    )


def test_candidate_support(output):
    assert output["oos_model_count"].between(4, 10).all()
    assert output["oos_model_count"].min() == 4
    assert output["oos_model_count"].max() == 10


def test_rank_and_priority_bounds(output):
    assert output["oos_replicate_rank_mean"].between(1, 5800).all()
    assert output["oos_replicate_rank_median"].between(1, 5800).all()
    assert output["oos_replicate_rank_p25"].between(1, 5800).all()
    assert output["oos_replicate_rank_p75"].between(1, 5800).all()
    for field in (
        "oos_priority_mean",
        "oos_priority_median",
        "oos_priority_std",
        "oos_priority_iqr",
    ):
        assert output[field].between(0, 1).all()


def test_dispersion_is_nonnegative(output):
    for field in (
        "oos_replicate_rank_std",
        "oos_replicate_rank_iqr",
        "oos_priority_std",
        "oos_priority_iqr",
    ):
        assert (output[field] >= 0).all()


def test_iqr_matches_quartiles(output):
    np.testing.assert_allclose(
        output["oos_replicate_rank_iqr"],
        output["oos_replicate_rank_p75"] - output["oos_replicate_rank_p25"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        output["oos_priority_iqr"],
        output["oos_replicate_rank_iqr"] / 5799.0,
        rtol=0,
        atol=1e-14,
    )


def test_normalized_priority_consistency(output):
    np.testing.assert_allclose(
        output["oos_priority_mean"],
        1.0 - (output["oos_replicate_rank_mean"] - 1.0) / 5799.0,
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        output["oos_priority_std"],
        output["oos_replicate_rank_std"] / 5799.0,
        atol=1e-14,
        rtol=0,
    )


def test_top_k_accounting(output):
    for label, k in (("top_1pct", 58), ("top_5pct", 290), ("top_10pct", 580)):
        frequency = output[f"oos_{label}_frequency"]
        assert frequency.between(0, 1).all()
        hits = frequency * output["oos_model_count"]
        np.testing.assert_allclose(hits, np.rint(hits), atol=1e-12, rtol=0)
        assert abs(float(hits.sum()) - 10 * k) <= 1e-10


def test_frequencies_are_nested(output):
    assert (output["oos_top_1pct_frequency"] <= output["oos_top_5pct_frequency"]).all()
    assert (output["oos_top_5pct_frequency"] <= output["oos_top_10pct_frequency"]).all()


def test_rank_matrix_and_canonical_crosscheck(module, reconstructed):
    frame, scores, ranks, audit, hits = reconstructed
    assert ranks.shape == (6400, 10)
    assert np.isnan(ranks[frame["known_positive"] == 1]).all()
    reference = pd.read_csv(module.REFERENCE_SCORES)
    result = module.build_rank_stability(frame, scores, ranks, audit, hits, reference)
    assert len(result) == 6300


def test_determinism_on_frozen_matrices(module, reconstructed):
    frame, scores, ranks, audit, hits = reconstructed
    reference = pd.read_csv(module.REFERENCE_SCORES)
    first = module.build_rank_stability(frame, scores, ranks, audit, hits, reference)
    second = module.build_rank_stability(frame, scores, ranks, audit, hits, reference)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_adversarial_reference_drift_is_rejected(module, reconstructed):
    frame, scores, ranks, audit, hits = reconstructed
    reference = pd.read_csv(module.REFERENCE_SCORES)
    index = reference.index[reference["known_positive"] == 0][0]
    reference.loc[index, "oos_replicate_rank_mean"] += 0.001
    with pytest.raises(RuntimeError, match="canonical drift"):
        module.build_rank_stability(frame, scores, ranks, audit, hits, reference)


def test_adversarial_rank_matrix_shape_is_rejected(module, reconstructed):
    frame, scores, ranks, audit, hits = reconstructed
    reference = pd.read_csv(module.REFERENCE_SCORES)
    with pytest.raises(RuntimeError, match="Rank matrix shape"):
        module.build_rank_stability(
            frame, scores, ranks[:, :-1], audit, hits, reference
        )


def test_phase_c_manifest(manifest):
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["phase"] == "C"
    assert manifest["status"] in {"QA_PENDING", "PASS"}
    assert manifest["ready_for_phase_d"] is (manifest["status"] == "PASS")
    assert manifest["artifact_rows"] == 6400
    assert manifest["candidate_rows"] == 6300
    assert manifest["known_positive_rows_excluded"] == 100
    assert manifest["oos_candidates_per_replicate"] == 5800
    assert manifest["frequency_denominator"] == "cell_specific_oos_model_count"
    assert manifest["oracle_loaded"] is False
    assert manifest["oracle_used"] is False
    if manifest["status"] == "QA_PENDING":
        assert manifest["qa"]["status"] == "PENDING"
        assert manifest["gates"]["LAB07_PHASE_C"] == "PENDING"
    else:
        assert manifest["qa"]["status"] == "PASS"
        assert manifest["qa"]["targeted_tests"] == 17
        assert manifest["qa"]["full_regression_tests"] >= 217
        assert all(value == "PASS" for value in manifest["gates"].values())
    assert manifest["interpretation_limitations"]


def test_output_integrity(module, manifest):
    assert module.sha256_lf(CSV) == manifest["output_sha256_lf"]["rank_stability_csv"]
    for name, path in (
        ("phase_a_manifest", module.PHASE_A),
        ("phase_b_manifest", module.PHASE_B),
        ("lab05_scores", module.REFERENCE_SCORES),
        ("lab05_evidence", module.EVIDENCE),
        ("lab05_rank_reconstruction_code", module.GATE05_SCRIPT),
    ):
        assert module.sha256_lf(path) == manifest["source_sha256_lf"][name]
    assert b"\r" not in CSV.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r\n" not in MANIFEST.read_bytes()


def test_no_extreme_quantiles_or_decorative_dispersion(module):
    assert not any("q05" in key or "q95" in key for key in module.ALL_COLUMNS)
    assert "rank_relative_dispersion" not in module.ALL_COLUMNS
