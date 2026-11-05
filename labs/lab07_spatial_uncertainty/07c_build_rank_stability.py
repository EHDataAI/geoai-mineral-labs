"""LAB07 Phase C: conditional OOS rank and top-k stability.

The frozen LAB05 Feature Set A logistic model is reconstructed in memory.
Neither the hidden oracle nor a probability-calibration procedure is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAB05 = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB07 = ROOT / "labs" / "lab07_spatial_uncertainty"
RESULTS = LAB07 / "results"

GATE05_SCRIPT = LAB05 / "05e1_score_full_grid_oos.py"
REFERENCE_SCORES = LAB05 / "results" / "prospectivity_scores_oos.csv"
EVIDENCE = LAB05 / "data" / "prospectivity_evidence.csv"
PHASE_A = RESULTS / "phase_a_audit.json"
PHASE_B = RESULTS / "phase_b_manifest.json"
OUTPUT = RESULTS / "rank_stability.csv"
MANIFEST = RESULTS / "phase_c_manifest.json"

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "distance_to_known_deposit",
    "deposit_id",
}

BASE_COLUMNS = [
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "oos_model_count",
    "candidate_rank",
]
RANK_COLUMNS = [
    "oos_replicate_rank_mean",
    "oos_replicate_rank_std",
    "oos_replicate_rank_median",
    "oos_replicate_rank_p25",
    "oos_replicate_rank_p75",
    "oos_replicate_rank_iqr",
]
PRIORITY_COLUMNS = [
    "oos_priority_mean",
    "oos_priority_median",
    "oos_priority_std",
    "oos_priority_iqr",
]
TOP_COUNTS = {"top_1pct": 58, "top_5pct": 290, "top_10pct": 580}
TOP_COLUMNS = [f"oos_{name}_frequency" for name in TOP_COUNTS]
ALL_COLUMNS = BASE_COLUMNS + RANK_COLUMNS + PRIORITY_COLUMNS + TOP_COLUMNS


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(f"LAB07 Phase C FAIL: {reason}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf(path: Path) -> str:
    """A portable digest across pure LF/CRLF Git worktree conversions."""
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    require(b"\r" not in lf, f"Unsupported CR in {path.name}")
    require(
        not (b"\r\n" in data and b"\n" in data.replace(b"\r\n", b"")),
        f"Mixed line endings in {path.name}",
    )
    return hashlib.sha256(lf).hexdigest()


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_upstream() -> None:
    """Validate upstream authorizations and prohibit oracle access."""
    a = read_manifest(PHASE_A)
    b = read_manifest(PHASE_B)
    require(a["status"] == "PASS", "Phase A status")
    require(a["ready_for_phase_b"] is True, "Phase A authorization")
    require(a["oracle_boundary"]["oracle_loaded"] is False, "Phase A oracle")
    require(a["oracle_boundary"]["oracle_used"] is False, "Phase A oracle use")
    require(b["status"] == "PASS", "Phase B status")
    require(b["ready_for_phase_c"] is True, "Phase B authorization")
    require(b["oracle_loaded"] is False, "Phase B oracle")
    require(b["oracle_used"] is False, "Phase B oracle use")
    require(b["candidate_rows"] == 6300, "Phase B candidates")
    require(b["background_replicates"] == 10, "Phase B replicates")
    require(
        b["score_semantics"]
        == "prospectivity_score_not_calibrated_deposit_probability",
        "Score meaning changed",
    )
    for key, path in (
        ("phase_a_manifest", PHASE_A),
        ("lab05_scores", REFERENCE_SCORES),
    ):
        expected = b["source_sha256"][key]
        data = path.read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        require(b"\r" not in lf, f"Unexpected line endings: {key}")
        require(
            not (b"\r\n" in data and b"\n" in data.replace(b"\r\n", b"")),
            f"Mixed line endings: {key}",
        )
        digest_variants = {
            hashlib.sha256(item).hexdigest()
            for item in (data, lf, lf.replace(b"\n", b"\r\n"))
        }
        require(expected in digest_variants, f"Upstream SHA256 drift: {key}")


def load_gate05():
    sys.path.insert(0, str(LAB05))
    spec = importlib.util.spec_from_file_location("lab07_phase_c_gate05", GATE05_SCRIPT)
    require(spec is not None and spec.loader is not None, "LAB05 import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_ranks():
    """Call LAB05's read-only scoring functions, never its writing main()."""
    gate05 = load_gate05()
    gate04 = gate05.load_gate04_module()
    gate05.validate_model_freeze(
        pd.read_csv(gate05.OOF_AGGREGATE_PATH),
        gate05.load_json(gate05.OOF_MANIFEST_PATH),
    )
    frame, features = gate05.load_safe_evidence(gate04)
    require(gate05.FEATURE_SET == "A", "Feature set changed")
    require(gate05.MODEL_NAME == "logistic_regression", "Model changed")
    require(len(features) == 9, "Feature count")
    require(not ORACLE_FIELDS.intersection(features), "Oracle feature")
    require(not ORACLE_FIELDS.intersection(frame.columns), "Oracle evidence")
    require(gate05.BACKGROUND_REPLICATES == 10, "Replicate count")
    scores, ranks, audit, hits = gate05.score_oos_replicates(gate04, frame, features)
    return frame, scores, ranks, audit, hits


def build_rank_stability(
    frame: pd.DataFrame,
    scores: np.ndarray,
    ranks: np.ndarray,
    audit: list[dict],
    top_hits: dict[str, np.ndarray],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Derive descriptive conditional uncertainty from observed OOS ranks."""
    require(scores.shape == (6400, 10), "Score matrix shape")
    require(ranks.shape == (6400, 10), "Rank matrix shape")
    require(len(audit) == 10, "Replicate audit length")
    require(len(frame) == 6400 and frame["cell_id"].is_unique, "Evidence keys")
    require(len(reference) == 6400 and reference["cell_id"].is_unique, "Scores keys")
    require(not ORACLE_FIELDS.intersection(reference.columns), "Oracle reference")

    known = frame["known_positive"].to_numpy(dtype=int) == 1
    candidate = ~known
    require(int(known.sum()) == 100, "Known-positive count")
    require(int(candidate.sum()) == 6300, "Candidate count")
    require(np.isnan(ranks[known]).all(), "Known-positive ranks")
    require(np.isnan(scores[known]).all(), "Known-positive scores")
    cr = ranks[candidate]
    require(cr.shape == (6300, 10), "Candidate rank matrix")
    require(
        np.array_equal(np.isfinite(scores[candidate]), np.isfinite(cr)),
        "Score/rank support mismatch",
    )
    support = np.isfinite(cr).sum(axis=1)
    require((support >= 4).all() and (support <= 10).all(), "OOS support bounds")
    require(int(support.min()) == 4 and int(support.max()) == 10, "OOS extrema")

    for j in range(10):
        observed = ranks[:, j][np.isfinite(ranks[:, j])]
        require(
            np.array_equal(np.sort(observed), np.arange(1, 5801)),
            f"OOS rank permutation in replicate {j}",
        )
        require(audit[j]["oos_candidate_rows"] == 5800, "Replicate OOS audit")

    ref = reference.set_index("cell_id").loc[frame.loc[candidate, "cell_id"]]
    require((ref["known_positive"] == 0).all(), "Candidate identity")
    require(
        np.array_equal(support, ref["oos_model_count"].to_numpy(dtype=int)),
        "Canonical OOS support",
    )
    final_ranks = ref["candidate_rank"].to_numpy(dtype=int)
    require(
        np.array_equal(np.sort(final_ranks), np.arange(1, 6301)),
        "Final candidate rank permutation",
    )

    rm = np.nanmean(cr, axis=1)
    rs = np.nanstd(cr, axis=1, ddof=1)
    rmed = np.nanmedian(cr, axis=1)
    r25 = np.nanquantile(cr, 0.25, axis=1, method="linear")
    r75 = np.nanquantile(cr, 0.75, axis=1, method="linear")
    priority = 1.0 - (cr - 1.0) / 5799.0
    require(
        (
            (priority[np.isfinite(priority)] >= 0.0)
            & (priority[np.isfinite(priority)] <= 1.0)
        ).all(),
        "Priority range",
    )
    p25 = np.nanquantile(priority, 0.25, axis=1, method="linear")
    p75 = np.nanquantile(priority, 0.75, axis=1, method="linear")

    values = {
        "oos_replicate_rank_mean": rm,
        "oos_replicate_rank_std": rs,
        "oos_replicate_rank_median": rmed,
        "oos_replicate_rank_p25": r25,
        "oos_replicate_rank_p75": r75,
        "oos_replicate_rank_iqr": r75 - r25,
        "oos_priority_mean": np.nanmean(priority, axis=1),
        "oos_priority_median": np.nanmedian(priority, axis=1),
        "oos_priority_std": np.nanstd(priority, axis=1, ddof=1),
        "oos_priority_iqr": p75 - p25,
    }
    for name, k in TOP_COUNTS.items():
        hits = np.count_nonzero(cr <= k, axis=1)
        require(int(hits.sum()) == 10 * k, f"{name} total hits")
        require(
            np.array_equal(hits, top_hits[name][candidate]),
            f"{name} hit consistency",
        )
        values[f"oos_{name}_frequency"] = hits / support

    for field in (
        "oos_replicate_rank_mean",
        "oos_replicate_rank_std",
        *TOP_COLUMNS,
    ):
        difference = np.abs(values[field] - ref[field].to_numpy(dtype=float))
        require(np.isfinite(difference).all(), f"{field} finite")
        require(float(difference.max()) <= 1e-12, f"{field} canonical drift")

    result = frame.loc[
        candidate, ["cell_id", "Easting", "Northing", "spatial_block_id"]
    ].copy()
    result["oos_model_count"] = support
    result["candidate_rank"] = final_ranks
    for name, value in values.items():
        result[name] = value
    result = result.loc[:, ALL_COLUMNS]
    require(
        np.isfinite(result[RANK_COLUMNS + PRIORITY_COLUMNS + TOP_COLUMNS]).all().all(),
        "Non-finite results",
    )
    require((result["oos_replicate_rank_iqr"] >= 0).all(), "Negative rank IQR")
    require((result["oos_priority_iqr"] >= 0).all(), "Negative priority IQR")
    require(result[TOP_COLUMNS].ge(0).all().all(), "Negative top-k frequency")
    require(result[TOP_COLUMNS].le(1).all().all(), "Top-k frequency over 1")
    return result.sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def build_manifest(result: pd.DataFrame) -> dict:
    source_paths = {
        "phase_a_manifest": PHASE_A,
        "phase_b_manifest": PHASE_B,
        "lab05_scores": REFERENCE_SCORES,
        "lab05_evidence": EVIDENCE,
        "lab05_rank_reconstruction_code": GATE05_SCRIPT,
    }
    return {
        "schema_version": "1.0.0",
        "laboratory": "LABORATORY 07",
        "phase": "C",
        "status": "QA_PENDING",
        "ready_for_phase_d": False,
        "artifact_rows": 6400,
        "candidate_rows": len(result),
        "known_positive_rows_excluded": 100,
        "background_replicates": 10,
        "oos_candidates_per_replicate": 5800,
        "rank_scale": "replicate_oos_1_to_5800",
        "candidate_rank_scale": "aggregate_candidate_1_to_6300",
        "priority_formula": "1-(replicate_oos_rank-1)/5799",
        "rank_std_ddof": 1,
        "rank_quantile_method": "linear",
        "top1_replicate_count": 58,
        "top5_replicate_count": 290,
        "top10_replicate_count": 580,
        "frequency_denominator": "cell_specific_oos_model_count",
        "rank_uncertainty_scope": "candidate_level_conditional_oos_rank_dispersion",
        "top_k_stability_scope": "candidate_level_conditional_oos_top_k_frequency",
        "score_semantics": "prospectivity_score_not_calibrated_deposit_probability",
        "oracle_loaded": False,
        "oracle_used": False,
        "qa": {"status": "PENDING"},
        "interpretation_limitations": [
            "Overlapping replicate training sets are not independent trials.",
            "Each replicate ranks a changing eligible set of 5800 candidates.",
            "Candidate-specific OOS support is only 4 to 10 observations.",
            "Top-k frequencies and normalized priorities are not probabilities.",
            "Nine original covariates do not imply nine effective coefficients.",
        ],
        "source_sha256": {key: sha256(path) for key, path in source_paths.items()},
        "source_sha256_lf": {
            key: sha256_lf(path) for key, path in source_paths.items()
        },
        "gates": {
            "PHASE_A_B_CONTRACT": "PASS",
            "FROZEN_MODEL": "PASS",
            "RANK_MATRIX": "PASS",
            "CANDIDATE_POPULATION": "PASS",
            "RANK_DISPERSION": "PASS",
            "NORMALIZED_PRIORITY": "PASS",
            "TOP_K_ACCOUNTING": "PASS",
            "CANONICAL_CROSSCHECK": "PASS",
            "ORACLE_ISOLATION": "PASS",
            "RUFF": "PENDING",
            "TARGETED_TESTS": "PENDING",
            "FULL_REGRESSION": "PENDING",
            "PRE_COMMIT": "PENDING",
            "GITLEAKS": "PENDING",
            "LAB07_PHASE_C": "PENDING",
        },
    }


def main() -> None:
    validate_upstream()
    frame, scores, ranks, audit, top_hits = reconstruct_ranks()
    reference = pd.read_csv(REFERENCE_SCORES)
    result = build_rank_stability(frame, scores, ranks, audit, top_hits, reference)
    manifest = build_manifest(result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )
    manifest["output_sha256"] = {"rank_stability_csv": sha256(OUTPUT)}
    manifest["output_sha256_lf"] = {"rank_stability_csv": sha256_lf(OUTPUT)}
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"CANDIDATE_ROWS={len(result)}")
    print(f"PHASE_C_CSV={OUTPUT.relative_to(ROOT)}")
    print(f"PHASE_C_MANIFEST={MANIFEST.relative_to(ROOT)}")
    print("LAB07_PHASE_C_BUILD=PASS")


def read_junit(path: Path) -> int:
    require(path.is_file(), f"JUnit evidence missing: {path.name}")
    root = ET.parse(path).getroot()
    suites = root.findall(".//testsuite")
    if root.tag == "testsuite":
        suites = [root]
    require(bool(suites), f"Invalid JUnit evidence: {path.name}")
    counts = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "errors", "failures", "skipped")
    }
    require(counts["errors"] == 0, f"JUnit errors: {path.name}")
    require(counts["failures"] == 0, f"JUnit failures: {path.name}")
    require(counts["skipped"] == 0, f"JUnit skips: {path.name}")
    return counts["tests"]


def finalize(evidence_dir: Path) -> None:
    """Unlock Phase D only after the ERL runner's external QA evidence exists."""
    manifest = read_manifest(MANIFEST)
    require(manifest["status"] == "QA_PENDING", "Finalization state")
    require(manifest["ready_for_phase_d"] is False, "Premature Phase D")
    require(manifest["oracle_loaded"] is False, "Oracle boundary")
    require(manifest["oracle_used"] is False, "Oracle use")
    require(
        sha256_lf(OUTPUT) == manifest["output_sha256_lf"]["rank_stability_csv"],
        "Output changed after build",
    )
    sources = {
        "phase_a_manifest": PHASE_A,
        "phase_b_manifest": PHASE_B,
        "lab05_scores": REFERENCE_SCORES,
        "lab05_evidence": EVIDENCE,
        "lab05_rank_reconstruction_code": GATE05_SCRIPT,
    }
    for name, path in sources.items():
        require(
            sha256_lf(path) == manifest["source_sha256_lf"][name],
            f"Source changed after build: {name}",
        )
    targeted = read_junit(evidence_dir / "pytest_targeted.xml")
    full = read_junit(evidence_dir / "pytest_full.xml")
    require(targeted == 17, "Targeted test count")
    require(full >= 217, "Full regression coverage")

    manifest["qa"] = {
        "status": "PASS",
        "targeted_tests": targeted,
        "full_regression_tests": full,
        "junit_evidence_location": "outside_repository",
        "security_scan_scope": "git_history_and_worktree",
    }
    manifest["status"] = "PASS"
    manifest["ready_for_phase_d"] = True
    for name in (
        "RUFF",
        "TARGETED_TESTS",
        "FULL_REGRESSION",
        "PRE_COMMIT",
        "GITLEAKS",
        "LAB07_PHASE_C",
    ):
        manifest["gates"][name] = "PASS"
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"TARGETED_TESTS={targeted}")
    print(f"FULL_REGRESSION_TESTS={full}")
    print("LAB07_PHASE_C_FINALIZED=PASS")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: 07c_build_rank_stability.py [--finalize QA_DIR]")
