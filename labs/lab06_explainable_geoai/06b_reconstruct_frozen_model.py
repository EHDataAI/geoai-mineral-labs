"""LABORATORY 06 Phase B: deterministic frozen-model reconstruction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LAB05_DIR = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB06_DIR = ROOT / "labs" / "lab06_explainable_geoai"
RESULTS_DIR = LAB06_DIR / "results"

GATE04_PATH = LAB05_DIR / "05d_run_spatial_benchmark.py"
GATE051_PATH = LAB05_DIR / "05e1_score_full_grid_oos.py"
REFERENCE_SCORES_PATH = (
    LAB05_DIR / "results" / "prospectivity_scores_oos.csv"
)

MANIFEST_PATH = RESULTS_DIR / "model_reconstruction_manifest.json"
RECONSTRUCTED_PATH = RESULTS_DIR / "reconstructed_oos_scores.csv"

EXPECTED_MODEL = "logistic_regression"
EXPECTED_FEATURE_SET = "A"
EXPECTED_ROWS = 6400
EXPECTED_CANDIDATES = 6300
EXPECTED_REPLICATES = 10
ABS_TOLERANCE = 1e-12


def load_module(path: Path, module_name: str):
    if str(LAB05_DIR) not in sys.path:
        sys.path.insert(0, str(LAB05_DIR))

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_reconstruction(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float]:
    columns = [
        "prospectivity_score_oos_mean",
        "prospectivity_score_oos_median",
        "prospectivity_score_oos_std",
        "prospectivity_score_oos_p25",
        "prospectivity_score_oos_p75",
        "prospectivity_score_oos_iqr",
        "oos_replicate_rank_mean",
        "oos_replicate_rank_std",
        "oos_top_1pct_frequency",
        "oos_top_5pct_frequency",
        "oos_top_10pct_frequency",
    ]

    left = reconstructed.sort_values("cell_id").reset_index(drop=True)
    right = reference.sort_values("cell_id").reset_index(drop=True)

    if left["cell_id"].tolist() != right["cell_id"].tolist():
        raise RuntimeError("cell_id order mismatch after deterministic sort.")

    diffs: dict[str, float] = {}

    for column in columns:
        a = left[column].to_numpy(dtype=float)
        b = right[column].to_numpy(dtype=float)

        both_nan = np.isnan(a) & np.isnan(b)
        finite = np.isfinite(a) & np.isfinite(b)

        if not np.all(both_nan | finite):
            raise RuntimeError(f"Finite/NaN pattern mismatch for {column}.")

        max_diff = 0.0
        if finite.any():
            max_diff = float(np.max(np.abs(a[finite] - b[finite])))

        diffs[column] = max_diff

        if max_diff > ABS_TOLERANCE:
            raise RuntimeError(
                f"Reconstruction drift for {column}: {max_diff}"
            )

    integer_columns = [
        "oos_model_count",
        "candidate_rank",
    ]

    for column in integer_columns:
        a = pd.Series(
            pd.array(left[column], dtype="Int64"),
            name=column,
        )
        b = pd.Series(
            pd.array(right[column], dtype="Int64"),
            name=column,
        )

        if not a.equals(b):
            mismatch = ~(a.eq(b) | (a.isna() & b.isna()))
            raise RuntimeError(
                "Exact reconstruction mismatch for "
                f"{column}: {int(mismatch.sum())} rows."
            )

    boolean_columns = [
        "candidate_top_1pct",
        "candidate_top_5pct",
        "candidate_top_10pct",
    ]

    for column in boolean_columns:
        a = left[column].astype(bool).to_numpy()
        b = right[column].astype(bool).to_numpy()

        mismatch_count = int(np.count_nonzero(a != b))

        if mismatch_count:
            raise RuntimeError(
                "Exact reconstruction mismatch for "
                f"{column}: {mismatch_count} rows."
            )

    return diffs


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gate04 = load_module(
        GATE04_PATH,
        "lab05_gate04_for_lab06_phase_b",
    )
    gate051 = load_module(
        GATE051_PATH,
        "lab05_gate051_for_lab06_phase_b",
    )

    aggregate = pd.read_csv(gate051.OOF_AGGREGATE_PATH)
    oof_manifest = gate051.load_json(gate051.OOF_MANIFEST_PATH)
    gate051.validate_model_freeze(aggregate, oof_manifest)

    frame, feature_columns = gate051.load_safe_evidence(gate04)

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError("Unexpected evidence row count.")

    score_matrix, rank_matrix, audit, top_hits = (
        gate051.score_oos_replicates(
            gate04,
            frame,
            feature_columns,
        )
    )

    reconstructed = gate051.build_output(
        frame,
        score_matrix,
        rank_matrix,
        top_hits,
    )

    reference = pd.read_csv(REFERENCE_SCORES_PATH)
    if len(reference) != EXPECTED_ROWS:
        raise RuntimeError("Unexpected reference score row count.")

    diffs = compare_reconstruction(reconstructed, reference)

    reconstructed[
        [
            "cell_id",
            "known_positive",
            "oos_model_count",
            "prospectivity_score_oos_mean",
            "candidate_rank",
        ]
    ].to_csv(RECONSTRUCTED_PATH, index=False)

    candidate_count = int((reconstructed["known_positive"] == 0).sum())

    if candidate_count != EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected reconstructed candidate count.")

    if len(audit) != EXPECTED_REPLICATES:
        raise RuntimeError("Unexpected replicate count.")

    manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "B",
        "purpose": "deterministic_frozen_model_reconstruction",
        "status": "PASS",
        "primary_feature_set": EXPECTED_FEATURE_SET,
        "primary_model": EXPECTED_MODEL,
        "evidence_rows": int(len(frame)),
        "candidate_rows": candidate_count,
        "background_replicates": int(len(audit)),
        "feature_columns": feature_columns,
        "absolute_tolerance": ABS_TOLERANCE,
        "maximum_absolute_differences": diffs,
        "reference_artifact": str(
            REFERENCE_SCORES_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "oracle_loaded": False,
        "oracle_used_for_training": False,
        "oracle_used_for_scoring": False,
        "model_reselected": False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("FROZEN_MODEL_RECONSTRUCTION = PASS")
    print(f"RECONSTRUCTED_ROWS = {len(reconstructed)}")
    print(f"CANDIDATE_ROWS = {candidate_count}")
    print(f"BACKGROUND_REPLICATES = {len(audit)}")
    print("MAX_ABSOLUTE_DIFF = " f"{max(diffs.values()):.16g}")
    print("ORACLE_USED = NO")
    print("MODEL_RESELECTED = NO")
    print("LAB06_PHASE_B_RECONSTRUCTION = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
