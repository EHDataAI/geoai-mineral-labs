from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LAB05_SCORES = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "results"
    / "prospectivity_scores_oos.csv"
)

PHASE_A_MANIFEST = (
    ROOT / "labs" / "lab07_spatial_uncertainty" / "results" / "phase_a_audit.json"
)

RESULTS_DIR = ROOT / "labs" / "lab07_spatial_uncertainty" / "results"

OUTPUT_CSV = RESULTS_DIR / "score_uncertainty.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "phase_b_manifest.json"

CANDIDATE_COUNT = 6300
ARTIFACT_ROWS = 6400
KNOWN_POSITIVES = 100
BACKGROUND_REPLICATES = 10

SCORE_FIELDS = [
    "prospectivity_score_oos_mean",
    "prospectivity_score_oos_median",
    "prospectivity_score_oos_std",
    "prospectivity_score_oos_p25",
    "prospectivity_score_oos_p75",
    "prospectivity_score_oos_iqr",
]

ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
    "distance_to_known_deposit",
    "deposit_id",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_phase_a() -> dict:
    data = json.loads(
        PHASE_A_MANIFEST.read_text(
            encoding="utf-8",
        )
    )

    require(
        data["status"] == "PASS",
        "Phase A status is not PASS",
    )
    require(
        data["ready_for_phase_b"] is True,
        "Phase A did not authorize Phase B",
    )
    require(
        data["gates"]["LAB07_PHASE_A"] == "READY",
        "Phase A gate is not READY",
    )

    oracle = data["oracle_boundary"]

    require(
        oracle["oracle_loaded"] is False,
        "Phase A oracle boundary violated",
    )
    require(
        oracle["oracle_used"] is False,
        "Phase A oracle use detected",
    )

    return data


def load_candidates() -> pd.DataFrame:
    frame = pd.read_csv(LAB05_SCORES)

    require(
        len(frame) == ARTIFACT_ROWS,
        "Unexpected LAB05 artifact row count",
    )
    require(
        frame["cell_id"].is_unique,
        "cell_id is not unique",
    )

    forbidden = ORACLE_FIELDS.intersection(frame.columns)

    require(
        not forbidden,
        f"Oracle fields present: {sorted(forbidden)}",
    )

    missing = {
        "cell_id",
        "Easting",
        "Northing",
        "spatial_block_id",
        "known_positive",
        "oos_model_count",
        *SCORE_FIELDS,
    }.difference(frame.columns)

    require(
        not missing,
        f"Missing required fields: {sorted(missing)}",
    )

    known = frame.loc[frame["known_positive"] == 1]

    candidates = frame.loc[frame["known_positive"] == 0].copy()

    require(
        len(known) == KNOWN_POSITIVES,
        "Unexpected known-positive count",
    )
    require(
        len(candidates) == CANDIDATE_COUNT,
        "Unexpected candidate count",
    )
    require(
        (known["oos_model_count"] == 0).all(),
        "Known positives have OOS support",
    )
    require(
        candidates["oos_model_count"].between(4, 10).all(),
        "Invalid candidate OOS support",
    )
    require(
        candidates[SCORE_FIELDS].notna().all().all(),
        "Missing candidate score values",
    )

    return candidates


def build_uncertainty(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates[
        [
            "cell_id",
            "Easting",
            "Northing",
            "spatial_block_id",
            "oos_model_count",
            *SCORE_FIELDS,
        ]
    ].copy()

    p25 = result["prospectivity_score_oos_p25"].to_numpy(dtype=float)

    p75 = result["prospectivity_score_oos_p75"].to_numpy(dtype=float)

    stored_iqr = result["prospectivity_score_oos_iqr"].to_numpy(dtype=float)

    calculated_iqr = p75 - p25

    require(
        np.allclose(
            stored_iqr,
            calculated_iqr,
            rtol=0.0,
            atol=1e-14,
        ),
        "Stored IQR differs from p75-p25",
    )

    require(
        (result["prospectivity_score_oos_std"] >= 0).all(),
        "Negative score standard deviation",
    )

    require(
        (result["prospectivity_score_oos_iqr"] >= 0).all(),
        "Negative score IQR",
    )

    result["oos_support_fraction"] = result["oos_model_count"] / BACKGROUND_REPLICATES

    result["score_mean_median_abs_gap"] = (
        result["prospectivity_score_oos_mean"]
        - result["prospectivity_score_oos_median"]
    ).abs()

    result["score_std_percentile"] = result["prospectivity_score_oos_std"].rank(
        method="average",
        pct=True,
    )

    result["score_iqr_percentile"] = result["prospectivity_score_oos_iqr"].rank(
        method="average",
        pct=True,
    )

    require(
        result["oos_support_fraction"].between(0.4, 1.0).all(),
        "Invalid support fraction",
    )

    require(
        result["score_std_percentile"].between(0.0, 1.0).all(),
        "Invalid STD percentile",
    )

    require(
        result["score_iqr_percentile"].between(0.0, 1.0).all(),
        "Invalid IQR percentile",
    )

    return result.sort_values(
        "cell_id",
        kind="mergesort",
    ).reset_index(drop=True)


def build_manifest(
    result: pd.DataFrame,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "laboratory": "LABORATORY 07",
        "phase": "B",
        "status": "PASS",
        "ready_for_phase_c": True,
        "artifact_rows": ARTIFACT_ROWS,
        "candidate_rows": len(result),
        "known_positive_rows_excluded": (KNOWN_POSITIVES),
        "background_replicates": (BACKGROUND_REPLICATES),
        "score_semantics": ("prospectivity_score_not_calibrated_deposit_probability"),
        "uncertainty_scope": ("candidate_level_oos_score_uncertainty"),
        "rank_uncertainty_in_scope": False,
        "top_k_uncertainty_in_scope": False,
        "oracle_loaded": False,
        "oracle_used": False,
        "derived_fields": [
            "oos_support_fraction",
            "score_mean_median_abs_gap",
            "score_std_percentile",
            "score_iqr_percentile",
        ],
        "support": {
            "minimum": int(result["oos_model_count"].min()),
            "maximum": int(result["oos_model_count"].max()),
            "mean": float(result["oos_model_count"].mean()),
        },
        "score_dispersion": {
            "std_median": float(result["prospectivity_score_oos_std"].median()),
            "std_maximum": float(result["prospectivity_score_oos_std"].max()),
            "iqr_median": float(result["prospectivity_score_oos_iqr"].median()),
            "iqr_maximum": float(result["prospectivity_score_oos_iqr"].max()),
        },
        "source_sha256": {
            "phase_a_manifest": sha256(PHASE_A_MANIFEST),
            "lab05_scores": sha256(LAB05_SCORES),
        },
        "gates": {
            "PHASE_A_CONTRACT": "PASS",
            "CANDIDATE_POPULATION": "PASS",
            "SCORE_DISPERSION": "PASS",
            "SUPPORT_ACCOUNTING": "PASS",
            "ORACLE_ISOLATION": "PASS",
            "LAB07_PHASE_B": "PASS",
        },
    }


def main() -> None:
    load_phase_a()

    candidates = load_candidates()

    result = build_uncertainty(candidates)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )

    manifest = build_manifest(result)

    manifest["output_sha256"] = {"score_uncertainty_csv": sha256(OUTPUT_CSV)}

    OUTPUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for gate, status in manifest["gates"].items():
        print(f"{gate}={status}")

    print(f"CANDIDATE_ROWS={len(result)}")
    print(f"PHASE_B_ARTIFACT={OUTPUT_CSV.relative_to(ROOT)}")
    print(f"PHASE_B_MANIFEST={OUTPUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
