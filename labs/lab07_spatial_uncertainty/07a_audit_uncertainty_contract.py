from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

FEATURES = [
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

FIELDS = [
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

ORACLE = {
    "true_deposit",
    "latent_favourability",
    "distance_to_known_deposit",
    "true_deposit_probability",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    require(math.isfinite(value), f"Non-finite {field}")
    return value


def audit(repo: Path) -> dict:
    lab05 = repo / "labs" / "lab05_mineral_prospectivity"
    lab06 = repo / "labs" / "lab06_explainable_geoai"

    paths = {
        "lab05_contract": lab05 / "prospectivity_contract.json",
        "lab05_model": (
            lab05
            / "results"
            / "prospectivity_model_manifest_oos.json"
        ),
        "lab05_scores": (
            lab05
            / "results"
            / "prospectivity_scores_oos.csv"
        ),
        "lab06_reconstruction": (
            lab06
            / "results"
            / "model_reconstruction_manifest.json"
        ),
        "lab06_stability": (
            lab06
            / "results"
            / "explanation_stability_manifest.json"
        ),
        "lab06_spatial": (
            lab06
            / "results"
            / "spatial_explainability_manifest.json"
        ),
    }

    for name, path in paths.items():
        require(path.is_file(), f"Missing {name}")

    contract = read_json(paths["lab05_contract"])
    model = read_json(paths["lab05_model"])
    reconstruction = read_json(paths["lab06_reconstruction"])
    stability = read_json(paths["lab06_stability"])
    spatial = read_json(paths["lab06_spatial"])

    groups = contract["feature_groups"]
    features = (
        groups["geology"]
        + groups["structure"]
        + groups["geophysics"]
        + groups["alteration"]
    )

    require(features == FEATURES, "Feature Set A changed")
    require(model["primary_feature_set"] == "A", "Feature set changed")
    require(
        model["primary_model"] == "logistic_regression",
        "Model changed",
    )
    require(model["model_frozen"] is True, "Model not frozen")
    require(model["dgp_v2_frozen"] is True, "DGP not frozen")
    require(
        model["candidate_rows"] == 6300,
        "Candidate count changed",
    )
    require(
        model["background_replicates"] == 10,
        "Replicate count changed",
    )
    require(
        model["oos_candidates_per_replicate"] == 5800,
        "OOS rows changed",
    )
    require(
        model["oos_model_count_min"] == 4,
        "Support min changed",
    )
    require(
        model["oos_model_count_max"] == 10,
        "Support max changed",
    )
    require(
        model["frequency_denominator"]
        == "cell_specific_oos_model_count",
        "Frequency denominator changed",
    )
    require(
        model["score_semantics"]
        == "prospectivity_score_not_calibrated_deposit_probability",
        "Score semantics changed",
    )
    require(
        model["oracle_used_for_scoring"] is False,
        "Oracle scoring",
    )
    require(
        model["oracle_used_for_training"] is False,
        "Oracle training",
    )
    require(
        model["oracle_values_loaded"] is False,
        "Oracle loaded",
    )

    require(
        reconstruction["status"] == "PASS",
        "LAB06 reconstruction failed",
    )
    require(
        reconstruction["candidate_rows"] == 6300,
        "LAB06 candidate count",
    )
    require(
        reconstruction["background_replicates"] == 10,
        "LAB06 replicate count",
    )
    require(
        reconstruction["feature_columns"] == FEATURES,
        "LAB06 feature set",
    )
    require(
        reconstruction["primary_model"] == "logistic_regression",
        "LAB06 model",
    )
    require(
        reconstruction["primary_feature_set"] == "A",
        "LAB06 feature set id",
    )
    require(
        reconstruction["model_reselected"] is False,
        "LAB06 reselection",
    )
    require(
        reconstruction["oracle_loaded"] is False,
        "LAB06 oracle",
    )

    tolerance = float(reconstruction["absolute_tolerance"])
    differences = reconstruction["maximum_absolute_differences"]

    require(
        tolerance <= 1e-12,
        "Tolerance too loose",
    )
    require(
        set(FIELDS).issubset(differences),
        "Reconstruction coverage",
    )

    maximum_difference = max(
        float(value)
        for value in differences.values()
    )

    require(
        maximum_difference <= tolerance,
        "Reconstruction tolerance exceeded",
    )

    require(stability["status"] == "PASS", "LAB06 stability")
    require(
        stability["oracle_loaded"] is False,
        "Stability oracle",
    )
    require(
        stability["model_reselected"] is False,
        "Stability reselection",
    )
    require(spatial["status"] == "PASS", "LAB06 spatial")
    require(
        spatial["candidate_rows"] == 6300,
        "Spatial candidates",
    )
    require(
        spatial["support_min"] == 4,
        "Spatial support min",
    )
    require(
        spatial["support_max"] == 10,
        "Spatial support max",
    )
    require(
        spatial["oracle_loaded"] is False,
        "Spatial oracle",
    )

    with paths["lab05_scores"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        all_rows = list(reader)

    required = {
        "cell_id",
        "known_positive",
        "oos_model_count",
        *FIELDS,
    }

    require(
        not required.difference(header),
        "Missing uncertainty columns",
    )

    forbidden = sorted(ORACLE.intersection(header))

    require(
        not forbidden,
        f"Oracle columns present: {forbidden}",
    )
    require(
        len(all_rows) == 6400,
        "Expected 6400 scoring rows",
    )

    candidate_rows = [
        row
        for row in all_rows
        if int(float(row["known_positive"])) == 0
    ]

    known_rows = [
        row
        for row in all_rows
        if int(float(row["known_positive"])) == 1
    ]

    require(
        len(candidate_rows) == 6300,
        "Expected 6300 candidates",
    )

    require(
        len(known_rows) == 100,
        "Expected 100 known positives",
    )

    require(
        all(
            int(row["oos_model_count"]) == 0
            for row in known_rows
        ),
        "Known positives must have zero OOS support",
    )

    seen: set[int] = set()
    supports: list[int] = []

    for row in candidate_rows:
        cell_id = int(row["cell_id"])

        require(
            cell_id not in seen,
            f"Duplicate cell_id {cell_id}",
        )
        seen.add(cell_id)

        require(
            int(float(row["known_positive"])) == 0,
            "Positive leakage",
        )

        support = int(row["oos_model_count"])
        supports.append(support)

        require(
            4 <= support <= 10,
            f"Invalid support {cell_id}",
        )

        p25 = finite(row, "prospectivity_score_oos_p25")
        median = finite(row, "prospectivity_score_oos_median")
        p75 = finite(row, "prospectivity_score_oos_p75")
        score_std = finite(row, "prospectivity_score_oos_std")
        score_iqr = finite(row, "prospectivity_score_oos_iqr")
        rank_mean = finite(row, "oos_replicate_rank_mean")
        rank_std = finite(row, "oos_replicate_rank_std")

        require(
            p25 <= median <= p75,
            f"Invalid quantiles {cell_id}",
        )
        require(
            score_std >= 0 and score_iqr >= 0,
            f"Invalid dispersion {cell_id}",
        )
        require(
            1 <= rank_mean <= 5800,
            f"Invalid rank {cell_id}",
        )
        require(
            rank_std >= 0,
            f"Invalid rank std {cell_id}",
        )

        for field in FIELDS[-3:]:
            value = finite(row, field)
            require(
                0 <= value <= 1,
                f"Invalid frequency {field}",
            )

    require(
        min(supports) == 4,
        "Observed support min changed",
    )
    require(
        max(supports) == 10,
        "Observed support max changed",
    )

    gates = {
        "LAB05_FROZEN_MODEL": "PASS",
        "FEATURE_SET_A_CONTRACT": "PASS",
        "CANDIDATE_POPULATION": "PASS",
        "OOS_ENSEMBLE_CONTRACT": "PASS",
        "LAB06_RECONSTRUCTION": "PASS",
        "UNCERTAINTY_INPUT_CONTRACT": "PASS",
        "ORACLE_ISOLATION": "PASS",
        "SCORE_SEMANTICS": "PASS",
        "LAB07_PHASE_A": "READY",
    }

    source_files = [
        path.relative_to(repo).as_posix()
        for path in paths.values()
    ]

    source_hashes = {
        name: sha256(path)
        for name, path in sorted(paths.items())
    }

    return {
        "schema_version": "1.0.0",
        "laboratory": "LABORATORY 07",
        "phase": "A",
        "status": "PASS",
        "ready_for_phase_b": True,
        "frozen_model": {
            "feature_set": "A",
            "model": "logistic_regression",
            "feature_columns": FEATURES,
            "model_reselected": False,
        },
        "uncertainty_input": {
            "candidate_rows": len(candidate_rows),
            "background_replicates": model["background_replicates"],
            "oos_candidates_per_replicate": (
                model["oos_candidates_per_replicate"]
            ),
            "oos_model_count_min": min(supports),
            "oos_model_count_max": max(supports),
            "uncertainty_fields": FIELDS,
            "score_semantics": model["score_semantics"],
            "frequency_denominator": (
                model["frequency_denominator"]
            ),
        },
        "lab06_crosscheck": {
            "absolute_tolerance": tolerance,
            "maximum_observed_difference": maximum_difference,
            "reconstruction_status": reconstruction["status"],
            "explanation_stability_status": stability["status"],
            "spatial_explainability_status": spatial["status"],
        },
        "oracle_boundary": {
            "oracle_loaded": False,
            "oracle_used": False,
            "oracle_artifacts_read": [],
            "forbidden_columns_present": forbidden,
        },
        "source_files_read": source_files,
        "source_sha256": source_hashes,
        "gates": gates,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    manifest = audit(repo)

    output = (
        Path(__file__).parent
        / "results"
        / "phase_a_audit.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for gate, result in manifest["gates"].items():
        print(f"{gate} = {result}")

    print(
        "PHASE_A_MANIFEST = "
        f"{output.relative_to(repo)}"
    )


if __name__ == "__main__":
    main()
