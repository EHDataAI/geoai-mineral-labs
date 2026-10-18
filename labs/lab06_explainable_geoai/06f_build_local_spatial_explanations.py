"""LABORATORY 06 Phase D: local and spatial OOS explainability."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
LAB05 = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB06 = ROOT / "labs" / "lab06_explainable_geoai"
RESULTS = LAB06 / "results"

G04_PATH = LAB05 / "05d_run_spatial_benchmark.py"
G051_PATH = LAB05 / "05e1_score_full_grid_oos.py"
SCORES_PATH = LAB05 / "results" / "prospectivity_scores_oos.csv"

LOCAL_CASES_PATH = RESULTS / "local_explanation_cases.csv"
LOCAL_REPLICATES_PATH = RESULTS / "local_shap_replicates.csv"
LOCAL_SUMMARY_PATH = RESULTS / "local_shap_contributions.csv"
SPATIAL_PATH = RESULTS / "spatial_shap_contributions.csv.gz"
DOMINANT_PATH = RESULTS / "dominant_driver_by_cell.csv"
LOCAL_MANIFEST_PATH = RESULTS / "local_explainability_manifest.json"
SPATIAL_MANIFEST_PATH = RESULTS / "spatial_explainability_manifest.json"

EXPECTED_REPLICATES = 10
EXPECTED_CANDIDATES = 6300
EXPECTED_OOS_PER_REPLICATE = 5800
EXPECTED_RAW_FEATURES = 9
EXPECTED_TRANSFORMED_FEATURES = 12
SHAP_VERSION = "0.53.0"
ADDITIVITY_TOLERANCE = 1e-8

CASE_RANKS = {
    "top_candidate": 1,
    "top_1pct_boundary": 63,
    "top_5pct_boundary": 315,
    "median_candidate": 3150,
    "lowest_candidate": 6300,
}

DRIVER_FAMILY = {
    "lithology": "lithology",
    "distance_to_intrusive_m": "intrusive",
    "intrusive_affinity": "intrusive",
    "distance_to_fault_m": "structure",
    "distance_to_fault_intersection_m": "structure",
    "distance_to_contact_m": "structure",
    "magnetic_anomaly": "geophysics",
    "gravity_anomaly": "geophysics",
    "alteration_index": "alteration",
}


def load_module(path: Path, name: str):
    if str(LAB05) not in sys.path:
        sys.path.insert(0, str(LAB05))

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_name(transformed_name: str) -> str:
    if transformed_name.startswith("lithology_"):
        return "lithology"
    return transformed_name


def select_cases(scores: pd.DataFrame) -> pd.DataFrame:
    candidates = scores.loc[scores["known_positive"] == 0].copy()

    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected candidate count.")

    rows: list[dict[str, object]] = []

    for label, rank in CASE_RANKS.items():
        match = candidates.loc[candidates["candidate_rank"] == rank]

        if len(match) != 1:
            raise RuntimeError(
                f"Expected one candidate at rank {rank}, observed {len(match)}."
            )

        row = match.iloc[0]

        rows.append(
            {
                "case_label": label,
                "selection_rule": f"candidate_rank_equals_{rank}",
                "cell_id": str(row["cell_id"]),
                "Easting": float(row["Easting"]),
                "Northing": float(row["Northing"]),
                "candidate_rank": int(row["candidate_rank"]),
                "prospectivity_score_oos_mean": float(
                    row["prospectivity_score_oos_mean"]
                ),
                "oos_model_count": int(row["oos_model_count"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    if importlib.metadata.version("shap") != SHAP_VERSION:
        raise RuntimeError("Unexpected SHAP version.")

    RESULTS.mkdir(parents=True, exist_ok=True)

    g04 = load_module(G04_PATH, "lab05_g04_for_lab06_phase_d")
    g051 = load_module(G051_PATH, "lab05_g051_for_lab06_phase_d")

    frame, features = g051.load_safe_evidence(g04)

    if len(features) != EXPECTED_RAW_FEATURES:
        raise RuntimeError("Unexpected raw Feature Set A width.")

    if set(features) != set(DRIVER_FAMILY):
        raise RuntimeError("Driver-family contract does not match Feature Set A.")

    scores = pd.read_csv(SCORES_PATH)
    cases = select_cases(scores)

    case_lookup = {
        str(row.cell_id): str(row.case_label)
        for row in cases.itertuples(index=False)
    }

    raw_feature_index = {feature: i for i, feature in enumerate(features)}

    signed_sum = np.zeros((len(frame), len(features)), dtype=float)
    abs_sum = np.zeros((len(frame), len(features)), dtype=float)
    support = np.zeros((len(frame), len(features)), dtype=int)

    candidate_mask = frame[g04.TARGET].to_numpy(dtype=int) == 0
    candidate_indices = np.flatnonzero(candidate_mask)
    cell_ids = frame["cell_id"].astype(str).to_numpy()

    local_rows: list[dict[str, object]] = []
    canonical_names: list[str] | None = None
    max_additivity_error = 0.0
    total_explanations = 0

    for replicate in range(EXPECTED_REPLICATES):
        training = g04.sample_background(frame, replicate)

        training_background_ids = set(
            training.loc[
                training[g04.TARGET] == 0,
                "cell_id",
            ].astype(str)
        )

        eligible = candidate_indices[
            np.array(
                [
                    cell_ids[i] not in training_background_ids
                    for i in candidate_indices
                ],
                dtype=bool,
            )
        ]

        if len(eligible) != EXPECTED_OOS_PER_REPLICATE:
            raise RuntimeError("Unexpected OOS candidate count.")

        model_seed = g04.SEED + replicate * 1000

        pipeline = g04.build_model_pipeline(
            "logistic_regression",
            features,
            model_seed,
        )

        pipeline.fit(
            training[features],
            training[g04.TARGET].to_numpy(dtype=int),
        )

        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]

        transformed_names = list(preprocessor.get_feature_names_out())

        if len(transformed_names) != EXPECTED_TRANSFORMED_FEATURES:
            raise RuntimeError("Unexpected transformed feature count.")

        if canonical_names is None:
            canonical_names = transformed_names
        elif transformed_names != canonical_names:
            raise RuntimeError("Transformed feature space changed.")

        background_t = preprocessor.transform(training[features])
        oos_t = preprocessor.transform(frame.iloc[eligible][features])

        masker = shap.maskers.Independent(
            background_t,
            max_samples=len(background_t),
        )

        explanation = shap.LinearExplainer(
            estimator,
            masker,
        )(oos_t)

        values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)

        if values.shape != (
            EXPECTED_OOS_PER_REPLICATE,
            EXPECTED_TRANSFORMED_FEATURES,
        ):
            raise RuntimeError("Unexpected SHAP matrix shape.")

        if base_values.ndim == 0:
            base_values = np.full(
                EXPECTED_OOS_PER_REPLICATE,
                float(base_values),
            )
        elif base_values.ndim > 1:
            base_values = base_values.reshape(
                EXPECTED_OOS_PER_REPLICATE,
                -1,
            )[:, 0]

        decision = estimator.decision_function(oos_t)
        error = float(
            np.max(
                np.abs(
                    base_values
                    + values.sum(axis=1)
                    - decision
                )
            )
        )

        max_additivity_error = max(max_additivity_error, error)

        if error > ADDITIVITY_TOLERANCE:
            raise RuntimeError(f"SHAP additivity failed: {error}")

        grouped = np.zeros(
            (EXPECTED_OOS_PER_REPLICATE, len(features)),
            dtype=float,
        )

        for transformed_index, transformed_name in enumerate(
            transformed_names
        ):
            feature = raw_name(transformed_name)
            grouped[:, raw_feature_index[feature]] += values[
                :,
                transformed_index,
            ]

        signed_sum[eligible] += grouped
        abs_sum[eligible] += np.abs(grouped)
        support[eligible] += 1

        eligible_ids = cell_ids[eligible]
        local_positions = {
            cell_id: position
            for position, cell_id in enumerate(eligible_ids)
            if cell_id in case_lookup
        }

        for cell_id, position in local_positions.items():
            frame_index = int(eligible[position])

            for feature_index, feature in enumerate(features):
                local_rows.append(
                    {
                        "case_label": case_lookup[cell_id],
                        "cell_id": cell_id,
                        "replicate": replicate,
                        "model_seed": model_seed,
                        "feature": feature,
                        "driver_family": DRIVER_FAMILY[feature],
                        "shap_value": float(
                            grouped[position, feature_index]
                        ),
                        "absolute_shap_value": float(
                            abs(grouped[position, feature_index])
                        ),
                        "base_value": float(base_values[position]),
                        "decision_function": float(decision[position]),
                        "Easting": float(frame.iloc[frame_index]["Easting"]),
                        "Northing": float(frame.iloc[frame_index]["Northing"]),
                    }
                )

        total_explanations += len(eligible)

        print(
            "SPATIAL_SHAP_REPLICATE_COMPLETE = "
            f"{replicate + 1}/{EXPECTED_REPLICATES}"
        )

    candidate_support = support[candidate_indices]

    if int(candidate_support.min()) < 4:
        raise RuntimeError("Unexpected low OOS explanation support.")

    if int(candidate_support.max()) > EXPECTED_REPLICATES:
        raise RuntimeError("Unexpected high OOS explanation support.")

    spatial_rows: list[dict[str, object]] = []

    for row_index in candidate_indices:
        cell_id = cell_ids[row_index]

        for feature_index, feature in enumerate(features):
            n = int(support[row_index, feature_index])

            if n <= 0:
                raise RuntimeError(
                    f"Candidate {cell_id} lacks explanation support."
                )

            spatial_rows.append(
                {
                    "cell_id": cell_id,
                    "Easting": float(frame.iloc[row_index]["Easting"]),
                    "Northing": float(frame.iloc[row_index]["Northing"]),
                    "feature": feature,
                    "driver_family": DRIVER_FAMILY[feature],
                    "oos_explanation_count": n,
                    "mean_signed_shap": float(
                        signed_sum[row_index, feature_index] / n
                    ),
                    "mean_abs_shap": float(
                        abs_sum[row_index, feature_index] / n
                    ),
                }
            )

    spatial = pd.DataFrame(spatial_rows)

    if len(spatial) != EXPECTED_CANDIDATES * EXPECTED_RAW_FEATURES:
        raise RuntimeError("Unexpected spatial contribution row count.")

    spatial.to_csv(SPATIAL_PATH, index=False, compression="gzip")

    local_replicates = pd.DataFrame(local_rows)

    expected_case_ids = set(cases["cell_id"].astype(str))
    observed_case_ids = set(local_replicates["cell_id"].astype(str))

    if observed_case_ids != expected_case_ids:
        raise RuntimeError("Not all deterministic local cases were explained.")

    local_summary = (
        local_replicates.groupby(
            [
                "case_label",
                "cell_id",
                "feature",
                "driver_family",
            ],
            as_index=False,
        )
        .agg(
            oos_explanation_count=("replicate", "nunique"),
            mean_signed_shap=("shap_value", "mean"),
            median_signed_shap=("shap_value", "median"),
            std_signed_shap=("shap_value", "std"),
            mean_abs_shap=("absolute_shap_value", "mean"),
        )
    )

    local_summary["direction"] = np.where(
        local_summary["mean_signed_shap"] > 0,
        "positive",
        np.where(
            local_summary["mean_signed_shap"] < 0,
            "negative",
            "neutral",
        ),
    )

    cases.to_csv(LOCAL_CASES_PATH, index=False)
    local_replicates.to_csv(LOCAL_REPLICATES_PATH, index=False)
    local_summary.to_csv(LOCAL_SUMMARY_PATH, index=False)

    spatial_with_score = spatial.merge(
        scores[
            [
                "cell_id",
                "candidate_rank",
                "prospectivity_score_oos_mean",
                "candidate_top_1pct",
                "candidate_top_5pct",
                "candidate_top_10pct",
            ]
        ].assign(cell_id=lambda x: x["cell_id"].astype(str)),
        on="cell_id",
        how="left",
        validate="many_to_one",
    )

    dominant_feature = (
        spatial_with_score.sort_values(
            [
                "cell_id",
                "mean_abs_shap",
                "feature",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("cell_id", as_index=False)
        .first()
        .rename(
            columns={
                "feature": "dominant_feature",
                "driver_family": "dominant_feature_family",
                "mean_signed_shap": "dominant_feature_mean_signed_shap",
                "mean_abs_shap": "dominant_feature_mean_abs_shap",
                "oos_explanation_count": "oos_explanation_count",
            }
        )
    )

    family = (
        spatial_with_score.groupby(
            [
                "cell_id",
                "Easting",
                "Northing",
                "driver_family",
            ],
            as_index=False,
        )
        .agg(
            family_mean_abs_shap=("mean_abs_shap", "sum"),
            family_mean_signed_shap=("mean_signed_shap", "sum"),
        )
    )

    dominant_family = (
        family.sort_values(
            [
                "cell_id",
                "family_mean_abs_shap",
                "driver_family",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("cell_id", as_index=False)
        .first()
        .rename(
            columns={
                "driver_family": "dominant_family",
                "family_mean_abs_shap": "dominant_family_mean_abs_shap",
                "family_mean_signed_shap": "dominant_family_mean_signed_shap",
            }
        )
    )

    dominant = dominant_feature.merge(
        dominant_family[
            [
                "cell_id",
                "dominant_family",
                "dominant_family_mean_abs_shap",
                "dominant_family_mean_signed_shap",
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    if len(dominant) != EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected dominant-driver row count.")

    dominant.to_csv(DOMINANT_PATH, index=False)

    local_manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "D",
        "status": "PASS",
        "method": "deterministic_local_oos_shap",
        "selection_rules": CASE_RANKS,
        "selected_case_count": len(cases),
        "oracle_loaded": False,
        "model_reselected": False,
        "causal_interpretation_allowed": False,
    }

    spatial_manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "D",
        "status": "PASS",
        "method": "oos_ensemble_spatial_shap_aggregation",
        "shap_version": SHAP_VERSION,
        "background_replicates": EXPECTED_REPLICATES,
        "oos_candidates_per_replicate": EXPECTED_OOS_PER_REPLICATE,
        "total_oos_explanations": total_explanations,
        "candidate_rows": EXPECTED_CANDIDATES,
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "spatial_long_rows": len(spatial),
        "spatial_storage": "gzip_csv",
        "dominant_driver_rows": len(dominant),
        "support_min": int(candidate_support.min()),
        "support_max": int(candidate_support.max()),
        "maximum_additivity_error": max_additivity_error,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "output_space": "log_odds",
        "aggregation_semantics": (
            "per-cell mean contribution across replicate models "
            "where the candidate was out of sample"
        ),
        "dominant_feature_rule": "maximum_mean_absolute_oos_shap",
        "dominant_family_rule": (
            "maximum_sum_of_feature_mean_absolute_oos_shap_within_family"
        ),
        "oracle_loaded": False,
        "model_reselected": False,
        "causal_interpretation_allowed": False,
        "correlated_predictor_attribution_requires_caution": True,
    }

    LOCAL_MANIFEST_PATH.write_text(
        json.dumps(local_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    SPATIAL_MANIFEST_PATH.write_text(
        json.dumps(spatial_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("LOCAL_EXPLAINABILITY = PASS")
    print(f"LOCAL_CASES = {len(cases)}")
    print("LOCAL_SELECTION = DETERMINISTIC_RANK_RULES")
    print("SPATIAL_EXPLAINABILITY = PASS")
    print(f"SPATIAL_LONG_ROWS = {len(spatial)}")
    print(f"DOMINANT_DRIVER_ROWS = {len(dominant)}")
    print(f"OOS_SUPPORT_MIN = {int(candidate_support.min())}")
    print(f"OOS_SUPPORT_MAX = {int(candidate_support.max())}")
    print(
        "MAXIMUM_ADDITIVITY_ERROR = "
        f"{max_additivity_error:.16g}"
    )
    print("ORACLE_USED = NO")
    print("MODEL_RESELECTED = NO")
    print("LAB06_PHASE_D = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
