"""LABORATORY 06 Phase E: explanation stability and method agreement."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LAB05 = ROOT / "labs" / "lab05_mineral_prospectivity"
LAB06 = ROOT / "labs" / "lab06_explainable_geoai"
RESULTS = LAB06 / "results"

G04_PATH = LAB05 / "05d_run_spatial_benchmark.py"
G051_PATH = LAB05 / "05e1_score_full_grid_oos.py"

COEFFICIENT_PATH = RESULTS / "coefficient_summary.csv"
PERMUTATION_PATH = RESULTS / "permutation_importance_summary.csv"
SHAP_GLOBAL_PATH = RESULTS / "shap_global_raw.csv"
RUFF_BASELINE_PATH = RESULTS / "ruff_legacy_baseline_phase_d.json"

SHAP_REPLICATE_PATH = RESULTS / "shap_replicate_stability.csv"
STABILITY_SUMMARY_PATH = RESULTS / "explanation_stability_summary.csv"
RANK_AGREEMENT_PATH = RESULTS / "explanation_rank_agreement.csv"
METHOD_AGREEMENT_PATH = RESULTS / "explanation_method_agreement.csv"
DRIVER_FAMILY_PATH = RESULTS / "driver_family_stability.csv"
MANIFEST_PATH = RESULTS / "explanation_stability_manifest.json"

EXPECTED_REPLICATES = 10
EXPECTED_CANDIDATES = 6300
EXPECTED_OOS_PER_REPLICATE = 5800
EXPECTED_RAW_FEATURES = 9
EXPECTED_TRANSFORMED_FEATURES = 12
SHAP_VERSION = "0.53.0"
ADDITIVITY_TOLERANCE = 1e-8
TOP_K_VALUES = (3, 5)

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


def coefficient_raw_summary() -> pd.DataFrame:
    coefficients = pd.read_csv(COEFFICIENT_PATH)

    coefficients["raw_feature"] = coefficients[
        "transformed_feature"
    ].map(raw_name)

    grouped = (
        coefficients.groupby("raw_feature", as_index=False)
        .agg(
            coefficient_importance=(
                "absolute_coefficient_mean",
                "sum",
            ),
            coefficient_signed=(
                "coefficient_mean",
                "sum",
            ),
        )
    )

    grouped["coefficient_rank"] = (
        grouped["coefficient_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return grouped


def topk_pairwise_jaccard(
    replicate_frame: pd.DataFrame,
    k: int,
) -> tuple[float, float, float]:
    replicate_top: dict[int, set[str]] = {}

    for replicate, group in replicate_frame.groupby("replicate"):
        selected = set(
            group.nsmallest(k, "rank")["feature"].astype(str)
        )
        replicate_top[int(replicate)] = selected

    values: list[float] = []

    for left, right in itertools.combinations(
        sorted(replicate_top),
        2,
    ):
        a = replicate_top[left]
        b = replicate_top[right]
        values.append(len(a & b) / len(a | b))

    arr = np.asarray(values, dtype=float)

    return (
        float(arr.mean()),
        float(np.median(arr)),
        float(arr.min()),
    )


def main() -> int:
    if importlib.metadata.version("shap") != SHAP_VERSION:
        raise RuntimeError("Unexpected SHAP version.")

    RESULTS.mkdir(parents=True, exist_ok=True)

    g04 = load_module(G04_PATH, "lab05_g04_for_lab06_phase_e")
    g051 = load_module(G051_PATH, "lab05_g051_for_lab06_phase_e")

    frame, features = g051.load_safe_evidence(g04)

    if len(features) != EXPECTED_RAW_FEATURES:
        raise RuntimeError("Unexpected raw Feature Set A width.")

    if set(features) != set(DRIVER_FAMILY):
        raise RuntimeError("Driver family contract mismatch.")

    candidate_mask = frame[g04.TARGET].to_numpy(dtype=int) == 0
    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) != EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected candidate count.")

    cell_ids = frame["cell_id"].astype(str).to_numpy()
    raw_feature_index = {
        feature: index
        for index, feature in enumerate(features)
    }

    rows: list[dict[str, object]] = []
    canonical_names: list[str] | None = None
    maximum_additivity_error = 0.0
    total_oos_explanations = 0

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
                    cell_ids[index] not in training_background_ids
                    for index in candidate_indices
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

        transformed_names = list(
            preprocessor.get_feature_names_out()
        )

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
        additivity_error = float(
            np.max(
                np.abs(
                    base_values
                    + values.sum(axis=1)
                    - decision
                )
            )
        )

        maximum_additivity_error = max(
            maximum_additivity_error,
            additivity_error,
        )

        if additivity_error > ADDITIVITY_TOLERANCE:
            raise RuntimeError(
                f"SHAP additivity failed: {additivity_error}"
            )

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

        importance = np.mean(np.abs(grouped), axis=0)
        signed = np.mean(grouped, axis=0)

        order = np.argsort(-importance, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(features) + 1)

        for feature_index, feature in enumerate(features):
            rows.append(
                {
                    "replicate": replicate,
                    "model_seed": model_seed,
                    "feature": feature,
                    "driver_family": DRIVER_FAMILY[feature],
                    "mean_abs_shap": float(
                        importance[feature_index]
                    ),
                    "mean_signed_shap": float(
                        signed[feature_index]
                    ),
                    "sign": (
                        "positive"
                        if signed[feature_index] > 0
                        else "negative"
                        if signed[feature_index] < 0
                        else "neutral"
                    ),
                    "rank": int(ranks[feature_index]),
                    "oos_candidate_count": len(eligible),
                }
            )

        total_oos_explanations += len(eligible)

        print(
            "STABILITY_SHAP_REPLICATE_COMPLETE = "
            f"{replicate + 1}/{EXPECTED_REPLICATES}"
        )

    replicate_frame = pd.DataFrame(rows)

    if len(replicate_frame) != (
        EXPECTED_REPLICATES * EXPECTED_RAW_FEATURES
    ):
        raise RuntimeError("Unexpected SHAP replicate row count.")

    replicate_frame.to_csv(SHAP_REPLICATE_PATH, index=False)

    stability = (
        replicate_frame.groupby(
            ["feature", "driver_family"],
            as_index=False,
        )
        .agg(
            importance_mean=("mean_abs_shap", "mean"),
            importance_std=("mean_abs_shap", "std"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
            positive_fraction=(
                "mean_signed_shap",
                lambda x: float((x > 0).mean()),
            ),
            negative_fraction=(
                "mean_signed_shap",
                lambda x: float((x < 0).mean()),
            ),
        )
    )

    stability["importance_cv"] = np.where(
        stability["importance_mean"] > 0,
        stability["importance_std"] / stability["importance_mean"],
        np.nan,
    )

    stability["sign_stability"] = stability[
        ["positive_fraction", "negative_fraction"]
    ].max(axis=1)

    stability["stable_top5_fraction"] = stability["feature"].map(
        replicate_frame.assign(
            is_top5=lambda x: x["rank"] <= 5
        )
        .groupby("feature")["is_top5"]
        .mean()
    )

    stability = stability.sort_values(
        ["rank_mean", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)

    stability.to_csv(STABILITY_SUMMARY_PATH, index=False)

    coefficient = coefficient_raw_summary()
    permutation = pd.read_csv(PERMUTATION_PATH).rename(
        columns={
            "rank": "permutation_rank",
            "importance_mean": "permutation_importance",
        }
    )
    shap_global = pd.read_csv(SHAP_GLOBAL_PATH).rename(
        columns={
            "raw_feature": "feature",
            "rank": "shap_rank",
            "mean_abs_shap": "shap_importance",
        }
    )

    agreement = (
        coefficient.rename(columns={"raw_feature": "feature"})
        .merge(
            permutation[
                [
                    "feature",
                    "permutation_importance",
                    "permutation_rank",
                ]
            ],
            on="feature",
            validate="one_to_one",
        )
        .merge(
            shap_global[
                [
                    "feature",
                    "shap_importance",
                    "shap_rank",
                ]
            ],
            on="feature",
            validate="one_to_one",
        )
    )

    agreement = agreement[
        [
            "feature",
            "coefficient_importance",
            "coefficient_rank",
            "permutation_importance",
            "permutation_rank",
            "shap_importance",
            "shap_rank",
        ]
    ].sort_values(
        ["shap_rank", "feature"],
        kind="mergesort",
    )

    agreement.to_csv(RANK_AGREEMENT_PATH, index=False)

    method_pairs = [
        (
            "coefficient",
            "permutation",
            "coefficient_rank",
            "permutation_rank",
        ),
        (
            "coefficient",
            "shap",
            "coefficient_rank",
            "shap_rank",
        ),
        (
            "permutation",
            "shap",
            "permutation_rank",
            "shap_rank",
        ),
    ]

    method_rows: list[dict[str, object]] = []

    for left, right, left_rank, right_rank in method_pairs:
        rho = float(
            spearmanr(
                agreement[left_rank],
                agreement[right_rank],
            ).statistic
        )

        record: dict[str, object] = {
            "method_left": left,
            "method_right": right,
            "spearman_rank_correlation": rho,
        }

        for k in TOP_K_VALUES:
            a = set(
                agreement.nsmallest(k, left_rank)["feature"]
            )
            b = set(
                agreement.nsmallest(k, right_rank)["feature"]
            )
            record[f"top{k}_jaccard"] = len(a & b) / len(a | b)

        method_rows.append(record)

    method_agreement = pd.DataFrame(method_rows)
    method_agreement.to_csv(METHOD_AGREEMENT_PATH, index=False)

    family = (
        replicate_frame.groupby(
            ["replicate", "driver_family"],
            as_index=False,
        )
        .agg(
            family_mean_abs_shap=("mean_abs_shap", "sum"),
            family_mean_signed_shap=("mean_signed_shap", "sum"),
        )
    )

    family["family_rank"] = family.groupby("replicate")[
        "family_mean_abs_shap"
    ].rank(method="first", ascending=False).astype(int)

    family_summary = (
        family.groupby("driver_family", as_index=False)
        .agg(
            family_importance_mean=("family_mean_abs_shap", "mean"),
            family_importance_std=("family_mean_abs_shap", "std"),
            family_rank_mean=("family_rank", "mean"),
            family_rank_std=("family_rank", "std"),
            positive_fraction=(
                "family_mean_signed_shap",
                lambda x: float((x > 0).mean()),
            ),
            negative_fraction=(
                "family_mean_signed_shap",
                lambda x: float((x < 0).mean()),
            ),
        )
    )

    family_summary["family_sign_stability"] = family_summary[
        ["positive_fraction", "negative_fraction"]
    ].max(axis=1)

    family_summary = family_summary.sort_values(
        ["family_rank_mean", "driver_family"],
        kind="mergesort",
    )

    family_summary.to_csv(DRIVER_FAMILY_PATH, index=False)

    topk_summary: dict[str, dict[str, float]] = {}

    for k in TOP_K_VALUES:
        mean_j, median_j, min_j = topk_pairwise_jaccard(
            replicate_frame,
            k,
        )
        topk_summary[f"top{k}"] = {
            "pairwise_jaccard_mean": mean_j,
            "pairwise_jaccard_median": median_j,
            "pairwise_jaccard_min": min_j,
        }

    minimum_sign_stability = float(
        stability["sign_stability"].min()
    )
    median_rank_std = float(
        stability["rank_std"].median()
    )
    minimum_method_rho = float(
        method_agreement["spearman_rank_correlation"].min()
    )

    manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "E",
        "status": "PASS",
        "method": "oos_shap_stability_and_cross_method_agreement",
        "shap_version": SHAP_VERSION,
        "background_replicates": EXPECTED_REPLICATES,
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "total_oos_explanations": total_oos_explanations,
        "maximum_additivity_error": maximum_additivity_error,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "topk_pairwise_stability": topk_summary,
        "minimum_feature_sign_stability": minimum_sign_stability,
        "median_feature_rank_std": median_rank_std,
        "minimum_method_spearman_rho": minimum_method_rho,
        "method_agreement_pairs": len(method_agreement),
        "driver_family_count": len(family_summary),
        "oracle_loaded": False,
        "model_reselected": False,
        "causal_interpretation_allowed": False,
        "correlated_predictor_attribution_requires_caution": True,
        "interpretation": (
            "Stability measures describe robustness of attribution "
            "under the frozen OOS ensemble and do not establish causality."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("EXPLANATION_STABILITY = PASS")
    print(f"SHAP_REPLICATE_ROWS = {len(replicate_frame)}")
    print(f"STABILITY_FEATURE_ROWS = {len(stability)}")
    print(f"METHOD_AGREEMENT_PAIRS = {len(method_agreement)}")
    print(f"DRIVER_FAMILY_ROWS = {len(family_summary)}")
    print(
        "MINIMUM_FEATURE_SIGN_STABILITY = "
        f"{minimum_sign_stability:.6f}"
    )
    print(
        "MINIMUM_METHOD_SPEARMAN_RHO = "
        f"{minimum_method_rho:.6f}"
    )
    print(
        "MAXIMUM_ADDITIVITY_ERROR = "
        f"{maximum_additivity_error:.16g}"
    )
    print("ORACLE_USED = NO")
    print("MODEL_RESELECTED = NO")
    print("LAB06_PHASE_E = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
