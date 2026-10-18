"""LABORATORY 06 Phase B: baseline interpretability for the frozen ensemble."""

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

COEFFICIENTS_PATH = RESULTS_DIR / "transformed_coefficients.csv"
SUMMARY_PATH = RESULTS_DIR / "coefficient_summary.csv"
MANIFEST_PATH = RESULTS_DIR / "baseline_interpretability_manifest.json"

EXPECTED_MODEL = "logistic_regression"
EXPECTED_REPLICATES = 10


def load_module(path: Path, module_name: str):
    if str(LAB05_DIR) not in sys.path:
        sys.path.insert(0, str(LAB05_DIR))

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gate04 = load_module(
        GATE04_PATH,
        "lab05_gate04_for_lab06_baseline",
    )
    gate051 = load_module(
        GATE051_PATH,
        "lab05_gate051_for_lab06_baseline",
    )

    frame, feature_columns = gate051.load_safe_evidence(gate04)

    rows: list[dict[str, object]] = []
    canonical_names: list[str] | None = None

    for replicate in range(EXPECTED_REPLICATES):
        training = gate04.sample_background(frame, replicate)
        model_seed = gate04.SEED + replicate * 1000

        pipeline = gate04.build_model_pipeline(
            EXPECTED_MODEL,
            feature_columns,
            model_seed,
        )

        pipeline.fit(
            training[feature_columns],
            training[gate04.TARGET].to_numpy(dtype=int),
        )

        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]

        names = list(preprocessor.get_feature_names_out())
        coefficients = estimator.coef_.reshape(-1)

        if len(names) != len(coefficients):
            raise RuntimeError(
                "Transformed feature names and coefficients differ in length."
            )

        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise RuntimeError(
                "Transformed feature space changed across background replicates."
            )

        for name, coefficient in zip(names, coefficients, strict=True):
            coefficient = float(coefficient)
            rows.append(
                {
                    "replicate": replicate,
                    "model_seed": model_seed,
                    "transformed_feature": name,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "sign": (
                        "positive"
                        if coefficient > 0
                        else "negative"
                        if coefficient < 0
                        else "zero"
                    ),
                }
            )

    coefficients = pd.DataFrame(rows)

    if coefficients["replicate"].nunique() != EXPECTED_REPLICATES:
        raise RuntimeError("Unexpected coefficient replicate count.")

    feature_counts = coefficients.groupby(
        "replicate"
    )["transformed_feature"].nunique()

    if feature_counts.nunique() != 1:
        raise RuntimeError("Transformed feature count is not stable.")

    summary = (
        coefficients.groupby("transformed_feature", as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_median=("coefficient", "median"),
            coefficient_std=("coefficient", "std"),
            absolute_coefficient_mean=("absolute_coefficient", "mean"),
            absolute_coefficient_median=("absolute_coefficient", "median"),
            positive_fraction=(
                "coefficient",
                lambda values: float((values > 0).mean()),
            ),
            negative_fraction=(
                "coefficient",
                lambda values: float((values < 0).mean()),
            ),
        )
        .sort_values(
            ["absolute_coefficient_mean", "transformed_feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    summary["rank_by_mean_absolute_coefficient"] = np.arange(
        1,
        len(summary) + 1,
        dtype=int,
    )

    summary["sign_stability"] = np.maximum(
        summary["positive_fraction"].to_numpy(dtype=float),
        summary["negative_fraction"].to_numpy(dtype=float),
    )

    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "B",
        "purpose": "baseline_interpretability",
        "status": "PASS",
        "model": EXPECTED_MODEL,
        "background_replicates": EXPECTED_REPLICATES,
        "raw_feature_count": len(feature_columns),
        "transformed_feature_count": int(
            coefficients["transformed_feature"].nunique()
        ),
        "transformed_feature_names": canonical_names,
        "coefficient_semantics": (
            "standardized_numeric_and_one_hot_encoded_logistic_regression"
        ),
        "causal_interpretation_allowed": False,
        "oracle_loaded": False,
        "model_reselected": False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("TRANSFORMED_FEATURE_SPACE = PASS")
    print(
        "TRANSFORMED_FEATURE_COUNT = "
        f"{manifest['transformed_feature_count']}"
    )
    print("COEFFICIENT_EXTRACTION = PASS")
    print("COEFFICIENT_SIGN_STABILITY = PASS")
    print("CAUSAL_INTERPRETATION = NO")
    print("ORACLE_USED = NO")
    print("LAB06_PHASE_B_BASELINE = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
