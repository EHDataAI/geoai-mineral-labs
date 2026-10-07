"""Evaluate sensitivity to simple below-LOD replacement rules for LAB04."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from geochemistry_config import (
    get_detection_limits,
    get_elements,
    get_replacement_fractions,
    load_contract,
)

from geoai.composition import (
    closure,
    clr,
    replace_below_detection_limit,
)


LAB_DIR = Path(__file__).resolve().parent
DATA_PATH = LAB_DIR / "data" / "synthetic_geochemistry.csv"
RESULTS_DIR = LAB_DIR / "results"

CONTRACT = load_contract()
ELEMENTS = get_elements()
LOD = get_detection_limits()

REPLACEMENT_RULES = get_replacement_fractions()


def run_clr_pca(
    frame: pd.DataFrame,
    replacement_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply replacement, CLR transform, and PCA."""
    observed = frame[ELEMENTS].to_numpy(dtype=float)

    replaced = replace_below_detection_limit(
        observed,
        detection_limits=LOD,
        fraction=replacement_fraction,
    )

    complete_mask = np.isfinite(replaced).all(axis=1)
    complete_values = replaced[complete_mask]

    closed_values = closure(
        complete_values,
        total=CONTRACT["closure_total"],
    )

    clr_values = clr(closed_values)

    model = PCA()
    scores = model.fit_transform(clr_values)

    result = pd.DataFrame(
        {
            "sample_id": frame.loc[
                complete_mask,
                "sample_id",
            ].to_numpy(),
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2],
        }
    )

    return result, model.explained_variance_ratio_


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATA_PATH)

    scores_by_rule: dict[str, pd.DataFrame] = {}
    variance_by_rule: dict[str, np.ndarray] = {}

    for name, fraction in REPLACEMENT_RULES.items():
        scores, explained = run_clr_pca(
            frame,
            replacement_fraction=fraction,
        )

        scores_by_rule[name] = scores
        variance_by_rule[name] = explained

    merged = scores_by_rule["lod_half"].merge(
        scores_by_rule["lod_sqrt2"],
        on="sample_id",
        suffixes=("_lod_half", "_lod_sqrt2"),
        validate="one_to_one",
    )

    score_correlations = {}

    for component in ["PC1", "PC2", "PC3"]:
        score_correlations[component] = float(
            merged[
                [
                    f"{component}_lod_half",
                    f"{component}_lod_sqrt2",
                ]
            ].corr().iloc[0, 1]
        )

    variance_rows = []

    for rule, explained in variance_by_rule.items():
        for index, value in enumerate(explained, start=1):
            variance_rows.append(
                {
                    "replacement_rule": rule,
                    "component": f"PC{index}",
                    "explained_variance_ratio": float(value),
                }
            )

    variance_table = pd.DataFrame(variance_rows)

    variance_table.to_csv(
        RESULTS_DIR / "lod_sensitivity_variance.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    merged.to_csv(
        RESULTS_DIR / "lod_sensitivity_scores.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    pc1_difference = abs(
        variance_by_rule["lod_half"][0]
        - variance_by_rule["lod_sqrt2"][0]
    )

    summary = {
        "rules": {
            name: float(value)
            for name, value in REPLACEMENT_RULES.items()
        },
        "common_samples": int(len(merged)),
        "pc1_score_correlation": score_correlations["PC1"],
        "pc2_score_correlation": score_correlations["PC2"],
        "pc3_score_correlation": score_correlations["PC3"],
        "pc1_explained_variance_lod_half": float(
            variance_by_rule["lod_half"][0]
        ),
        "pc1_explained_variance_lod_sqrt2": float(
            variance_by_rule["lod_sqrt2"][0]
        ),
        "pc1_explained_variance_absolute_difference": float(
            pc1_difference
        ),
    }

    with open(
        RESULTS_DIR / "lod_sensitivity_summary.json",
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print("LAB04_LOD_SENSITIVITY = PASS")
    print("COMMON_SAMPLES =", summary["common_samples"])
    print(
        "PC1_SCORE_CORRELATION =",
        round(summary["pc1_score_correlation"], 6),
    )
    print(
        "PC2_SCORE_CORRELATION =",
        round(summary["pc2_score_correlation"], 6),
    )
    print(
        "PC3_SCORE_CORRELATION =",
        round(summary["pc3_score_correlation"], 6),
    )
    print(
        "PC1_VARIANCE_DIFFERENCE =",
        round(
            summary[
                "pc1_explained_variance_absolute_difference"
            ],
            8,
        ),
    )


if __name__ == "__main__":
    main()
