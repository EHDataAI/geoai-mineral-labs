"""Run the core compositional analysis for LAB04.

This script compares conventional geochemical representations with
compositional log-ratio representations using a controlled synthetic dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from geochemistry_config import (
    get_detection_limits,
    get_elements,
    get_replacement_fractions,
    load_contract,
)

from geoai.composition import (
    closure,
    clr,
    ilr,
    replace_below_detection_limit,
)


LAB_DIR = Path(__file__).resolve().parent
DATA_PATH = LAB_DIR / "data" / "synthetic_geochemistry.csv"
RESULTS_DIR = LAB_DIR / "results"

CONTRACT = load_contract()
ELEMENTS = get_elements()
LOD = get_detection_limits()


def run_pca(
    values: np.ndarray,
    feature_names: list[str],
    prefix: str,
    standardize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run PCA with an explicit scaling policy.

    Conventional concentration representations are standardized because
    elements occur at very different numerical scales. CLR and ILR
    representations are not variance-standardized so the Aitchison geometry
    is preserved.
    """
    if standardize:
        analysis_values = StandardScaler().fit_transform(values)
    else:
        analysis_values = values

    model = PCA()
    scores = model.fit_transform(analysis_values)

    score_columns = [
        f"{prefix}_PC{index}"
        for index in range(1, scores.shape[1] + 1)
    ]

    scores_df = pd.DataFrame(
        scores,
        columns=score_columns,
    )

    loadings_df = pd.DataFrame(
        model.components_.T,
        index=feature_names,
        columns=score_columns,
    )

    loadings_df.index.name = "feature"

    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                model.explained_variance_ratio_
            ),
            "eigenvalue": model.explained_variance_,
        }
    )

    return scores_df, loadings_df, explained


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATA_PATH)

    observed = frame[ELEMENTS].to_numpy(dtype=float)

    replaced = replace_below_detection_limit(
        observed,
        detection_limits=LOD,
        fraction=get_replacement_fractions()["lod_half"],
    )

    complete_mask = np.isfinite(replaced).all(axis=1)

    analytical_frame = frame.loc[complete_mask].reset_index(drop=True)
    complete_values = replaced[complete_mask]

    closed_values = closure(
        complete_values,
        total=CONTRACT["closure_total"],
    )

    clr_values = clr(closed_values)
    ilr_values = ilr(closed_values)

    raw_scores, raw_loadings, raw_explained = run_pca(
        complete_values,
        ELEMENTS,
        "RAW",
        standardize=True,
    )

    closed_scores, closed_loadings, closed_explained = run_pca(
        closed_values,
        ELEMENTS,
        "CLOSED",
        standardize=True,
    )

    clr_scores, clr_loadings, clr_explained = run_pca(
        clr_values,
        ELEMENTS,
        "CLR",
        standardize=False,
    )

    ilr_names = [
        f"ILR_{index}"
        for index in range(1, ilr_values.shape[1] + 1)
    ]

    ilr_scores, ilr_loadings, ilr_explained = run_pca(
        ilr_values,
        ilr_names,
        "ILR",
        standardize=False,
    )

    analytical_output = analytical_frame[
        [
            "sample_id",
            "easting",
            "northing",
            "lithology",
            "anomaly_class",
            "latent_mineralization",
            "latent_pathfinder",
            "latent_alteration",
        ]
    ].copy()

    for index, element in enumerate(ELEMENTS):
        analytical_output[f"{element}_replaced"] = complete_values[:, index]
        analytical_output[f"{element}_closed"] = closed_values[:, index]
        analytical_output[f"{element}_clr"] = clr_values[:, index]

    for index, name in enumerate(ilr_names):
        analytical_output[name] = ilr_values[:, index]

    analytical_output = pd.concat(
        [
            analytical_output,
            raw_scores,
            closed_scores,
            clr_scores,
            ilr_scores,
        ],
        axis=1,
    )

    analytical_output.to_csv(
        RESULTS_DIR / "analytical_dataset.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    explained = pd.concat(
        [
            raw_explained.assign(representation="raw"),
            closed_explained.assign(representation="closed"),
            clr_explained.assign(representation="clr"),
            ilr_explained.assign(representation="ilr"),
        ],
        ignore_index=True,
    )

    explained.to_csv(
        RESULTS_DIR / "pca_explained_variance.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    raw_loadings.to_csv(
        RESULTS_DIR / "pca_loadings_raw.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    closed_loadings.to_csv(
        RESULTS_DIR / "pca_loadings_closed.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    clr_loadings.to_csv(
        RESULTS_DIR / "pca_loadings_clr.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    ilr_loadings.to_csv(
        RESULTS_DIR / "pca_loadings_ilr.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    raw_corr = pd.DataFrame(
        complete_values,
        columns=ELEMENTS,
    ).corr()

    closed_corr = pd.DataFrame(
        closed_values,
        columns=ELEMENTS,
    ).corr()

    clr_corr = pd.DataFrame(
        clr_values,
        columns=ELEMENTS,
    ).corr()

    raw_corr.to_csv(
        RESULTS_DIR / "correlation_raw.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    closed_corr.to_csv(
        RESULTS_DIR / "correlation_closed.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    clr_corr.to_csv(
        RESULTS_DIR / "correlation_clr.csv",
        encoding="utf-8",
        lineterminator="\n",
    )

    clr_nonzero_eigenvalues = (
        clr_explained["eigenvalue"]
        .to_numpy(dtype=float)[: ilr_values.shape[1]]
    )

    ilr_eigenvalues = ilr_explained["eigenvalue"].to_numpy(dtype=float)

    coda_eigenvalues_match = np.allclose(
        clr_nonzero_eigenvalues,
        ilr_eigenvalues,
        rtol=1e-10,
        atol=1e-12,
    )

    if not coda_eigenvalues_match:
        raise RuntimeError(
            "CLR and ILR PCA eigenvalues are not equivalent within tolerance."
        )

    summary = {
        "input_rows": int(len(frame)),
        "complete_rows": int(complete_mask.sum()),
        "excluded_missing_rows": int((~complete_mask).sum()),
        "elements": len(ELEMENTS),
        "clr_dimension": int(clr_values.shape[1]),
        "clr_rank": int(np.linalg.matrix_rank(clr_values)),
        "ilr_dimension": int(ilr_values.shape[1]),
        "coda_eigenvalues_match": bool(coda_eigenvalues_match),
        "raw_pc1_variance": float(
            raw_explained.loc[0, "explained_variance_ratio"]
        ),
        "closed_pc1_variance": float(
            closed_explained.loc[0, "explained_variance_ratio"]
        ),
        "clr_pc1_variance": float(
            clr_explained.loc[0, "explained_variance_ratio"]
        ),
        "ilr_pc1_variance": float(
            ilr_explained.loc[0, "explained_variance_ratio"]
        ),
    }

    with open(
        RESULTS_DIR / "analysis_summary.json",
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

    print("LAB04_CODA_ANALYSIS = PASS")
    print("INPUT_ROWS =", summary["input_rows"])
    print("COMPLETE_ROWS =", summary["complete_rows"])
    print("EXCLUDED_MISSING_ROWS =", summary["excluded_missing_rows"])
    print("CLR_DIMENSION =", summary["clr_dimension"])
    print("CLR_RANK =", summary["clr_rank"])
    print("ILR_DIMENSION =", summary["ilr_dimension"])
    print("CODA_EIGENVALUES_MATCH =", summary["coda_eigenvalues_match"])
    print()
    print("PC1_EXPLAINED_VARIANCE")
    print(
        explained.groupby("representation")
        .first()["explained_variance_ratio"]
        .round(4)
    )


if __name__ == "__main__":
    main()
