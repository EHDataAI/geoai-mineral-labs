"""LAB05 Gate 3.1: synthetic geological plausibility audit.

Diagnostic only:
- does not modify evidence data;
- does not load oracle fields;
- does not select a model;
- does not open hidden deposits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from prospectivity_config import KNOWN_POSITIVE_COUNT


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

DATA_PATH = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

RESULTS_DIR = LAB_DIR / "results"

METRICS_PATH = (
    RESULTS_DIR
    / "synthetic_plausibility_metrics.csv"
)

SPATIAL_PATH = (
    RESULTS_DIR
    / "synthetic_spatial_diagnostics.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "synthetic_plausibility_manifest.json"
)


GEOCHEMICAL = [
    "Cu",
    "Mo",
    "Au",
    "As",
    "Sb",
    "Fe",
    "Mg",
    "Ca",
    "Al",
    "K",
]

PATHFINDER_METALS = [
    "Cu",
    "Mo",
    "Au",
    "As",
    "Sb",
]

CONTEXT_CONTINUOUS = [
    "distance_to_intrusive_m",
    "intrusive_affinity",
    "distance_to_fault_m",
    "distance_to_fault_intersection_m",
    "distance_to_contact_m",
    "magnetic_anomaly",
    "gravity_anomaly",
    "alteration_index",
]

SAFE_METADATA = [
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "known_positive",
    "lithology",
]

SAFE_COLUMNS = list(
    dict.fromkeys(
        [
            *SAFE_METADATA,
            *CONTEXT_CONTINUOUS,
            *GEOCHEMICAL,
        ]
    )
)

FORBIDDEN_ORACLE_FIELDS = {
    "true_deposit",
    "latent_favourability",
    "deposit_id",
    "distance_to_known_deposit",
    "true_deposit_probability",
}

EXPECTED_ROWS = 6400
EXPECTED_KNOWN_POSITIVES = KNOWN_POSITIVE_COUNT


def load_safe_data() -> pd.DataFrame:
    """Load only explicitly safe evidence fields."""
    header = pd.read_csv(
        DATA_PATH,
        nrows=0,
    )

    available = set(
        header.columns
    )

    missing = (
        set(SAFE_COLUMNS)
        - available
    )

    if missing:
        raise RuntimeError(
            "Missing safe evidence columns: "
            f"{sorted(missing)}"
        )

    frame = pd.read_csv(
        DATA_PATH,
        usecols=SAFE_COLUMNS,
    )

    if (
        set(frame.columns)
        & FORBIDDEN_ORACLE_FIELDS
    ):
        raise RuntimeError(
            "Oracle field loaded unexpectedly."
        )

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected grid row count."
        )

    if int(
        frame[
            "known_positive"
        ].sum()
    ) != EXPECTED_KNOWN_POSITIVES:
        raise RuntimeError(
            "Unexpected known-positive count."
        )

    if frame[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "cell_id must be unique."
        )

    return frame


def safe_skew(
    values: pd.Series,
) -> float:
    return float(
        values.skew()
    )


def safe_kurtosis(
    values: pd.Series,
) -> float:
    return float(
        values.kurt()
    )


def marginal_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Describe safe continuous variables without oracle information."""
    variables = [
        *CONTEXT_CONTINUOUS,
        *GEOCHEMICAL,
    ]

    rows: list[dict] = []

    for variable in variables:
        values = pd.to_numeric(
            frame[variable],
            errors="coerce",
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.empty:
            raise RuntimeError(
                f"No finite values for {variable}."
            )

        quantiles = finite.quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )

        positive = finite[
            finite > 0
        ]

        if len(
            positive
        ) >= 10:
            log_values = np.log10(
                positive
            )

            log_skew = float(
                pd.Series(
                    log_values
                ).skew()
            )
        else:
            log_skew = math.nan

        rows.append(
            {
                "analysis": (
                    "marginal_distribution"
                ),
                "variable": variable,
                "n": int(
                    len(finite)
                ),
                "missing_fraction": float(
                    values.isna().mean()
                ),
                "zero_fraction": float(
                    (
                        finite
                        == 0
                    ).mean()
                ),
                "min": float(
                    finite.min()
                ),
                "p01": float(
                    quantiles.loc[
                        0.01
                    ]
                ),
                "p05": float(
                    quantiles.loc[
                        0.05
                    ]
                ),
                "p25": float(
                    quantiles.loc[
                        0.25
                    ]
                ),
                "median": float(
                    quantiles.loc[
                        0.50
                    ]
                ),
                "p75": float(
                    quantiles.loc[
                        0.75
                    ]
                ),
                "p95": float(
                    quantiles.loc[
                        0.95
                    ]
                ),
                "p99": float(
                    quantiles.loc[
                        0.99
                    ]
                ),
                "max": float(
                    finite.max()
                ),
                "mean": float(
                    finite.mean()
                ),
                "std": float(
                    finite.std(
                        ddof=1
                    )
                ),
                "skewness": safe_skew(
                    finite
                ),
                "excess_kurtosis": safe_kurtosis(
                    finite
                ),
                "positive_log10_skewness": (
                    log_skew
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def robust_effect_size(
    frame: pd.DataFrame,
    variable: str,
) -> dict:
    """Known-positive contrast used only as observed-label diagnostic."""
    positive = frame.loc[
        frame[
            "known_positive"
        ]
        == 1,
        variable,
    ].astype(
        float
    )

    unlabeled = frame.loc[
        frame[
            "known_positive"
        ]
        == 0,
        variable,
    ].astype(
        float
    )

    pooled = frame[
        variable
    ].astype(
        float
    )

    iqr = float(
        pooled.quantile(
            0.75
        )
        - pooled.quantile(
            0.25
        )
    )

    median_delta = float(
        positive.median()
        - unlabeled.median()
    )

    if iqr > 0:
        robust_delta = (
            median_delta
            / iqr
        )
    else:
        robust_delta = math.nan

    return {
        "analysis": (
            "known_vs_unlabeled_contrast"
        ),
        "variable": variable,
        "known_median": float(
            positive.median()
        ),
        "unlabeled_median": float(
            unlabeled.median()
        ),
        "median_difference": median_delta,
        "pooled_iqr": iqr,
        "robust_median_effect_iqr": (
            robust_delta
        ),
    }


def observed_label_contrasts(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for variable in [
        *CONTEXT_CONTINUOUS,
        *GEOCHEMICAL,
    ]:
        rows.append(
            robust_effect_size(
                frame,
                variable,
            )
        )

    return pd.DataFrame(
        rows
    )


def lithology_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify geochemical heterogeneity by lithology."""
    rows: list[dict] = []

    groups = frame.groupby(
        "lithology",
        observed=True,
    )

    for variable in GEOCHEMICAL:
        global_median = float(
            frame[
                variable
            ].median()
        )

        global_iqr = float(
            frame[
                variable
            ].quantile(
                0.75
            )
            - frame[
                variable
            ].quantile(
                0.25
            )
        )

        medians = groups[
            variable
        ].median()

        range_medians = float(
            medians.max()
            - medians.min()
        )

        normalized = (
            range_medians
            / global_iqr
            if global_iqr > 0
            else math.nan
        )

        rows.append(
            {
                "analysis": (
                    "lithology_dependence"
                ),
                "variable": variable,
                "lithology_levels": int(
                    frame[
                        "lithology"
                    ].nunique()
                ),
                "global_median": (
                    global_median
                ),
                "global_iqr": (
                    global_iqr
                ),
                "range_of_lithology_medians": (
                    range_medians
                ),
                "normalized_median_range": (
                    normalized
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def correlation_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Inspect raw and log-transformed metal correlation structure."""
    raw = frame[
        GEOCHEMICAL
    ].astype(
        float
    )

    positive = raw.clip(
        lower=np.finfo(
            float
        ).tiny
    )

    log_values = np.log10(
        positive
    )

    rows: list[dict] = []

    for method_name, matrix_input in [
        (
            "raw",
            raw,
        ),
        (
            "log10",
            log_values,
        ),
    ]:
        corr = (
            matrix_input.corr()
        )

        for i, left in enumerate(
            GEOCHEMICAL
        ):
            for right in (
                GEOCHEMICAL[
                    i + 1:
                ]
            ):
                rows.append(
                    {
                        "analysis": (
                            "pairwise_correlation"
                        ),
                        "transform": (
                            method_name
                        ),
                        "variable_a": left,
                        "variable_b": right,
                        "correlation": float(
                            corr.loc[
                                left,
                                right,
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def grid_indices(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    east = np.sort(
        frame[
            "Easting"
        ].unique()
    )

    north = np.sort(
        frame[
            "Northing"
        ].unique()
    )

    if (
        len(east)
        * len(north)
        != len(frame)
    ):
        raise RuntimeError(
            "Grid is not rectangular."
        )

    e_map = {
        value: index
        for index, value
        in enumerate(east)
    }

    n_map = {
        value: index
        for index, value
        in enumerate(north)
    }

    e_idx = frame[
        "Easting"
    ].map(
        e_map
    ).to_numpy(
        dtype=int
    )

    n_idx = frame[
        "Northing"
    ].map(
        n_map
    ).to_numpy(
        dtype=int
    )

    return (
        e_idx,
        n_idx,
    )


def four_neighbor_moran(
    frame: pd.DataFrame,
    variable: str,
) -> float:
    """Simple rook-neighbor Moran-like diagnostic on rectangular grid."""
    e_idx, n_idx = grid_indices(
        frame
    )

    values = frame[
        variable
    ].to_numpy(
        dtype=float
    )

    centered = (
        values
        - values.mean()
    )

    lookup = {
        (
            int(e),
            int(n),
        ): idx
        for idx, (
            e,
            n,
        ) in enumerate(
            zip(
                e_idx,
                n_idx,
                strict=True,
            )
        )
    }

    numerator = 0.0
    weights = 0

    for idx, (
        e,
        n,
    ) in enumerate(
        zip(
            e_idx,
            n_idx,
            strict=True,
        )
    ):
        for de, dn in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ):
            neighbor = lookup.get(
                (
                    int(e + de),
                    int(n + dn),
                )
            )

            if neighbor is None:
                continue

            numerator += (
                centered[idx]
                * centered[
                    neighbor
                ]
            )

            weights += 1

    denominator = float(
        np.sum(
            centered ** 2
        )
    )

    if (
        denominator <= 0
        or weights == 0
    ):
        return math.nan

    n_rows = len(
        frame
    )

    return float(
        (
            n_rows
            / weights
        )
        * (
            numerator
            / denominator
        )
    )


def lag_1_semivariance(
    frame: pd.DataFrame,
    variable: str,
) -> float:
    """Mean semivariance between horizontal/vertical adjacent cells."""
    e_idx, n_idx = grid_indices(
        frame
    )

    values = frame[
        variable
    ].to_numpy(
        dtype=float
    )

    lookup = {
        (
            int(e),
            int(n),
        ): idx
        for idx, (
            e,
            n,
        ) in enumerate(
            zip(
                e_idx,
                n_idx,
                strict=True,
            )
        )
    }

    semivariances = []

    for idx, (
        e,
        n,
    ) in enumerate(
        zip(
            e_idx,
            n_idx,
            strict=True,
        )
    ):
        for de, dn in (
            (1, 0),
            (0, 1),
        ):
            neighbor = lookup.get(
                (
                    int(e + de),
                    int(n + dn),
                )
            )

            if neighbor is None:
                continue

            semivariances.append(
                0.5
                * (
                    values[idx]
                    - values[
                        neighbor
                    ]
                ) ** 2
            )

    if not semivariances:
        return math.nan

    return float(
        np.mean(
            semivariances
        )
    )


def spatial_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    variables = [
        "magnetic_anomaly",
        "gravity_anomaly",
        "alteration_index",
        *GEOCHEMICAL,
    ]

    rows = []

    for variable in variables:
        variance = float(
            frame[
                variable
            ].var(
                ddof=1
            )
        )

        semivariance = (
            lag_1_semivariance(
                frame,
                variable,
            )
        )

        rows.append(
            {
                "variable": variable,
                "rook_moran_like": (
                    four_neighbor_moran(
                        frame,
                        variable,
                    )
                ),
                "lag1_semivariance": (
                    semivariance
                ),
                "global_variance": variance,
                "lag1_semivariance_variance_ratio": (
                    semivariance
                    / variance
                    if variance > 0
                    else math.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def distance_relationships(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman associations used as plausibility diagnostics only."""
    distance_columns = [
        "distance_to_intrusive_m",
        "distance_to_fault_m",
        "distance_to_fault_intersection_m",
        "distance_to_contact_m",
    ]

    response_columns = [
        "alteration_index",
        "magnetic_anomaly",
        "gravity_anomaly",
        *PATHFINDER_METALS,
    ]

    rows = []

    for distance in (
        distance_columns
    ):
        for response in (
            response_columns
        ):
            rho = frame[
                [
                    distance,
                    response,
                ]
            ].corr(
                method="spearman"
            ).iloc[
                0,
                1,
            ]

            rows.append(
                {
                    "analysis": (
                        "distance_relationship"
                    ),
                    "distance_variable": (
                        distance
                    ),
                    "response_variable": (
                        response
                    ),
                    "spearman_rho": float(
                        rho
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_flags(
    marginal: pd.DataFrame,
    correlations: pd.DataFrame,
    spatial: pd.DataFrame,
    lithology: pd.DataFrame,
) -> list[dict]:
    """Create conservative review flags, not model-tuning rules."""
    flags: list[dict] = []

    metal_marginal = marginal.loc[
        marginal[
            "variable"
        ].isin(
            PATHFINDER_METALS
        )
    ]

    nearly_symmetric = int(
        (
            metal_marginal[
                "skewness"
            ].abs()
            < 0.25
        ).sum()
    )

    if nearly_symmetric == len(
        PATHFINDER_METALS
    ):
        flags.append(
            {
                "severity": "REVIEW",
                "check": (
                    "all_pathfinder_metals_nearly_symmetric"
                ),
                "detail": (
                    "All Cu/Mo/Au/As/Sb have "
                    "|skewness| < 0.25."
                ),
            }
        )

    raw_corr = correlations.loc[
        correlations[
            "transform"
        ]
        == "raw"
    ]

    very_high_pairs = raw_corr.loc[
        raw_corr[
            "correlation"
        ].abs()
        >= 0.95
    ]

    if not very_high_pairs.empty:
        flags.append(
            {
                "severity": "REVIEW",
                "check": (
                    "extreme_raw_geochemical_correlation"
                ),
                "detail": (
                    f"{len(very_high_pairs)} pair(s) "
                    "have |r| >= 0.95."
                ),
            }
        )

    key_spatial = spatial.loc[
        spatial[
            "variable"
        ].isin(
            [
                "magnetic_anomaly",
                "gravity_anomaly",
                "alteration_index",
            ]
        )
    ]

    if (
        key_spatial[
            "rook_moran_like"
        ].abs()
        < 0.05
    ).all():
        flags.append(
            {
                "severity": "REVIEW",
                "check": (
                    "weak_context_spatial_structure"
                ),
                "detail": (
                    "Magnetic, gravity, and alteration "
                    "all have |rook Moran-like| < 0.05."
                ),
            }
        )

    if (
        lithology[
            "normalized_median_range"
        ]
        < 0.05
    ).all():
        flags.append(
            {
                "severity": "REVIEW",
                "check": (
                    "weak_lithological_geochemical_dependence"
                ),
                "detail": (
                    "All geochemical variables show very "
                    "small median differences by lithology."
                ),
            }
        )

    return flags


def main() -> int:
    print(
        "LAB05_GATE03_1_AUDIT_START"
    )

    frame = load_safe_data()

    marginal = marginal_metrics(
        frame
    )

    label_contrasts = (
        observed_label_contrasts(
            frame
        )
    )

    lithology = lithology_metrics(
        frame
    )

    correlations = (
        correlation_metrics(
            frame
        )
    )

    spatial = spatial_metrics(
        frame
    )

    relationships = (
        distance_relationships(
            frame
        )
    )

    flags = build_flags(
        marginal,
        correlations,
        spatial,
        lithology,
    )

    combined = pd.concat(
        [
            marginal,
            label_contrasts,
            lithology,
            correlations,
            relationships,
        ],
        ignore_index=True,
        sort=False,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        METRICS_PATH,
        index=False,
    )

    spatial.to_csv(
        SPATIAL_PATH,
        index=False,
    )

    manifest = {
        "gate": "3.1",
        "purpose": (
            "diagnostic_synthetic_geological_plausibility_audit"
        ),
        "dataset_modified": False,
        "model_selected": False,
        "oracle_values_loaded": False,
        "oracle_fields_forbidden": sorted(
            FORBIDDEN_ORACLE_FIELDS
        ),
        "rows": int(
            len(frame)
        ),
        "known_positive_rows": int(
            frame[
                "known_positive"
            ].sum()
        ),
        "unlabeled_rows": int(
            (
                frame[
                    "known_positive"
                ]
                == 0
            ).sum()
        ),
        "geochemical_variables": (
            GEOCHEMICAL
        ),
        "context_variables": (
            CONTEXT_CONTINUOUS
        ),
        "review_flags": flags,
        "review_flag_count": int(
            len(flags)
        ),
        "interpretation": (
            "Flags require scientific review and do "
            "not automatically imply dataset regeneration."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"ROWS = {len(frame)}"
    )

    print(
        "KNOWN_POSITIVES = "
        f"{int(frame['known_positive'].sum())}"
    )

    print(
        "UNLABELED = "
        f"{int((frame['known_positive'] == 0).sum())}"
    )

    print(
        "ORACLE_VALUES_LOADED = NO"
    )

    print(
        "DATASET_MODIFIED = NO"
    )

    print(
        "MODEL_SELECTED = NO"
    )

    print(
        "REVIEW_FLAGS = "
        f"{len(flags)}"
    )

    for flag in flags:
        print(
            "FLAG = "
            f"{flag['severity']} | "
            f"{flag['check']} | "
            f"{flag['detail']}"
        )

    print(
        "LAB05_GATE03_1_AUDIT = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
