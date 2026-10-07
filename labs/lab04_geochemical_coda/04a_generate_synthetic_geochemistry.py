"""Generate a controlled synthetic geochemical dataset for LAB04.

The experiment is designed to support compositional data analysis while keeping
the ground-truth latent processes available for validation.

The dataset is synthetic and contains no real mineral exploration data.
"""

from __future__ import annotations

from pathlib import Path

from geochemistry_config import (
    get_detection_limit_mapping,
    get_elements,
    load_contract,
)

import numpy as np
import pandas as pd


CONTRACT = load_contract()
SEED = CONTRACT["seed"]
N_SAMPLES = CONTRACT["expected_samples"]

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
OUTPUT_PATH = DATA_DIR / "synthetic_geochemistry.csv"

ELEMENTS = get_elements()
LOD = get_detection_limit_mapping()


def radial_influence(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    scale_m: float,
) -> np.ndarray:
    """Return a Gaussian radial influence field."""
    distance_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    return np.exp(-0.5 * distance_sq / scale_m**2)


def main() -> None:
    rng = np.random.default_rng(SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    easting = rng.uniform(350_000.0, 430_000.0, N_SAMPLES)
    northing = rng.uniform(7_100_000.0, 7_180_000.0, N_SAMPLES)

    lithology_code = rng.choice(
        ["intrusive", "volcanic", "sedimentary"],
        size=N_SAMPLES,
        p=[0.40, 0.35, 0.25],
    )

    lithology_intrusive = (lithology_code == "intrusive").astype(float)
    lithology_volcanic = (lithology_code == "volcanic").astype(float)
    lithology_sedimentary = (lithology_code == "sedimentary").astype(float)

    porphyry_core = radial_influence(
        easting,
        northing,
        center_x=392_000.0,
        center_y=7_144_000.0,
        scale_m=11_000.0,
    )

    epithermal_halo = radial_influence(
        easting,
        northing,
        center_x=405_000.0,
        center_y=7_158_000.0,
        scale_m=8_000.0,
    )

    alteration = np.clip(
        0.55 * porphyry_core
        + 0.35 * epithermal_halo
        + rng.normal(0.0, 0.08, N_SAMPLES),
        0.0,
        None,
    )

    mineralization = np.clip(
        0.75 * porphyry_core
        + 0.45 * epithermal_halo
        + 0.25 * alteration
        + rng.normal(0.0, 0.10, N_SAMPLES),
        0.0,
        None,
    )

    pathfinder = np.clip(
        0.65 * epithermal_halo
        + 0.25 * alteration
        + rng.normal(0.0, 0.10, N_SAMPLES),
        0.0,
        None,
    )

    lithogenic_mafic = np.clip(
        0.70 * lithology_volcanic
        + 0.25 * lithology_intrusive
        + rng.normal(0.0, 0.10, N_SAMPLES),
        0.0,
        None,
    )

    lithogenic_felsic = np.clip(
        0.75 * lithology_intrusive
        + 0.20 * lithology_sedimentary
        + rng.normal(0.0, 0.10, N_SAMPLES),
        0.0,
        None,
    )

    latent_log = pd.DataFrame(
        {
            "Cu": 3.5 + 1.30 * mineralization + 0.30 * alteration,
            "Mo": 1.3 + 1.15 * mineralization + 0.20 * lithogenic_felsic,
            "Au": -4.5 + 1.00 * mineralization + 0.85 * pathfinder,
            "As": 1.7 + 1.25 * pathfinder + 0.20 * alteration,
            "Sb": 0.3 + 1.10 * pathfinder,
            "Fe": 8.2 + 0.60 * lithogenic_mafic + 0.25 * alteration,
            "Mg": 7.6 + 0.75 * lithogenic_mafic,
            "Ca": 7.5 + 0.65 * lithology_sedimentary + 0.20 * lithogenic_mafic,
            "Al": 8.0 + 0.55 * lithogenic_felsic,
            "K": 7.2 + 0.75 * lithogenic_felsic + 0.15 * alteration,
        }
    )

    analytical_noise = rng.normal(
        loc=0.0,
        scale=0.18,
        size=(N_SAMPLES, len(ELEMENTS)),
    )

    latent_positive = np.exp(
        latent_log[ELEMENTS].to_numpy(dtype=float) + analytical_noise
    )

    latent_df = pd.DataFrame(
        latent_positive,
        columns=[f"{element}_latent" for element in ELEMENTS],
    )

    observed_df = pd.DataFrame(index=np.arange(N_SAMPLES))
    censored_df = pd.DataFrame(index=np.arange(N_SAMPLES))

    for index, element in enumerate(ELEMENTS):
        values = latent_positive[:, index].copy()
        below_lod = values < LOD[element]

        censored_df[f"{element}_below_lod"] = below_lod

        observed = values.copy()
        observed[below_lod] = 0.0

        observed_df[element] = observed

    missing_mask = rng.random((N_SAMPLES, len(ELEMENTS))) < 0.01

    for index, element in enumerate(ELEMENTS):
        observed_df.loc[missing_mask[:, index], element] = np.nan

    anomaly_score = (
        0.50 * mineralization
        + 0.30 * pathfinder
        + 0.20 * alteration
    )

    anomaly_class = np.select(
        [
            anomaly_score >= 0.60,
            anomaly_score >= 0.30,
        ],
        [
            "anomalous",
            "halo",
        ],
        default="background",
    )

    frame = pd.DataFrame(
        {
            "sample_id": [
                f"GEO_{value:05d}"
                for value in range(1, N_SAMPLES + 1)
            ],
            "easting": easting,
            "northing": northing,
            "lithology": lithology_code,
            "anomaly_class": anomaly_class,
            "latent_mineralization": mineralization,
            "latent_pathfinder": pathfinder,
            "latent_alteration": alteration,
            "latent_lithogenic_mafic": lithogenic_mafic,
            "latent_lithogenic_felsic": lithogenic_felsic,
        }
    )

    frame = pd.concat(
        [
            frame,
            observed_df,
            censored_df,
            latent_df,
        ],
        axis=1,
    )

    frame.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print("LAB04_SYNTHETIC_DATA = PASS")
    print("SEED =", SEED)
    print("ROWS =", len(frame))
    print("ELEMENTS =", len(ELEMENTS))
    print("OUTPUT =", OUTPUT_PATH)
    print()
    print("ANOMALY_CLASS_COUNTS")
    print(frame["anomaly_class"].value_counts().sort_index())
    print()
    print("BELOW_LOD_COUNTS")
    print(
        frame[
            [f"{element}_below_lod" for element in ELEMENTS]
        ].sum()
    )
    print()
    print(
        "MISSING_OBSERVED_VALUES =",
        int(frame[ELEMENTS].isna().sum().sum()),
    )


if __name__ == "__main__":
    main()
