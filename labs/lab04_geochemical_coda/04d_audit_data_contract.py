
"""Audit LAB04 synthetic geochemical data and source consistency.

This audit checks data integrity, censoring semantics, source definitions,
and consistency of analytical detection limits.

It does not establish physical calibration or real-world geological validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geochemistry_config import (
    get_detection_limit_mapping,
    get_detection_limits,
    get_elements,
    load_contract,
)


LAB_DIR = Path(__file__).resolve().parent
DATA_PATH = LAB_DIR / "data" / "synthetic_geochemistry.csv"
RESULTS_DIR = LAB_DIR / "results"

def audit() -> dict:
    """Validate data integrity and the LAB04 analytical contract."""
    contract = load_contract()
    elements = get_elements()
    generator_lod = get_detection_limit_mapping()
    limits = get_detection_limits()

    if not np.array_equal(
        limits,
        np.asarray([generator_lod[element] for element in elements]),
    ):
        raise AssertionError("Detection-limit mapping is misaligned.")

    frame = pd.read_csv(DATA_PATH)

    if len(frame) != contract["expected_samples"]:
        raise AssertionError("Unexpected sample count.")

    if frame["sample_id"].isna().any():
        raise AssertionError("Missing sample identifier.")

    if frame["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample identifiers.")

    for coordinate in ["easting", "northing"]:
        if not np.isfinite(frame[coordinate]).all():
            raise AssertionError(f"Invalid {coordinate} coordinate.")

    if not set(frame["anomaly_class"].unique()) == {
        "background", "halo", "anomalous"
    }:
        raise AssertionError("Unexpected anomaly classes.")

    values = frame[elements].to_numpy(dtype=float)

    flags = frame[
        [f"{element}_below_lod" for element in elements]
    ].to_numpy(dtype=bool)

    latent = frame[
        [f"{element}_latent" for element in elements]
    ].to_numpy(dtype=float)

    finite = np.isfinite(values)
    missing = np.isnan(values)

    if np.isinf(values).any():
        raise AssertionError("Infinite observed concentrations.")

    if not np.isfinite(latent).all() or np.any(latent <= 0):
        raise AssertionError("Invalid latent concentrations.")

    if not np.array_equal(flags, latent < limits):
        raise AssertionError("Censor flags disagree with latent values.")

    if not np.all(values[finite & flags] == contract["censored_observation_encoding"]):
        raise AssertionError("Censored observations must be encoded as zero.")

    if not np.all(values[finite & ~flags] >= np.broadcast_to(
        limits, values.shape
    )[finite & ~flags]):
        raise AssertionError("Uncensored values below detection limits.")

    if not np.allclose(
        values[finite & ~flags],
        latent[finite & ~flags],
    ):
        raise AssertionError("Observed and latent detected values disagree.")

    if not missing.any():
        raise AssertionError("Expected missing values are absent.")

    if not flags.any():
        raise AssertionError("Expected censoring is absent.")

    if np.any(finite & ~flags & (values <= 0)):
        raise AssertionError("Invalid nonpositive uncensored values.")

    censor_rates = {
        element: float(flags[:, index].mean())
        for index, element in enumerate(elements)
    }

    if not all(0 < rate < 0.25 for rate in censor_rates.values()):
        raise AssertionError("Censoring rate outside experimental bounds.")

    summary = {
        "status": "PASS",
        "samples": int(len(frame)),
        "elements": elements,
        "element_units": contract["unit"],
        "unit_status": contract["unit_status"],
        "composition_scope": contract["composition_scope"],
        "closure_total": contract["closure_total"],
        "closure_interpretation": contract["closure_interpretation"],
        "detection_limits": generator_lod,
        "censoring_rates": censor_rates,
        "missing_cells": int(missing.sum()),
        "missing_rows": int(missing.any(axis=1).sum()),
        "censored_cells": int(flags.sum()),
        "data_sources": "synthetic_only",
        "latent_columns": "diagnostic_ground_truth_only",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = RESULTS_DIR / "data_contract_audit.json"

    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return summary


if __name__ == "__main__":
    result = audit()
    print("LAB04_DATA_CONTRACT_AUDIT =", result["status"])
    print("SAMPLES =", result["samples"])
    print("ELEMENTS =", len(result["elements"]))
    print("MISSING_CELLS =", result["missing_cells"])
    print("MISSING_ROWS =", result["missing_rows"])
    print("CENSORED_CELLS =", result["censored_cells"])
    print("UNIT_STATUS =", result["unit_status"])
    print("UNIT =", result["element_units"])
    print("COMPOSITION_SCOPE =", result["composition_scope"])
