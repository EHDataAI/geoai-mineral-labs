
"""Centralized configuration for LAB04 synthetic geochemistry.

The JSON data contract is the single source of truth for element
ordering, synthetic concentration units, and detection limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


LAB_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = LAB_DIR / "geochemistry_contract.json"

EXPECTED_ELEMENTS = (
    "Cu", "Mo", "Au", "As", "Sb",
    "Fe", "Mg", "Ca", "Al", "K",
)


def load_contract() -> dict:
    """Load and validate the LAB04 geochemical contract."""
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        contract = json.load(handle)

    if contract.get("contract_version") != "1.0.0":
        raise ValueError("Unsupported geochemical contract version.")

    elements = contract.get("elements")

    if elements != list(EXPECTED_ELEMENTS):
        raise ValueError("Unexpected element ordering.")

    if contract.get("unit") != "ppm":
        raise ValueError("Expected synthetic ppm convention.")

    if contract.get("unit_status") != "synthetic_convention":
        raise ValueError("Invalid unit provenance.")

    if contract.get("whole_rock_composition") is not False:
        raise ValueError("Whole-rock composition must be false.")

    if contract.get("composition_scope") != (
        "selected_10_element_subcomposition"
    ):
        raise ValueError("Unexpected composition scope.")

    if contract.get("crs") != "EPSG:32719":
        raise ValueError("Unexpected coordinate reference system.")

    if contract.get("expected_samples") != 1200:
        raise ValueError("Unexpected sample count.")

    if contract.get("seed") != 12345:
        raise ValueError("Unexpected random seed.")

    limits = contract.get("detection_limits")

    if not isinstance(limits, dict):
        raise ValueError("Detection limits must be a dictionary.")

    if set(limits) != set(elements):
        raise ValueError("Detection limit elements do not match.")

    values = np.asarray(
        [limits[element] for element in elements],
        dtype=float,
    )

    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Detection limits must be positive and finite.")

    if contract.get("closure_total") != 100.0:
        raise ValueError("Unexpected relative closure total.")

    if contract.get("primary_replacement") != "LOD/2":
        raise ValueError("Unexpected primary replacement strategy.")

    if contract.get("sensitivity_replacement") != "LOD/sqrt(2)":
        raise ValueError("Unexpected sensitivity strategy.")

    return contract


def get_elements() -> list[str]:
    """Return the ordered geochemical element names."""
    return list(load_contract()["elements"])


def get_detection_limits() -> NDArray[np.float64]:
    """Return detection limits aligned to the element ordering."""
    contract = load_contract()

    return np.asarray(
        [
            contract["detection_limits"][element]
            for element in contract["elements"]
        ],
        dtype=float,
    )


def get_detection_limit_mapping() -> dict[str, float]:
    """Return the element-to-LOD mapping."""
    contract = load_contract()

    return {
        element: float(contract["detection_limits"][element])
        for element in contract["elements"]
    }


def get_replacement_fractions() -> dict[str, float]:
    """Return the two contract-defined replacement fractions."""
    contract = load_contract()
    methods = {
        "LOD/2": 0.5,
        "LOD/sqrt(2)": float(1.0 / np.sqrt(2.0)),
    }
    return {
        "lod_half": methods[contract["primary_replacement"]],
        "lod_sqrt2": methods[contract["sensitivity_replacement"]],
    }


if __name__ == "__main__":
    contract = load_contract()

    print("LAB04_CONTRACT_LOADER = PASS")
    print("VERSION =", contract["contract_version"])
    print("ELEMENTS =", len(get_elements()))
    print("UNIT =", contract["unit"])
    print("COMPOSITION_SCOPE =", contract["composition_scope"])
    print("LOD =", get_detection_limits().tolist())
