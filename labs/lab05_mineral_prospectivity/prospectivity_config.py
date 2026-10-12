"""Central configuration loader for LAB05 Mineral Prospectivity Mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = LAB_DIR / "prospectivity_contract.json"


def load_contract() -> dict[str, Any]:
    """Load and validate the LAB05 prospectivity contract."""
    with CONTRACT_PATH.open("r", encoding="utf-8") as file:
        contract = json.load(file)

    required_top_level = {
        "contract_version",
        "laboratory",
        "title",
        "dataset_type",
        "random_seed",
        "study_area",
        "prediction_problem",
        "label_semantics",
        "feature_groups",
        "coordinate_columns",
        "grouping_columns",
        "label_columns",
        "diagnostic_only_columns",
        "forbidden_model_features",
        "models",
        "validation",
        "metrics",
        "background_sampling",
        "geochemistry",
        "synthetic_oracle",
    }

    missing = required_top_level.difference(contract)

    if missing:
        raise ValueError(
            f"LAB05 contract missing required sections: {sorted(missing)}"
        )

    study_area = contract["study_area"]

    expected_grid_cells = (
        (study_area["easting_max"] - study_area["easting_min"])
        // study_area["grid_cell_size_m"]
    ) * (
        (study_area["northing_max"] - study_area["northing_min"])
        // study_area["grid_cell_size_m"]
    )

    if expected_grid_cells != study_area["expected_grid_cells"]:
        raise ValueError(
            "LAB05 contract grid dimensions do not match expected_grid_cells."
        )

    forbidden = set(contract["forbidden_model_features"])
    diagnostic = set(contract["diagnostic_only_columns"])
    coordinates = set(contract["coordinate_columns"])
    grouping = set(contract["grouping_columns"])
    labels = set(contract["label_columns"])

    required_forbidden = diagnostic | coordinates | grouping | labels

    if not required_forbidden.issubset(forbidden):
        missing_forbidden = required_forbidden.difference(forbidden)

        raise ValueError(
            "LAB05 contract does not forbid all protected columns: "
            f"{sorted(missing_forbidden)}"
        )

    if contract["synthetic_oracle"]["allowed_during_model_selection"]:
        raise ValueError(
            "Synthetic oracle must not be available during model selection."
        )

    known_positive_count = contract["synthetic_oracle"].get(
        "known_positive_count"
    )

    if not isinstance(known_positive_count, int):
        raise ValueError(
            "synthetic_oracle.known_positive_count must be an integer."
        )

    if known_positive_count <= 0:
        raise ValueError(
            "synthetic_oracle.known_positive_count must be positive."
        )

    known_positive_selection = contract["synthetic_oracle"].get(
        "known_positive_selection"
    )

    if (
        known_positive_selection
        != "fixed_count_random_without_replacement"
    ):
        raise ValueError(
            "Unexpected synthetic oracle known-positive selection policy."
        )
    if contract["synthetic_oracle"]["allowed_during_feature_selection"]:
        raise ValueError(
            "Synthetic oracle must not be available during feature selection."
        )

    return contract


CONTRACT = load_contract()

SEED = int(CONTRACT["random_seed"])
STUDY_AREA = CONTRACT["study_area"]
KNOWN_POSITIVE_COUNT = int(
    CONTRACT["synthetic_oracle"]["known_positive_count"]
)

BACKGROUND_RATIO = int(
    CONTRACT["background_sampling"]["positive_to_background_ratio"]
)

BACKGROUND_REPLICATES = int(
    CONTRACT["background_sampling"]["sensitivity_replicates"]
)

BACKGROUND_PER_REPLICATE = (
    KNOWN_POSITIVE_COUNT
    * BACKGROUND_RATIO
)

CV_FOLDS = int(
    CONTRACT["validation"]["n_splits"]
)

ROWS_PER_REPLICATE = (
    KNOWN_POSITIVE_COUNT
    + BACKGROUND_PER_REPLICATE
)
FEATURE_GROUPS = CONTRACT["feature_groups"]
FORBIDDEN_MODEL_FEATURES = frozenset(
    CONTRACT["forbidden_model_features"]
)
DIAGNOSTIC_ONLY_COLUMNS = frozenset(
    CONTRACT["diagnostic_only_columns"]
)


if __name__ == "__main__":
    print("LAB05_CONTRACT_VALIDATION = PASS")
    print(f"SEED = {SEED}")
    print(f"CRS = {STUDY_AREA['crs']}")
    print(f"GRID_CELLS = {STUDY_AREA['expected_grid_cells']}")
    print(
        "FORBIDDEN_MODEL_FEATURES = "
        f"{len(FORBIDDEN_MODEL_FEATURES)}"
    )
