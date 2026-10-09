"""Create, build, run and validate LAB05 Gate 03.

This script automates:
- Gate 03 builder creation
- contract patching
- evidence-layer code creation
- test creation
- runner creation
- syntax QA
- Gate 03 execution
- focused pytest
- full pytest
- git diff --check

No predictive model is fitted in this gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_BRANCH = "feat/lab05-mineral-prospectivity"

ROOT = Path(__file__).resolve().parents[2]

PYTHON = Path(
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

BUILDER_PATH = (
    ROOT
    / "scripts"
    / "lab05"
    / "build_gate03.py"
)

RUNNER_PATH = (
    ROOT
    / "scripts"
    / "lab05"
    / "run_gate03.py"
)

EVIDENCE_PATH = (
    LAB_DIR
    / "05c_build_evidence_layers.py"
)

TEST_PATH = (
    ROOT
    / "tests"
    / "test_lab05_evidence_layers.py"
)


EVIDENCE_CODE = r'''"""Build LAB05 evidence layers and compositional geochemistry.

Synthetic geochemistry is generated from observable geological,
structural, geophysical and alteration evidence plus independent noise.

Synthetic oracle variables are retained only as protected metadata and
must not be used as model predictors or to generate geochemical
predictors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoai.composition import closure, clr, ilr
from prospectivity_config import (
    CONTRACT,
    FORBIDDEN_MODEL_FEATURES,
    SEED,
)


LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"

INPUT_PATH = DATA_DIR / "synthetic_mineral_system.csv"
OUTPUT_PATH = DATA_DIR / "prospectivity_evidence.csv"

MANIFEST_PATH = RESULTS_DIR / "evidence_manifest.json"
DIAGNOSTICS_PATH = RESULTS_DIR / "geochemistry_diagnostics.json"


ELEMENTS = [
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

LOD = {
    "Cu": 0.040,
    "Mo": 0.020,
    "Au": 0.012,
    "As": 0.030,
    "Sb": 0.018,
    "Fe": 0.080,
    "Mg": 0.060,
    "Ca": 0.060,
    "Al": 0.080,
    "K": 0.050,
}

BASE_ABUNDANCE = {
    "Cu": 0.16,
    "Mo": 0.08,
    "Au": 0.035,
    "As": 0.10,
    "Sb": 0.055,
    "Fe": 1.40,
    "Mg": 0.95,
    "Ca": 1.10,
    "Al": 1.55,
    "K": 0.85,
}


def observable_signals(
    data: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build signals exclusively from observable evidence."""
    intrusive = data[
        "intrusive_affinity"
    ].to_numpy(dtype=float)

    alteration = data[
        "alteration_index"
    ].to_numpy(dtype=float)

    pathway = np.exp(
        -data[
            "distance_to_fault_m"
        ].to_numpy(dtype=float)
        / 6500.0
    )

    contact = np.exp(
        -data[
            "distance_to_contact_m"
        ].to_numpy(dtype=float)
        / 5500.0
    )

    intersection = np.exp(
        -data[
            "distance_to_fault_intersection_m"
        ].to_numpy(dtype=float)
        / 11000.0
    )

    lithology = data["lithology"].astype(str)

    intrusive_lithology = (
        lithology.eq("intrusive")
        .to_numpy(dtype=float)
    )

    volcanic_lithology = (
        lithology.eq("volcanic")
        .to_numpy(dtype=float)
    )

    return {
        "intrusive": intrusive,
        "alteration": alteration,
        "pathway": pathway,
        "contact": contact,
        "intersection": intersection,
        "intrusive_lithology": intrusive_lithology,
        "volcanic_lithology": volcanic_lithology,
    }


def generate_geochemistry(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Generate positive synthetic geochemistry with LOD censoring."""
    result = data.copy()

    signal = observable_signals(result)

    coefficients = {
        "Cu": (1.15, 0.90, 0.70, 0.40),
        "Mo": (1.25, 0.65, 0.45, 0.35),
        "Au": (0.70, 0.95, 0.85, 0.55),
        "As": (0.30, 1.05, 0.80, 0.65),
        "Sb": (0.20, 0.85, 0.75, 0.70),
        "Fe": (0.35, 0.30, 0.20, 0.10),
        "Mg": (-0.20, -0.10, 0.05, 0.10),
        "Ca": (-0.15, -0.15, 0.05, 0.15),
        "Al": (0.10, 0.05, 0.05, 0.05),
        "K": (0.25, 0.35, 0.10, 0.15),
    }

    censored_counts: dict[str, int] = {}

    for element in ELEMENTS:
        source_w, alt_w, path_w, contact_w = (
            coefficients[element]
        )

        lithology_effect = (
            0.22 * signal["intrusive_lithology"]
            + 0.10 * signal["volcanic_lithology"]
        )

        log_abundance = (
            np.log(BASE_ABUNDANCE[element])
            + source_w * signal["intrusive"]
            + alt_w * signal["alteration"]
            + path_w * signal["pathway"]
            + contact_w * signal["contact"]
            + 0.15 * signal["intersection"]
            + lithology_effect
            + rng.normal(
                0.0,
                0.55,
                len(result),
            )
        )

        abundance = np.exp(log_abundance)

        below_lod = abundance < LOD[element]

        censored_counts[element] = int(
            below_lod.sum()
        )

        abundance = np.where(
            below_lod,
            LOD[element] / 2.0,
            abundance,
        )

        result[element] = abundance

    return result, censored_counts


def add_compositional_transforms(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add closed, CLR and ILR representations."""
    result = data.copy()

    raw = result[
        ELEMENTS
    ].to_numpy(dtype=float)

    if not np.isfinite(raw).all():
        raise ValueError(
            "Raw geochemistry contains non-finite values."
        )

    if not (raw > 0.0).all():
        raise ValueError(
            "LOD-treated geochemistry must be strictly positive."
        )

    closed = closure(raw)
    clr_values = clr(closed)
    ilr_values = ilr(closed)

    for index, element in enumerate(ELEMENTS):
        result[
            f"{element}_closed"
        ] = closed[:, index]

        result[
            f"clr_{element}"
        ] = clr_values[:, index]

    expected_ilr_dim = len(ELEMENTS) - 1

    if ilr_values.shape[1] != expected_ilr_dim:
        raise ValueError(
            "Unexpected ILR dimension."
        )

    for index in range(expected_ilr_dim):
        result[
            f"ilr_{index + 1}"
        ] = ilr_values[:, index]

    return result


def feature_sets() -> dict[str, list[str]]:
    """Return the contract-driven modelling feature sets."""
    groups = CONTRACT["feature_groups"]

    base = (
        list(groups["geology"])
        + list(groups["structure"])
        + list(groups["geophysics"])
        + list(groups["alteration"])
    )

    raw = list(
        groups["geochemistry_raw"]
    )

    ilr_columns = [
        f"ilr_{index}"
        for index in range(
            1,
            len(ELEMENTS),
        )
    ]

    return {
        "A_non_geochemical": base,
        "B_raw_geochemistry": (
            base + raw
        ),
        "C_ilr_geochemistry": (
            base + ilr_columns
        ),
    }


def validate_feature_sets(
    sets: dict[str, list[str]],
) -> None:
    """Reject protected fields from all modelling feature sets."""
    forbidden = set(
        FORBIDDEN_MODEL_FEATURES
    )

    for name, columns in sets.items():
        duplicates = {
            column
            for column in columns
            if columns.count(column) > 1
        }

        if duplicates:
            raise ValueError(
                f"{name} contains duplicate features: "
                f"{sorted(duplicates)}"
            )

        leaked = set(columns) & forbidden

        if leaked:
            raise ValueError(
                f"{name} contains forbidden features: "
                f"{sorted(leaked)}"
            )


def validate_transforms(
    data: pd.DataFrame,
) -> dict[str, float | int]:
    """Validate CoDA invariants."""
    raw = data[
        ELEMENTS
    ].to_numpy(dtype=float)

    closed_columns = [
        f"{element}_closed"
        for element in ELEMENTS
    ]

    clr_columns = [
        f"clr_{element}"
        for element in ELEMENTS
    ]

    ilr_columns = [
        f"ilr_{index}"
        for index in range(
            1,
            len(ELEMENTS),
        )
    ]

    closed = data[
        closed_columns
    ].to_numpy(dtype=float)

    clr_values = data[
        clr_columns
    ].to_numpy(dtype=float)

    ilr_values = data[
        ilr_columns
    ].to_numpy(dtype=float)

    closure_error = float(
        np.max(
            np.abs(
                closed.sum(axis=1) - 1.0
            )
        )
    )

    clr_sum_error = float(
        np.max(
            np.abs(
                clr_values.sum(axis=1)
            )
        )
    )

    if closure_error > 1e-10:
        raise ValueError(
            "Closure invariant failed."
        )

    if clr_sum_error > 1e-10:
        raise ValueError(
            "CLR zero-sum invariant failed."
        )

    if ilr_values.shape != (
        len(data),
        len(ELEMENTS) - 1,
    ):
        raise ValueError(
            "ILR dimensionality invariant failed."
        )

    return {
        "rows": int(len(data)),
        "elements": len(ELEMENTS),
        "ilr_dimensions": int(
            ilr_values.shape[1]
        ),
        "raw_minimum": float(
            raw.min()
        ),
        "closure_max_abs_error": (
            closure_error
        ),
        "clr_max_abs_row_sum": (
            clr_sum_error
        ),
    }


def validate_identity(
    input_data: pd.DataFrame,
    output_data: pd.DataFrame,
) -> None:
    """Ensure Gate 3 preserves Gate 1 cell identity and order."""
    if len(input_data) != len(
        output_data
    ):
        raise ValueError(
            "Gate 3 changed row count."
        )

    if not input_data[
        "cell_id"
    ].equals(
        output_data["cell_id"]
    ):
        raise ValueError(
            "Gate 3 changed cell_id order."
        )


def main() -> None:
    """Build and validate Gate 3 evidence layers."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            "Gate 1 dataset is missing."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_data = pd.read_csv(
        INPUT_PATH
    )

    rng = np.random.default_rng(
        SEED + 300
    )

    evidence, censored = (
        generate_geochemistry(
            input_data,
            rng,
        )
    )

    evidence = (
        add_compositional_transforms(
            evidence
        )
    )

    sets = feature_sets()

    validate_feature_sets(sets)

    diagnostics = validate_transforms(
        evidence
    )

    validate_identity(
        input_data,
        evidence,
    )

    total_censored = int(
        sum(censored.values())
    )

    total_geochemical_values = int(
        len(evidence) * len(ELEMENTS)
    )

    censored_fraction = float(
        total_censored
        / total_geochemical_values
    )

    if total_censored <= 0:
        raise ValueError(
            "LOD treatment was not exercised: "
            "no values were censored."
        )

    if censored_fraction >= 0.10:
        raise ValueError(
            "Synthetic LOD censoring is excessive: "
            f"{censored_fraction:.6f}."
        )

    diagnostics[
        "total_censored_values"
    ] = total_censored

    diagnostics[
        "total_geochemical_values"
    ] = total_geochemical_values

    diagnostics[
        "censored_fraction"
    ] = censored_fraction

    diagnostics[
        "censored_by_element"
    ] = censored

    manifest = {
        "feature_sets": sets,
        "forbidden_model_features": sorted(
            FORBIDDEN_MODEL_FEATURES
        ),
        "geochemistry_elements": ELEMENTS,
        "lod": LOD,
        "clr_columns": [
            f"clr_{element}"
            for element in ELEMENTS
        ],
        "ilr_columns": [
            f"ilr_{index}"
            for index in range(
                1,
                len(ELEMENTS),
            )
        ],
        "primary_geochemical_representation": (
            "ILR"
        ),
        "clr_role": "diagnostic_comparison",
        "oracle_used_to_generate_geochemistry": False,
    }

    evidence.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    DIAGNOSTICS_PATH.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("LAB05_GATE03 = PASS")
    print(
        f"ROWS = {diagnostics['rows']}"
    )
    print(
        f"ELEMENTS = {diagnostics['elements']}"
    )
    print(
        "ILR_DIMENSIONS = "
        f"{diagnostics['ilr_dimensions']}"
    )
    print(
        "RAW_MINIMUM = "
        f"{diagnostics['raw_minimum']}"
    )
    print(
        "CLOSURE_MAX_ABS_ERROR = "
        f"{diagnostics['closure_max_abs_error']}"
    )
    print(
        "CLR_MAX_ABS_ROW_SUM = "
        f"{diagnostics['clr_max_abs_row_sum']}"
    )
    print(
        "TOTAL_CENSORED_VALUES = "
        f"{diagnostics['total_censored_values']}"
    )
    print(
        "CENSORED_FRACTION = "
        f"{diagnostics['censored_fraction']}"
    )
    print(
        "FEATURE_SET_A = "
        f"{len(sets['A_non_geochemical'])}"
    )
    print(
        "FEATURE_SET_B = "
        f"{len(sets['B_raw_geochemistry'])}"
    )
    print(
        "FEATURE_SET_C = "
        f"{len(sets['C_ilr_geochemistry'])}"
    )
    print(f"EVIDENCE = {OUTPUT_PATH}")
    print(f"MANIFEST = {MANIFEST_PATH}")
    print(
        f"DIAGNOSTICS = {DIAGNOSTICS_PATH}"
    )


if __name__ == "__main__":
    main()
'''


TEST_CODE = r'''"""Tests for LAB05 Gate 3 evidence layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

EVIDENCE_PATH = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

MANIFEST_PATH = (
    LAB_DIR
    / "results"
    / "evidence_manifest.json"
)

DIAGNOSTICS_PATH = (
    LAB_DIR
    / "results"
    / "geochemistry_diagnostics.json"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

ELEMENTS = [
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


def load_evidence() -> pd.DataFrame:
    return pd.read_csv(
        EVIDENCE_PATH
    )


def load_manifest() -> dict:
    return json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_gate03_outputs_exist() -> None:
    assert EVIDENCE_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    assert DIAGNOSTICS_PATH.is_file()


def test_gate03_preserves_6400_cells() -> None:
    data = load_evidence()

    assert len(data) == 6400
    assert data["cell_id"].nunique() == 6400


def test_lod_treated_geochemistry_is_positive() -> None:
    data = load_evidence()

    values = data[
        ELEMENTS
    ].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert (values > 0.0).all()


def test_closed_composition_sums_to_one() -> None:
    data = load_evidence()

    columns = [
        f"{element}_closed"
        for element in ELEMENTS
    ]

    row_sums = data[
        columns
    ].sum(axis=1)

    assert np.allclose(
        row_sums,
        1.0,
        atol=1e-10,
    )


def test_clr_rows_sum_to_zero() -> None:
    data = load_evidence()

    columns = [
        f"clr_{element}"
        for element in ELEMENTS
    ]

    row_sums = data[
        columns
    ].sum(axis=1)

    assert np.allclose(
        row_sums,
        0.0,
        atol=1e-10,
    )


def test_ilr_dimension_is_d_minus_one() -> None:
    data = load_evidence()

    columns = [
        column
        for column in data.columns
        if column.startswith("ilr_")
    ]

    assert len(columns) == 9


def test_feature_sets_exist() -> None:
    manifest = load_manifest()

    sets = manifest[
        "feature_sets"
    ]

    assert set(sets) == {
        "A_non_geochemical",
        "B_raw_geochemistry",
        "C_ilr_geochemistry",
    }


def test_raw_feature_set_contains_ten_elements() -> None:
    manifest = load_manifest()

    raw_set = set(
        manifest[
            "feature_sets"
        ][
            "B_raw_geochemistry"
        ]
    )

    assert set(ELEMENTS).issubset(
        raw_set
    )


def test_ilr_feature_set_contains_nine_coordinates() -> None:
    manifest = load_manifest()

    ilr_set = set(
        manifest[
            "feature_sets"
        ][
            "C_ilr_geochemistry"
        ]
    )

    expected = {
        f"ilr_{index}"
        for index in range(
            1,
            10,
        )
    }

    assert expected.issubset(
        ilr_set
    )


def test_no_forbidden_features_enter_model_sets() -> None:
    manifest = load_manifest()

    forbidden = set(
        manifest[
            "forbidden_model_features"
        ]
    )

    for columns in manifest[
        "feature_sets"
    ].values():
        assert not (
            set(columns)
            & forbidden
        )


def test_cell_id_is_model_forbidden() -> None:
    contract = json.loads(
        CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "cell_id"
        in contract[
            "forbidden_model_features"
        ]
    )


def test_oracle_did_not_generate_geochemistry() -> None:
    manifest = load_manifest()

    assert (
        manifest[
            "oracle_used_to_generate_geochemistry"
        ]
        is False
    )


def test_lod_censoring_is_exercised_but_limited() -> None:
    diagnostics = json.loads(
        DIAGNOSTICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        diagnostics["total_censored_values"]
        > 0
    )

    assert (
        diagnostics["censored_fraction"]
        < 0.10
    )


def test_ilr_is_primary_coda_representation() -> None:
    manifest = load_manifest()

    assert (
        manifest[
            "primary_geochemical_representation"
        ]
        == "ILR"
    )

    assert (
        manifest["clr_role"]
        == "diagnostic_comparison"
    )
'''


RUNNER_CODE = r'''"""Execute and validate LAB05 Gate 3."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PYTHON = Path(
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

EVIDENCE = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "05c_build_evidence_layers.py"
)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run one Gate 3 command."""
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def main() -> int:
    """Execute Gate 3 QA."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    environment["PYTHONUNBUFFERED"] = "1"
    environment["GIT_PAGER"] = "cat"
    environment["PAGER"] = "cat"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["CI"] = "1"

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(EVIDENCE),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(EVIDENCE),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "tests/test_lab05_evidence_layers.py",
            "-q",
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
        ],
        env=environment,
    )

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--check",
        ],
        env=environment,
    )

    print()
    print("LAB05_GATE03_QA = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


BUILDER_CODE = r'''"""Build LAB05 Gate 3 assets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


EXPECTED_BRANCH = "feat/lab05-mineral-prospectivity"

ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

EVIDENCE_PATH = (
    LAB_DIR
    / "05c_build_evidence_layers.py"
)

TEST_PATH = (
    ROOT
    / "tests"
    / "test_lab05_evidence_layers.py"
)

RUNNER_PATH = (
    ROOT
    / "scripts"
    / "lab05"
    / "run_gate03.py"
)


def current_branch() -> str:
    result = subprocess.run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def patch_contract() -> None:
    contract = json.loads(
        CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    forbidden = list(
        contract[
            "forbidden_model_features"
        ]
    )

    if "cell_id" not in forbidden:
        forbidden.append("cell_id")

        contract[
            "forbidden_model_features"
        ] = forbidden

        CONTRACT_PATH.write_text(
            json.dumps(
                contract,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "CONTRACT_PATCHED_CELL_ID = YES"
        )
    else:
        print(
            "CONTRACT_PATCHED_CELL_ID = NO"
        )


def write_file(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )

    print(
        f"WRITTEN = "
        f"{path.relative_to(ROOT)}"
    )


def main() -> int:
    branch = current_branch()

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Gate 3 builder requires branch "
            f"{EXPECTED_BRANCH!r}; "
            f"current branch is {branch!r}."
        )

    patch_contract()

    write_file(
        EVIDENCE_PATH,
        EVIDENCE_CODE,
    )

    write_file(
        TEST_PATH,
        TEST_CODE,
    )

    write_file(
        RUNNER_PATH,
        RUNNER_CODE,
    )

    print(
        "LAB05_GATE03_BUILD = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_text(
    path: Path,
    content: str,
) -> None:
    """Write normalized UTF-8 LF text."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized = (
        content.replace(
            "\r\n",
            "\n",
        )
        .rstrip()
        + "\n"
    )

    path.write_text(
        normalized,
        encoding="utf-8",
        newline="\n",
    )


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run command and fail immediately on error."""
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def main() -> int:
    """Create and execute complete Gate 3 automation."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    environment["PYTHONUNBUFFERED"] = "1"
    environment["GIT_PAGER"] = "cat"
    environment["PAGER"] = "cat"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["CI"] = "1"

    branch = subprocess.run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Gate 3 automation requires branch "
            f"{EXPECTED_BRANCH!r}; "
            f"current branch is {branch!r}."
        )

    print(
        "LAB05_GATE03_AUTOMATION_START"
    )

    builder_code = BUILDER_CODE.replace(
        "EVIDENCE_CODE,",
        "EVIDENCE_CODE,\n",
    )

    namespace = {
        "EVIDENCE_CODE": EVIDENCE_CODE,
        "TEST_CODE": TEST_CODE,
        "RUNNER_CODE": RUNNER_CODE,
    }

    builder_prefix = (
        "EVIDENCE_CODE = "
        + repr(EVIDENCE_CODE)
        + "\n\n"
        + "TEST_CODE = "
        + repr(TEST_CODE)
        + "\n\n"
        + "RUNNER_CODE = "
        + repr(RUNNER_CODE)
        + "\n\n"
    )

    safe_builder_code = BUILDER_CODE.replace(
        "from __future__ import annotations\n\n",
        "",
        1,
    )

    write_text(
        BUILDER_PATH,
        builder_prefix + safe_builder_code,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(BUILDER_PATH),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(BUILDER_PATH),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(EVIDENCE_PATH),
            str(RUNNER_PATH),
            str(TEST_PATH),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(RUNNER_PATH),
        ],
        env=environment,
    )

    contract = json.loads(
        CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    forbidden = contract[
        "forbidden_model_features"
    ]

    if "cell_id" not in forbidden:
        raise RuntimeError(
            "cell_id was not protected."
        )

    print()
    print(
        "FORBIDDEN_MODEL_FEATURES = "
        f"{len(forbidden)}"
    )

    print(
        "CELL_ID_PROTECTED = YES"
    )

    print(
        "LAB05_GATE03_AUTOMATION = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
