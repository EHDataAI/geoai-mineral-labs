EVIDENCE_CODE = '"""Build LAB05 evidence layers and compositional geochemistry.\n\nSynthetic geochemistry is generated from observable geological,\nstructural, geophysical and alteration evidence plus independent noise.\n\nSynthetic oracle variables are retained only as protected metadata and\nmust not be used as model predictors or to generate geochemical\npredictors.\n"""\n\nfrom __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nfrom geoai.composition import closure, clr, ilr\nfrom prospectivity_config import (\n    CONTRACT,\n    FORBIDDEN_MODEL_FEATURES,\n    SEED,\n)\n\n\nLAB_DIR = Path(__file__).resolve().parent\nDATA_DIR = LAB_DIR / "data"\nRESULTS_DIR = LAB_DIR / "results"\n\nINPUT_PATH = DATA_DIR / "synthetic_mineral_system.csv"\nOUTPUT_PATH = DATA_DIR / "prospectivity_evidence.csv"\n\nMANIFEST_PATH = RESULTS_DIR / "evidence_manifest.json"\nDIAGNOSTICS_PATH = RESULTS_DIR / "geochemistry_diagnostics.json"\n\n\nELEMENTS = [\n    "Cu",\n    "Mo",\n    "Au",\n    "As",\n    "Sb",\n    "Fe",\n    "Mg",\n    "Ca",\n    "Al",\n    "K",\n]\n\nLOD = {\n    "Cu": 0.040,\n    "Mo": 0.020,\n    "Au": 0.012,\n    "As": 0.030,\n    "Sb": 0.018,\n    "Fe": 0.080,\n    "Mg": 0.060,\n    "Ca": 0.060,\n    "Al": 0.080,\n    "K": 0.050,\n}\n\nBASE_ABUNDANCE = {\n    "Cu": 0.16,\n    "Mo": 0.08,\n    "Au": 0.035,\n    "As": 0.10,\n    "Sb": 0.055,\n    "Fe": 1.40,\n    "Mg": 0.95,\n    "Ca": 1.10,\n    "Al": 1.55,\n    "K": 0.85,\n}\n\n\ndef observable_signals(\n    data: pd.DataFrame,\n) -> dict[str, np.ndarray]:\n    """Build signals exclusively from observable evidence."""\n    intrusive = data[\n        "intrusive_affinity"\n    ].to_numpy(dtype=float)\n\n    alteration = data[\n        "alteration_index"\n    ].to_numpy(dtype=float)\n\n    pathway = np.exp(\n        -data[\n            "distance_to_fault_m"\n        ].to_numpy(dtype=float)\n        / 6500.0\n    )\n\n    contact = np.exp(\n        -data[\n            "distance_to_contact_m"\n        ].to_numpy(dtype=float)\n        / 5500.0\n    )\n\n    intersection = np.exp(\n        -data[\n            "distance_to_fault_intersection_m"\n        ].to_numpy(dtype=float)\n        / 11000.0\n    )\n\n    lithology = data["lithology"].astype(str)\n\n    intrusive_lithology = (\n        lithology.eq("intrusive")\n        .to_numpy(dtype=float)\n    )\n\n    volcanic_lithology = (\n        lithology.eq("volcanic")\n        .to_numpy(dtype=float)\n    )\n\n    return {\n        "intrusive": intrusive,\n        "alteration": alteration,\n        "pathway": pathway,\n        "contact": contact,\n        "intersection": intersection,\n        "intrusive_lithology": intrusive_lithology,\n        "volcanic_lithology": volcanic_lithology,\n    }\n\n\ndef generate_geochemistry(\n    data: pd.DataFrame,\n    rng: np.random.Generator,\n) -> tuple[pd.DataFrame, dict[str, int]]:\n    """Generate positive synthetic geochemistry with LOD censoring."""\n    result = data.copy()\n\n    signal = observable_signals(result)\n\n    coefficients = {\n        "Cu": (1.15, 0.90, 0.70, 0.40),\n        "Mo": (1.25, 0.65, 0.45, 0.35),\n        "Au": (0.70, 0.95, 0.85, 0.55),\n        "As": (0.30, 1.05, 0.80, 0.65),\n        "Sb": (0.20, 0.85, 0.75, 0.70),\n        "Fe": (0.35, 0.30, 0.20, 0.10),\n        "Mg": (-0.20, -0.10, 0.05, 0.10),\n        "Ca": (-0.15, -0.15, 0.05, 0.15),\n        "Al": (0.10, 0.05, 0.05, 0.05),\n        "K": (0.25, 0.35, 0.10, 0.15),\n    }\n\n    censored_counts: dict[str, int] = {}\n\n    for element in ELEMENTS:\n        source_w, alt_w, path_w, contact_w = (\n            coefficients[element]\n        )\n\n        lithology_effect = (\n            0.22 * signal["intrusive_lithology"]\n            + 0.10 * signal["volcanic_lithology"]\n        )\n\n        log_abundance = (\n            np.log(BASE_ABUNDANCE[element])\n            + source_w * signal["intrusive"]\n            + alt_w * signal["alteration"]\n            + path_w * signal["pathway"]\n            + contact_w * signal["contact"]\n            + 0.15 * signal["intersection"]\n            + lithology_effect\n            + rng.normal(\n                0.0,\n                0.55,\n                len(result),\n            )\n        )\n\n        abundance = np.exp(log_abundance)\n\n        below_lod = abundance < LOD[element]\n\n        censored_counts[element] = int(\n            below_lod.sum()\n        )\n\n        abundance = np.where(\n            below_lod,\n            LOD[element] / 2.0,\n            abundance,\n        )\n\n        result[element] = abundance\n\n    return result, censored_counts\n\n\ndef add_compositional_transforms(\n    data: pd.DataFrame,\n) -> pd.DataFrame:\n    """Add closed, CLR and ILR representations."""\n    result = data.copy()\n\n    raw = result[\n        ELEMENTS\n    ].to_numpy(dtype=float)\n\n    if not np.isfinite(raw).all():\n        raise ValueError(\n            "Raw geochemistry contains non-finite values."\n        )\n\n    if not (raw > 0.0).all():\n        raise ValueError(\n            "LOD-treated geochemistry must be strictly positive."\n        )\n\n    closed = closure(raw)\n    clr_values = clr(closed)\n    ilr_values = ilr(closed)\n\n    for index, element in enumerate(ELEMENTS):\n        result[\n            f"{element}_closed"\n        ] = closed[:, index]\n\n        result[\n            f"clr_{element}"\n        ] = clr_values[:, index]\n\n    expected_ilr_dim = len(ELEMENTS) - 1\n\n    if ilr_values.shape[1] != expected_ilr_dim:\n        raise ValueError(\n            "Unexpected ILR dimension."\n        )\n\n    for index in range(expected_ilr_dim):\n        result[\n            f"ilr_{index + 1}"\n        ] = ilr_values[:, index]\n\n    return result\n\n\ndef feature_sets() -> dict[str, list[str]]:\n    """Return the contract-driven modelling feature sets."""\n    groups = CONTRACT["feature_groups"]\n\n    base = (\n        list(groups["geology"])\n        + list(groups["structure"])\n        + list(groups["geophysics"])\n        + list(groups["alteration"])\n    )\n\n    raw = list(\n        groups["geochemistry_raw"]\n    )\n\n    ilr_columns = [\n        f"ilr_{index}"\n        for index in range(\n            1,\n            len(ELEMENTS),\n        )\n    ]\n\n    return {\n        "A_non_geochemical": base,\n        "B_raw_geochemistry": (\n            base + raw\n        ),\n        "C_ilr_geochemistry": (\n            base + ilr_columns\n        ),\n    }\n\n\ndef validate_feature_sets(\n    sets: dict[str, list[str]],\n) -> None:\n    """Reject protected fields from all modelling feature sets."""\n    forbidden = set(\n        FORBIDDEN_MODEL_FEATURES\n    )\n\n    for name, columns in sets.items():\n        duplicates = {\n            column\n            for column in columns\n            if columns.count(column) > 1\n        }\n\n        if duplicates:\n            raise ValueError(\n                f"{name} contains duplicate features: "\n                f"{sorted(duplicates)}"\n            )\n\n        leaked = set(columns) & forbidden\n\n        if leaked:\n            raise ValueError(\n                f"{name} contains forbidden features: "\n                f"{sorted(leaked)}"\n            )\n\n\ndef validate_transforms(\n    data: pd.DataFrame,\n) -> dict[str, float | int]:\n    """Validate CoDA invariants."""\n    raw = data[\n        ELEMENTS\n    ].to_numpy(dtype=float)\n\n    closed_columns = [\n        f"{element}_closed"\n        for element in ELEMENTS\n    ]\n\n    clr_columns = [\n        f"clr_{element}"\n        for element in ELEMENTS\n    ]\n\n    ilr_columns = [\n        f"ilr_{index}"\n        for index in range(\n            1,\n            len(ELEMENTS),\n        )\n    ]\n\n    closed = data[\n        closed_columns\n    ].to_numpy(dtype=float)\n\n    clr_values = data[\n        clr_columns\n    ].to_numpy(dtype=float)\n\n    ilr_values = data[\n        ilr_columns\n    ].to_numpy(dtype=float)\n\n    closure_error = float(\n        np.max(\n            np.abs(\n                closed.sum(axis=1) - 1.0\n            )\n        )\n    )\n\n    clr_sum_error = float(\n        np.max(\n            np.abs(\n                clr_values.sum(axis=1)\n            )\n        )\n    )\n\n    if closure_error > 1e-10:\n        raise ValueError(\n            "Closure invariant failed."\n        )\n\n    if clr_sum_error > 1e-10:\n        raise ValueError(\n            "CLR zero-sum invariant failed."\n        )\n\n    if ilr_values.shape != (\n        len(data),\n        len(ELEMENTS) - 1,\n    ):\n        raise ValueError(\n            "ILR dimensionality invariant failed."\n        )\n\n    return {\n        "rows": int(len(data)),\n        "elements": len(ELEMENTS),\n        "ilr_dimensions": int(\n            ilr_values.shape[1]\n        ),\n        "raw_minimum": float(\n            raw.min()\n        ),\n        "closure_max_abs_error": (\n            closure_error\n        ),\n        "clr_max_abs_row_sum": (\n            clr_sum_error\n        ),\n    }\n\n\ndef validate_identity(\n    input_data: pd.DataFrame,\n    output_data: pd.DataFrame,\n) -> None:\n    """Ensure Gate 3 preserves Gate 1 cell identity and order."""\n    if len(input_data) != len(\n        output_data\n    ):\n        raise ValueError(\n            "Gate 3 changed row count."\n        )\n\n    if not input_data[\n        "cell_id"\n    ].equals(\n        output_data["cell_id"]\n    ):\n        raise ValueError(\n            "Gate 3 changed cell_id order."\n        )\n\n\ndef main() -> None:\n    """Build and validate Gate 3 evidence layers."""\n    if not INPUT_PATH.is_file():\n        raise FileNotFoundError(\n            "Gate 1 dataset is missing."\n        )\n\n    DATA_DIR.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n\n    RESULTS_DIR.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n\n    input_data = pd.read_csv(\n        INPUT_PATH\n    )\n\n    rng = np.random.default_rng(\n        SEED + 300\n    )\n\n    evidence, censored = (\n        generate_geochemistry(\n            input_data,\n            rng,\n        )\n    )\n\n    evidence = (\n        add_compositional_transforms(\n            evidence\n        )\n    )\n\n    sets = feature_sets()\n\n    validate_feature_sets(sets)\n\n    diagnostics = validate_transforms(\n        evidence\n    )\n\n    validate_identity(\n        input_data,\n        evidence,\n    )\n\n    total_censored = int(\n        sum(censored.values())\n    )\n\n    total_geochemical_values = int(\n        len(evidence) * len(ELEMENTS)\n    )\n\n    censored_fraction = float(\n        total_censored\n        / total_geochemical_values\n    )\n\n    if total_censored <= 0:\n        raise ValueError(\n            "LOD treatment was not exercised: "\n            "no values were censored."\n        )\n\n    if censored_fraction >= 0.10:\n        raise ValueError(\n            "Synthetic LOD censoring is excessive: "\n            f"{censored_fraction:.6f}."\n        )\n\n    diagnostics[\n        "total_censored_values"\n    ] = total_censored\n\n    diagnostics[\n        "total_geochemical_values"\n    ] = total_geochemical_values\n\n    diagnostics[\n        "censored_fraction"\n    ] = censored_fraction\n\n    diagnostics[\n        "censored_by_element"\n    ] = censored\n\n    manifest = {\n        "feature_sets": sets,\n        "forbidden_model_features": sorted(\n            FORBIDDEN_MODEL_FEATURES\n        ),\n        "geochemistry_elements": ELEMENTS,\n        "lod": LOD,\n        "clr_columns": [\n            f"clr_{element}"\n            for element in ELEMENTS\n        ],\n        "ilr_columns": [\n            f"ilr_{index}"\n            for index in range(\n                1,\n                len(ELEMENTS),\n            )\n        ],\n        "primary_geochemical_representation": (\n            "ILR"\n        ),\n        "clr_role": "diagnostic_comparison",\n        "oracle_used_to_generate_geochemistry": False,\n    }\n\n    evidence.to_csv(\n        OUTPUT_PATH,\n        index=False,\n        encoding="utf-8",\n    )\n\n    MANIFEST_PATH.write_text(\n        json.dumps(\n            manifest,\n            indent=2,\n            sort_keys=True,\n        )\n        + "\\n",\n        encoding="utf-8",\n    )\n\n    DIAGNOSTICS_PATH.write_text(\n        json.dumps(\n            diagnostics,\n            indent=2,\n            sort_keys=True,\n        )\n        + "\\n",\n        encoding="utf-8",\n    )\n\n    print("LAB05_GATE03 = PASS")\n    print(\n        f"ROWS = {diagnostics[\'rows\']}"\n    )\n    print(\n        f"ELEMENTS = {diagnostics[\'elements\']}"\n    )\n    print(\n        "ILR_DIMENSIONS = "\n        f"{diagnostics[\'ilr_dimensions\']}"\n    )\n    print(\n        "RAW_MINIMUM = "\n        f"{diagnostics[\'raw_minimum\']}"\n    )\n    print(\n        "CLOSURE_MAX_ABS_ERROR = "\n        f"{diagnostics[\'closure_max_abs_error\']}"\n    )\n    print(\n        "CLR_MAX_ABS_ROW_SUM = "\n        f"{diagnostics[\'clr_max_abs_row_sum\']}"\n    )\n    print(\n        "TOTAL_CENSORED_VALUES = "\n        f"{diagnostics[\'total_censored_values\']}"\n    )\n    print(\n        "CENSORED_FRACTION = "\n        f"{diagnostics[\'censored_fraction\']}"\n    )\n    print(\n        "FEATURE_SET_A = "\n        f"{len(sets[\'A_non_geochemical\'])}"\n    )\n    print(\n        "FEATURE_SET_B = "\n        f"{len(sets[\'B_raw_geochemistry\'])}"\n    )\n    print(\n        "FEATURE_SET_C = "\n        f"{len(sets[\'C_ilr_geochemistry\'])}"\n    )\n    print(f"EVIDENCE = {OUTPUT_PATH}")\n    print(f"MANIFEST = {MANIFEST_PATH}")\n    print(\n        f"DIAGNOSTICS = {DIAGNOSTICS_PATH}"\n    )\n\n\nif __name__ == "__main__":\n    main()\n'

TEST_CODE = '"""Tests for LAB05 Gate 3 evidence layers."""\n\nfrom __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\n\nROOT = Path(__file__).resolve().parents[1]\n\nLAB_DIR = (\n    ROOT\n    / "labs"\n    / "lab05_mineral_prospectivity"\n)\n\nEVIDENCE_PATH = (\n    LAB_DIR\n    / "data"\n    / "prospectivity_evidence.csv"\n)\n\nMANIFEST_PATH = (\n    LAB_DIR\n    / "results"\n    / "evidence_manifest.json"\n)\n\nDIAGNOSTICS_PATH = (\n    LAB_DIR\n    / "results"\n    / "geochemistry_diagnostics.json"\n)\n\nCONTRACT_PATH = (\n    LAB_DIR\n    / "prospectivity_contract.json"\n)\n\nELEMENTS = [\n    "Cu",\n    "Mo",\n    "Au",\n    "As",\n    "Sb",\n    "Fe",\n    "Mg",\n    "Ca",\n    "Al",\n    "K",\n]\n\n\ndef load_evidence() -> pd.DataFrame:\n    return pd.read_csv(\n        EVIDENCE_PATH\n    )\n\n\ndef load_manifest() -> dict:\n    return json.loads(\n        MANIFEST_PATH.read_text(\n            encoding="utf-8"\n        )\n    )\n\n\ndef test_gate03_outputs_exist() -> None:\n    assert EVIDENCE_PATH.is_file()\n    assert MANIFEST_PATH.is_file()\n    assert DIAGNOSTICS_PATH.is_file()\n\n\ndef test_gate03_preserves_6400_cells() -> None:\n    data = load_evidence()\n\n    assert len(data) == 6400\n    assert data["cell_id"].nunique() == 6400\n\n\ndef test_lod_treated_geochemistry_is_positive() -> None:\n    data = load_evidence()\n\n    values = data[\n        ELEMENTS\n    ].to_numpy(dtype=float)\n\n    assert np.isfinite(values).all()\n    assert (values > 0.0).all()\n\n\ndef test_closed_composition_sums_to_one() -> None:\n    data = load_evidence()\n\n    columns = [\n        f"{element}_closed"\n        for element in ELEMENTS\n    ]\n\n    row_sums = data[\n        columns\n    ].sum(axis=1)\n\n    assert np.allclose(\n        row_sums,\n        1.0,\n        atol=1e-10,\n    )\n\n\ndef test_clr_rows_sum_to_zero() -> None:\n    data = load_evidence()\n\n    columns = [\n        f"clr_{element}"\n        for element in ELEMENTS\n    ]\n\n    row_sums = data[\n        columns\n    ].sum(axis=1)\n\n    assert np.allclose(\n        row_sums,\n        0.0,\n        atol=1e-10,\n    )\n\n\ndef test_ilr_dimension_is_d_minus_one() -> None:\n    data = load_evidence()\n\n    columns = [\n        column\n        for column in data.columns\n        if column.startswith("ilr_")\n    ]\n\n    assert len(columns) == 9\n\n\ndef test_feature_sets_exist() -> None:\n    manifest = load_manifest()\n\n    sets = manifest[\n        "feature_sets"\n    ]\n\n    assert set(sets) == {\n        "A_non_geochemical",\n        "B_raw_geochemistry",\n        "C_ilr_geochemistry",\n    }\n\n\ndef test_raw_feature_set_contains_ten_elements() -> None:\n    manifest = load_manifest()\n\n    raw_set = set(\n        manifest[\n            "feature_sets"\n        ][\n            "B_raw_geochemistry"\n        ]\n    )\n\n    assert set(ELEMENTS).issubset(\n        raw_set\n    )\n\n\ndef test_ilr_feature_set_contains_nine_coordinates() -> None:\n    manifest = load_manifest()\n\n    ilr_set = set(\n        manifest[\n            "feature_sets"\n        ][\n            "C_ilr_geochemistry"\n        ]\n    )\n\n    expected = {\n        f"ilr_{index}"\n        for index in range(\n            1,\n            10,\n        )\n    }\n\n    assert expected.issubset(\n        ilr_set\n    )\n\n\ndef test_no_forbidden_features_enter_model_sets() -> None:\n    manifest = load_manifest()\n\n    forbidden = set(\n        manifest[\n            "forbidden_model_features"\n        ]\n    )\n\n    for columns in manifest[\n        "feature_sets"\n    ].values():\n        assert not (\n            set(columns)\n            & forbidden\n        )\n\n\ndef test_cell_id_is_model_forbidden() -> None:\n    contract = json.loads(\n        CONTRACT_PATH.read_text(\n            encoding="utf-8"\n        )\n    )\n\n    assert (\n        "cell_id"\n        in contract[\n            "forbidden_model_features"\n        ]\n    )\n\n\ndef test_oracle_did_not_generate_geochemistry() -> None:\n    manifest = load_manifest()\n\n    assert (\n        manifest[\n            "oracle_used_to_generate_geochemistry"\n        ]\n        is False\n    )\n\n\ndef test_lod_censoring_is_exercised_but_limited() -> None:\n    diagnostics = json.loads(\n        DIAGNOSTICS_PATH.read_text(\n            encoding="utf-8"\n        )\n    )\n\n    assert (\n        diagnostics["total_censored_values"]\n        > 0\n    )\n\n    assert (\n        diagnostics["censored_fraction"]\n        < 0.10\n    )\n\n\ndef test_ilr_is_primary_coda_representation() -> None:\n    manifest = load_manifest()\n\n    assert (\n        manifest[\n            "primary_geochemical_representation"\n        ]\n        == "ILR"\n    )\n\n    assert (\n        manifest["clr_role"]\n        == "diagnostic_comparison"\n    )\n'

RUNNER_CODE = '"""Execute and validate LAB05 Gate 3."""\n\nfrom __future__ import annotations\n\nimport os\nimport subprocess\nimport sys\nfrom pathlib import Path\n\n\nROOT = Path(__file__).resolve().parents[2]\n\nPYTHON = Path(\n    "/home/erickh/.venvs/"\n    "geoai-mineral-qa/bin/python"\n)\n\nEVIDENCE = (\n    ROOT\n    / "labs"\n    / "lab05_mineral_prospectivity"\n    / "05c_build_evidence_layers.py"\n)\n\n\ndef run(\n    command: list[str],\n    *,\n    env: dict[str, str],\n) -> None:\n    """Run one Gate 3 command."""\n    print()\n    print(\n        "RUN = "\n        + " ".join(command)\n    )\n\n    subprocess.run(\n        command,\n        cwd=ROOT,\n        check=True,\n        env=env,\n        stdin=subprocess.DEVNULL,\n    )\n\n\ndef main() -> int:\n    """Execute Gate 3 QA."""\n    environment = os.environ.copy()\n\n    environment["PYTHONPATH"] = str(\n        ROOT / "src"\n    )\n\n    environment["PYTHONUNBUFFERED"] = "1"\n    environment["GIT_PAGER"] = "cat"\n    environment["PAGER"] = "cat"\n    environment["GIT_TERMINAL_PROMPT"] = "0"\n    environment["CI"] = "1"\n\n    run(\n        [\n            str(PYTHON),\n            "-m",\n            "py_compile",\n            str(EVIDENCE),\n        ],\n        env=environment,\n    )\n\n    run(\n        [\n            str(PYTHON),\n            str(EVIDENCE),\n        ],\n        env=environment,\n    )\n\n    run(\n        [\n            str(PYTHON),\n            "-m",\n            "pytest",\n            "tests/test_lab05_evidence_layers.py",\n            "-q",\n        ],\n        env=environment,\n    )\n\n    run(\n        [\n            str(PYTHON),\n            "-m",\n            "pytest",\n            "-q",\n        ],\n        env=environment,\n    )\n\n    run(\n        [\n            "git",\n            "--no-pager",\n            "diff",\n            "--check",\n        ],\n        env=environment,\n    )\n\n    print()\n    print("LAB05_GATE03_QA = PASS")\n\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n'

"""Build LAB05 Gate 3 assets."""

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
