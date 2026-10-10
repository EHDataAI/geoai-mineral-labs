"""LAB05 Gate 4 preflight.

Read-only validation before spatial benchmarking.

Checks:
- repository branch
- Gate 3 evidence availability
- evidence manifest
- feature sets A/B/C
- target and spatial group columns
- forbidden-feature leakage
- class counts
- spatial group support
- required sklearn splitters
- non-interactive execution
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)


ROOT = Path(__file__).resolve().parents[2]

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

CONTRACT_PATH = (
    LAB_DIR
    / "prospectivity_contract.json"
)

EXPECTED_BRANCH = (
    "feat/lab05-mineral-prospectivity"
)

TARGET = "known_positive"

GROUP = "spatial_block_id"

EXPECTED_FEATURE_SETS = (
    "A",
    "B",
    "C",
)

EXPECTED_ROWS = 6400

EXPECTED_POSITIVES = 100

BACKGROUND_PER_REPLICATE = 500

BACKGROUND_REPLICATES = 10

CV_FOLDS = 5

EXPECTED_MODELS = (
    "dummy",
    "logistic_regression",
    "random_forest",
)


def git_output(
    *arguments: str,
) -> str:
    """Run read-only Git command without interactive stdin."""
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    return result.stdout.strip()


def load_json(
    path: Path,
) -> dict:
    """Load JSON object."""
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


FEATURE_SET_KEYS = {
    "A": "A_non_geochemical",
    "B": "B_raw_geochemistry",
    "C": "C_ilr_geochemistry",
}


def extract_feature_sets(
    manifest: dict,
) -> dict[str, list[str]]:
    """Read canonical Gate 3 feature sets A/B/C."""
    raw_feature_sets = manifest.get(
        "feature_sets"
    )

    if not isinstance(
        raw_feature_sets,
        dict,
    ):
        raise RuntimeError(
            "Manifest field 'feature_sets' "
            "must be a dictionary."
        )

    result: dict[str, list[str]] = {}

    for short_name, canonical_name in (
        FEATURE_SET_KEYS.items()
    ):
        columns = raw_feature_sets.get(
            canonical_name
        )

        if not isinstance(
            columns,
            list,
        ):
            raise RuntimeError(
                "Missing canonical feature set "
                f"{canonical_name!r}."
            )

        if not all(
            isinstance(column, str)
            for column in columns
        ):
            raise RuntimeError(
                f"Feature set {canonical_name!r} "
                "contains non-string columns."
            )

        result[short_name] = list(
            columns
        )

    expected_manifest_keys = set(
        FEATURE_SET_KEYS.values()
    )

    actual_manifest_keys = set(
        raw_feature_sets
    )

    if (
        actual_manifest_keys
        != expected_manifest_keys
    ):
        raise RuntimeError(
            "Unexpected Gate 3 feature-set "
            "contract. Expected "
            f"{sorted(expected_manifest_keys)}, "
            "found "
            f"{sorted(actual_manifest_keys)}."
        )

    return result


def main() -> int:
    """Execute Gate 4 read-only preflight."""
    print(
        "LAB05_GATE04_PREFLIGHT_START"
    )

    branch = git_output(
        "branch",
        "--show-current",
    )

    print(
        f"BRANCH = {branch}"
    )

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Unexpected branch: "
            f"{branch!r}"
        )

    porcelain = git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    status_lines = [
        line
        for line in porcelain.splitlines()
        if line.strip()
    ]

    allowed_self_status = (
        "?? scripts/lab05/preflight_gate04.py"
    )

    unexpected_status = [
        line
        for line in status_lines
        if line != allowed_self_status
    ]

    if unexpected_status:
        raise RuntimeError(
            "Unexpected working-tree changes "
            "before Gate 4: "
            f"{unexpected_status}"
        )

    if status_lines == [
        allowed_self_status
    ]:
        print(
            "WORKTREE_CLEAN_EXCEPT_PREFLIGHT = YES"
        )
    elif not status_lines:
        print(
            "WORKTREE_CLEAN = YES"
        )
    else:
        raise RuntimeError(
            "Unexpected Gate 4 preflight "
            "repository state."
        )

    if not EVIDENCE_PATH.is_file():
        raise FileNotFoundError(
            EVIDENCE_PATH
        )

    evidence = pd.read_csv(
        EVIDENCE_PATH
    )

    manifest = load_json(
        MANIFEST_PATH
    )

    contract = load_json(
        CONTRACT_PATH
    )

    print(
        f"SKLEARN_VERSION = "
        f"{sklearn.__version__}"
    )

    print(
        f"ROWS = {len(evidence)}"
    )

    print(
        f"COLUMNS = {len(evidence.columns)}"
    )

    if len(evidence) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected evidence row count: "
            f"{len(evidence)}"
        )

    required = {
        TARGET,
        GROUP,
        "cell_id",
    }

    missing_required = (
        required
        - set(evidence.columns)
    )

    if missing_required:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_required)}"
        )

    positives = int(
        evidence[TARGET].sum()
    )

    unlabeled = int(
        len(evidence)
        - positives
    )

    print(
        f"KNOWN_POSITIVES = {positives}"
    )

    print(
        f"UNLABELED_BACKGROUND = {unlabeled}"
    )

    if positives != EXPECTED_POSITIVES:
        raise RuntimeError(
            "Unexpected known-positive count: "
            f"{positives}"
        )

    if unlabeled < BACKGROUND_PER_REPLICATE:
        raise RuntimeError(
            "Insufficient unlabeled/background cells."
        )

    groups = int(
        evidence[GROUP].nunique()
    )

    positive_groups = int(
        evidence.loc[
            evidence[TARGET] == 1,
            GROUP,
        ].nunique()
    )

    print(
        f"SPATIAL_GROUPS = {groups}"
    )

    print(
        f"POSITIVE_SPATIAL_GROUPS = "
        f"{positive_groups}"
    )

    if positive_groups < CV_FOLDS:
        raise RuntimeError(
            "Insufficient positive spatial groups "
            "for five-fold grouped CV."
        )

    feature_sets = extract_feature_sets(
        manifest
    )

    forbidden = set(
        contract.get(
            "forbidden_model_features",
            []
        )
    )

    print(
        f"FORBIDDEN_MODEL_FEATURES = "
        f"{len(forbidden)}"
    )

    for name in EXPECTED_FEATURE_SETS:
        columns = feature_sets[name]

        missing = (
            set(columns)
            - set(evidence.columns)
        )

        leakage = (
            set(columns)
            & forbidden
        )

        duplicates = (
            len(columns)
            != len(set(columns))
        )

        print(
            f"FEATURE_SET_{name}_COUNT = "
            f"{len(columns)}"
        )

        print(
            f"FEATURE_SET_{name} = "
            + ",".join(columns)
        )

        if missing:
            raise RuntimeError(
                f"Feature set {name} contains "
                f"missing columns: {sorted(missing)}"
            )

        if leakage:
            raise RuntimeError(
                f"Feature set {name} leaks "
                f"forbidden variables: "
                f"{sorted(leakage)}"
            )

        if duplicates:
            raise RuntimeError(
                f"Feature set {name} contains "
                "duplicate columns."
            )

    expected_counts = {
        "A": 9,
        "B": 19,
        "C": 18,
    }

    for name, expected in (
        expected_counts.items()
    ):
        actual = len(
            feature_sets[name]
        )

        if actual != expected:
            raise RuntimeError(
                f"Feature set {name}: expected "
                f"{expected}, found {actual}."
            )

    oracle_names = {
        "true_deposit",
        "latent_favourability",
        "true_deposit_probability",
        "deposit_id",
        "distance_to_known_deposit",
    }

    all_model_features = set().union(
        *(
            set(columns)
            for columns
            in feature_sets.values()
        )
    )

    oracle_overlap = (
        all_model_features
        & oracle_names
    )

    if oracle_overlap:
        raise RuntimeError(
            "Oracle leakage detected: "
            f"{sorted(oracle_overlap)}"
        )

    print(
        "ORACLE_FEATURE_LEAKAGE = 0"
    )

    spatial_cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=12345,
    )

    random_cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=12345,
    )

    if spatial_cv.n_splits != CV_FOLDS:
        raise RuntimeError(
            "Spatial CV initialization failed."
        )

    if random_cv.n_splits != CV_FOLDS:
        raise RuntimeError(
            "Random CV initialization failed."
        )

    expected_fit_count = (
        len(EXPECTED_MODELS)
        * len(EXPECTED_FEATURE_SETS)
        * BACKGROUND_REPLICATES
        * CV_FOLDS
        * 2
    )

    print(
        f"BACKGROUND_REPLICATES = "
        f"{BACKGROUND_REPLICATES}"
    )

    print(
        f"BACKGROUND_PER_REPLICATE = "
        f"{BACKGROUND_PER_REPLICATE}"
    )

    print(
        f"CV_FOLDS = {CV_FOLDS}"
    )

    print(
        f"MODELS = "
        + ",".join(EXPECTED_MODELS)
    )

    print(
        f"PLANNED_MODEL_FITS = "
        f"{expected_fit_count}"
    )

    if expected_fit_count != 900:
        raise RuntimeError(
            "Unexpected benchmark fit count."
        )

    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            str(
                EVIDENCE_PATH.relative_to(ROOT)
            ),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    if ignored.returncode != 0:
        raise RuntimeError(
            "Generated evidence dataset must "
            "remain ignored by Git."
        )

    print(
        "HEAVY_EVIDENCE_GIT_POLICY = PASS"
    )

    print(
        "SPATIAL_CV_AVAILABLE = YES"
    )

    print(
        "RANDOM_CV_DIAGNOSTIC_AVAILABLE = YES"
    )

    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "LAB05_GATE04_PREFLIGHT = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
