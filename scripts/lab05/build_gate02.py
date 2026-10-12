"""Build Gate 02 assets for LAB05.

Gate 02 validates label semantics and spatial cross-validation
feasibility before any predictive modelling is allowed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EXPECTED_BRANCH = "feat/lab05-mineral-prospectivity"

AUDIT_PATH = (
    "labs/lab05_mineral_prospectivity/"
    "05b_audit_labels_and_spatial_cv.py"
)

TEST_PATH = "tests/test_lab05_spatial_cv_audit.py"

RUNNER_PATH = "scripts/lab05/run_gate02.py"


AUDIT = r'''"""Audit LAB05 labels and spatial CV feasibility."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from prospectivity_config import CONTRACT, SEED


LAB_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    LAB_DIR
    / "data"
    / "synthetic_mineral_system.csv"
)

RESULTS_DIR = LAB_DIR / "results"

FOLD_PATH = (
    RESULTS_DIR
    / "spatial_cv_fold_audit.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "spatial_cv_audit_summary.json"
)


def build_fold_audit(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Construct deterministic grouped spatial folds."""
    validation = CONTRACT["validation"]

    n_splits = int(
        validation["n_splits"]
    )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    y = data["known_positive"].to_numpy()

    groups = data[
        validation["group_column"]
    ].to_numpy()

    records: list[dict[str, int | float]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            X=data,
            y=y,
            groups=groups,
        ),
        start=1,
    ):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        train_groups = set(
            train["spatial_block_id"]
        )

        test_groups = set(
            test["spatial_block_id"]
        )

        overlap = (
            train_groups
            & test_groups
        )

        train_positive = int(
            train["known_positive"].sum()
        )

        test_positive = int(
            test["known_positive"].sum()
        )

        records.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_positive": train_positive,
                "test_positive": test_positive,
                "train_groups": len(
                    train_groups
                ),
                "test_groups": len(
                    test_groups
                ),
                "group_overlap": len(
                    overlap
                ),
                "test_positive_rate": float(
                    test_positive
                    / len(test)
                ),
            }
        )

    return pd.DataFrame(records)


def validate_fold_audit(
    audit: pd.DataFrame,
    data: pd.DataFrame,
) -> dict[str, int | float]:
    """Validate Gate 2 label and split invariants."""
    expected_splits = int(
        CONTRACT["validation"]["n_splits"]
    )

    if len(audit) != expected_splits:
        raise ValueError(
            "Unexpected number of CV folds."
        )

    if (
        audit["group_overlap"] != 0
    ).any():
        raise ValueError(
            "Spatial group leakage detected."
        )

    if (
        audit["train_positive"] <= 0
    ).any():
        raise ValueError(
            "A training fold has no positive labels."
        )

    if (
        audit["test_positive"] <= 0
    ).any():
        raise ValueError(
            "A test fold has no positive labels."
        )

    known_positive = int(
        data["known_positive"].sum()
    )

    true_deposits = int(
        data["true_deposit"].sum()
    )

    hidden_deposits = int(
        (
            (data["true_deposit"] == 1)
            & (data["known_positive"] == 0)
        ).sum()
    )

    positive_blocks = int(
        data.loc[
            data["known_positive"] == 1,
            "spatial_block_id",
        ].nunique()
    )

    if positive_blocks < expected_splits:
        raise ValueError(
            "Too few positive spatial blocks for CV."
        )

    if known_positive >= true_deposits:
        raise ValueError(
            "Synthetic oracle has no hidden deposits."
        )

    summary = {
        "rows": int(len(data)),
        "known_positives": known_positive,
        "true_deposits": true_deposits,
        "hidden_deposits": hidden_deposits,
        "positive_blocks": positive_blocks,
        "spatial_blocks": int(
            data["spatial_block_id"].nunique()
        ),
        "n_splits": expected_splits,
        "min_test_positives": int(
            audit["test_positive"].min()
        ),
        "max_test_positives": int(
            audit["test_positive"].max()
        ),
        "min_test_groups": int(
            audit["test_groups"].min()
        ),
        "max_test_groups": int(
            audit["test_groups"].max()
        ),
        "total_group_overlap": int(
            audit["group_overlap"].sum()
        ),
    }

    return summary


def main() -> None:
    """Run LAB05 Gate 2 audit."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "Gate 1 dataset does not exist."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(DATA_PATH)

    audit = build_fold_audit(data)

    summary = validate_fold_audit(
        audit,
        data,
    )

    audit.to_csv(
        FOLD_PATH,
        index=False,
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("LAB05_GATE02 = PASS")

    for key, value in summary.items():
        print(
            f"{key.upper()} = {value}"
        )

    print()
    print(audit.to_string(index=False))

    print()
    print(f"FOLD_AUDIT = {FOLD_PATH}")
    print(f"SUMMARY = {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
'''


TESTS = r'''"""Tests for LAB05 Gate 2 spatial CV audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

AUDIT_PATH = (
    LAB_DIR
    / "results"
    / "spatial_cv_fold_audit.csv"
)

SUMMARY_PATH = (
    LAB_DIR
    / "results"
    / "spatial_cv_audit_summary.json"
)


def load_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_PATH)


def load_summary() -> dict:
    return json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_gate02_outputs_exist() -> None:
    assert AUDIT_PATH.is_file()
    assert SUMMARY_PATH.is_file()


def test_five_spatial_folds_exist() -> None:
    audit = load_audit()

    assert len(audit) == 5
    assert set(audit["fold"]) == {
        1,
        2,
        3,
        4,
        5,
    }


def test_no_group_overlap() -> None:
    audit = load_audit()

    assert audit[
        "group_overlap"
    ].eq(0).all()


def test_every_test_fold_has_positives() -> None:
    audit = load_audit()

    assert audit[
        "test_positive"
    ].gt(0).all()


def test_every_train_fold_has_positives() -> None:
    audit = load_audit()

    assert audit[
        "train_positive"
    ].gt(0).all()


def test_positive_blocks_are_sufficient() -> None:
    summary = load_summary()

    assert (
        summary["positive_blocks"]
        >= summary["n_splits"]
    )


def test_hidden_oracle_deposits_remain() -> None:
    summary = load_summary()

    assert (
        summary["hidden_deposits"]
        > 0
    )


def test_total_group_overlap_is_zero() -> None:
    summary = load_summary()

    assert (
        summary["total_group_overlap"]
        == 0
    )
'''


RUNNER = r'''"""Execute and validate LAB05 Gate 2."""

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

AUDIT = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "05b_audit_labels_and_spatial_cv.py"
)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run one Gate 2 command."""
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
    )


def main() -> int:
    """Execute Gate 2 QA."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    environment["GIT_PAGER"] = "cat"
    environment["PAGER"] = "cat"

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(AUDIT),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(AUDIT),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "tests/test_lab05_spatial_cv_audit.py",
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
    print("LAB05_GATE02_QA = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def repo_root() -> Path:
    """Resolve repository root."""
    return Path(__file__).resolve().parents[2]


def current_branch(root: Path) -> str:
    """Return current Git branch."""
    result = subprocess.run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def write_file(
    root: Path,
    relative: str,
    content: str,
) -> None:
    """Write canonical UTF-8 file."""
    path = root / relative

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )

    print(f"WRITTEN = {relative}")


def main() -> int:
    """Build Gate 2 assets."""
    root = repo_root()

    branch = current_branch(root)

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Gate 2 builder requires branch "
            f"{EXPECTED_BRANCH!r}; "
            f"current branch is {branch!r}."
        )

    write_file(
        root,
        AUDIT_PATH,
        AUDIT,
    )

    write_file(
        root,
        TEST_PATH,
        TESTS,
    )

    write_file(
        root,
        RUNNER_PATH,
        RUNNER,
    )

    print("LAB05_GATE02_BUILD = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
