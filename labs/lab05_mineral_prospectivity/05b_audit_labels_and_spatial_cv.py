"""Audit LAB05 labels and spatial CV feasibility."""

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
