"""Tests for LAB05 Gate 2 spatial CV audit."""

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
