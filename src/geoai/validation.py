from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FoldSplit:
    """Indices defining one train/test cross-validation split."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def _validate_n_splits(n_items: int, n_splits: int) -> None:
    if n_items < 2:
        raise ValueError("At least two items are required.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_splits > n_items:
        raise ValueError(
            "n_splits cannot exceed the number of available items."
        )


def _as_finite_vector(
    values: Iterable[float],
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def random_kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    random_state: int = 111,
) -> tuple[FoldSplit, ...]:
    """
    Create reproducible random K-fold splits.

    Python's standard-library PRNG is used deliberately so split generation
    does not depend on NumPy's random module.
    """
    _validate_n_splits(n_samples, n_splits)

    indices = list(range(n_samples))
    rng = random.Random(random_state)
    rng.shuffle(indices)

    base_size, remainder = divmod(n_samples, n_splits)

    fold_sizes = [
        base_size + (1 if fold < remainder else 0)
        for fold in range(n_splits)
    ]

    all_indices = np.arange(n_samples, dtype=int)
    splits: list[FoldSplit] = []

    start = 0

    for fold, fold_size in enumerate(fold_sizes):
        stop = start + fold_size

        test_idx = np.sort(
            np.asarray(indices[start:stop], dtype=int)
        )

        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        train_idx = all_indices[mask]

        splits.append(
            FoldSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )

        start = stop

    return tuple(splits)


def assign_spatial_blocks(
    x: Iterable[float],
    y: Iterable[float],
    block_size: float,
    *,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> np.ndarray:
    """
    Assign observations to regular square spatial blocks.

    Coordinates are assumed to be expressed in a metric projected CRS.
    """
    x_arr = _as_finite_vector(x, "x")
    y_arr = _as_finite_vector(y, "y")

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")

    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and greater than zero.")

    if origin_x is None:
        origin_x = (
            np.floor(x_arr.min() / block_size)
            * block_size
        )

    if origin_y is None:
        origin_y = (
            np.floor(y_arr.min() / block_size)
            * block_size
        )

    if not np.isfinite(origin_x) or not np.isfinite(origin_y):
        raise ValueError("Block origins must be finite.")

    block_col = np.floor(
        (x_arr - origin_x) / block_size
    ).astype(int)

    block_row = np.floor(
        (y_arr - origin_y) / block_size
    ).astype(int)

    return np.asarray(
        [
            f"c{col}_r{row}"
            for col, row in zip(
                block_col,
                block_row,
                strict=True,
            )
        ],
        dtype=str,
    )


def spatial_group_kfold_indices(
    groups: Iterable[object],
    n_splits: int = 5,
) -> tuple[FoldSplit, ...]:
    """
    Create deterministic group-preserving spatial folds.

    Entire spatial blocks remain in either train or test for each fold,
    preventing direct block leakage.

    Groups are assigned greedily to balance the number of observations
    across folds.
    """
    group_arr = np.asarray(list(groups))

    if group_arr.ndim != 1:
        raise ValueError("groups must be one-dimensional.")

    if group_arr.size == 0:
        raise ValueError("groups cannot be empty.")

    unique_groups, counts = np.unique(
        group_arr,
        return_counts=True,
    )

    _validate_n_splits(
        int(unique_groups.size),
        n_splits,
    )

    order = sorted(
        range(unique_groups.size),
        key=lambda idx: (
            -int(counts[idx]),
            str(unique_groups[idx]),
        ),
    )

    fold_loads = np.zeros(n_splits, dtype=int)
    group_to_fold: dict[object, int] = {}

    for group_idx in order:
        fold = int(np.argmin(fold_loads))
        group = unique_groups[group_idx]

        group_to_fold[group] = fold
        fold_loads[fold] += int(counts[group_idx])

    sample_folds = np.asarray(
        [
            group_to_fold[group]
            for group in group_arr
        ],
        dtype=int,
    )

    all_indices = np.arange(
        group_arr.size,
        dtype=int,
    )

    splits: list[FoldSplit] = []

    for fold in range(n_splits):
        test_mask = sample_folds == fold

        splits.append(
            FoldSplit(
                fold=fold,
                train_idx=all_indices[~test_mask],
                test_idx=all_indices[test_mask],
            )
        )

    return tuple(splits)


def nearest_test_to_train_distance(
    x: Iterable[float],
    y: Iterable[float],
    train_idx: Iterable[int],
    test_idx: Iterable[int],
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Compute the nearest training-point distance for each test observation.

    This diagnostic measures geometric train/test separation and is useful
    for evaluating how restrictive a spatial validation design actually is.
    """
    x_arr = _as_finite_vector(x, "x")
    y_arr = _as_finite_vector(y, "y")

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")

    train = np.asarray(list(train_idx), dtype=int)
    test = np.asarray(list(test_idx), dtype=int)

    if train.size == 0 or test.size == 0:
        raise ValueError(
            "train_idx and test_idx must both contain observations."
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if (
        train.min() < 0
        or test.min() < 0
        or train.max() >= x_arr.size
        or test.max() >= x_arr.size
    ):
        raise IndexError(
            "Split indices are outside the coordinate arrays."
        )

    result = np.empty(test.size, dtype=float)

    train_x = x_arr[train]
    train_y = y_arr[train]

    for start in range(0, test.size, chunk_size):
        stop = min(
            start + chunk_size,
            test.size,
        )

        test_x = x_arr[test[start:stop]][:, None]
        test_y = y_arr[test[start:stop]][:, None]

        distances = np.hypot(
            test_x - train_x[None, :],
            test_y - train_y[None, :],
        )

        result[start:stop] = distances.min(
            axis=1
        )

    return result


def summarize_fold_geometry(
    x: Iterable[float],
    y: Iterable[float],
    splits: Iterable[FoldSplit],
    *,
    groups: Iterable[object] | None = None,
) -> tuple[dict[str, float | int], ...]:
    """
    Summarize geometric train/test separation for cross-validation folds.
    """
    x_arr = _as_finite_vector(x, "x")
    y_arr = _as_finite_vector(y, "y")

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")

    group_arr = (
        None
        if groups is None
        else np.asarray(list(groups))
    )

    if (
        group_arr is not None
        and group_arr.size != x_arr.size
    ):
        raise ValueError(
            "groups must have the same length as x and y."
        )

    summaries: list[dict[str, float | int]] = []

    for split in splits:
        distances = nearest_test_to_train_distance(
            x_arr,
            y_arr,
            split.train_idx,
            split.test_idx,
        )

        group_overlap = 0

        if group_arr is not None:
            train_groups = set(
                group_arr[split.train_idx].tolist()
            )
            test_groups = set(
                group_arr[split.test_idx].tolist()
            )

            group_overlap = len(
                train_groups.intersection(
                    test_groups
                )
            )

        summaries.append(
            {
                "fold": int(split.fold),
                "train_n": int(split.train_idx.size),
                "test_n": int(split.test_idx.size),
                "min_test_to_train_distance": float(
                    distances.min()
                ),
                "median_test_to_train_distance": float(
                    np.median(distances)
                ),
                "mean_test_to_train_distance": float(
                    distances.mean()
                ),
                "group_overlap": int(group_overlap),
            }
        )

    return tuple(summaries)
