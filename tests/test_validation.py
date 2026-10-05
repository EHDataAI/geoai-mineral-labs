import numpy as np
import pytest

from geoai.validation import (
    assign_spatial_blocks,
    nearest_test_to_train_distance,
    random_kfold_indices,
    spatial_group_kfold_indices,
    summarize_fold_geometry,
)


def test_random_kfold_is_reproducible_and_complete():
    first = random_kfold_indices(
        n_samples=11,
        n_splits=3,
        random_state=111,
    )

    second = random_kfold_indices(
        n_samples=11,
        n_splits=3,
        random_state=111,
    )

    assert len(first) == 3

    for split_a, split_b in zip(
        first,
        second,
        strict=True,
    ):
        assert np.array_equal(
            split_a.train_idx,
            split_b.train_idx,
        )
        assert np.array_equal(
            split_a.test_idx,
            split_b.test_idx,
        )

        assert set(split_a.train_idx).isdisjoint(
            set(split_a.test_idx)
        )

    all_test = np.concatenate(
        [split.test_idx for split in first]
    )

    assert sorted(all_test.tolist()) == list(
        range(11)
    )

    fold_sizes = [
        split.test_idx.size
        for split in first
    ]

    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_assign_spatial_blocks_known_geometry():
    x = np.array(
        [0.0, 9.9, 10.0, 19.9]
    )
    y = np.array(
        [0.0, 9.9, 10.0, 19.9]
    )

    groups = assign_spatial_blocks(
        x,
        y,
        block_size=10.0,
        origin_x=0.0,
        origin_y=0.0,
    )

    assert groups.tolist() == [
        "c0_r0",
        "c0_r0",
        "c1_r1",
        "c1_r1",
    ]


def test_assign_spatial_blocks_rejects_invalid_size():
    with pytest.raises(
        ValueError,
        match="block_size",
    ):
        assign_spatial_blocks(
            [0.0, 1.0],
            [0.0, 1.0],
            block_size=0.0,
        )


def test_spatial_group_kfold_prevents_group_leakage():
    groups = np.array(
        [
            "A", "A",
            "B", "B",
            "C", "C",
            "D", "D",
        ]
    )

    splits = spatial_group_kfold_indices(
        groups,
        n_splits=4,
    )

    all_test = []

    for split in splits:
        train_groups = set(
            groups[split.train_idx]
        )
        test_groups = set(
            groups[split.test_idx]
        )

        assert train_groups.isdisjoint(
            test_groups
        )

        all_test.extend(
            split.test_idx.tolist()
        )

    assert sorted(all_test) == list(
        range(groups.size)
    )


def test_spatial_group_kfold_rejects_too_many_folds():
    groups = ["A", "A", "B", "B"]

    with pytest.raises(
        ValueError,
        match="n_splits",
    ):
        spatial_group_kfold_indices(
            groups,
            n_splits=3,
        )


def test_nearest_test_to_train_distance_known_geometry():
    x = np.array(
        [0.0, 3.0, 10.0]
    )
    y = np.zeros(3)

    distances = nearest_test_to_train_distance(
        x,
        y,
        train_idx=[0, 2],
        test_idx=[1],
    )

    assert distances.shape == (1,)
    assert distances[0] == pytest.approx(
        3.0
    )


def test_spatial_summary_reports_zero_group_overlap():
    x = np.array(
        [0.0, 1.0, 10.0, 11.0]
    )
    y = np.zeros(4)

    groups = np.array(
        ["A", "A", "B", "B"]
    )

    splits = spatial_group_kfold_indices(
        groups,
        n_splits=2,
    )

    summaries = summarize_fold_geometry(
        x,
        y,
        splits,
        groups=groups,
    )

    assert len(summaries) == 2

    assert all(
        row["group_overlap"] == 0
        for row in summaries
    )

    assert all(
        row["min_test_to_train_distance"] > 0
        for row in summaries
    )
