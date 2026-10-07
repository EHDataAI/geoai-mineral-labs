"""Tests for compositional data analysis utilities."""

import numpy as np
import pytest

from geoai.composition import (
    aitchison_distance,
    closure,
    clr,
    clr_inverse,
    helmert_submatrix,
    ilr,
    ilr_inverse,
)


def test_closure_sums_to_requested_total():
    data = np.array(
        [
            [2.0, 3.0, 5.0],
            [4.0, 4.0, 2.0],
        ]
    )

    closed = closure(data, total=100.0)

    assert np.allclose(closed.sum(axis=1), 100.0)


def test_closure_is_scale_invariant():
    composition = np.array([[2.0, 3.0, 5.0]])

    assert np.allclose(
        closure(composition),
        closure(composition * 1000.0),
    )


def test_clr_rows_sum_to_zero():
    data = np.array(
        [
            [2.0, 3.0, 5.0],
            [10.0, 4.0, 1.0],
        ]
    )

    transformed = clr(data)

    assert np.allclose(transformed.sum(axis=1), 0.0)


def test_clr_is_scale_invariant():
    composition = np.array([[2.0, 3.0, 5.0]])

    assert np.allclose(
        clr(composition),
        clr(composition * 100.0),
    )


def test_clr_inverse_recovers_closed_composition():
    data = np.array(
        [
            [2.0, 3.0, 5.0],
            [10.0, 4.0, 1.0],
        ]
    )

    recovered = clr_inverse(clr(data), total=1.0)

    assert np.allclose(recovered, closure(data))


def test_helmert_basis_is_orthonormal():
    basis = helmert_submatrix(5)

    assert basis.shape == (4, 5)
    assert np.allclose(basis @ basis.T, np.eye(4))
    assert np.allclose(basis.sum(axis=1), 0.0)


def test_ilr_has_d_minus_one_coordinates():
    data = np.array(
        [
            [2.0, 3.0, 5.0, 7.0],
            [10.0, 4.0, 1.0, 2.0],
        ]
    )

    transformed = ilr(data)

    assert transformed.shape == (2, 3)


def test_ilr_inverse_recovers_closed_composition():
    data = np.array(
        [
            [2.0, 3.0, 5.0, 7.0],
            [10.0, 4.0, 1.0, 2.0],
        ]
    )

    recovered = ilr_inverse(ilr(data))

    assert np.allclose(recovered, closure(data))


def test_aitchison_distance_is_scale_invariant():
    x = np.array([[2.0, 3.0, 5.0]])
    y = np.array([[5.0, 2.0, 3.0]])

    distance_original = aitchison_distance(x, y)
    distance_scaled = aitchison_distance(
        x * 10.0,
        y * 1000.0,
    )

    assert np.allclose(distance_original, distance_scaled)


def test_aitchison_distance_zero_for_equivalent_compositions():
    x = np.array([[2.0, 3.0, 5.0]])
    y = np.array([[20.0, 30.0, 50.0]])

    assert np.allclose(aitchison_distance(x, y), 0.0)


@pytest.mark.parametrize(
    "data",
    [
        [[0.0, 1.0, 2.0]],
        [[-1.0, 2.0, 3.0]],
        [[np.nan, 2.0, 3.0]],
        [[np.inf, 2.0, 3.0]],
    ],
)
def test_logratio_functions_reject_invalid_compositions(data):
    with pytest.raises(ValueError):
        clr(data)

def test_replace_below_detection_limit_replaces_zero():
    from geoai.composition import replace_below_detection_limit

    data = np.array([[0.0, 4.0, 10.0]])
    lod = np.array([2.0, 2.0, 2.0])

    replaced = replace_below_detection_limit(
        data,
        lod,
        fraction=0.5,
    )

    assert np.allclose(
        replaced,
        [[1.0, 4.0, 10.0]],
    )


def test_replace_below_detection_limit_preserves_missing():
    from geoai.composition import replace_below_detection_limit

    data = np.array([[0.0, np.nan, 10.0]])
    lod = np.array([2.0, 2.0, 2.0])

    replaced = replace_below_detection_limit(data, lod)

    assert replaced[0, 0] == 1.0
    assert np.isnan(replaced[0, 1])
    assert replaced[0, 2] == 10.0


def test_replace_below_detection_limit_rejects_invalid_fraction():
    from geoai.composition import replace_below_detection_limit

    with pytest.raises(ValueError):
        replace_below_detection_limit(
            [[0.0, 2.0]],
            [1.0, 1.0],
            fraction=1.0,
        )
