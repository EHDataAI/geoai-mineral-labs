"""Core compositional data analysis utilities.

The functions in this module implement a minimal, auditable subset of
Aitchison geometry for strictly positive compositional data.

They intentionally avoid hidden zero replacement or closure assumptions.
Input validation is explicit so analytical decisions remain traceable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_2d_float_array(values: ArrayLike) -> FloatArray:
    """Return values as a validated two-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError("Compositional data must be one- or two-dimensional.")

    if array.shape[1] < 2:
        raise ValueError("A composition must contain at least two components.")

    if not np.isfinite(array).all():
        raise ValueError("Compositional data must contain only finite values.")

    if np.any(array <= 0.0):
        raise ValueError(
            "Log-ratio transformations require strictly positive components."
        )

    return array


def closure(values: ArrayLike, total: float = 1.0) -> FloatArray:
    """Close positive compositions to a constant sum."""
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Closure total must be a positive finite value.")

    array = _as_2d_float_array(values)
    row_sums = array.sum(axis=1, keepdims=True)

    return total * array / row_sums


def geometric_mean(values: ArrayLike) -> FloatArray:
    """Compute the row-wise geometric mean of positive compositions."""
    array = _as_2d_float_array(values)
    return np.exp(np.mean(np.log(array), axis=1))


def clr(values: ArrayLike) -> FloatArray:
    """Centered log-ratio transformation."""
    array = _as_2d_float_array(values)
    log_values = np.log(array)
    row_log_mean = np.mean(log_values, axis=1, keepdims=True)

    return log_values - row_log_mean


def clr_inverse(values: ArrayLike, total: float = 1.0) -> FloatArray:
    """Inverse centered log-ratio transformation."""
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError("CLR coordinates must be one- or two-dimensional.")

    if not np.isfinite(array).all():
        raise ValueError("CLR coordinates must contain only finite values.")

    if not np.allclose(array.sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("CLR coordinates must sum to zero within each row.")

    return closure(np.exp(array), total=total)


def helmert_submatrix(n_components: int) -> FloatArray:
    """Construct an orthonormal Helmert basis for the simplex."""
    if n_components < 2:
        raise ValueError("At least two components are required.")

    basis = np.zeros((n_components - 1, n_components), dtype=float)

    for row in range(1, n_components):
        scale = np.sqrt(row * (row + 1.0))
        basis[row - 1, :row] = 1.0 / scale
        basis[row - 1, row] = -row / scale

    return basis


def ilr(values: ArrayLike) -> FloatArray:
    """Isometric log-ratio transformation using a Helmert basis."""
    clr_values = clr(values)
    basis = helmert_submatrix(clr_values.shape[1])

    return clr_values @ basis.T


def ilr_inverse(values: ArrayLike, total: float = 1.0) -> FloatArray:
    """Inverse ILR transformation for the Helmert basis."""
    coordinates = np.asarray(values, dtype=float)

    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, -1)

    if coordinates.ndim != 2:
        raise ValueError("ILR coordinates must be one- or two-dimensional.")

    if not np.isfinite(coordinates).all():
        raise ValueError("ILR coordinates must contain only finite values.")

    n_components = coordinates.shape[1] + 1
    basis = helmert_submatrix(n_components)
    clr_values = coordinates @ basis

    return clr_inverse(clr_values, total=total)


def aitchison_distance(
    x: ArrayLike,
    y: ArrayLike,
) -> FloatArray:
    """Compute row-wise Aitchison distance between two compositions."""
    x_clr = clr(x)
    y_clr = clr(y)

    if x_clr.shape != y_clr.shape:
        raise ValueError("x and y must have the same compositional shape.")

    return np.linalg.norm(x_clr - y_clr, axis=1)

def replace_below_detection_limit(
    values: ArrayLike,
    detection_limits: ArrayLike,
    fraction: float = 0.5,
) -> FloatArray:
    """Replace non-positive censored values with a fraction of their LOD.

    Missing values are preserved. This function performs a simple deterministic
    replacement intended for controlled sensitivity experiments, not as a
    universal treatment of censored geochemical data.
    """
    array = np.asarray(values, dtype=float).copy()
    limits = np.asarray(detection_limits, dtype=float)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError("Values must be one- or two-dimensional.")

    if limits.ndim != 1 or limits.shape[0] != array.shape[1]:
        raise ValueError(
            "Detection limits must contain one value per compositional component."
        )

    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError("Detection limits must be positive and finite.")

    if not 0.0 < fraction < 1.0:
        raise ValueError("Replacement fraction must lie strictly between 0 and 1.")

    censored = np.isfinite(array) & (array <= 0.0)

    replacement = limits * fraction
    array[censored] = np.broadcast_to(
        replacement,
        array.shape,
    )[censored]

    return array
