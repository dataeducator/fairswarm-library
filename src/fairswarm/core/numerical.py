"""
Numerical stability utilities for FairSwarm.

This module provides functions for ensuring numerical stability
in FairSwarm computations, particularly for distributions and gradients.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Default epsilon for numerical stability
DEFAULT_EPS = 1e-10


def safe_normalize(
    x: NDArray[np.float64],
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """
    Safely normalize an array to sum to 1.

    Args:
        x: Input array (must be non-negative)
        eps: Small value for numerical stability

    Returns:
        Normalized array summing to 1. If input is all zeros,
        returns uniform distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0, None)  # Ensure non-negative
    total: float = float(np.sum(x))

    # If all zeros, return uniform distribution
    if total <= eps:
        return np.ones_like(x) / len(x)

    return x / total


def safe_log(
    x: NDArray[np.float64],
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """
    Compute log with clipping to avoid log(0).

    Args:
        x: Input array
        eps: Minimum value for clipping

    Returns:
        log(max(x, eps))
    """
    x_safe = np.clip(x, eps, None)
    return np.log(x_safe)


def safe_divide(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """
    Safely divide with protection against division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Small value added to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def check_distribution(
    p: NDArray[np.float64],
    eps: float = 1e-6,
) -> tuple[bool, str]:
    """
    Check if an array is a valid probability distribution.

    Args:
        p: Array to check
        eps: Tolerance for sum check

    Returns:
        Tuple of (is_valid, error_message)
    """
    p = np.asarray(p)

    # Check for NaN
    if bool(np.any(np.isnan(p))):
        return False, "Distribution contains NaN values"

    # Check for Inf
    if bool(np.any(np.isinf(p))):
        return False, "Distribution contains Inf values"

    # Check non-negativity
    if bool(np.any(p < 0)):
        return False, f"Distribution has negative values: min={p.min()}"

    # Check sum to 1
    total = float(np.sum(p))
    if abs(total - 1.0) > eps:
        return False, f"Distribution sums to {total}, not 1.0"

    return True, "Valid distribution"


def repair_distribution(
    p: NDArray[np.float64],
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """
    Repair an invalid distribution to make it valid.

    Handles:
    - NaN values (replaced with eps)
    - Inf values (replaced with eps)
    - Negative values (clipped to 0)
    - Normalization to sum to 1

    Args:
        p: Input array
        eps: Small value for replacements

    Returns:
        Valid probability distribution
    """
    p = np.asarray(p, dtype=np.float64).copy()

    # Replace NaN and Inf
    p = np.nan_to_num(p, nan=eps, posinf=eps, neginf=0.0)

    # Clip negative values
    p = np.clip(p, 0, None)

    # If all zeros, make uniform
    if np.sum(p) <= eps:
        p = np.ones_like(p)

    # Normalize
    return p / np.sum(p)


def check_gradient(
    gradient: NDArray[np.float64],
    max_norm: float = 100.0,
) -> tuple[bool, str]:
    """
    Check if a gradient is numerically healthy.

    Args:
        gradient: Gradient array
        max_norm: Maximum allowed L2 norm

    Returns:
        Tuple of (is_healthy, message)
    """
    gradient = np.asarray(gradient)

    # Check for NaN
    if bool(np.any(np.isnan(gradient))):
        return False, "Gradient contains NaN values"

    # Check for Inf
    if bool(np.any(np.isinf(gradient))):
        return False, "Gradient contains Inf values"

    # Check norm
    norm = float(np.linalg.norm(gradient))
    if norm > max_norm:
        return False, f"Gradient norm {norm:.2f} exceeds max {max_norm}"

    return True, f"Healthy gradient (norm={norm:.4f})"


def clip_gradient(
    gradient: NDArray[np.float64],
    max_norm: float = 1.0,
) -> NDArray[np.float64]:
    """
    Clip gradient to maximum norm.

    Args:
        gradient: Input gradient
        max_norm: Maximum L2 norm

    Returns:
        Clipped gradient
    """
    gradient = np.asarray(gradient, dtype=np.float64)

    # Handle NaN/Inf
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip norm
    norm = np.linalg.norm(gradient)
    if norm > max_norm and norm > 0:
        gradient = gradient * (max_norm / norm)

    return gradient


__all__ = [
    "DEFAULT_EPS",
    "safe_normalize",
    "safe_log",
    "safe_divide",
    "check_distribution",
    "repair_distribution",
    "check_gradient",
    "clip_gradient",
]
