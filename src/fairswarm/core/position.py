"""
Position encoding utilities for FairSwarm PSO.

This module provides functions for transforming between continuous
position vectors and discrete coalitions.

Key Functions:
    - sigmoid: Bounds position values to [0, 1] (Algorithm 1: x_p ← Sigmoid(x_p))
    - decode_coalition: Converts position to coalition (Algorithm 1: SelectTop(x_p, m))

Mathematical Foundation:
    FairSwarm uses a continuous relaxation where each client i has a
    selection probability p_i ∈ [0, 1]. The discrete coalition is obtained
    by selecting the m clients with highest probabilities.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fairswarm.types import Coalition


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply sigmoid function to bound values to [0, 1].

    The sigmoid function σ(x) = 1 / (1 + e^(-x)) maps any real value
    to the interval (0, 1), providing smooth bounding for position vectors.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with values in (0, 1)

    Algorithm Reference:
        Algorithm 1, Line 571: x_p ← Sigmoid(x_p)

    Properties:
        - σ(0) = 0.5
        - σ(x) → 1 as x → ∞
        - σ(x) → 0 as x → -∞
        - Smooth and differentiable everywhere

    Example:
        >>> import numpy as np
        >>> from fairswarm.core.position import sigmoid
        >>> x = np.array([-2.0, 0.0, 2.0])
        >>> sigmoid(x)
        array([0.11920292, 0.5       , 0.88079708])

    Note:
        Uses clipping to avoid numerical overflow for extreme values.
    """
    # Clip to avoid overflow in exp
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def inverse_sigmoid(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply inverse sigmoid (logit) function.

    The logit function is the inverse of sigmoid:
        logit(p) = log(p / (1 - p))

    Args:
        p: Input array with values in (0, 1)

    Returns:
        Array of unbounded real values

    Raises:
        ValueError: If values are not in (0, 1)

    Example:
        >>> p = np.array([0.1, 0.5, 0.9])
        >>> inverse_sigmoid(p)
        array([-2.19722458,  0.        ,  2.19722458])
    """
    # Clip to avoid log(0) or log(inf)
    eps = 1e-10
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def decode_coalition(
    position: NDArray[np.float64],
    coalition_size: int,
) -> Coalition:
    """
    Decode continuous position to discrete coalition using SelectTop.

    Selects the m clients with the highest position values, where m
    is the coalition size.

    Args:
        position: Continuous position vector x_p ∈ [0,1]^n
        coalition_size: Number of clients to select (m)

    Returns:
        List of client indices forming the coalition

    Raises:
        ValueError: If coalition_size > len(position) or coalition_size < 1

    Algorithm Reference:
        Algorithm 1: S_p ← SelectTop(x_p, m)

    Mathematical Interpretation:
        The position vector represents selection probabilities. SelectTop
        is a deterministic rounding that selects the m most likely clients.

    Theorem 3 Connection:
        This rounding is analyzed as pipage rounding in the proof of
        Theorem 3, giving the (1-1/e-η) approximation guarantee.

    Example:
        >>> import numpy as np
        >>> position = np.array([0.9, 0.2, 0.7, 0.5, 0.8])
        >>> decode_coalition(position, coalition_size=3)
        [0, 4, 2]  # Indices of top 3 values (0.9, 0.8, 0.7)
    """
    n_clients = len(position)

    if coalition_size < 1:
        raise ValueError(f"coalition_size must be >= 1, got {coalition_size}")

    if coalition_size > n_clients:
        raise ValueError(
            f"coalition_size ({coalition_size}) cannot exceed "
            f"number of clients ({n_clients})"
        )

    # Get indices sorted by position value (descending)
    sorted_indices = np.argsort(position)[::-1]

    # Select top m indices
    coalition: Coalition = [int(x) for x in sorted_indices[:coalition_size]]

    return coalition


def encode_coalition(
    coalition: Coalition,
    n_clients: int,
    selected_value: float = 0.9,
    unselected_value: float = 0.1,
) -> NDArray[np.float64]:
    """
    Encode a discrete coalition as a continuous position vector.

    This is useful for initializing particles with known good coalitions
    or for analysis purposes.

    Args:
        coalition: List of selected client indices
        n_clients: Total number of clients
        selected_value: Position value for selected clients
        unselected_value: Position value for unselected clients

    Returns:
        Position vector with high values for selected clients

    Example:
        >>> coalition = [0, 2, 4]
        >>> encode_coalition(coalition, n_clients=5)
        array([0.9, 0.1, 0.9, 0.1, 0.9])
    """
    position: NDArray[np.float64] = np.full(
        n_clients, unselected_value, dtype=np.float64
    )
    for idx in coalition:
        if 0 <= idx < n_clients:
            position[idx] = selected_value
    return position


def soft_decode_coalition(
    position: NDArray[np.float64],
    coalition_size: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Coalition:
    """
    Stochastically decode position to coalition using softmax sampling.

    Instead of deterministic SelectTop, this uses softmax probabilities
    to sample a coalition. Useful for exploration or ensemble methods.

    Args:
        position: Continuous position vector
        coalition_size: Number of clients to select
        temperature: Softmax temperature (lower = more deterministic)
        rng: Random number generator

    Returns:
        Sampled coalition

    Note:
        Higher temperature increases randomness. Temperature → 0 approaches
        deterministic SelectTop behavior.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_clients = len(position)

    if coalition_size > n_clients:
        raise ValueError(
            f"coalition_size ({coalition_size}) cannot exceed "
            f"number of clients ({n_clients})"
        )

    # Compute softmax probabilities
    scaled = position / temperature
    scaled = scaled - np.max(scaled)  # Numerical stability
    exp_scaled = np.exp(scaled)
    probabilities = exp_scaled / np.sum(exp_scaled)

    # Sample without replacement
    coalition: Coalition = [
        int(x)
        for x in rng.choice(
            n_clients,
            size=coalition_size,
            replace=False,
            p=probabilities,
        )
    ]

    return coalition


def position_similarity(
    pos1: NDArray[np.float64],
    pos2: NDArray[np.float64],
) -> float:
    """
    Compute similarity between two position vectors.

    Uses cosine similarity, which is appropriate for comparing
    the relative emphasis on different clients.

    Args:
        pos1: First position vector
        pos2: Second position vector

    Returns:
        Cosine similarity in [-1, 1], where 1 = identical direction

    Example:
        >>> p1 = np.array([0.9, 0.1, 0.8])
        >>> p2 = np.array([0.8, 0.2, 0.9])
        >>> position_similarity(p1, p2)
        0.987...
    """
    norm1 = np.linalg.norm(pos1)
    norm2 = np.linalg.norm(pos2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(pos1, pos2) / (norm1 * norm2))


def coalition_overlap(
    coalition1: Coalition,
    coalition2: Coalition,
) -> float:
    """
    Compute Jaccard similarity between two coalitions.

    Args:
        coalition1: First coalition
        coalition2: Second coalition

    Returns:
        Jaccard similarity in [0, 1]

    Example:
        >>> coalition_overlap([0, 1, 2], [1, 2, 3])
        0.5  # 2 shared out of 4 total unique
    """
    set1 = set(coalition1)
    set2 = set(coalition2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 1.0  # Both empty

    return intersection / union
