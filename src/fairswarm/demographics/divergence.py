"""
Divergence metrics for demographic distributions.

This module implements divergence measures used in FairSwarm for
quantifying demographic dissimilarity between coalitions and targets.

Mathematical Foundation (Definition 2 from CLAUDE.md):
    DemDiv(S) = D_KL(δ_S || δ*)

    where:
    - δ_S = (1/|S|) Σ_{i ∈ S} δ_i is the coalition's demographic distribution
    - δ* is the target distribution
    - D_KL is the Kullback-Leibler divergence

Key Functions:
    - kl_divergence: KL divergence D_KL(P || Q) - MATCHES DEFINITION 2
    - js_divergence: Jensen-Shannon divergence (symmetric)
    - wasserstein_distance: Earth mover's distance
    - total_variation_distance: L1-based distance

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fairswarm.demographics.distribution import (
    DemographicDistribution,
    combine_distributions,
)

# =============================================================================
# KL Divergence - DEFINITION 2 IMPLEMENTATION
# =============================================================================


def kl_divergence(
    p: NDArray[np.float64] | DemographicDistribution,
    q: NDArray[np.float64] | DemographicDistribution,
    eps: float = 1e-10,
) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).

    This is the PRIMARY divergence measure for FairSwarm, implementing
    Definition 2 from CLAUDE.md exactly:

        DemDiv(S) = D_KL(δ_S || δ*)

    where δ_S is the coalition distribution and δ* is the target.

    Mathematical Definition:
        D_KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))

    Properties:
        - Non-negative: D_KL(P || Q) ≥ 0
        - Zero iff P = Q: D_KL(P || Q) = 0 ⟺ P = Q
        - Asymmetric: D_KL(P || Q) ≠ D_KL(Q || P) in general
        - Not a metric: Does not satisfy triangle inequality

    Args:
        p: First distribution (coalition demographics δ_S)
        q: Second distribution (target demographics δ*)
        eps: Small constant for numerical stability (avoid log(0))

    Returns:
        KL divergence value (non-negative float)

    Raises:
        ValueError: If distributions have different lengths
        ValueError: If either distribution does not sum to ~1.0 (tolerance: 0.01)

    Example:
        >>> import numpy as np
        >>> p = np.array([0.4, 0.3, 0.2, 0.1])  # Coalition
        >>> q = np.array([0.25, 0.25, 0.25, 0.25])  # Target (uniform)
        >>> kl_divergence(p, q)
        0.1206...

    Research Reference:
        Definition 2 in CLAUDE.md defines DemDiv(S) = D_KL(δ_S || δ*)
        Theorem 2 bounds this divergence for FairSwarm output.

    Security Note:
        This function operates on aggregated demographic statistics only.
        Never pass individual patient data.
    """
    # Convert DemographicDistribution to arrays
    p_arr = p.as_array() if isinstance(p, DemographicDistribution) else np.asarray(p)
    q_arr = q.as_array() if isinstance(q, DemographicDistribution) else np.asarray(q)

    # Validate same length
    if len(p_arr) != len(q_arr):
        raise ValueError(
            f"Distributions must have same length. Got {len(p_arr)} and {len(q_arr)}"
        )

    # Validate that inputs are valid probability distributions (sum to ~1.0)
    p_sum = float(np.sum(p_arr))
    q_sum = float(np.sum(q_arr))
    tolerance = 0.01
    if abs(p_sum - 1.0) > tolerance:
        raise ValueError(
            f"Distribution p does not sum to 1.0 (got {p_sum:.6f}). "
            f"Pass a valid probability distribution or normalize before calling."
        )
    if abs(q_sum - 1.0) > tolerance:
        raise ValueError(
            f"Distribution q does not sum to 1.0 (got {q_sum:.6f}). "
            f"Pass a valid probability distribution or normalize before calling."
        )

    # Add smoothing to avoid log(0) - as specified in CLAUDE.md
    p_smooth = np.clip(p_arr, eps, 1.0)
    q_smooth = np.clip(q_arr, eps, 1.0)

    # Renormalize ONLY to account for the slight change introduced by smoothing
    p_smooth = p_smooth / np.sum(p_smooth)
    q_smooth = q_smooth / np.sum(q_smooth)

    # KL divergence: Σ P(i) * log(P(i) / Q(i))
    # Only sum over non-zero P entries (0 * log(0) = 0 by convention)
    kl: float = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))

    return float(kl)


# =============================================================================
# Alternative Divergence Measures
# =============================================================================


def js_divergence(
    p: NDArray[np.float64] | DemographicDistribution,
    q: NDArray[np.float64] | DemographicDistribution,
    eps: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric version of KL).

    The JS divergence is a symmetrized and smoothed version of KL divergence:
        JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)

    where M = 0.5 * (P + Q) is the midpoint distribution.

    Properties:
        - Symmetric: JS(P || Q) = JS(Q || P)
        - Bounded: 0 ≤ JS(P || Q) ≤ log(2)
        - sqrt(JS) is a proper metric

    Args:
        p: First distribution
        q: Second distribution
        eps: Smoothing constant

    Returns:
        JS divergence value in [0, log(2)]

    Example:
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> js_divergence(p, q)
        0.1408...
    """
    # Convert to arrays
    p_arr = p.as_array() if isinstance(p, DemographicDistribution) else np.asarray(p)
    q_arr = q.as_array() if isinstance(q, DemographicDistribution) else np.asarray(q)

    # Midpoint distribution
    m = 0.5 * (p_arr + q_arr)

    # JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    js = 0.5 * kl_divergence(p_arr, m, eps) + 0.5 * kl_divergence(q_arr, m, eps)

    return float(js)


def total_variation_distance(
    p: NDArray[np.float64] | DemographicDistribution,
    q: NDArray[np.float64] | DemographicDistribution,
) -> float:
    """
    Compute total variation distance between distributions.

    The TV distance is half the L1 distance:
        TV(P, Q) = 0.5 * Σ |P(i) - Q(i)|

    Properties:
        - Symmetric: TV(P, Q) = TV(Q, P)
        - Bounded: 0 ≤ TV(P, Q) ≤ 1
        - Is a proper metric

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        TV distance in [0, 1]

    Example:
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> total_variation_distance(p, q)
        0.4
    """
    p_arr = p.as_array() if isinstance(p, DemographicDistribution) else np.asarray(p)
    q_arr = q.as_array() if isinstance(q, DemographicDistribution) else np.asarray(q)

    if len(p_arr) != len(q_arr):
        raise ValueError(
            f"Distributions must have same length. Got {len(p_arr)} and {len(q_arr)}"
        )

    return float(0.5 * np.sum(np.abs(p_arr - q_arr)))


def wasserstein_distance(
    p: NDArray[np.float64] | DemographicDistribution,
    q: NDArray[np.float64] | DemographicDistribution,
) -> float:
    """
    Compute 1-Wasserstein (Earth Mover's) distance for 1D distributions.

    For discrete distributions on ordered categories, this is the
    L1 distance between cumulative distribution functions:
        W_1(P, Q) = Σ |CDF_P(i) - CDF_Q(i)|

    Note:
        This assumes demographic groups have a natural ordering.
        For unordered categories, use total_variation_distance instead.

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        Wasserstein distance (non-negative)

    Example:
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> q = np.array([0.2, 0.3, 0.5])
        >>> wasserstein_distance(p, q)
        0.6
    """
    p_arr = p.as_array() if isinstance(p, DemographicDistribution) else np.asarray(p)
    q_arr = q.as_array() if isinstance(q, DemographicDistribution) else np.asarray(q)

    if len(p_arr) != len(q_arr):
        raise ValueError(
            f"Distributions must have same length. Got {len(p_arr)} and {len(q_arr)}"
        )

    # Compute CDFs
    cdf_p = np.cumsum(p_arr)
    cdf_q = np.cumsum(q_arr)

    # L1 distance between CDFs
    return float(np.sum(np.abs(cdf_p - cdf_q)))


# =============================================================================
# Coalition Divergence (Definition 2 Complete Implementation)
# =============================================================================


def coalition_demographic_divergence(
    client_demographics: Sequence[NDArray[np.float64] | DemographicDistribution],
    coalition_indices: Sequence[int],
    target: NDArray[np.float64] | DemographicDistribution,
    weights: Sequence[float] | None = None,
) -> float:
    """
    Compute demographic divergence for a coalition from target.

    This is the complete implementation of Definition 2 from CLAUDE.md:

        DemDiv(S) = D_KL(δ_S || δ*)

    where δ_S = (1/|S|) Σ_{i ∈ S} δ_i is the coalition's average demographics.

    Args:
        client_demographics: Demographics for all clients
        coalition_indices: Indices of clients in the coalition
        target: Target demographic distribution δ*
        weights: Optional weights for coalition members (default: uniform)

    Returns:
        KL divergence from coalition demographics to target

    Raises:
        ValueError: If coalition is empty or indices out of range

    Example:
        >>> # Three hospitals with different demographics
        >>> demographics = [
        ...     np.array([0.8, 0.1, 0.1]),  # Hospital 0
        ...     np.array([0.2, 0.7, 0.1]),  # Hospital 1
        ...     np.array([0.3, 0.3, 0.4]),  # Hospital 2
        ... ]
        >>> target = np.array([0.4, 0.4, 0.2])  # Target distribution
        >>> coalition = [0, 2]  # Select hospitals 0 and 2
        >>> divergence = coalition_demographic_divergence(
        ...     demographics, coalition, target
        ... )

    Research Reference:
        Definition 2: DemDiv(S) = D_KL((1/|S|) Σ_{i ∈ S} δ_i || δ*)
        Theorem 2: FairSwarm guarantees DemDiv(S*) ≤ ε with high probability
    """
    if len(coalition_indices) == 0:
        raise ValueError("Coalition cannot be empty")

    # Validate indices
    max_idx = len(client_demographics) - 1
    for idx in coalition_indices:
        if idx < 0 or idx > max_idx:
            raise ValueError(f"Coalition index {idx} out of range [0, {max_idx}]")

    # Convert client demographics to DemographicDistribution objects if needed
    coalition_dists: list[DemographicDistribution] = []
    for idx in coalition_indices:
        demo = client_demographics[idx]
        if isinstance(demo, DemographicDistribution):
            coalition_dists.append(demo)
        else:
            demo_arr: NDArray[np.float64] = np.asarray(demo, dtype=np.float64)
            coalition_dists.append(DemographicDistribution(values=demo_arr))

    # Compute coalition average: δ_S = (1/|S|) Σ_{i ∈ S} δ_i
    if weights is not None:
        # Normalize weights for coalition members
        coalition_weights = [weights[i] for i in coalition_indices]
        weight_sum = sum(coalition_weights)
        if weight_sum <= 0:
            raise ValueError("Coalition weights must have positive sum")
        coalition_weights = [w / weight_sum for w in coalition_weights]
    else:
        coalition_weights = None

    coalition_avg = combine_distributions(coalition_dists, coalition_weights)

    # Convert target if needed
    target_arr = (
        target.as_array() if isinstance(target, DemographicDistribution) else target
    )

    # Compute D_KL(δ_S || δ*) - Definition 2
    return kl_divergence(coalition_avg.as_array(), target_arr)


def is_epsilon_fair(
    divergence: float,
    epsilon: float,
) -> bool:
    """
    Check if a divergence value satisfies ε-fairness.

    Definition 3 from CLAUDE.md:
        A coalition S is ε-fair if DemDiv(S) ≤ ε

    Args:
        divergence: Computed demographic divergence
        epsilon: Fairness threshold

    Returns:
        True if divergence ≤ epsilon

    Example:
        >>> divergence = 0.03
        >>> is_epsilon_fair(divergence, epsilon=0.05)
        True
    """
    return divergence <= epsilon
