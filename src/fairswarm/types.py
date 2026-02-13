"""
Core type definitions for FairSwarm.

This module defines the fundamental types used throughout the FairSwarm library.
All types are designed to be compatible with static type checkers (mypy) and
provide clear semantic meaning for the algorithm's mathematical constructs.

Type Aliases:
    ClientId: Unique identifier for a federated learning client
    Coalition: Set of client indices forming a coalition
    DemographicVector: Probability distribution over demographic groups
    FitnessValue: Scalar fitness value from optimization

Research Foundation:
    These types map directly to the mathematical notation in CLAUDE.md:
    - ClientId → c_i in C = {c_1, ..., c_n}
    - Coalition → S ⊆ C
    - DemographicVector → δ_i ∈ Δ^(k-1)
    - FitnessValue → F(S) ∈ ℝ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NewType, Sequence, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Demographics Container
# =============================================================================


@dataclass
class Demographics:
    """
    Named demographic proportions container.

    Stores named demographic attributes (e.g., age, gender, race) as
    proportions. Values are normalized to a probability distribution
    when converted to an array.

    Attributes:
        age: Age-related demographic proportion
        gender: Gender-related demographic proportion
        race: Race-related demographic proportion
    """

    age: float = 0.0
    gender: float = 0.0
    race: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        """Convert to a normalized numpy array (probability distribution)."""
        arr = np.array([self.age, self.gender, self.race], dtype=np.float64)
        total = arr.sum()
        if total > 0:
            return arr / total
        return np.ones(3, dtype=np.float64) / 3  # uniform if all zeros

    def to_labels(self) -> tuple:
        """Return the demographic group labels."""
        return ("age", "gender", "race")


# =============================================================================
# Semantic Type Aliases
# =============================================================================

# Client identifier - typically a hospital or institution ID
ClientId = NewType("ClientId", str)

# Coalition as a list of client indices (positions in the client array)
Coalition = List[int]

# Demographic vector: probability distribution over k demographic groups
# Must sum to 1.0 and contain non-negative values
DemographicVector = NDArray[np.float64]

# Fitness value: scalar result of fitness function evaluation
FitnessValue = float

# Position vector in PSO: continuous values in [0, 1]^n representing
# selection probabilities for each client
PositionVector = NDArray[np.float64]

# Velocity vector in PSO: continuous values representing rate of change
VelocityVector = NDArray[np.float64]


# =============================================================================
# Validation Functions
# =============================================================================


def validate_demographic_vector(
    vector: DemographicVector,
    tolerance: float = 1e-6,
) -> bool:
    """
    Validate that a vector is a proper probability distribution.

    A valid demographic vector must:
    1. Contain only non-negative values
    2. Sum to 1.0 (within tolerance)

    Args:
        vector: The demographic vector to validate
        tolerance: Numerical tolerance for sum check

    Returns:
        True if valid, False otherwise

    Example:
        >>> valid = np.array([0.6, 0.2, 0.15, 0.05])
        >>> validate_demographic_vector(valid)
        True
        >>> invalid = np.array([0.6, 0.2, 0.1, 0.05])  # sums to 0.95
        >>> validate_demographic_vector(invalid)
        False
    """
    if np.any(vector < 0):
        return False
    if not np.isclose(np.sum(vector), 1.0, atol=tolerance):
        return False
    return True


def validate_coalition(
    coalition: Coalition,
    n_clients: int,
) -> bool:
    """
    Validate that a coalition contains valid, unique client indices.

    Args:
        coalition: List of client indices
        n_clients: Total number of clients

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_coalition([0, 2, 5], n_clients=10)
        True
        >>> validate_coalition([0, 2, 15], n_clients=10)  # 15 out of range
        False
        >>> validate_coalition([0, 2, 2], n_clients=10)  # duplicate
        False
    """
    if len(coalition) != len(set(coalition)):
        return False  # Duplicates
    if any(idx < 0 or idx >= n_clients for idx in coalition):
        return False  # Out of range
    return True


def validate_position_vector(
    position: PositionVector,
    n_clients: int,
) -> bool:
    """
    Validate that a position vector has correct shape and bounds.

    Position vectors in FairSwarm represent selection probabilities
    and must be in [0, 1]^n after sigmoid transformation.

    Args:
        position: The position vector to validate
        n_clients: Expected dimension (number of clients)

    Returns:
        True if valid, False otherwise
    """
    if position.shape != (n_clients,):
        return False
    if np.any(position < 0) or np.any(position > 1):
        return False
    return True


# =============================================================================
# Type Conversion Utilities
# =============================================================================


def normalize_to_distribution(
    values: Union[Sequence[float], NDArray[np.float64]],
) -> DemographicVector:
    """
    Normalize a sequence of non-negative values to a probability distribution.

    Args:
        values: Non-negative values to normalize

    Returns:
        Normalized demographic vector summing to 1.0

    Raises:
        ValueError: If values contain negative numbers or sum to zero

    Example:
        >>> normalize_to_distribution([60, 20, 15, 5])
        array([0.6 , 0.2 , 0.15, 0.05])
    """
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("Values must be non-negative")
    total = np.sum(arr)
    if total == 0:
        raise ValueError("Values must have positive sum")
    return arr / total
