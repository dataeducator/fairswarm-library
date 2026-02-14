"""
Demographic distribution representation.

This module provides the DemographicDistribution class for handling
probability distributions over demographic groups.

Research Foundation:
    Each client c_i has demographic distribution δ_i ∈ Δ^(k-1)
    where Δ^(k-1) is the (k-1)-dimensional probability simplex.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fairswarm.types import Demographics


@dataclass(frozen=True)
class DemographicDistribution:
    """
    Represents a probability distribution over demographic groups.

    A demographic distribution δ ∈ Δ^(k-1) is a vector of k non-negative
    values that sum to 1, where k is the number of demographic groups.

    Attributes:
        values: The probability values for each demographic group
        labels: Optional names for each demographic group

    Mathematical Properties:
        - All values are non-negative: δ_i ≥ 0
        - Values sum to 1: Σ δ_i = 1
        - Immutable (frozen dataclass)

    Example:
        >>> dist = DemographicDistribution.from_dict({
        ...     "white": 0.60, "black": 0.20, "hispanic": 0.15, "other": 0.05
        ... })
        >>> dist.as_array()
        array([0.6 , 0.2 , 0.15, 0.05])
        >>> dist["white"]
        0.6

    Security Note:
        Demographics should be aggregated population-level data,
        never individual patient information.
    """

    values: NDArray[np.float64]
    labels: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate that values form a valid probability distribution."""
        # Ensure numpy array
        if not isinstance(self.values, np.ndarray):
            object.__setattr__(
                self, "values", np.asarray(self.values, dtype=np.float64)
            )

        # Check non-negative
        if bool(np.any(self.values < 0)):
            raise ValueError(
                "Demographic distribution values must be non-negative. "
                f"Got minimum value: {float(np.min(self.values))}"
            )

        # Check sum to 1 (with tolerance)
        total = float(np.sum(self.values))
        if not bool(np.isclose(total, 1.0, atol=1e-6)):
            raise ValueError(
                f"Demographic distribution must sum to 1.0, got {total:.6f}. "
                "Use DemographicDistribution.from_counts() for unnormalized data."
            )

        # Validate labels match values length
        if self.labels is not None and len(self.labels) != len(self.values):
            raise ValueError(
                f"Labels length ({len(self.labels)}) must match "
                f"values length ({len(self.values)})"
            )

    @property
    def n_groups(self) -> int:
        """Number of demographic groups."""
        return len(self.values)

    @property
    def entropy(self) -> float:
        """
        Shannon entropy of the distribution.

        Higher entropy means more uniform distribution.
        Maximum entropy = log(k) for uniform distribution over k groups.

        Returns:
            Entropy in nats (natural log)
        """
        # Avoid log(0) by filtering zero probabilities
        nonzero = self.values[self.values > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy for this number of groups."""
        return float(np.log(self.n_groups))

    @property
    def normalized_entropy(self) -> float:
        """
        Entropy normalized to [0, 1] range.

        0 = all probability in one group (minimum diversity)
        1 = uniform distribution (maximum diversity)
        """
        if self.max_entropy == 0:
            return 1.0  # Single group
        return self.entropy / self.max_entropy

    def as_array(self) -> NDArray[np.float64]:
        """
        Get the distribution as a numpy array.

        Returns:
            Copy of the probability values

        Note:
            Returns a copy to maintain immutability.
        """
        return self.values.copy()

    def as_dict(self) -> dict[str, float]:
        """
        Get the distribution as a dictionary.

        Returns:
            Dictionary mapping group labels to probabilities

        Raises:
            ValueError: If no labels are defined
        """
        if self.labels is None:
            raise ValueError(
                "Cannot convert to dict: no labels defined. "
                "Use from_dict() or provide labels at construction."
            )
        return {label: float(val) for label, val in zip(self.labels, self.values)}

    def __getitem__(self, key: int | str) -> float:
        """
        Get probability for a demographic group by index or label.

        Args:
            key: Integer index or string label

        Returns:
            Probability value

        Raises:
            KeyError: If string label not found
            IndexError: If integer index out of range
        """
        if isinstance(key, str):
            if self.labels is None:
                raise KeyError(f"No labels defined, cannot lookup '{key}'")
            if key not in self.labels:
                raise KeyError(f"Label '{key}' not found. Available: {self.labels}")
            idx = self.labels.index(key)
            return float(self.values[idx])
        else:
            return float(self.values[key])

    def __len__(self) -> int:
        """Number of demographic groups."""
        return self.n_groups

    def __iter__(self) -> Iterator[float]:
        """Iterate over probability values."""
        return iter(float(v) for v in self.values)

    def items(self) -> Iterator[tuple[str, float]]:
        """
        Iterate over (label, probability) pairs.

        Yields:
            Tuples of (label, probability)

        Raises:
            ValueError: If no labels defined
        """
        if self.labels is None:
            raise ValueError("No labels defined")
        for label, val in zip(self.labels, self.values):
            yield label, float(val)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, float],
        normalize: bool = False,
    ) -> DemographicDistribution:
        """
        Create a distribution from a dictionary.

        Args:
            data: Dictionary mapping group names to values
            normalize: If True, normalize values to sum to 1

        Returns:
            New DemographicDistribution instance

        Example:
            >>> dist = DemographicDistribution.from_dict({
            ...     "white": 0.60, "black": 0.25, "hispanic": 0.15
            ... })
        """
        labels = tuple(data.keys())
        values = np.array(list(data.values()), dtype=np.float64)

        if normalize:
            total = float(np.sum(values))
            if total <= 0:
                raise ValueError("Cannot normalize: sum is zero or negative")
            values = values / total

        return cls(values=values, labels=labels)

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, int] | Sequence[int],
        labels: Sequence[str] | None = None,
    ) -> DemographicDistribution:
        """
        Create a distribution from raw counts (automatically normalizes).

        Args:
            counts: Dictionary or sequence of counts per group
            labels: Labels for sequence input (ignored for dict input)

        Returns:
            Normalized DemographicDistribution

        Example:
            >>> dist = DemographicDistribution.from_counts({
            ...     "white": 600, "black": 200, "hispanic": 150, "other": 50
            ... })
            >>> dist["white"]
            0.6
        """
        if isinstance(counts, dict):
            return cls.from_dict(
                {k: float(v) for k, v in counts.items()},
                normalize=True,
            )
        else:
            values = np.array(counts, dtype=np.float64)
            total = float(np.sum(values))
            if total <= 0:
                raise ValueError("Cannot normalize: sum is zero or negative")
            values = values / total
            return cls(
                values=values,
                labels=tuple(labels) if labels else None,
            )

    @classmethod
    def uniform(
        cls,
        n_groups: int,
        labels: Sequence[str] | None = None,
    ) -> DemographicDistribution:
        """
        Create a uniform distribution over n groups.

        Args:
            n_groups: Number of demographic groups
            labels: Optional labels for groups

        Returns:
            Uniform distribution

        Example:
            >>> uniform = DemographicDistribution.uniform(4)
            >>> uniform.as_array()
            array([0.25, 0.25, 0.25, 0.25])
        """
        values = np.ones(n_groups, dtype=np.float64) / n_groups
        return cls(
            values=values,
            labels=tuple(labels) if labels else None,
        )

    @classmethod
    def from_demographics(
        cls,
        demographics: Demographics,
    ) -> DemographicDistribution:
        """
        Create a distribution from a Demographics object.

        The Demographics values are normalized to form a valid
        probability distribution.

        Args:
            demographics: Demographics object with named proportions

        Returns:
            Normalized DemographicDistribution
        """
        from fairswarm.types import Demographics as DemoType

        if not isinstance(demographics, DemoType):
            raise TypeError(f"Expected Demographics, got {type(demographics)}")
        values = demographics.to_array()
        labels = demographics.to_labels()
        return cls(values=values, labels=labels)

    def reorder(
        self,
        new_labels: Sequence[str],
    ) -> DemographicDistribution:
        """
        Reorder distribution to match a new label ordering.

        Args:
            new_labels: Desired order of labels

        Returns:
            New distribution with reordered values

        Raises:
            ValueError: If labels don't match
        """
        if self.labels is None:
            raise ValueError("Cannot reorder: no labels defined")

        if set(new_labels) != set(self.labels):
            raise ValueError(
                f"New labels {set(new_labels)} don't match "
                f"existing labels {set(self.labels)}"
            )

        new_values = np.array([self[label] for label in new_labels])
        return DemographicDistribution(
            values=new_values,
            labels=tuple(new_labels),
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.labels:
            pairs = ", ".join(f"{k}={v:.3f}" for k, v in zip(self.labels, self.values))
            return f"DemographicDistribution({pairs})"
        else:
            return f"DemographicDistribution({self.values})"


def combine_distributions(
    distributions: Sequence[DemographicDistribution],
    weights: Sequence[float] | None = None,
) -> DemographicDistribution:
    """
    Combine multiple distributions into a weighted average.

    This computes δ_S = Σ w_i δ_i for coalition demographic aggregation.
    If no weights provided, uses uniform weighting (simple average).

    Args:
        distributions: Sequence of distributions to combine
        weights: Optional weights for each distribution (must sum to 1)

    Returns:
        Combined distribution

    Mathematical Reference:
        From Definition 2: δ_S = (1/|S|) Σ_{i ∈ S} δ_i

    Example:
        >>> d1 = DemographicDistribution(np.array([0.8, 0.2]))
        >>> d2 = DemographicDistribution(np.array([0.4, 0.6]))
        >>> combined = combine_distributions([d1, d2])
        >>> combined.as_array()
        array([0.6, 0.4])
    """
    if len(distributions) == 0:
        raise ValueError("Cannot combine empty list of distributions")

    n_groups = distributions[0].n_groups
    for i, d in enumerate(distributions):
        if d.n_groups != n_groups:
            raise ValueError(
                f"All distributions must have same number of groups. "
                f"Distribution 0 has {n_groups}, distribution {i} has {d.n_groups}"
            )

    if weights is None:
        weights = [1.0 / len(distributions)] * len(distributions)
    else:
        weights = list(weights)
        if len(weights) != len(distributions):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of distributions ({len(distributions)})"
            )
        if not bool(np.isclose(sum(weights), 1.0, atol=1e-6)):
            raise ValueError(f"Weights must sum to 1, got {sum(weights)}")

    # Weighted combination
    combined: NDArray[np.float64] = np.zeros(n_groups, dtype=np.float64)
    for dist, weight in zip(distributions, weights):
        combined += weight * dist.values

    # Preserve labels from first distribution if available
    labels = distributions[0].labels

    return DemographicDistribution(values=combined, labels=labels)
