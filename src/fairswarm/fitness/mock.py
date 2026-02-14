"""
Mock fitness functions for testing FairSwarm.

This module provides deterministic fitness functions for unit testing
and development, allowing reproducible evaluation without actual
federated learning computations.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

__all__ = [
    "MockFitness",
    "ConstantFitness",
    "DeterministicFitness",
    "DataQualityFitness",
]


class MockFitness(FitnessFunction):
    """
    Mock fitness function for testing.

    Provides deterministic fitness values based on coalition properties,
    useful for unit testing the PSO optimization logic.

    Fitness modes:
        - "size": Fitness proportional to coalition size
        - "sum": Sum of client indices (predictable ordering)
        - "mean_quality": Average data quality of clients
        - "custom": User-provided function

    Attributes:
        mode: The fitness computation mode
        custom_fn: Custom fitness function if mode="custom"

    Example:
        >>> fitness = MockFitness(mode="size")
        >>> result = fitness.evaluate([0, 1, 2], clients)
        >>> assert result.value == 3  # Coalition size
    """

    MODES = ["size", "sum", "mean_quality", "diversity", "custom"]

    def __init__(
        self,
        mode: str = "mean_quality",
        custom_fn: Callable[[Coalition, list[Client]], float] | None = None,
    ):
        """
        Initialize MockFitness.

        Args:
            mode: Fitness computation mode
            custom_fn: Custom function if mode="custom"

        Raises:
            ValueError: If mode is unknown or custom_fn missing for custom mode
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {self.MODES}")

        if mode == "custom" and custom_fn is None:
            raise ValueError("custom_fn required when mode='custom'")

        self.mode = mode
        self.custom_fn = custom_fn

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate mock fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with deterministic fitness value
        """
        if not coalition:
            return FitnessResult(
                value=0.0,
                components={"mock": 0.0},
                coalition=coalition,
            )

        if self.mode == "size":
            # Fitness = coalition size
            fitness = float(len(coalition))

        elif self.mode == "sum":
            # Fitness = sum of indices (predictable for testing)
            fitness = float(sum(coalition))

        elif self.mode == "mean_quality":
            # Fitness = average data quality
            qualities = [
                clients[i].data_quality for i in coalition if 0 <= i < len(clients)
            ]
            fitness = float(np.mean(qualities)) if qualities else 0.0

        elif self.mode == "diversity":
            # Fitness = number of unique demographic groups represented
            groups: set[int] = set()
            for i in coalition:
                if 0 <= i < len(clients):
                    # Use the dominant demographic group
                    demo = np.asarray(clients[i].demographics, dtype=np.float64)
                    dominant = int(np.argmax(demo))
                    groups.add(dominant)
            fitness = float(len(groups))

        elif self.mode == "custom" and self.custom_fn is not None:
            fitness = self.custom_fn(coalition, clients)

        else:
            fitness = 0.0

        return FitnessResult(
            value=fitness,
            components={"mock": fitness, "mode": hash(self.mode) % 1000 / 1000},
            coalition=coalition,
            metadata={"mode": self.mode},
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute mock gradient.

        Returns uniform gradient for size mode, quality-based for others.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Mock gradient vector
        """
        n_clients = len(clients)

        if self.mode == "size":
            # All clients equally good for size
            return np.ones(n_clients) / n_clients

        elif self.mode == "mean_quality":
            # Gradient proportional to data quality
            gradient = np.array([c.data_quality for c in clients], dtype=np.float64)
            norm = np.linalg.norm(gradient)
            if norm > 1e-10:
                gradient = gradient / norm
            return gradient

        elif self.mode == "sum":
            # Higher indices are better
            gradient = np.arange(n_clients, dtype=np.float64)
            norm = np.linalg.norm(gradient)
            if norm > 1e-10:
                gradient = gradient / norm
            return gradient

        else:
            # Default: uniform
            return np.ones(n_clients) / n_clients

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "mode": self.mode,
            "has_custom_fn": self.custom_fn is not None,
        }


class ConstantFitness(FitnessFunction):
    """
    Fitness function that returns a constant value.

    Useful for testing and as a baseline.

    Attributes:
        value: The constant fitness value

    Example:
        >>> fitness = ConstantFitness(value=0.5)
        >>> result = fitness.evaluate([0, 1], clients)
        >>> assert result.value == 0.5
    """

    def __init__(self, value: float = 1.0):
        """
        Initialize ConstantFitness.

        Args:
            value: The constant fitness value to return
        """
        self.constant_value = value

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate constant fitness.

        Args:
            coalition: List of client indices (ignored)
            clients: List of all clients (ignored)

        Returns:
            FitnessResult with constant value
        """
        return FitnessResult(
            value=self.constant_value,
            components={"constant": self.constant_value},
            coalition=coalition,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for constant fitness.

        Returns zero gradient (constant has no preference).

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Zero gradient vector
        """
        return np.zeros(len(clients))

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "value": self.constant_value,
        }


class DeterministicFitness(FitnessFunction):
    """
    Fitness function with predetermined values per coalition.

    Useful for testing specific optimization scenarios.

    Example:
        >>> fitness = DeterministicFitness({
        ...     frozenset([0, 1]): 0.8,
        ...     frozenset([0, 2]): 0.9,
        ...     frozenset([1, 2]): 0.7,
        ... })
        >>> result = fitness.evaluate([0, 2], clients)
        >>> assert result.value == 0.9
    """

    def __init__(
        self,
        values: dict[frozenset[int], float],
        default_value: float = 0.0,
    ):
        """
        Initialize DeterministicFitness.

        Args:
            values: Mapping from coalition (as frozenset) to fitness
            default_value: Fitness for coalitions not in mapping
        """
        self.values = values
        self.default_value = default_value

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Look up predetermined fitness value.

        Args:
            coalition: List of client indices
            clients: List of all clients (ignored)

        Returns:
            FitnessResult with predetermined value
        """
        key = frozenset(coalition)
        fitness = self.values.get(key, self.default_value)

        return FitnessResult(
            value=fitness,
            components={
                "predetermined": fitness,
                "is_known": float(key in self.values),
            },
            coalition=coalition,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute approximate gradient from known values.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Approximate gradient vector
        """
        # For predetermined fitness, gradient approximation is complex
        # Return zero gradient as default
        return np.zeros(len(clients))

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "n_known_coalitions": len(self.values),
            "default_value": self.default_value,
        }


class DataQualityFitness(FitnessFunction):
    """
    Fitness based on client data quality attributes.

    Combines multiple quality metrics (sample size, data quality score)
    into a single fitness value.

    Fitness = w_quality * avg_quality + w_size * avg_normalized_size

    Attributes:
        quality_weight: Weight for data quality score
        size_weight: Weight for sample size (normalized)

    Example:
        >>> fitness = DataQualityFitness(
        ...     quality_weight=0.5,
        ...     size_weight=0.3,
        ... )
    """

    def __init__(
        self,
        quality_weight: float = 0.5,
        size_weight: float = 0.3,
    ):
        """
        Initialize DataQualityFitness.

        Args:
            quality_weight: Weight for data quality score
            size_weight: Weight for normalized sample size
        """
        self.quality_weight = quality_weight
        self.size_weight = size_weight

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate data quality fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with quality-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=0.0,
                components={"quality": 0.0, "size": 0.0},
                coalition=coalition,
            )

        # Collect metrics
        qualities = []
        sizes = []

        for idx in coalition:
            if 0 <= idx < len(clients):
                client = clients[idx]
                qualities.append(client.data_quality)
                sizes.append(client.dataset_size)

        if not qualities:
            return FitnessResult(
                value=0.0,
                components={"quality": 0.0, "size": 0.0},
                coalition=coalition,
            )

        # Normalize sizes (0 to 1 based on max in coalition)
        max_size = max(sizes) if sizes else 1
        normalized_sizes = [s / max_size for s in sizes] if max_size > 0 else sizes

        # Compute components
        avg_quality = float(np.mean(qualities))
        avg_size = float(np.mean(normalized_sizes))

        # Combined fitness
        fitness = self.quality_weight * avg_quality + self.size_weight * avg_size

        return FitnessResult(
            value=fitness,
            components={
                "quality": avg_quality,
                "size": avg_size,
                "quality_contribution": self.quality_weight * avg_quality,
                "size_contribution": self.size_weight * avg_size,
            },
            coalition=coalition,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient based on data quality.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Gradient vector favoring high-quality clients
        """
        n_clients = len(clients)

        # Compute quality score for each client
        gradient = np.zeros(n_clients)
        max_samples = max(c.dataset_size for c in clients) if clients else 1

        for i, client in enumerate(clients):
            normalized_size = (
                client.dataset_size / max_samples if max_samples > 0 else 0
            )

            gradient[i] = (
                self.quality_weight * client.data_quality
                + self.size_weight * normalized_size
            )

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "quality_weight": self.quality_weight,
            "size_weight": self.size_weight,
        }
