"""
Composite fitness functions for multi-objective optimization.

This module provides utilities for combining multiple fitness
functions into a single objective for PSO optimization.

Mathematical Foundation:
    The composite fitness function combines multiple objectives:

    F(S) = Σ_i w_i · f_i(S)

    Where w_i are weights and f_i are component fitness functions.

    For FairSwarm, the typical formulation is:
    F(S) = w_1·ValAcc(S) - w_2·DemDiv(S) - w_3·CommCost(S)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

__all__ = [
    "WeightedComponent",
    "WeightedFitness",
    "CompositeFitness",
    "CommunicationCostFitness",
]


@dataclass
class WeightedComponent:
    """
    A weighted fitness component.

    Attributes:
        name: Identifier for this component
        fitness: The fitness function
        weight: Weight for this component (positive = reward, negative = penalty)
    """

    name: str
    fitness: FitnessFunction
    weight: float


class WeightedFitness(FitnessFunction):
    """
    Weighted sum of multiple fitness functions.

    Combines multiple fitness functions with specified weights.
    Positive weights reward higher values, negative weights penalize.

    F(S) = Σ_i w_i · f_i(S)

    Attributes:
        components: List of (name, fitness_fn, weight) tuples

    Example:
        >>> accuracy_fitness = AccuracyFitness()
        >>> fairness_fitness = DemographicFitness(target)
        >>> cost_fitness = CommunicationCostFitness()
        >>>
        >>> composite = WeightedFitness([
        ...     ("accuracy", accuracy_fitness, 0.5),
        ...     ("fairness", fairness_fitness, -0.3),  # Penalty
        ...     ("cost", cost_fitness, -0.2),          # Penalty
        ... ])
    """

    def __init__(
        self,
        components: list[tuple[str, FitnessFunction, float]],
    ):
        """
        Initialize WeightedFitness.

        Args:
            components: List of (name, fitness_function, weight) tuples
        """
        self.components = [
            WeightedComponent(name=name, fitness=fn, weight=weight)
            for name, fn, weight in components
        ]

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate weighted sum of component fitness values.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with combined fitness and component breakdown
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={c.name: float("-inf") for c in self.components},
                coalition=coalition,
            )

        total_fitness = 0.0
        component_values: dict[str, float] = {}
        all_metadata: dict[str, Any] = {}

        for comp in self.components:
            result = comp.fitness.evaluate(coalition, clients)
            raw_value = result.value

            # Apply weight
            weighted_value = comp.weight * raw_value
            total_fitness += weighted_value

            # Store both raw and weighted values
            component_values[comp.name] = raw_value
            component_values[f"{comp.name}_weighted"] = weighted_value

            # Merge component details
            for key, value in result.components.items():
                component_values[f"{comp.name}_{key}"] = value

            # Collect metadata
            if result.metadata:
                all_metadata[comp.name] = result.metadata

        return FitnessResult(
            value=total_fitness,
            components=component_values,
            coalition=coalition,
            metadata=all_metadata,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute weighted sum of component gradients.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Combined gradient vector
        """
        n_clients = len(clients)
        total_gradient = np.zeros(n_clients)

        for comp in self.components:
            comp_gradient = comp.fitness.compute_gradient(
                position, clients, coalition_size
            )
            total_gradient += comp.weight * comp_gradient

        # Normalize if non-zero
        norm = np.linalg.norm(total_gradient)
        if norm > 1e-10:
            total_gradient = total_gradient / norm

        return total_gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "components": [
                {
                    "name": comp.name,
                    "weight": comp.weight,
                    "config": comp.fitness.get_config(),
                }
                for comp in self.components
            ],
        }


class CompositeFitness(FitnessFunction):
    """
    Flexible composite fitness with customizable aggregation.

    Supports various aggregation strategies beyond weighted sum:
    - "weighted_sum": Standard weighted combination
    - "min": Conservative (Rawlsian) approach
    - "product": Multiplicative combination
    - "lexicographic": Priority-based ordering

    Attributes:
        components: List of weighted fitness components
        aggregation: Aggregation strategy

    Example:
        >>> composite = CompositeFitness(
        ...     components=[
        ...         ("accuracy", accuracy_fn, 1.0),
        ...         ("fairness", fairness_fn, 1.0),
        ...     ],
        ...     aggregation="min",  # Optimize worst-case objective
        ... )
    """

    AGGREGATION_METHODS = ["weighted_sum", "min", "product", "lexicographic"]

    def __init__(
        self,
        components: list[tuple[str, FitnessFunction, float]],
        aggregation: str = "weighted_sum",
    ):
        """
        Initialize CompositeFitness.

        Args:
            components: List of (name, fitness_function, weight) tuples
            aggregation: Aggregation method (default: weighted_sum)

        Raises:
            ValueError: If aggregation method is not recognized
        """
        if aggregation not in self.AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown aggregation method: {aggregation}. "
                f"Choose from: {self.AGGREGATION_METHODS}"
            )

        self.components = [
            WeightedComponent(name=name, fitness=fn, weight=weight)
            for name, fn, weight in components
        ]
        self.aggregation = aggregation

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate using specified aggregation method.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with aggregated fitness
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={c.name: float("-inf") for c in self.components},
                coalition=coalition,
            )

        # Evaluate all components
        component_results: dict[str, FitnessResult] = {}
        weighted_values: list[float] = []

        for comp in self.components:
            result = comp.fitness.evaluate(coalition, clients)
            component_results[comp.name] = result
            weighted_values.append(comp.weight * result.value)

        # Aggregate based on method
        if self.aggregation == "weighted_sum":
            total_fitness = sum(weighted_values)
        elif self.aggregation == "min":
            total_fitness = min(weighted_values)
        elif self.aggregation == "product":
            # Shift values to be positive for product
            shifted = [max(v + 1, 0.001) for v in weighted_values]
            total_fitness = float(np.prod(shifted)) - 1
        elif self.aggregation == "lexicographic":
            # Use large multipliers to enforce priority
            total_fitness = 0.0
            multiplier = 1.0
            for v in weighted_values:
                total_fitness += multiplier * v
                multiplier *= 0.001  # Each subsequent objective has less weight
        else:
            total_fitness = sum(weighted_values)

        # Build component breakdown
        component_values: dict[str, float] = {}
        all_metadata: dict[str, Any] = {"aggregation": self.aggregation}

        for comp in self.components:
            result = component_results[comp.name]
            component_values[comp.name] = result.value
            component_values[f"{comp.name}_weighted"] = comp.weight * result.value

        return FitnessResult(
            value=total_fitness,
            components=component_values,
            coalition=coalition,
            metadata=all_metadata,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for the composite fitness.

        For weighted_sum, this is the weighted sum of gradients.
        For other methods, uses weighted_sum as approximation.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Combined gradient vector
        """
        n_clients = len(clients)
        total_gradient = np.zeros(n_clients)

        for comp in self.components:
            comp_gradient = comp.fitness.compute_gradient(
                position, clients, coalition_size
            )
            total_gradient += comp.weight * comp_gradient

        # Normalize
        norm = np.linalg.norm(total_gradient)
        if norm > 1e-10:
            total_gradient = total_gradient / norm

        return total_gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "aggregation": self.aggregation,
            "components": [
                {
                    "name": comp.name,
                    "weight": comp.weight,
                    "config": comp.fitness.get_config(),
                }
                for comp in self.components
            ],
        }


class CommunicationCostFitness(FitnessFunction):
    """
    Fitness based on communication/computation cost.

    Evaluates coalitions based on their total communication cost,
    which is important for efficient federated learning.

    Fitness = -CommCost(S) = -Σ_{i∈S} cost_i

    Lower cost leads to higher fitness (less negative).

    Attributes:
        normalize: Whether to normalize by coalition size

    Example:
        >>> cost_fitness = CommunicationCostFitness(normalize=True)
        >>> result = cost_fitness.evaluate(coalition, clients)
        >>> print(f"Average cost per client: {-result.value:.4f}")
    """

    def __init__(self, normalize: bool = False):
        """
        Initialize CommunicationCostFitness.

        Args:
            normalize: If True, normalize cost by coalition size
        """
        self.normalize = normalize

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate communication cost fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with cost-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=0.0,
                components={"cost": 0.0},
                coalition=coalition,
            )

        # Sum communication costs
        total_cost = 0.0
        for idx in coalition:
            if 0 <= idx < len(clients):
                total_cost += clients[idx].communication_cost

        # Optionally normalize by coalition size
        if self.normalize and coalition:
            total_cost /= len(coalition)

        # Fitness is negative cost (lower cost = higher fitness)
        fitness = -total_cost

        return FitnessResult(
            value=fitness,
            components={
                "cost": total_cost,
                "cost_penalty": -total_cost,
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
        Compute gradient for cost reduction.

        Gradient points toward lower-cost clients.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Gradient vector (negative for high-cost clients)
        """
        len(clients)
        costs = np.array([c.communication_cost for c in clients])

        # Gradient: negative for high-cost clients
        # (we want to reduce selection of expensive clients)
        gradient = -costs

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "normalize": self.normalize,
        }
