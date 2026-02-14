"""
Abstract base class for FairSwarm fitness functions.

This module defines the interface for fitness evaluation in the
FairSwarm PSO algorithm.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.types import Coalition, FitnessValue

if TYPE_CHECKING:
    from fairswarm.core.client import Client


@dataclass(frozen=True)
class FitnessResult:
    """
    Result of fitness evaluation with component breakdown.

    Provides transparency into fitness computation by exposing
    individual components (accuracy, fairness, cost, etc.).

    Attributes:
        value: Total fitness value (higher is better)
        components: Dictionary of component contributions
        coalition: The evaluated coalition
        metadata: Additional evaluation information

    Example:
        >>> result = fitness_fn.evaluate(coalition, clients)
        >>> print(f"Total: {result.value:.4f}")
        >>> print(f"Accuracy contribution: {result.components['accuracy']:.4f}")
        >>> print(f"Fairness penalty: {result.components['fairness']:.4f}")
    """

    value: FitnessValue
    components: dict[str, float] = field(default_factory=dict)
    coalition: Coalition = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"FitnessResult(value={self.value:.4f}, "
            f"components={list(self.components.keys())})"
        )


class FitnessFunction(ABC):
    """
    Abstract base class for fitness functions.

    Fitness functions evaluate the quality of a coalition for the
    FairSwarm PSO optimization. Higher fitness values indicate better
    coalitions.

    Subclasses must implement:
        - evaluate(): Compute fitness for a coalition
        - compute_gradient(): Compute gradient for fairness-aware update

    Algorithm Reference:
        Algorithm 1 uses fitness to update personal and global bests:
            fit_p ← Fitness(S_p)
            If fit_p > pBestFit_p:
                pBest_p ← x_p
                pBestFit_p ← fit_p

    Example:
        >>> class MyFitness(FitnessFunction):
        ...     def evaluate(self, coalition, clients):
        ...         accuracy = compute_accuracy(coalition, clients)
        ...         return FitnessResult(value=accuracy, components={'accuracy': accuracy})
        ...
        ...     def compute_gradient(self, position, clients, coalition_size):
        ...         return np.zeros(len(clients))
    """

    @abstractmethod
    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate the fitness of a coalition.

        Args:
            coalition: List of client indices in the coalition
            clients: List of all available clients

        Returns:
            FitnessResult with value and component breakdown

        Note:
            Implementation should be deterministic for reproducibility.
            Stochastic evaluation can be achieved through explicit randomness.
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for fairness-aware velocity update.

        The gradient guides particles toward fairer regions of the
        search space. This is the novel contribution of FairSwarm.

        Args:
            position: Current particle position (selection probabilities)
            clients: List of all available clients
            coalition_size: Target coalition size m

        Returns:
            Gradient vector of same dimension as position

        Algorithm Reference:
            v_fairness ← c₃ · ∇_fair

        Mathematical Foundation:
            The gradient points toward positions that reduce demographic
            divergence while maintaining coalition quality.
        """
        pass

    def evaluate_batch(
        self,
        coalitions: list[Coalition],
        clients: list[Client],
    ) -> list[FitnessResult]:
        """
        Evaluate multiple coalitions.

        Default implementation calls evaluate() for each coalition.
        Override for vectorized implementations.

        Args:
            coalitions: List of coalitions to evaluate
            clients: List of all available clients

        Returns:
            List of FitnessResult for each coalition
        """
        return [self.evaluate(c, clients) for c in coalitions]

    def is_feasible(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> bool:
        """
        Check if a coalition satisfies constraints.

        Default implementation returns True. Override to add
        constraint checking (e.g., minimum coverage, budget limits).

        Args:
            coalition: Coalition to check
            clients: List of all available clients

        Returns:
            True if coalition is feasible, False otherwise
        """
        return True

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration for reproducibility.

        Returns:
            Dictionary of configuration parameters
        """
        return {"class": self.__class__.__name__}
