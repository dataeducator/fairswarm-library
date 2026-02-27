"""
Greedy Selection Baseline for FairSwarm Comparison.

This module implements greedy coalition selection algorithms for
baseline comparison in experiments.

Greedy Strategies:
    1. Fitness Greedy: Select clients that maximize marginal fitness
    2. Fairness Greedy: Select clients that minimize demographic divergence
    3. Quality Greedy: Select clients with highest data quality
    4. Balanced Greedy: Alternates between fitness and fairness

Theoretical Background:
    For submodular objectives, greedy achieves (1-1/e) approximation.
    This makes it a strong baseline for comparing FairSwarm.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.base import FitnessFunction
from fairswarm.fitness.fairness import compute_coalition_demographics
from fairswarm.types import Coalition


class GreedyCriterion(Enum):
    """Greedy selection criterion."""

    FITNESS = "fitness"
    FAIRNESS = "fairness"
    QUALITY = "quality"
    BALANCED = "balanced"
    COVERAGE = "coverage"


@dataclass
class GreedyConfig:
    """Configuration for greedy baseline."""

    criterion: GreedyCriterion = GreedyCriterion.FITNESS
    coalition_size: int = 10
    fairness_weight: float = 0.5  # For balanced mode
    seed: Optional[int] = None


@dataclass
class GreedyResult:
    """Result from greedy selection."""

    coalition: Coalition
    fitness: float
    demographic_divergence: float
    selection_order: List[int]
    marginal_gains: List[float]
    iterations: int
    criterion: str


class GreedyBaseline:
    """
    Greedy coalition selection baseline.

    Implements various greedy strategies for comparison with FairSwarm.
    The greedy algorithm iteratively selects the client with the
    highest marginal gain until the coalition is full.

    Algorithm:
        S = {}
        for i = 1 to k:
            c* = argmax_{c not in S} MarginalGain(S, c)
            S = S ∪ {c*}
        return S

    Properties:
        - For submodular functions, achieves (1-1/e) approximation
        - Time complexity: O(n * k) fitness evaluations
        - Deterministic (no randomness)

    Example:
        >>> from experiments.baselines.greedy import GreedyBaseline, GreedyConfig
        >>> from fairswarm.fitness.fairness import DemographicFitness
        >>>
        >>> config = GreedyConfig(
        ...     criterion=GreedyCriterion.FITNESS,
        ...     coalition_size=10,
        ... )
        >>> baseline = GreedyBaseline(
        ...     clients=clients,
        ...     fitness_fn=fitness_fn,
        ...     target_distribution=target,
        ...     config=config,
        ... )
        >>> result = baseline.run()
        >>> print(f"Coalition: {result.coalition}")
    """

    def __init__(
        self,
        clients: List[Client],
        fitness_fn: Optional[FitnessFunction] = None,
        target_distribution: Optional[DemographicDistribution] = None,
        config: Optional[GreedyConfig] = None,
    ):
        """
        Initialize GreedyBaseline.

        Args:
            clients: List of FL clients
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics
            config: Greedy configuration
        """
        self.clients = clients
        self.fitness_fn = fitness_fn
        self.target_distribution = target_distribution
        self.config = config or GreedyConfig()

        # Random state (for tie-breaking)
        self.rng = np.random.default_rng(self.config.seed)

    def run(self) -> GreedyResult:
        """
        Run greedy selection.

        Returns:
            GreedyResult with selected coalition
        """
        if self.config.criterion == GreedyCriterion.FITNESS:
            return self._greedy_fitness()
        elif self.config.criterion == GreedyCriterion.FAIRNESS:
            return self._greedy_fairness()
        elif self.config.criterion == GreedyCriterion.QUALITY:
            return self._greedy_quality()
        elif self.config.criterion == GreedyCriterion.BALANCED:
            return self._greedy_balanced()
        elif self.config.criterion == GreedyCriterion.COVERAGE:
            return self._greedy_coverage()
        else:
            raise ValueError(f"Unknown criterion: {self.config.criterion}")

    def _greedy_fitness(self) -> GreedyResult:
        """
        Greedy selection maximizing fitness.

        Selects client with highest marginal fitness gain at each step.
        """
        selected: List[int] = []
        remaining = set(range(len(self.clients)))
        marginal_gains: List[float] = []

        for _ in range(min(self.config.coalition_size, len(self.clients))):
            best_gain = float("-inf")
            best_client = None

            current_fitness = 0.0
            if selected and self.fitness_fn:
                result = self.fitness_fn.evaluate(selected, self.clients)
                current_fitness = result.value

            for client_idx in remaining:
                # Compute marginal gain
                candidate = selected + [client_idx]

                if self.fitness_fn:
                    result = self.fitness_fn.evaluate(candidate, self.clients)
                    gain = result.value - current_fitness
                else:
                    # Default: use data quality
                    gain = self.clients[client_idx].data_quality

                if gain > best_gain or (gain == best_gain and self.rng.random() > 0.5):
                    best_gain = gain
                    best_client = client_idx

            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
                marginal_gains.append(best_gain)

        return self._create_result(selected, marginal_gains)

    def _greedy_fairness(self) -> GreedyResult:
        """
        Greedy selection minimizing demographic divergence.

        At each step, selects client that brings coalition closest
        to target distribution.
        """
        if self.target_distribution is None:
            raise ValueError("Target distribution required for fairness greedy")

        selected: List[int] = []
        remaining = set(range(len(self.clients)))
        marginal_gains: List[float] = []
        target = self.target_distribution.as_array()

        for _ in range(min(self.config.coalition_size, len(self.clients))):
            best_divergence = float("inf")
            best_client = None

            for client_idx in remaining:
                candidate = selected + [client_idx]

                # Compute coalition demographics
                coalition_demo = compute_coalition_demographics(candidate, self.clients)
                divergence = kl_divergence(coalition_demo, target)

                if divergence < best_divergence:
                    best_divergence = divergence
                    best_client = client_idx

            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
                # Gain is negative divergence (reduction is good)
                marginal_gains.append(-best_divergence)

        return self._create_result(selected, marginal_gains)

    def _greedy_quality(self) -> GreedyResult:
        """
        Greedy selection by data quality.

        Simply selects clients with highest data quality.
        """
        # Sort by data quality
        quality_order = sorted(
            range(len(self.clients)),
            key=lambda i: self.clients[i].data_quality,
            reverse=True,
        )

        selected = quality_order[: self.config.coalition_size]
        marginal_gains = [self.clients[i].data_quality for i in selected]

        return self._create_result(selected, marginal_gains)

    def _greedy_balanced(self) -> GreedyResult:
        """
        Balanced greedy alternating between fitness and fairness.

        Alternates between selecting for fitness and fairness.
        """
        if self.target_distribution is None:
            raise ValueError("Target distribution required for balanced greedy")

        selected: List[int] = []
        remaining = set(range(len(self.clients)))
        marginal_gains: List[float] = []
        target = self.target_distribution.as_array()

        for step in range(min(self.config.coalition_size, len(self.clients))):
            use_fitness = step % 2 == 0

            best_score = float("-inf") if use_fitness else float("inf")
            best_client = None

            for client_idx in remaining:
                candidate = selected + [client_idx]

                if use_fitness and self.fitness_fn:
                    result = self.fitness_fn.evaluate(candidate, self.clients)
                    score = result.value
                    is_better = score > best_score
                else:
                    # Fairness: minimize divergence
                    coalition_demo = compute_coalition_demographics(
                        candidate, self.clients
                    )
                    score = kl_divergence(coalition_demo, target)
                    is_better = score < best_score

                if is_better:
                    best_score = score
                    best_client = client_idx

            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
                marginal_gains.append(best_score if use_fitness else -best_score)

        return self._create_result(selected, marginal_gains)

    def _greedy_coverage(self) -> GreedyResult:
        """
        Greedy selection maximizing demographic coverage.

        Selects clients to maximize diversity of demographic groups covered.
        """
        selected: List[int] = []
        remaining = set(range(len(self.clients)))
        marginal_gains: List[float] = []

        # Track covered demographic groups
        covered_weight: NDArray[np.float64] = np.zeros(4)  # Assuming 4 groups

        for _ in range(min(self.config.coalition_size, len(self.clients))):
            best_coverage_gain = float("-inf")
            best_client = None

            for client_idx in remaining:
                client_demo = self.clients[client_idx].demographics.as_array()

                # Compute coverage gain (emphasize underrepresented groups)
                gain = 0.0
                for i, weight in enumerate(client_demo):
                    if i < len(covered_weight):
                        # More gain for underrepresented groups
                        current = (
                            covered_weight[i] / (len(selected) + 1) if selected else 0
                        )
                        gain += weight * (1.0 - current)

                if gain > best_coverage_gain:
                    best_coverage_gain = gain
                    best_client = client_idx

            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
                marginal_gains.append(best_coverage_gain)

                # Update coverage
                client_demo = self.clients[best_client].demographics.as_array()
                for i, weight in enumerate(client_demo):
                    if i < len(covered_weight):
                        covered_weight[i] += weight

        return self._create_result(selected, marginal_gains)

    def _create_result(
        self,
        selected: List[int],
        marginal_gains: List[float],
    ) -> GreedyResult:
        """Create GreedyResult from selection."""
        # Compute final fitness
        fitness = 0.0
        if self.fitness_fn and selected:
            result = self.fitness_fn.evaluate(selected, self.clients)
            fitness = result.value

        # Compute final divergence
        divergence = float("inf")
        if self.target_distribution and selected:
            coalition_demo = compute_coalition_demographics(selected, self.clients)
            target = self.target_distribution.as_array()
            divergence = kl_divergence(coalition_demo, target)

        return GreedyResult(
            coalition=selected,
            fitness=fitness,
            demographic_divergence=divergence,
            selection_order=selected.copy(),
            marginal_gains=marginal_gains,
            iterations=len(selected),
            criterion=self.config.criterion.value,
        )


def run_greedy_baseline(
    clients: List[Client],
    coalition_size: int,
    fitness_fn: Optional[FitnessFunction] = None,
    target_distribution: Optional[DemographicDistribution] = None,
    criterion: str = "fitness",
    seed: Optional[int] = None,
) -> GreedyResult:
    """
    Convenience function to run greedy baseline.

    Args:
        clients: List of clients
        coalition_size: Target coalition size
        fitness_fn: Fitness function
        target_distribution: Target demographics
        criterion: Greedy criterion ("fitness", "fairness", "quality", "balanced")
        seed: Random seed

    Returns:
        GreedyResult
    """
    config = GreedyConfig(
        criterion=GreedyCriterion(criterion),
        coalition_size=coalition_size,
        seed=seed,
    )

    baseline = GreedyBaseline(
        clients=clients,
        fitness_fn=fitness_fn,
        target_distribution=target_distribution,
        config=config,
    )

    return baseline.run()
