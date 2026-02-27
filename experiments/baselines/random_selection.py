"""
Random Selection Baseline for FairSwarm Comparison.

Uniform random coalition selection without fairness consideration.
Provides lower bound for selection algorithm performance.

Author: Tenicka Norwood
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from fairswarm import Client
from fairswarm.fitness.base import FitnessFunction
from fairswarm.types import Coalition


@dataclass
class RandomSelectionConfig:
    """
    Configuration for random selection baseline.

    Attributes:
        coalition_size: Number of clients to select
        n_iterations: Number of random selections to try
        seed: Random seed
    """

    coalition_size: int = 10
    n_iterations: int = 100
    seed: Optional[int] = None


@dataclass
class RandomSelectionResult:
    """
    Result from random selection baseline.

    Attributes:
        coalition: Best coalition found
        fitness: Best fitness achieved
        fairness_divergence: Demographic divergence of best coalition
        avg_fitness: Average fitness across all tries
        avg_fairness: Average fairness across all tries
        metrics: Additional metrics
    """

    coalition: Coalition = field(default_factory=list)
    fitness: float = 0.0
    fairness_divergence: float = 0.0
    avg_fitness: float = 0.0
    avg_fairness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class RandomSelectionBaseline:
    """
    Random coalition selection baseline.

    Selects coalitions uniformly at random and keeps the best one.
    No optimization or fairness awareness.

    Usage:
        >>> random_sel = RandomSelectionBaseline(
        ...     clients, config=RandomSelectionConfig(coalition_size=10)
        ... )
        >>> result = random_sel.run(fitness_fn)

    Comparison Purpose:
        - Lower bound for optimization-based selection
        - Expected performance without intelligent selection
        - Baseline fairness (what random chance achieves)
    """

    def __init__(
        self,
        clients: List[Client],
        config: Optional[RandomSelectionConfig] = None,
    ):
        """
        Initialize random selection baseline.

        Args:
            clients: List of federated learning clients
            config: Configuration
        """
        self.clients = clients
        self.config = config or RandomSelectionConfig()
        self.n_clients = len(clients)

        self.rng = np.random.default_rng(self.config.seed)

    def run(
        self,
        fitness_fn: FitnessFunction,
        target_distribution: Optional[NDArray[np.float64]] = None,
    ) -> RandomSelectionResult:
        """
        Run random selection baseline.

        Args:
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics for fairness measurement

        Returns:
            RandomSelectionResult with best coalition and metrics
        """
        best_coalition: Coalition = []
        best_fitness = float("-inf")
        best_fairness = float("inf")

        all_fitness: List[float] = []
        all_fairness: List[float] = []

        for _ in range(self.config.n_iterations):
            # Random selection
            coalition = self.rng.choice(
                self.n_clients,
                size=min(self.config.coalition_size, self.n_clients),
                replace=False,
            ).tolist()

            # Evaluate
            result = fitness_fn.evaluate(coalition, self.clients)
            all_fitness.append(result.value)

            # Compute fairness
            if target_distribution is not None:
                fairness = self._compute_fairness(coalition, target_distribution)
            else:
                fairness = 0.0
            all_fairness.append(fairness)

            # Track best
            if result.value > best_fitness:
                best_fitness = result.value
                best_coalition = coalition
                best_fairness = fairness

        return RandomSelectionResult(
            coalition=best_coalition,
            fitness=best_fitness,
            fairness_divergence=best_fairness,
            avg_fitness=float(np.mean(all_fitness)),
            avg_fairness=float(np.mean(all_fairness)) if all_fairness else 0.0,
            metrics={
                "n_iterations": self.config.n_iterations,
                "coalition_size": self.config.coalition_size,
                "fitness_std": float(np.std(all_fitness)),
                "fairness_std": float(np.std(all_fairness)) if all_fairness else 0.0,
                "fitness_range": (float(min(all_fitness)), float(max(all_fitness))),
            },
        )

    def _compute_fairness(
        self,
        coalition: Coalition,
        target_distribution: NDArray[np.float64],
    ) -> float:
        """Compute demographic divergence from target."""
        coalition_clients = [self.clients[i] for i in coalition]
        sizes = np.array([c.dataset_size for c in coalition_clients])
        weights = sizes / sizes.sum()

        n_groups = len(target_distribution)
        coalition_demo = np.zeros(n_groups)

        for client, weight in zip(coalition_clients, weights):
            demo = client.demographics
            n = min(len(demo), n_groups)
            coalition_demo[:n] += weight * demo[:n]

        coalition_demo = coalition_demo / (coalition_demo.sum() + 1e-8)

        # KL divergence
        eps = 1e-10
        p = np.clip(coalition_demo, eps, 1)
        q = np.clip(target_distribution, eps, 1)

        return float(np.sum(p * np.log(p / q)))
