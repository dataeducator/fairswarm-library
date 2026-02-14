"""
Random selection baseline for FairSwarm comparison.

This is the lower bound baseline: selects m clients uniformly at random.
Expected demographic divergence is O(1/sqrt(m)) by concentration inequalities.

Reference: CLAUDE.md Section 4.2 (Baselines)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client


class RandomSelection:
    """
    Baseline: uniform random coalition selection.

    Selects m clients uniformly at random without any optimization.
    Serves as the lower-bound baseline in experiments.

    Attributes:
        n_clients: Total number of available clients
        coalition_size: Number of clients to select
    """

    def __init__(
        self,
        clients: list[Client],
        coalition_size: int,
        seed: int | None = None,
    ):
        self.clients = clients
        self.n_clients = len(clients)
        self.coalition_size = coalition_size
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        fitness_fn: FitnessFunction,
        n_trials: int = 100,
    ) -> tuple[Coalition, FitnessResult]:
        """
        Select the best random coalition over n_trials.

        Args:
            fitness_fn: Fitness function for evaluation
            n_trials: Number of random coalitions to try

        Returns:
            Tuple of (best_coalition, best_result)
        """
        best_coalition: Coalition = []
        best_result: FitnessResult | None = None

        for _ in range(n_trials):
            indices = self.rng.choice(
                self.n_clients, size=self.coalition_size, replace=False
            ).tolist()
            result = fitness_fn.evaluate(indices, self.clients)

            if best_result is None or result.value > best_result.value:
                best_coalition = indices
                best_result = result

        assert best_result is not None
        return best_coalition, best_result
