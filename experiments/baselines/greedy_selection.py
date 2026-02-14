"""
Greedy selection baseline for FairSwarm comparison.

Myopic baseline that greedily adds the client with the highest
marginal fitness gain at each step.

Reference: CLAUDE.md Section 4.2 (Baselines)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client


class GreedySelection:
    """
    Baseline: greedy coalition selection.

    At each step, adds the client that maximizes the marginal gain
    in the fitness function. For submodular objectives, this gives
    a (1 - 1/e) approximation (Nemhauser et al., 1978).

    Attributes:
        clients: List of available clients
        coalition_size: Target coalition size
    """

    def __init__(
        self,
        clients: list[Client],
        coalition_size: int,
    ):
        self.clients = clients
        self.n_clients = len(clients)
        self.coalition_size = coalition_size

    def select(
        self,
        fitness_fn: FitnessFunction,
    ) -> tuple[Coalition, FitnessResult]:
        """
        Greedily build a coalition by marginal gain.

        Args:
            fitness_fn: Fitness function for evaluation

        Returns:
            Tuple of (coalition, final_result)
        """
        selected: list[int] = []
        remaining = set(range(self.n_clients))

        for _ in range(self.coalition_size):
            best_candidate = -1
            best_gain = float("-inf")

            for candidate in remaining:
                trial = selected + [candidate]
                result = fitness_fn.evaluate(trial, self.clients)
                gain = result.value

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            if best_candidate >= 0:
                selected.append(best_candidate)
                remaining.discard(best_candidate)

        final_result = fitness_fn.evaluate(selected, self.clients)
        return selected, final_result
