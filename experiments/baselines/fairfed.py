"""
FairFed Baseline for FairSwarm Comparison.

Implements the FairFed algorithm (Ezzeldin et al., AAAI 2023) for fair
federated learning via aggregation reweighting.

FairFed adjusts client aggregation weights based on local fairness metrics
(demographic parity gap) so that the global model's per-group performance
converges toward equality. Unlike FairSwarm, which operates at the
*selection* level (choosing which clients participate), FairFed operates
at the *aggregation* level (reweighting client contributions).

For head-to-head comparison, we use FairFed's reweighting to select the
top-k clients by adjusted weight, then evaluate via real FL training.

Reference:
    Ezzeldin et al., "FairFed: Enabling Group Fairness in Federated
    Learning" (AAAI 2023)

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
class FairFedConfig:
    """Configuration for FairFed baseline.

    Attributes:
        coalition_size: Number of clients to select
        n_rounds: Number of reweighting rounds
        beta: Fairness-performance tradeoff parameter (higher = more fair)
        seed: Random seed
    """

    coalition_size: int = 10
    n_rounds: int = 50
    beta: float = 1.0
    seed: Optional[int] = None


@dataclass
class FairFedResult:
    """Result from FairFed baseline.

    Attributes:
        coalition: Final selected coalition
        fitness: Final fitness value
        fairness_divergence: Demographic divergence achieved
        convergence_round: Round at which converged
        metrics: Additional metrics
    """

    coalition: Coalition = field(default_factory=list)
    fitness: float = 0.0
    fairness_divergence: float = 0.0
    convergence_round: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class FairFedBaseline:
    """FairFed: Enabling Group Fairness in Federated Learning.

    Implements the core FairFed mechanism: adjusting client aggregation
    weights based on the gap between each client's local demographic
    distribution and the global target. Clients whose demographics are
    underrepresented get higher aggregation weights.

    For coalition selection comparison: we rank clients by their
    FairFed-adjusted weight and select the top-k.

    Algorithm Overview:
        1. Compute each client's local fairness metric (demographic
           parity gap relative to target)
        2. Compute global fairness metric
        3. Adjust aggregation weights: w_i = w_i * exp(-beta * gap_i)
        4. Normalize weights and select top-k clients

    Comparison Purpose:
        - Most direct comparison for fair FL
        - Tests whether aggregation-level fairness (FairFed) or
          selection-level fairness (FairSwarm) is more effective
    """

    def __init__(
        self,
        clients: List[Client],
        config: Optional[FairFedConfig] = None,
    ):
        self.clients = clients
        self.config = config or FairFedConfig()
        self.n_clients = len(clients)
        self.rng = np.random.default_rng(self.config.seed)

        # Tracking
        self._fitness_history: List[float] = []
        self._fairness_history: List[float] = []
        self._weight_history: List[NDArray[np.float64]] = []

    def run(
        self,
        fitness_fn: FitnessFunction,
        target_distribution: Optional[NDArray[np.float64]] = None,
    ) -> FairFedResult:
        """Run FairFed client selection.

        Args:
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics for fairness

        Returns:
            FairFedResult with selected coalition and metrics
        """
        if target_distribution is None:
            target_distribution = np.ones(4) / 4

        # Initialize uniform aggregation weights
        weights = np.ones(self.n_clients) / self.n_clients

        best_coalition: Coalition = []
        best_fitness = float("-inf")
        convergence_round = None

        for round_num in range(self.config.n_rounds):
            # Step 1: Compute local fairness gaps
            local_gaps = self._compute_local_fairness_gaps(target_distribution)

            # Step 2: Compute global fairness (mean gap)
            global_gap = np.mean(local_gaps)

            # Step 3: Adjust weights using FairFed mechanism
            # Clients with smaller gaps (more fair) get higher weights
            # w_i = w_i * exp(-beta * (local_gap_i - global_gap))
            adjustments = np.exp(
                -self.config.beta * (local_gaps - global_gap)
            )
            weights = weights * adjustments

            # Normalize weights
            weights = weights / (weights.sum() + 1e-10)
            self._weight_history.append(weights.copy())

            # Step 4: Select top-k clients by adjusted weight
            coalition = self._select_coalition(weights)

            # Evaluate
            result = fitness_fn.evaluate(coalition, self.clients)
            fairness = self._compute_fairness(coalition, target_distribution)

            self._fitness_history.append(result.value)
            self._fairness_history.append(fairness)

            if result.value > best_fitness:
                best_fitness = result.value
                best_coalition = coalition

            # Check convergence
            if len(self._fitness_history) > 10:
                recent_var = np.var(self._fitness_history[-10:])
                if recent_var < 1e-6 and convergence_round is None:
                    convergence_round = round_num

        final_fairness = self._fairness_history[-1] if self._fairness_history else 0.0

        return FairFedResult(
            coalition=best_coalition,
            fitness=best_fitness,
            fairness_divergence=final_fairness,
            convergence_round=convergence_round,
            metrics={
                "n_rounds": self.config.n_rounds,
                "coalition_size": self.config.coalition_size,
                "beta": self.config.beta,
                "final_weights": weights.tolist(),
                "fitness_history": self._fitness_history.copy(),
                "fairness_history": self._fairness_history.copy(),
            },
        )

    def _compute_local_fairness_gaps(
        self,
        target_distribution: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute each client's demographic parity gap from target.

        The gap is the L1 distance between the client's demographic
        distribution and the target. Smaller gap = more fair client.

        Args:
            target_distribution: Target demographic distribution

        Returns:
            Array of fairness gaps per client
        """
        gaps = np.zeros(self.n_clients)
        n_groups = len(target_distribution)

        for i, client in enumerate(self.clients):
            demo = client.demographics
            n = min(len(demo), n_groups)

            client_dist = np.zeros(n_groups)
            client_dist[:n] = demo[:n]
            client_dist = client_dist / (client_dist.sum() + 1e-10)

            # L1 distance (total variation distance)
            gaps[i] = 0.5 * np.sum(np.abs(client_dist - target_distribution))

        return gaps

    def _select_coalition(
        self,
        weights: NDArray[np.float64],
    ) -> Coalition:
        """Select top-k clients by FairFed-adjusted weight.

        Args:
            weights: FairFed-adjusted aggregation weights

        Returns:
            Coalition of selected client indices
        """
        k = min(self.config.coalition_size, self.n_clients)
        top_indices = np.argsort(weights)[-k:]
        return top_indices.tolist()

    def _compute_fairness(
        self,
        coalition: Coalition,
        target_distribution: NDArray[np.float64],
    ) -> float:
        """Compute demographic divergence (KL) from target."""
        coalition_clients = [self.clients[i] for i in coalition]
        sizes = np.array([c.dataset_size for c in coalition_clients])
        weights = sizes / (sizes.sum() + 1e-10)

        n_groups = len(target_distribution)
        coalition_demo = np.zeros(n_groups)

        for client, weight in zip(coalition_clients, weights):
            demo = client.demographics
            n = min(len(demo), n_groups)
            coalition_demo[:n] += weight * demo[:n]

        coalition_demo = coalition_demo / (coalition_demo.sum() + 1e-8)

        eps = 1e-10
        p = np.clip(coalition_demo, eps, 1)
        q = np.clip(target_distribution, eps, 1)
        return float(np.sum(p * np.log(p / q)))

    def reset(self) -> None:
        """Reset baseline for new run."""
        self._fitness_history = []
        self._fairness_history = []
        self._weight_history = []
        self.rng = np.random.default_rng(self.config.seed)
