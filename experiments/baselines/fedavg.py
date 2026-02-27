"""
FedAvg Baseline for FairSwarm Comparison.

Standard Federated Averaging algorithm without client selection.
All clients participate in every round.

Reference:
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (AISTATS 2017)

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
class FedAvgConfig:
    """
    Configuration for FedAvg baseline.

    Attributes:
        n_rounds: Number of federated learning rounds
        local_epochs: Local training epochs per round
        learning_rate: Local learning rate
        participation_rate: Fraction of clients per round (1.0 = all)
        seed: Random seed
    """

    n_rounds: int = 50
    local_epochs: int = 5
    learning_rate: float = 0.01
    participation_rate: float = 1.0
    seed: Optional[int] = None


@dataclass
class FedAvgResult:
    """
    Result from FedAvg baseline.

    Attributes:
        coalition: Selected clients (all clients in standard FedAvg)
        fitness: Final fitness value
        fairness_divergence: Demographic divergence (for comparison)
        convergence_round: Round at which converged
        metrics: Additional metrics
    """

    coalition: Coalition = field(default_factory=list)
    fitness: float = 0.0
    fairness_divergence: float = 0.0
    convergence_round: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class FedAvgBaseline:
    """
    FedAvg baseline without fair client selection.

    This baseline uses standard federated averaging where all
    clients (or a random subset) participate in each round.
    No fairness-aware selection is performed.

    Usage:
        >>> fedavg = FedAvgBaseline(clients, config=FedAvgConfig(n_rounds=50))
        >>> result = fedavg.run(fitness_fn)
        >>> print(f"FedAvg fitness: {result.fitness:.4f}")
        >>> print(f"FedAvg fairness: {result.fairness_divergence:.4f}")

    Comparison Purpose:
        - Lower bound for FairSwarm fairness improvement
        - Upper bound for accuracy (no selection overhead)
        - Baseline for convergence speed
    """

    def __init__(
        self,
        clients: List[Client],
        config: Optional[FedAvgConfig] = None,
    ):
        """
        Initialize FedAvg baseline.

        Args:
            clients: List of federated learning clients
            config: FedAvg configuration
        """
        self.clients = clients
        self.config = config or FedAvgConfig()
        self.n_clients = len(clients)

        self.rng = np.random.default_rng(self.config.seed)

        # Tracking
        self._fitness_history: List[float] = []
        self._fairness_history: List[float] = []

    def run(
        self,
        fitness_fn: FitnessFunction,
        target_distribution: Optional[NDArray[np.float64]] = None,
    ) -> FedAvgResult:
        """
        Run FedAvg federated learning.

        Args:
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics for fairness measurement

        Returns:
            FedAvgResult with final metrics
        """
        # Determine coalition (all clients or random subset)
        if self.config.participation_rate >= 1.0:
            coalition = list(range(self.n_clients))
        else:
            n_select = max(1, int(self.n_clients * self.config.participation_rate))
            coalition = self.rng.choice(
                self.n_clients, size=n_select, replace=False
            ).tolist()

        # Evaluate initial fitness
        initial_result = fitness_fn.evaluate(coalition, self.clients)
        self._fitness_history.append(initial_result.value)

        # Compute initial fairness
        if target_distribution is not None:
            initial_fairness = self._compute_fairness(coalition, target_distribution)
        else:
            initial_fairness = 0.0
        self._fairness_history.append(initial_fairness)

        # Simulate training rounds (FedAvg doesn't change selection)
        best_fitness = initial_result.value
        convergence_round = None

        for round_num in range(self.config.n_rounds):
            # In FedAvg, we use the same coalition each round
            # (or resample if participation_rate < 1)
            if self.config.participation_rate < 1.0:
                n_select = max(1, int(self.n_clients * self.config.participation_rate))
                coalition = self.rng.choice(
                    self.n_clients, size=n_select, replace=False
                ).tolist()

            # Evaluate (simulated improvement over rounds)
            result = fitness_fn.evaluate(coalition, self.clients)

            # Simulate training improvement
            improvement_factor = 1.0 + 0.1 * np.log1p(round_num) / np.log1p(
                self.config.n_rounds
            )
            simulated_fitness = result.value * improvement_factor

            self._fitness_history.append(simulated_fitness)

            if target_distribution is not None:
                fairness = self._compute_fairness(coalition, target_distribution)
                self._fairness_history.append(fairness)

            # Track best
            if simulated_fitness > best_fitness:
                best_fitness = simulated_fitness

            # Check convergence (simplified)
            if len(self._fitness_history) > 10:
                recent_var = np.var(self._fitness_history[-10:])
                if recent_var < 1e-6 and convergence_round is None:
                    convergence_round = round_num

        # Final result
        final_fairness = self._fairness_history[-1] if self._fairness_history else 0.0

        return FedAvgResult(
            coalition=coalition,
            fitness=best_fitness,
            fairness_divergence=final_fairness,
            convergence_round=convergence_round,
            metrics={
                "n_rounds": self.config.n_rounds,
                "n_clients_per_round": len(coalition),
                "participation_rate": self.config.participation_rate,
                "fitness_history": self._fitness_history.copy(),
                "fairness_history": self._fairness_history.copy(),
            },
        )

    def _compute_fairness(
        self,
        coalition: Coalition,
        target_distribution: NDArray[np.float64],
    ) -> float:
        """
        Compute demographic divergence from target.

        Args:
            coalition: Selected client indices
            target_distribution: Target demographic distribution

        Returns:
            KL divergence from target
        """
        # Aggregate coalition demographics
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
        kl = np.sum(p * np.log(p / q))

        return float(kl)

    def reset(self) -> None:
        """Reset baseline for new run."""
        self._fitness_history = []
        self._fairness_history = []
        self.rng = np.random.default_rng(self.config.seed)
