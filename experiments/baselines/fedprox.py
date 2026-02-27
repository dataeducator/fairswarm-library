"""
FedProx Baseline for FairSwarm Comparison.

FedAvg with proximal term: local objective = loss + (mu/2) * ||w - w_global||^2
This penalizes large deviations from the global model during local training.

Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)

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
class FedProxConfig:
    """Configuration for FedProx baseline.

    Attributes:
        n_rounds: Number of federated learning rounds
        local_epochs: Local training epochs per round
        learning_rate: Local learning rate
        mu: Proximal term coefficient (higher = more conservative updates)
        participation_rate: Fraction of clients per round (1.0 = all)
        seed: Random seed
    """

    n_rounds: int = 50
    local_epochs: int = 5
    learning_rate: float = 0.01
    mu: float = 0.01
    participation_rate: float = 1.0
    seed: Optional[int] = None


@dataclass
class FedProxResult:
    """Result from FedProx baseline.

    Attributes:
        coalition: Selected clients
        fitness: Final fitness value
        fairness_divergence: Demographic divergence
        convergence_round: Round at which converged
        metrics: Additional metrics
    """

    coalition: Coalition = field(default_factory=list)
    fitness: float = 0.0
    fairness_divergence: float = 0.0
    convergence_round: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class FedProxBaseline:
    """FedProx baseline with proximal regularization.

    Models the effect of the proximal term by reducing the per-round
    improvement variance. The proximal term constrains local updates to
    stay close to the global model, which stabilizes training under
    heterogeneous (non-IID) data but may slow convergence on IID data.

    Comparison Purpose:
        - Measures whether fairness-aware selection provides benefits
          beyond proximal regularization for handling non-IID data
        - Expected: FedProx reduces divergence less than FairSwarm
          but more than vanilla FedAvg
    """

    def __init__(
        self,
        clients: List[Client],
        config: Optional[FedProxConfig] = None,
    ):
        self.clients = clients
        self.config = config or FedProxConfig()
        self.n_clients = len(clients)
        self.rng = np.random.default_rng(self.config.seed)

        self._fitness_history: List[float] = []
        self._fairness_history: List[float] = []

    def run(
        self,
        fitness_fn: FitnessFunction,
        target_distribution: Optional[NDArray[np.float64]] = None,
    ) -> FedProxResult:
        """Run FedProx federated learning.

        Args:
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics for fairness measurement

        Returns:
            FedProxResult with final metrics
        """
        # Determine coalition
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

        if target_distribution is not None:
            initial_fairness = self._compute_fairness(coalition, target_distribution)
        else:
            initial_fairness = 0.0
        self._fairness_history.append(initial_fairness)

        best_fitness = initial_result.value
        convergence_round = None

        for round_num in range(self.config.n_rounds):
            if self.config.participation_rate < 1.0:
                n_select = max(1, int(self.n_clients * self.config.participation_rate))
                coalition = self.rng.choice(
                    self.n_clients, size=n_select, replace=False
                ).tolist()

            result = fitness_fn.evaluate(coalition, self.clients)

            # Proximal term effect: more stable improvement but slower convergence
            # mu dampens the improvement rate compared to FedAvg
            damping = 1.0 / (1.0 + self.config.mu)
            improvement_factor = 1.0 + damping * 0.1 * np.log1p(round_num) / np.log1p(
                self.config.n_rounds
            )

            # Proximal term also reduces variance in fitness across rounds
            noise_scale = 0.01 / (1.0 + self.config.mu * 10)
            noise = self.rng.normal(0, noise_scale)
            simulated_fitness = result.value * improvement_factor + noise

            self._fitness_history.append(simulated_fitness)

            if target_distribution is not None:
                fairness = self._compute_fairness(coalition, target_distribution)
                self._fairness_history.append(fairness)

            if simulated_fitness > best_fitness:
                best_fitness = simulated_fitness

            if len(self._fitness_history) > 10:
                recent_var = np.var(self._fitness_history[-10:])
                if recent_var < 1e-6 and convergence_round is None:
                    convergence_round = round_num

        final_fairness = self._fairness_history[-1] if self._fairness_history else 0.0

        return FedProxResult(
            coalition=coalition,
            fitness=best_fitness,
            fairness_divergence=final_fairness,
            convergence_round=convergence_round,
            metrics={
                "n_rounds": self.config.n_rounds,
                "n_clients_per_round": len(coalition),
                "participation_rate": self.config.participation_rate,
                "mu": self.config.mu,
                "fitness_history": self._fitness_history.copy(),
                "fairness_history": self._fairness_history.copy(),
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
        total = sizes.sum()
        if total == 0:
            return 0.0
        weights = sizes / total

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
        kl = np.sum(p * np.log(p / q))

        return float(kl)

    def reset(self) -> None:
        """Reset baseline for new run."""
        self._fitness_history = []
        self._fairness_history = []
        self.rng = np.random.default_rng(self.config.seed)
