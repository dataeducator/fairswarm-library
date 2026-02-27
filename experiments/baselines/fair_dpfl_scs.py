"""
FedFDP Baseline for FairSwarm Comparison.

State-of-the-art fair client selection baseline inspired by:
"FedFDP: Fairness-Aware Federated Learning with Differential Privacy"
(Ling et al., 2024, arXiv:2402.16028)

This is a stub implementation for comparison purposes.
The actual algorithm uses:
- Gradient-based client utility estimation
- Fairness-aware gradient clipping
- Differential privacy for utility estimation

Reference:
    FedFDP (Ling et al., 2024) - State-of-the-art fairness competitor

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
class FairDPFLConfig:
    """
    Configuration for FedFDP baseline.

    Attributes:
        coalition_size: Number of clients to select
        n_rounds: Number of selection rounds
        fairness_threshold: Target fairness constraint (epsilon)
        privacy_budget: Differential privacy epsilon
        lambda_init: Initial Lagrangian multiplier
        lambda_lr: Learning rate for lambda updates
        seed: Random seed
    """

    coalition_size: int = 10
    n_rounds: int = 50
    fairness_threshold: float = 0.05
    privacy_budget: float = 4.0
    lambda_init: float = 0.1
    lambda_lr: float = 0.01
    seed: Optional[int] = None


@dataclass
class FairDPFLResult:
    """
    Result from FedFDP baseline.

    Attributes:
        coalition: Final selected coalition
        fitness: Final fitness value
        fairness_divergence: Demographic divergence achieved
        privacy_spent: Total privacy budget consumed
        convergence_round: Round at which converged
        metrics: Additional metrics
    """

    coalition: Coalition = field(default_factory=list)
    fitness: float = 0.0
    fairness_divergence: float = 0.0
    privacy_spent: float = 0.0
    convergence_round: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class FairDPFL_SCS:
    """
    FedFDP: Fairness-Aware Federated Learning with Differential Privacy.

    Stub implementation of the state-of-the-art baseline for
    fair client selection, inspired by Ling et al. (2024).

    Algorithm Overview:
        1. Estimate client utilities using gradient norms
        2. Add DP noise to utility estimates
        3. Select clients to maximize utility subject to fairness constraint
        4. Use Lagrangian relaxation to handle fairness constraint
        5. Update Lagrangian multiplier based on constraint violation

    Comparison Purpose:
        - State-of-the-art fairness comparison
        - Shows FairSwarm improvements over existing methods
        - Validates novel contributions

    Note:
        This is a simplified stub. Full implementation would require
        gradient-based utility estimation from actual training.
    """

    def __init__(
        self,
        clients: List[Client],
        config: Optional[FairDPFLConfig] = None,
    ):
        """
        Initialize FedFDP baseline.

        Args:
            clients: List of federated learning clients
            config: Configuration
        """
        self.clients = clients
        self.config = config or FairDPFLConfig()
        self.n_clients = len(clients)

        self.rng = np.random.default_rng(self.config.seed)

        # Lagrangian multiplier for fairness constraint
        self._lambda = self.config.lambda_init

        # Tracking
        self._fitness_history: List[float] = []
        self._fairness_history: List[float] = []
        self._lambda_history: List[float] = []

    def run(
        self,
        fitness_fn: FitnessFunction,
        target_distribution: Optional[NDArray[np.float64]] = None,
    ) -> FairDPFLResult:
        """
        Run FedFDP client selection.

        Args:
            fitness_fn: Fitness function for evaluation
            target_distribution: Target demographics for fairness

        Returns:
            FairDPFLResult with selected coalition and metrics
        """
        if target_distribution is None:
            target_distribution = np.ones(4) / 4  # Uniform default

        best_coalition: Coalition = []
        best_fitness = float("-inf")
        convergence_round = None
        total_privacy_spent = 0.0

        for round_num in range(self.config.n_rounds):
            # Step 1: Estimate client utilities (stub: use dataset size as proxy)
            utilities = self._estimate_utilities()

            # Step 2: Add DP noise
            noisy_utilities = self._add_dp_noise(utilities)
            total_privacy_spent += self.config.privacy_budget / self.config.n_rounds

            # Step 3: Compute fairness-adjusted scores
            fairness_bonuses = self._compute_fairness_bonuses(target_distribution)
            adjusted_scores = noisy_utilities + self._lambda * fairness_bonuses

            # Step 4: Select top-k clients
            coalition = self._select_coalition(adjusted_scores)

            # Evaluate
            result = fitness_fn.evaluate(coalition, self.clients)
            fairness = self._compute_fairness(coalition, target_distribution)

            self._fitness_history.append(result.value)
            self._fairness_history.append(fairness)
            self._lambda_history.append(self._lambda)

            # Step 5: Update Lagrangian multiplier
            constraint_violation = fairness - self.config.fairness_threshold
            self._lambda = max(
                0, self._lambda + self.config.lambda_lr * constraint_violation
            )

            # Track best
            if result.value > best_fitness:
                best_fitness = result.value
                best_coalition = coalition

            # Check convergence
            if len(self._fitness_history) > 10:
                recent_var = np.var(self._fitness_history[-10:])
                if recent_var < 1e-6 and convergence_round is None:
                    convergence_round = round_num

        final_fairness = self._fairness_history[-1] if self._fairness_history else 0.0

        return FairDPFLResult(
            coalition=best_coalition,
            fitness=best_fitness,
            fairness_divergence=final_fairness,
            privacy_spent=total_privacy_spent,
            convergence_round=convergence_round,
            metrics={
                "n_rounds": self.config.n_rounds,
                "coalition_size": self.config.coalition_size,
                "final_lambda": self._lambda,
                "fitness_history": self._fitness_history.copy(),
                "fairness_history": self._fairness_history.copy(),
                "lambda_history": self._lambda_history.copy(),
            },
        )

    def _estimate_utilities(self) -> NDArray[np.float64]:
        """
        Estimate client utilities.

        In full implementation, this would use gradient norms.
        Stub uses dataset size as proxy for utility.

        Returns:
            Array of client utilities
        """
        # Proxy: larger datasets are more useful
        utilities = np.array([np.log1p(c.dataset_size) for c in self.clients])

        # Normalize
        utilities = (utilities - utilities.min()) / (
            utilities.max() - utilities.min() + 1e-8
        )

        return utilities

    def _add_dp_noise(
        self,
        utilities: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Add differential privacy noise to utilities.

        Uses Laplace mechanism with sensitivity based on
        utility range.

        Args:
            utilities: Clean utility estimates

        Returns:
            Noisy utilities
        """
        sensitivity = 1.0  # Normalized utilities in [0, 1]
        scale = sensitivity / self.config.privacy_budget

        noise = self.rng.laplace(0, scale, size=utilities.shape)

        return utilities + noise

    def _compute_fairness_bonuses(
        self,
        target_distribution: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute fairness bonus for each client.

        Clients whose demographics are underrepresented get higher bonuses.

        Args:
            target_distribution: Target demographics

        Returns:
            Array of fairness bonuses per client
        """
        bonuses = np.zeros(self.n_clients)

        for i, client in enumerate(self.clients):
            demo = client.demographics
            n_groups = min(len(demo), len(target_distribution))

            # Bonus for representing underrepresented groups
            # Higher bonus if client has demographics that are underrepresented
            for g in range(n_groups):
                if demo[g] > 0:
                    bonuses[i] += target_distribution[g] * demo[g]

        # Normalize
        if bonuses.max() > 0:
            bonuses = bonuses / bonuses.max()

        return bonuses

    def _select_coalition(
        self,
        scores: NDArray[np.float64],
    ) -> Coalition:
        """
        Select top-k clients by score.

        Args:
            scores: Adjusted client scores

        Returns:
            Coalition of selected client indices
        """
        k = min(self.config.coalition_size, self.n_clients)
        top_indices = np.argsort(scores)[-k:]
        return top_indices.tolist()

    def _compute_fairness(
        self,
        coalition: Coalition,
        target_distribution: NDArray[np.float64],
    ) -> float:
        """Compute demographic divergence."""
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

    def reset(self) -> None:
        """Reset baseline for new run."""
        self._lambda = self.config.lambda_init
        self._fitness_history = []
        self._fairness_history = []
        self._lambda_history = []
        self.rng = np.random.default_rng(self.config.seed)
