"""
FairSwarm-DP: Differentially Private FairSwarm.

This module extends the FairSwarm algorithm with differential privacy
guarantees, implementing Theorem 4 (Privacy-Fairness Tradeoff).

Theorem 4: UtilityLoss ≥ Ω(√(k·log(1/δ))/(ε_DP·ε_F))

Key Modifications:
    1. Noisy fitness evaluation (budget-calibrated noise)
    2. Gradient clipping for stability (post-processing, no privacy cost)
    3. Privacy budget tracking via RDP accountant
    4. Auto-calibrated noise_multiplier to use full budget

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import (
    ConvergenceMetrics,
    OptimizationResult,
)
from fairswarm.core.config import FairSwarmConfig
from fairswarm.core.particle import Particle
from fairswarm.core.position import decode_coalition
from fairswarm.core.swarm import Swarm
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.fairness import compute_fairness_gradient
from fairswarm.privacy.accountant import PrivacyAccountant, RDPAccountant
from fairswarm.privacy.mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    NoiseMechanism,
    clip_gradient,
)
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """
    Differential privacy configuration for FairSwarm-DP.

    Attributes:
        epsilon: Privacy budget for entire optimization
        delta: Privacy failure probability
        noise_multiplier: Noise scale (σ/sensitivity). If auto_calibrate
            is True (default), this is overridden by budget-aware calibration.
        max_grad_norm: Maximum gradient norm for clipping (stability, not privacy)
        mechanism: Type of noise mechanism ("gaussian" or "laplace")
        accountant_type: Type of privacy accountant
        fitness_sensitivity: Known sensitivity of the fitness function.
            If None, sensitivity is estimated empirically by sampling.
        auto_calibrate: If True (default), noise_multiplier is computed
            from the privacy budget to ensure all iterations run with
            calibrated noise. This guarantees monotonic utility improvement
            as epsilon increases.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    mechanism: str = "gaussian"
    accountant_type: str = "rdp"
    fitness_sensitivity: float | None = None  # None = estimate automatically
    auto_calibrate: bool = True

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "mechanism": self.mechanism,
            "accountant_type": self.accountant_type,
            "fitness_sensitivity": self.fitness_sensitivity,
            "auto_calibrate": self.auto_calibrate,
        }


@dataclass
class DPResult:
    """
    Privacy metrics for FairSwarm-DP optimization.

    Attributes:
        epsilon_spent: Total epsilon consumed
        delta: Privacy failure probability
        n_queries: Number of private queries made
        privacy_satisfied: Whether within budget
    """

    epsilon_spent: float
    delta: float
    n_queries: int
    privacy_satisfied: bool
    accountant_config: dict[str, Any] = field(default_factory=dict)


class FairSwarmDP(FairSwarm):
    """
    Differentially Private FairSwarm optimizer.

    Extends FairSwarm with differential privacy guarantees:
    - Noisy fitness evaluations
    - Private fairness gradients
    - Privacy budget tracking

    Theorem 4 Connection:
        The privacy-fairness tradeoff bounds the minimum utility loss:
        UtilityLoss ≥ Ω(√(k·log(1/δ))/(ε_DP·ε_F))

    Attributes:
        dp_config: Differential privacy configuration
        accountant: Privacy budget tracker
        mechanism: Noise mechanism

    Example:
        >>> dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        >>> optimizer = FairSwarmDP(
        ...     clients=clients,
        ...     coalition_size=10,
        ...     target_distribution=target,
        ...     dp_config=dp_config,
        ... )
        >>> result = optimizer.optimize(fitness_fn, n_iterations=100)
        >>> print(f"Privacy spent: ε={result.dp_result.epsilon_spent:.2f}")
    """

    def __init__(
        self,
        clients: list[Client],
        coalition_size: int,
        config: FairSwarmConfig | None = None,
        target_distribution: DemographicDistribution | None = None,
        dp_config: DPConfig | None = None,
        seed: int | None = None,
    ):
        """
        Initialize FairSwarmDP.

        Args:
            clients: List of federated learning clients
            coalition_size: Number of clients to select
            config: PSO hyperparameters
            target_distribution: Target demographics
            dp_config: Differential privacy configuration
            seed: Random seed
        """
        super().__init__(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            target_distribution=target_distribution,
            seed=seed,
        )

        # DP configuration
        self.dp_config = dp_config or DPConfig()

        # Initialize privacy mechanism
        self.mechanism = self._create_mechanism()

        # Initialize privacy accountant
        self.accountant = self._create_accountant()

        # Query tracking
        self._n_queries = 0
        self._fitness_sensitivity = 1.0  # Will be set based on fitness function
        self._calibrated_noise_multiplier = self.dp_config.noise_multiplier

    def _create_mechanism(self) -> NoiseMechanism:
        """Create noise mechanism based on config."""
        if self.dp_config.mechanism == "laplace":
            return LaplaceMechanism(epsilon=self.dp_config.epsilon)
        else:
            return GaussianMechanism(
                epsilon=self.dp_config.epsilon,
                delta=self.dp_config.delta,
            )

    def _create_accountant(self) -> PrivacyAccountant:
        """Create privacy accountant based on config."""
        if self.dp_config.accountant_type == "rdp":
            return RDPAccountant()
        else:
            from fairswarm.privacy.accountant import SimpleAccountant

            return SimpleAccountant()

    def optimize(
        self,
        fitness_fn: FitnessFunction,
        n_iterations: int | None = None,
        convergence_threshold: float = 1e-6,
        convergence_window: int = 20,
        callback: Callable[[int, Swarm, FitnessResult], None] | None = None,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run FairSwarm-DP optimization with privacy guarantees.

        Args:
            fitness_fn: Fitness function for coalition evaluation
            n_iterations: Maximum iterations
            convergence_threshold: Stop if improvement < threshold
            convergence_window: Iterations to check for convergence
            callback: Optional callback function
            verbose: Print progress

        Returns:
            OptimizationResult with privacy metrics
        """
        n_iterations = n_iterations or self.config.max_iterations

        # Reset accountant for new optimization
        self.accountant.reset()
        self._n_queries = 0

        # Estimate fitness sensitivity for noise calibration
        self._fitness_sensitivity = self._estimate_sensitivity(fitness_fn)

        # Initialize swarm
        self._initialize_swarm()
        if self.swarm is None:
            raise RuntimeError("Swarm initialization failed: swarm is None")

        # Auto-calibrate noise_multiplier to use exactly the privacy budget
        # across all iterations. Each iteration has 1 fitness query per particle.
        if self.dp_config.auto_calibrate:
            queries_per_iter = len(self.swarm.particles)
            self._calibrated_noise_multiplier = self._calibrate_noise_multiplier(
                total_budget=self.dp_config.epsilon,
                n_iterations=n_iterations,
                queries_per_iteration=queries_per_iter,
                delta=self.dp_config.delta,
            )
            logger.info(
                f"Auto-calibrated noise_multiplier={self._calibrated_noise_multiplier:.4f} "
                f"for ε={self.dp_config.epsilon}, {n_iterations} iterations, "
                f"{queries_per_iter} queries/iter"
            )
        else:
            self._calibrated_noise_multiplier = self.dp_config.noise_multiplier

        # Track metrics
        fitness_history: list[float] = []
        diversity_history: list[float] = []
        global_best_updates: list[int] = []

        # Main optimization loop with privacy
        converged = False
        convergence_iteration = None

        for iteration in range(n_iterations):
            self._iteration = iteration

            # Check privacy budget
            current_epsilon = self.accountant.get_epsilon(self.dp_config.delta)
            if current_epsilon >= self.dp_config.epsilon:
                logger.info(f"Privacy budget exhausted at iteration {iteration}")
                break

            # Update particles with DP
            for particle in self.swarm.particles:
                self._update_particle_dp(particle, fitness_fn)

            # Update global best
            improved = self.swarm.update_global_best()
            if improved:
                global_best_updates.append(iteration)

            # Record metrics
            fitness_history.append(self.swarm.g_best_fitness)
            diversity_history.append(self.swarm.get_diversity())

            # Callback
            if callback and self.swarm.g_best is not None:
                g_best_coalition = decode_coalition(
                    self.swarm.g_best, self.coalition_size
                )
                result = fitness_fn.evaluate(g_best_coalition, self.clients)
                callback(iteration, self.swarm, result)

            # Verbose output
            if verbose and iteration % 10 == 0:
                eps_spent = self.accountant.get_epsilon(self.dp_config.delta)
                logger.info(
                    f"Iteration {iteration}: fitness={self.swarm.g_best_fitness:.4f}, "
                    f"ε_spent={eps_spent:.4f}"
                )

            # Check convergence: no meaningful improvement over recent window
            if len(fitness_history) >= convergence_window:
                recent = fitness_history[-convergence_window:]
                improvement = abs(recent[-1] - recent[0])
                if improvement < convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    break

        # Final result
        if self.swarm.g_best is None:
            raise RuntimeError(
                "Optimization failed: no global best position found. "
                "This may indicate an issue with the fitness function, "
                "swarm initialization, or privacy budget exhaustion."
            )
        final_coalition = decode_coalition(self.swarm.g_best, self.coalition_size)
        final_result = fitness_fn.evaluate(final_coalition, self.clients)

        # Build convergence metrics
        convergence_metrics = ConvergenceMetrics(
            iterations=self._iteration + 1,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            global_best_updates=global_best_updates,
            converged=converged,
            convergence_iteration=convergence_iteration,
        )

        # Build fairness metrics
        fairness_metrics = self._compute_fairness_metrics(final_coalition)

        # Build DP result
        dp_result = DPResult(
            epsilon_spent=self.accountant.get_epsilon(self.dp_config.delta),
            delta=self.dp_config.delta,
            n_queries=self._n_queries,
            privacy_satisfied=self.accountant.get_epsilon(self.dp_config.delta)
            <= self.dp_config.epsilon,
            accountant_config=self.accountant.get_config(),
        )

        # Build result
        opt_result = OptimizationResult(
            coalition=final_coalition,
            fitness=final_result.value,
            fitness_components=final_result.components,
            position=self.swarm.g_best.copy(),
            convergence=convergence_metrics,
            fairness=fairness_metrics,
            config={**dataclasses.asdict(self.config), "dp": self.dp_config.to_dict()},
            metadata={
                "n_clients": self.n_clients,
                "coalition_size": self.coalition_size,
                "seed": self.seed,
                "dp_result": {
                    "epsilon_spent": dp_result.epsilon_spent,
                    "delta": dp_result.delta,
                    "n_queries": dp_result.n_queries,
                    "privacy_satisfied": dp_result.privacy_satisfied,
                },
            },
        )

        return opt_result

    def _update_particle_dp(
        self,
        particle: Particle,
        fitness_fn: FitnessFunction,
    ) -> None:
        """
        Update particle with differential privacy.

        Privacy model:
        - Fitness evaluation: noisy (1 privacy query per particle)
        - Fairness gradient: post-processing of public demographic metadata
          (client demographics are known to the server in FL), so gradient
          computation has no privacy cost. Gradient is clipped for stability.
        """
        if self.swarm is None:
            raise RuntimeError("Cannot update particle: swarm is not initialized")
        # Compute fairness gradient (post-processing of public demographics)
        if self.target_distribution is not None:
            gradient_result = compute_fairness_gradient(
                position=particle.position,
                clients=self.clients,
                target_distribution=self.target_distribution,
                coalition_size=self.coalition_size,
            )
            fairness_gradient = gradient_result.gradient

            # Clip gradient for numerical stability (not a privacy operation)
            fairness_gradient, _ = clip_gradient(
                fairness_gradient, self.dp_config.max_grad_norm
            )
        else:
            fairness_gradient = np.zeros(self.n_clients)

        # Standard velocity update with clipped gradient
        particle.apply_velocity_update(
            inertia=self.config.inertia,
            cognitive=self.config.cognitive,
            social=self.config.social,
            fairness_coeff=self.config.fairness_coefficient,
            g_best=self.swarm.g_best,
            fairness_gradient=fairness_gradient,
            velocity_max=self.config.velocity_max,
            rng=self.rng,
        )

        # Position update
        particle.apply_position_update()

        # Decode coalition
        coalition = particle.decode(self.coalition_size)

        # Private fitness evaluation (the ONLY privacy query per particle)
        result = self._private_evaluate(fitness_fn, coalition)

        # Update personal best
        particle.update_personal_best(result.value, coalition)

    def _private_evaluate(
        self,
        fitness_fn: FitnessFunction,
        coalition: Coalition,
    ) -> FitnessResult:
        """
        Evaluate fitness with differential privacy.

        Only the aggregate fitness value is privatized (1 privacy query).
        Component values are redacted (set to the noisy aggregate) to
        prevent information leakage, but this does not cost additional
        privacy budget since components are deterministic functions of
        the coalition which is already determined by the position.

        Args:
            fitness_fn: Fitness function
            coalition: Coalition to evaluate

        Returns:
            Private fitness result
        """
        # Get true fitness
        result = fitness_fn.evaluate(coalition, self.clients)

        # Add calibrated noise to fitness value (1 privacy query)
        noise_sigma = self._calibrated_noise_multiplier * self._fitness_sensitivity
        noise = float(self.rng.normal(0.0, noise_sigma))
        noisy_value = float(result.value) + noise
        self._record_fitness_query()

        # Redact components (no additional privacy cost)
        private_components = dict.fromkeys(result.components, noisy_value)

        return FitnessResult(
            value=noisy_value,
            components=private_components,
            coalition=result.coalition,
            metadata={"private": True},
        )

    def _record_fitness_query(self) -> None:
        """Record a fitness query for privacy accounting."""
        self._n_queries += 1

        if isinstance(self.accountant, RDPAccountant):
            self.accountant.step(
                noise_multiplier=self._calibrated_noise_multiplier,
                sampling_rate=1.0,
            )
        else:
            epsilon_per_query = (
                self._fitness_sensitivity / self._calibrated_noise_multiplier
            )
            self.accountant.step(epsilon=epsilon_per_query)

    @staticmethod
    def _compute_rdp_epsilon(
        total_queries: int,
        noise_multiplier: float,
        delta: float,
    ) -> float:
        """
        Compute (ε, δ)-DP guarantee for Gaussian mechanism via RDP.

        For ``total_queries`` independent Gaussian queries with the given
        noise_multiplier, find the tightest ε by minimizing over RDP orders.

        Args:
            total_queries: Total number of privacy queries
            noise_multiplier: σ / Δf (noise scale relative to sensitivity)
            delta: Target failure probability

        Returns:
            Best (ε, δ)-DP epsilon across RDP orders
        """
        if total_queries == 0:
            return 0.0
        if noise_multiplier <= 0:
            return float("inf")

        best_eps = float("inf")
        # Search over RDP orders matching the accountant's range
        for alpha_10x in range(11, 1000):
            alpha = alpha_10x / 10.0
            total_rdp = total_queries * alpha / (2.0 * noise_multiplier**2)
            eps = total_rdp + math.log(1.0 / delta) / (alpha - 1.0)
            best_eps = min(best_eps, eps)
        for alpha in range(12, 64):
            total_rdp = total_queries * float(alpha) / (2.0 * noise_multiplier**2)
            eps = total_rdp + math.log(1.0 / delta) / (alpha - 1.0)
            best_eps = min(best_eps, eps)

        return best_eps

    def _calibrate_noise_multiplier(
        self,
        total_budget: float,
        n_iterations: int,
        queries_per_iteration: int,
        delta: float,
    ) -> float:
        """
        Compute the noise_multiplier that uses exactly the privacy budget.

        Binary searches for the smallest σ such that the total RDP-converted
        (ε, δ)-DP guarantee stays within ``total_budget``. Higher budgets
        yield lower noise_multiplier (less noise), guaranteeing monotonic
        utility improvement as ε increases.

        Args:
            total_budget: Total ε budget for optimization
            n_iterations: Number of planned iterations
            queries_per_iteration: Privacy queries per iteration
            delta: Target δ

        Returns:
            Calibrated noise_multiplier
        """
        total_queries = n_iterations * queries_per_iteration

        # Binary search: higher σ → more noise → lower ε
        lo, hi = 0.01, 10000.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            eps = self._compute_rdp_epsilon(total_queries, mid, delta)
            if eps > total_budget:
                lo = mid  # Need more noise to stay within budget
            else:
                hi = mid  # Can use less noise

        # Verify calibration: ensure returned σ stays within budget
        final_eps = self._compute_rdp_epsilon(total_queries, hi, delta)
        if final_eps > total_budget:
            # Use slightly more noise to guarantee budget compliance
            hi = lo
        return hi

    def _estimate_sensitivity(self, fitness_fn: FitnessFunction) -> float:
        """
        Estimate fitness function sensitivity.

        Sensitivity Δf = max |f(D) - f(D')| over neighboring datasets.

        If dp_config.fitness_sensitivity is provided, uses that value.
        Otherwise estimates empirically by sampling coalitions and computing
        the maximum difference when one client is swapped.

        Args:
            fitness_fn: Fitness function

        Returns:
            Estimated sensitivity
        """
        # Use configured value if available
        if self.dp_config.fitness_sensitivity is not None:
            return self.dp_config.fitness_sensitivity

        # Empirical estimation: sample coalitions and measure max change
        # when one client is swapped
        n_samples = min(50, self.n_clients * 2)
        max_diff = 0.0

        for _ in range(n_samples):
            # Random coalition
            indices = self.rng.choice(
                self.n_clients, size=self.coalition_size, replace=False
            ).tolist()
            base_result = fitness_fn.evaluate(indices, self.clients)

            # Try swapping each member with a non-member
            non_members = [i for i in range(self.n_clients) if i not in indices]
            if not non_members:
                continue

            swap_idx = int(self.rng.choice(len(indices)))
            new_member = int(self.rng.choice(non_members))
            modified = indices.copy()
            modified[swap_idx] = new_member
            modified_result = fitness_fn.evaluate(modified, self.clients)

            diff = abs(base_result.value - modified_result.value)
            max_diff = max(max_diff, diff)

        # Add safety margin and ensure positive
        return max(max_diff * 1.5, 0.01)

    def get_privacy_spent(self) -> tuple[float, float]:
        """
        Get current privacy expenditure.

        Returns:
            Tuple of (epsilon, delta)
        """
        epsilon = self.accountant.get_epsilon(self.dp_config.delta)
        return (epsilon, self.dp_config.delta)

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        spent = self.accountant.get_epsilon(self.dp_config.delta)
        return max(0.0, self.dp_config.epsilon - spent)

    def reset(self, seed: int | None = None) -> None:
        """Reset optimizer state including privacy accountant."""
        super().reset(seed)
        self.accountant.reset()
        self._n_queries = 0

    def __repr__(self) -> str:
        return (
            f"FairSwarmDP(n_clients={self.n_clients}, "
            f"coalition_size={self.coalition_size}, "
            f"ε={self.dp_config.epsilon}, δ={self.dp_config.delta})"
        )


def run_fairswarm_dp(
    clients: list[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
    target_distribution: DemographicDistribution | None = None,
    config: FairSwarmConfig | None = None,
    dp_config: DPConfig | None = None,
    n_iterations: int = 100,
    seed: int | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Convenience function to run FairSwarm-DP optimization.

    Args:
        clients: List of federated learning clients
        coalition_size: Number of clients to select
        fitness_fn: Fitness function for evaluation
        target_distribution: Target demographics
        config: PSO configuration
        dp_config: Differential privacy configuration
        n_iterations: Maximum iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        OptimizationResult with privacy metrics

    Example:
        >>> dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        >>> result = run_fairswarm_dp(
        ...     clients=clients,
        ...     coalition_size=10,
        ...     fitness_fn=fitness,
        ...     dp_config=dp_config,
        ... )
    """
    optimizer = FairSwarmDP(
        clients=clients,
        coalition_size=coalition_size,
        config=config,
        target_distribution=target_distribution,
        dp_config=dp_config,
        seed=seed,
    )

    return optimizer.optimize(
        fitness_fn=fitness_fn,
        n_iterations=n_iterations,
        verbose=verbose,
    )
