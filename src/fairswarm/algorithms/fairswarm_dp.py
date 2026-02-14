"""
FairSwarm-DP: Differentially Private FairSwarm.

This module extends the FairSwarm algorithm with differential privacy
guarantees, implementing Theorem 4 (Privacy-Fairness Tradeoff).

Theorem 4: UtilityLoss ≥ Ω(√(k·log(1/δ))/(ε_DP·ε_F))

Key Modifications:
    1. Noisy fitness evaluation
    2. Private fairness gradient
    3. Privacy budget tracking
    4. Clipped velocity updates

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

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
    add_noise_to_gradient,
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
        noise_multiplier: Noise scale (σ/sensitivity)
        max_grad_norm: Maximum gradient norm for clipping
        mechanism: Type of noise mechanism ("gaussian" or "laplace")
        accountant_type: Type of privacy accountant
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    mechanism: str = "gaussian"
    accountant_type: str = "rdp"

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
        assert self.swarm is not None

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

            # Check convergence
            if len(fitness_history) >= convergence_window:
                recent = fitness_history[-convergence_window:]
                improvement = max(recent) - min(recent)
                if improvement < convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    break

        # Final result
        assert self.swarm.g_best is not None
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

        Modifications from standard FairSwarm:
        1. Clip fairness gradient
        2. Add noise to gradient
        3. Add noise to fitness evaluation
        """
        assert self.swarm is not None
        # Compute fairness gradient
        if self.target_distribution is not None:
            gradient_result = compute_fairness_gradient(
                position=particle.position,
                clients=self.clients,
                target_distribution=self.target_distribution,
                coalition_size=self.coalition_size,
            )
            fairness_gradient = gradient_result.gradient

            # Apply DP to gradient
            fairness_gradient = self._privatize_gradient(fairness_gradient)
        else:
            fairness_gradient = np.zeros(self.n_clients)

        # Standard velocity update with private gradient
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

        # Private fitness evaluation
        result = self._private_evaluate(fitness_fn, coalition)

        # Update personal best
        particle.update_personal_best(result.value, coalition)

    def _privatize_gradient(
        self,
        gradient: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply differential privacy to gradient.

        Args:
            gradient: Raw gradient

        Returns:
            Private gradient
        """
        # Clip gradient
        clipped, _ = clip_gradient(gradient, self.dp_config.max_grad_norm)

        # Add noise
        private_grad: NDArray[np.float64]
        if isinstance(self.mechanism, GaussianMechanism):
            private_grad = add_noise_to_gradient(
                gradient=clipped,
                noise_multiplier=self.dp_config.noise_multiplier,
                max_norm=self.dp_config.max_grad_norm,
                rng=self.rng,
            )
        else:
            # Laplace noise
            noisy = self.mechanism.add_noise(
                clipped,
                sensitivity=self.dp_config.max_grad_norm,
                rng=self.rng,
            )
            private_grad = np.asarray(noisy, dtype=np.float64)

        # Record privacy cost
        self._record_gradient_query()

        return private_grad

    def _private_evaluate(
        self,
        fitness_fn: FitnessFunction,
        coalition: Coalition,
    ) -> FitnessResult:
        """
        Evaluate fitness with differential privacy.

        Args:
            fitness_fn: Fitness function
            coalition: Coalition to evaluate

        Returns:
            Private fitness result
        """
        # Get true fitness
        result = fitness_fn.evaluate(coalition, self.clients)

        # Add noise to fitness value
        noisy_value = float(
            self.mechanism.add_noise(
                result.value,
                sensitivity=self._fitness_sensitivity,
                rng=self.rng,
            )
        )

        # Record privacy cost
        self._record_fitness_query()

        # Return with noisy value
        return FitnessResult(
            value=noisy_value,
            components=result.components,
            coalition=result.coalition,
            metadata={**result.metadata, "private": True},
        )

    def _record_gradient_query(self) -> None:
        """Record a gradient query for privacy accounting."""
        self._n_queries += 1

        # Use RDP accounting for Gaussian mechanism
        if isinstance(self.accountant, RDPAccountant):
            self.accountant.step(
                noise_multiplier=self.dp_config.noise_multiplier,
                sampling_rate=1.0,
            )
        else:
            # Simple accounting
            epsilon_per_query = (
                self.dp_config.max_grad_norm / self.dp_config.noise_multiplier
            )
            self.accountant.step(epsilon=epsilon_per_query)

    def _record_fitness_query(self) -> None:
        """Record a fitness query for privacy accounting."""
        self._n_queries += 1

        if isinstance(self.accountant, RDPAccountant):
            self.accountant.step(
                noise_multiplier=self.dp_config.noise_multiplier,
                sampling_rate=1.0,
            )
        else:
            epsilon_per_query = (
                self._fitness_sensitivity / self.dp_config.noise_multiplier
            )
            self.accountant.step(epsilon=epsilon_per_query)

    def _estimate_sensitivity(self, fitness_fn: FitnessFunction) -> float:
        """
        Estimate fitness function sensitivity.

        Sensitivity Δf = max |f(D) - f(D')|

        Args:
            fitness_fn: Fitness function

        Returns:
            Estimated sensitivity
        """
        # Default sensitivity for bounded fitness functions
        # This should be calibrated based on the specific fitness function
        return 1.0

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
