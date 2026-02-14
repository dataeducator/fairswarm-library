"""
FairSwarm: Provably Fair Particle Swarm Optimization.

This module implements the core FairSwarm algorithm (Algorithm 1) for
fair coalition selection in federated learning.

Algorithm 1: FairSwarm PSO
    Input: Clients C, coalition size m, target demographics δ*
    Output: Coalition S* with bounded demographic divergence

    Initialize:
        For each particle p ∈ {1, ..., P}:
            x_p ← random vector in [0,1]^n
            v_p ← random vector in [-1,1]^n
            S_p ← SelectTop(x_p, m)
            pBest_p ← x_p
            pBestFit_p ← Fitness(S_p)
        gBest ← argmax_p pBestFit_p

    Main Loop:
        For t = 1 to T:
            For each particle p:
                # Velocity update with fairness gradient
                r_1, r_2 ← Uniform(0,1)
                v_cognitive ← c_1 · r_1 · (pBest_p - x_p)
                v_social ← c_2 · r_2 · (gBest - x_p)
                v_fairness ← c_3 · ∇_fair  # NOVEL
                v_p ← ω · v_p + v_cognitive + v_social + v_fairness
                v_p ← Clamp(v_p, -v_max, v_max)

                # Position update
                x_p ← x_p + v_p
                x_p ← Sigmoid(x_p)

                # Coalition and fitness
                S_p ← SelectTop(x_p, m)
                fit_p ← Fitness(S_p)

                # Update personal best
                If fit_p > pBestFit_p:
                    pBest_p ← x_p
                    pBestFit_p ← fit_p

            # Update global best
            If max_p pBestFit_p > gBestFit:
                gBest ← argmax_p pBest_p
                gBestFit ← max_p pBestFit_p

    Return: S* = SelectTop(gBest, m)

Theoretical Guarantees:
    - Theorem 1: Convergence when ω + (c_1+c_2)/2 < 2
    - Theorem 2: DemDiv(S*) ≤ ε with probability ≥ 1 - δ
    - Theorem 3: (1-1/e-η) approximation for submodular objectives

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
Institution: Meharry Medical College
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fairswarm.algorithms.result import (
    ConvergenceMetrics,
    FairnessMetrics,
    OptimizationResult,
)
from fairswarm.core.config import FairSwarmConfig
from fairswarm.core.particle import Particle
from fairswarm.core.position import decode_coalition
from fairswarm.core.swarm import Swarm, SwarmHistory
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.fairness import (
    compute_coalition_demographics,
    compute_fairness_gradient,
)
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

logger = logging.getLogger(__name__)


class FairSwarm:
    """
    FairSwarm: Provably Fair PSO for Federated Learning.

    Implements Algorithm 1 from CLAUDE.md, a novel particle swarm
    optimization algorithm that incorporates a fairness gradient
    to achieve provably fair coalition selection.

    Key Innovation:
        The fairness-aware velocity update:
        v_p ← ω·v_p + c_1·r_1·(pBest - x) + c_2·r_2·(gBest - x) + c_3·∇_fair

    Attributes:
        clients: List of federated learning clients
        coalition_size: Target number of clients to select (m)
        config: PSO hyperparameters
        target_distribution: Target demographic distribution (δ*)
        fitness_fn: Fitness function for coalition evaluation
        swarm: The particle swarm
        history: Optimization history

    Example:
        >>> from fairswarm import FairSwarm, FairSwarmConfig, Client
        >>> from fairswarm.demographics import CensusTarget
        >>> from fairswarm.fitness import DemographicFitness
        >>>
        >>> clients = create_synthetic_clients(n_clients=20)
        >>> target = CensusTarget.US_2020.as_distribution()
        >>>
        >>> optimizer = FairSwarm(
        ...     clients=clients,
        ...     coalition_size=10,
        ...     target_distribution=target,
        ... )
        >>>
        >>> fitness = DemographicFitness(target_distribution=target)
        >>> result = optimizer.optimize(fitness_fn=fitness, n_iterations=100)
        >>> print(result.summary())

    Research Reference:
        Algorithm 1 in CLAUDE.md, Theorems 1-3 for guarantees.
    """

    def __init__(
        self,
        clients: list[Client],
        coalition_size: int,
        config: FairSwarmConfig | None = None,
        target_distribution: DemographicDistribution | None = None,
        seed: int | None = None,
    ):
        """
        Initialize FairSwarm optimizer.

        Args:
            clients: List of federated learning clients
            coalition_size: Number of clients to select (m)
            config: PSO hyperparameters (defaults to standard config)
            target_distribution: Target demographics (δ*)
            seed: Random seed for reproducibility

        Raises:
            ValueError: If coalition_size > len(clients) or < 1
        """
        self.clients = clients
        self.coalition_size = coalition_size
        self.n_clients = len(clients)

        # Validate coalition size
        if coalition_size < 1:
            raise ValueError(f"coalition_size must be >= 1, got {coalition_size}")
        if coalition_size > self.n_clients:
            raise ValueError(
                f"coalition_size ({coalition_size}) cannot exceed "
                f"number of clients ({self.n_clients})"
            )

        # Configuration
        self.config = config or FairSwarmConfig()
        self.target_distribution = target_distribution

        # Random state
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Swarm (initialized in optimize())
        self.swarm: Swarm | None = None
        self.history: SwarmHistory | None = None

        # Logging
        self._iteration = 0
        self._best_fitness = float("-inf")

        # Adaptive fairness state
        self._current_fairness_weight = self.config.fairness_weight
        self._adaptive_alpha = 0.5  # Rate of increase when behind target
        self._adaptive_beta = 0.3  # Rate of decrease when at target

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
        Run FairSwarm optimization (Algorithm 1).

        Args:
            fitness_fn: Fitness function for coalition evaluation
            n_iterations: Maximum iterations (default: from config)
            convergence_threshold: Stop if improvement < threshold
            convergence_window: Iterations to check for convergence
            callback: Optional callback(iteration, swarm, best_result)
            verbose: Print progress information

        Returns:
            OptimizationResult with optimized coalition and metrics

        Algorithm Reference:
            Implements Algorithm 1 from CLAUDE.md
        """
        n_iterations = n_iterations or self.config.max_iterations

        # Initialize swarm (Algorithm 1: Initialization)
        self._initialize_swarm()
        if self.swarm is None:
            raise RuntimeError("Swarm initialization failed: swarm is None")

        # Track convergence
        fitness_history: list[float] = []
        diversity_history: list[float] = []
        global_best_updates: list[int] = []

        # Main optimization loop (Algorithm 1: Main Loop)
        converged = False
        convergence_iteration = None

        for iteration in range(n_iterations):
            self._iteration = iteration

            # Update all particles
            for particle in self.swarm.particles:
                self._update_particle(particle, fitness_fn)

            # Update global best
            improved = self.swarm.update_global_best()
            if improved:
                global_best_updates.append(iteration)
                self._best_fitness = self.swarm.g_best_fitness

            # Record metrics
            fitness_history.append(self.swarm.g_best_fitness)
            diversity_history.append(self.swarm.get_diversity())

            # Record in history
            if self.history and self.swarm.g_best is not None:
                g_best_coalition = decode_coalition(
                    self.swarm.g_best, self.coalition_size
                )
                # Compute fairness (demographic divergence) for this iteration
                fairness_value = 0.0
                if self.target_distribution is not None:
                    coalition_demo = compute_coalition_demographics(
                        g_best_coalition, self.clients
                    )
                    target = self.target_distribution.as_array()
                    fairness_value = float(kl_divergence(coalition_demo, target))

                self.history.record(
                    fitness=self.swarm.g_best_fitness,
                    fairness=fairness_value,
                    diversity=self.swarm.get_diversity(),
                    coalition=g_best_coalition,
                )

            # Callback
            if callback and self.swarm.g_best is not None:
                g_best_coalition = decode_coalition(
                    self.swarm.g_best, self.coalition_size
                )
                result = fitness_fn.evaluate(g_best_coalition, self.clients)
                callback(iteration, self.swarm, result)

            # Verbose output
            if verbose and iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"fitness={self.swarm.g_best_fitness:.6f}, "
                    f"diversity={self.swarm.get_diversity():.4f}"
                )

            # Adaptive fairness weight update (Algorithm 1: novel contribution)
            if self.config.adaptive_fairness and self.target_distribution is not None:
                current_fairness = self._compute_current_fairness()
                self._adapt_fairness_weight(
                    iteration=iteration,
                    max_iterations=n_iterations,
                    current_fairness=current_fairness,
                )

            # Check convergence: no meaningful improvement over recent window
            if len(fitness_history) >= convergence_window:
                recent = fitness_history[-convergence_window:]
                improvement = abs(recent[-1] - recent[0])
                if improvement < convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    if verbose:
                        logger.info(f"Converged at iteration {iteration}")
                    break

        # Extract final coalition
        if self.swarm.g_best is None:
            raise RuntimeError(
                "Optimization failed: no global best position found. "
                "This may indicate an issue with the fitness function "
                "or swarm initialization."
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

        # Build result
        opt_result = OptimizationResult(
            coalition=final_coalition,
            fitness=final_result.value,
            fitness_components=final_result.components,
            position=self.swarm.g_best.copy(),
            convergence=convergence_metrics,
            fairness=fairness_metrics,
            config=dataclasses.asdict(self.config),
            metadata={
                "n_clients": self.n_clients,
                "coalition_size": self.coalition_size,
                "seed": self.seed,
                "n_particles": self.config.swarm_size,
            },
        )

        return opt_result

    def _initialize_swarm(self) -> None:
        """
        Initialize particle swarm (Algorithm 1: Initialization).

        Creates P particles with random positions and velocities,
        and initializes personal bests.
        """
        self.swarm = Swarm.initialize(
            swarm_size=self.config.swarm_size,
            n_clients=self.n_clients,
            rng=self.rng,
        )
        self.history = SwarmHistory()

        logger.debug(
            f"Initialized swarm with {self.config.swarm_size} particles, "
            f"{self.n_clients} clients"
        )

    def _update_particle(
        self,
        particle: Particle,
        fitness_fn: FitnessFunction,
    ) -> None:
        """
        Update a single particle (Algorithm 1: Particle Update).

        Implements:
            1. Velocity update with fairness gradient
            2. Position update with sigmoid bounding
            3. Coalition decoding and fitness evaluation
            4. Personal best update

        Args:
            particle: The particle to update
            fitness_fn: Fitness function for evaluation
        """
        if self.swarm is None:
            raise RuntimeError("Cannot update particle: swarm is not initialized")
        # Compute fairness gradient (NOVEL contribution)
        if self.target_distribution is not None:
            gradient_result = compute_fairness_gradient(
                position=particle.position,
                clients=self.clients,
                target_distribution=self.target_distribution,
                coalition_size=self.coalition_size,
            )
            fairness_gradient = gradient_result.gradient
        else:
            fairness_gradient = np.zeros(self.n_clients)

        # Compute effective fairness coefficient (with adaptive scaling)
        # When adaptive_fairness is enabled, scale c₃ based on current fairness level
        effective_fairness_coeff = self.config.fairness_coefficient
        if self.config.adaptive_fairness:
            # Scale fairness coefficient by the ratio of current to base weight
            base_weight = self.config.fairness_weight
            if base_weight > 0:
                adaptive_scale = self._current_fairness_weight / base_weight
                effective_fairness_coeff = (
                    self.config.fairness_coefficient * adaptive_scale
                )

        # Velocity update (Algorithm 1: Lines 554-567)
        particle.apply_velocity_update(
            inertia=self.config.inertia,
            cognitive=self.config.cognitive,
            social=self.config.social,
            fairness_coeff=effective_fairness_coeff,
            g_best=self.swarm.g_best,
            fairness_gradient=fairness_gradient,
            velocity_max=self.config.velocity_max,
            rng=self.rng,
        )

        # Position update (Algorithm 1: Lines 569-571)
        particle.apply_position_update()

        # Decode coalition (Algorithm 1: S_p ← SelectTop(x_p, m))
        coalition = particle.decode(self.coalition_size)

        # Evaluate fitness
        result = fitness_fn.evaluate(coalition, self.clients)

        # Update personal best (Algorithm 1: If fit_p > pBestFit_p)
        particle.update_personal_best(result.value, coalition)

    def _compute_current_fairness(self) -> float:
        """
        Compute current demographic divergence for the global best coalition.

        Returns:
            Current DemDiv(S) value for the best coalition
        """
        if self.swarm is None or self.swarm.g_best is None:
            return float("inf")

        g_best_coalition = decode_coalition(self.swarm.g_best, self.coalition_size)

        if not g_best_coalition or self.target_distribution is None:
            return float("inf")

        coalition_demo = compute_coalition_demographics(g_best_coalition, self.clients)
        target = self.target_distribution.as_array()
        return float(kl_divergence(coalition_demo, target))

    def _adapt_fairness_weight(
        self,
        iteration: int,
        max_iterations: int,
        current_fairness: float,
    ) -> None:
        """
        Adapt fairness weight based on current fairness level.

        Implements Algorithm 1's AdaptFairnessWeight subroutine:
            If fairness > epsilon_target:
                lambda = lambda * (1 + alpha * t/T)  # Increase pressure
            Else:
                lambda = lambda * (1 - beta * t/T)   # Decrease pressure

        This novel contribution dynamically balances accuracy and fairness
        based on whether the algorithm is meeting fairness targets.

        Args:
            iteration: Current iteration number (t)
            max_iterations: Total iterations (T)
            current_fairness: Current DemDiv(S) value
        """
        progress = iteration / max_iterations if max_iterations > 0 else 0.0
        epsilon_target = self.config.epsilon_fair
        base_weight = self.config.fairness_weight

        if current_fairness > epsilon_target:
            # Behind target: increase fairness pressure over time
            # This ensures we don't give up on fairness goals
            adjustment = 1.0 + self._adaptive_alpha * progress
        else:
            # At or below target: can relax fairness pressure
            # This allows more focus on accuracy when fairness is achieved
            adjustment = 1.0 - self._adaptive_beta * progress

        # Update the current fairness weight with bounds
        self._current_fairness_weight = float(
            np.clip(
                base_weight * adjustment,
                0.1,  # Minimum: always consider some fairness
                0.9,  # Maximum: always consider some accuracy
            )
        )

        logger.debug(
            f"Iteration {iteration}: fairness={current_fairness:.4f}, "
            f"target={epsilon_target:.4f}, weight={self._current_fairness_weight:.4f}"
        )

    def _compute_fairness_metrics(
        self,
        coalition: Coalition,
    ) -> FairnessMetrics | None:
        """
        Compute fairness metrics for a coalition.

        Implements Definition 2: DemDiv(S) = D_KL(δ_S || δ*)

        Args:
            coalition: The coalition to evaluate

        Returns:
            FairnessMetrics or None if no target distribution
        """
        if self.target_distribution is None:
            return None

        # Compute coalition demographics: δ_S = (1/|S|) Σ_{i∈S} δ_i
        coalition_demo = compute_coalition_demographics(coalition, self.clients)
        target = self.target_distribution.as_array()

        # Compute divergence: DemDiv(S) = D_KL(δ_S || δ*)
        divergence = kl_divergence(coalition_demo, target)

        # Build distribution dictionaries for reporting
        labels = self.target_distribution.labels or [
            f"group_{i}" for i in range(len(target))
        ]
        coalition_dist = {
            cat: float(coalition_demo[i])
            for i, cat in enumerate(labels)
            if i < len(coalition_demo)
        }
        target_dist = {
            cat: float(target[i]) for i, cat in enumerate(labels) if i < len(target)
        }

        # Check ε-fairness (Theorem 2)
        epsilon = self.config.epsilon_fair
        epsilon_satisfied = divergence <= epsilon

        # Compute per-group representation
        group_representation = {}
        for i, cat in enumerate(labels):
            if i < len(coalition_demo):
                group_representation[cat] = float(coalition_demo[i])

        return FairnessMetrics(
            demographic_divergence=divergence,
            coalition_distribution=coalition_dist,
            target_distribution=target_dist,
            epsilon_satisfied=epsilon_satisfied,
            group_representation=group_representation,
        )

    def get_swarm_state(self) -> dict[str, Any]:
        """
        Get current swarm state for debugging/visualization.

        Returns:
            Dictionary with swarm state information
        """
        if self.swarm is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "n_particles": len(self.swarm.particles),
            "g_best_fitness": self.swarm.g_best_fitness,
            "g_best_coalition": (
                decode_coalition(self.swarm.g_best, self.coalition_size)
                if self.swarm.g_best is not None
                else None
            ),
            "diversity": self.swarm.get_diversity(),
            "iteration": self._iteration,
        }

    def reset(self, seed: int | None = None) -> None:
        """
        Reset optimizer state for a new run.

        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.swarm = None
        self.history = None
        self._iteration = 0
        self._best_fitness = float("-inf")
        self._current_fairness_weight = self.config.fairness_weight

    def __repr__(self) -> str:
        return (
            f"FairSwarm(n_clients={self.n_clients}, "
            f"coalition_size={self.coalition_size}, "
            f"n_particles={self.config.swarm_size})"
        )


def run_fairswarm(
    clients: list[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
    target_distribution: DemographicDistribution | None = None,
    config: FairSwarmConfig | None = None,
    n_iterations: int = 100,
    seed: int | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Convenience function to run FairSwarm optimization.

    Args:
        clients: List of federated learning clients
        coalition_size: Number of clients to select
        fitness_fn: Fitness function for evaluation
        target_distribution: Target demographics (optional)
        config: PSO configuration (optional)
        n_iterations: Maximum iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        OptimizationResult with optimized coalition

    Example:
        >>> from fairswarm.algorithms import run_fairswarm
        >>> from fairswarm.fitness import DemographicFitness
        >>> from fairswarm.demographics import CensusTarget
        >>>
        >>> target = CensusTarget.US_2020.as_distribution()
        >>> fitness = DemographicFitness(target_distribution=target)
        >>>
        >>> result = run_fairswarm(
        ...     clients=clients,
        ...     coalition_size=10,
        ...     fitness_fn=fitness,
        ...     target_distribution=target,
        ...     n_iterations=100,
        ... )
    """
    optimizer = FairSwarm(
        clients=clients,
        coalition_size=coalition_size,
        config=config,
        target_distribution=target_distribution,
        seed=seed,
    )

    return optimizer.optimize(
        fitness_fn=fitness_fn,
        n_iterations=n_iterations,
        verbose=verbose,
    )
