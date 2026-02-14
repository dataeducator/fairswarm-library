"""
Optimization result containers for FairSwarm.

This module provides structured containers for FairSwarm optimization
results, including convergence metrics and fairness analysis.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.types import Coalition, FitnessValue


@dataclass
class ConvergenceMetrics:
    """
    Metrics tracking PSO convergence.

    Used to analyze optimization dynamics and verify Theorem 1
    convergence guarantees.

    Attributes:
        iterations: Number of iterations completed
        fitness_history: Best fitness at each iteration
        diversity_history: Swarm diversity at each iteration
        global_best_updates: Iterations where global best improved
        converged: Whether optimization converged
        convergence_iteration: Iteration at which convergence detected
    """

    iterations: int
    fitness_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    global_best_updates: list[int] = field(default_factory=list)
    converged: bool = False
    convergence_iteration: int | None = None

    @property
    def improvement_rate(self) -> float:
        """Fraction of iterations with global best improvement."""
        if self.iterations == 0:
            return 0.0
        return len(self.global_best_updates) / self.iterations

    @property
    def final_diversity(self) -> float:
        """Swarm diversity at final iteration."""
        return self.diversity_history[-1] if self.diversity_history else 0.0

    def fitness_improvement(self, window: int = 10) -> float:
        """
        Average fitness improvement over recent iterations.

        Args:
            window: Number of recent iterations to consider

        Returns:
            Average improvement per iteration
        """
        if len(self.fitness_history) < 2:
            return 0.0

        recent = self.fitness_history[-window:]
        if len(recent) < 2:
            return 0.0

        improvements = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return float(np.mean(improvements))


@dataclass
class FairnessMetrics:
    """
    Fairness metrics for the optimized coalition.

    Used to verify Theorem 2 (ε-fairness) guarantees.

    Attributes:
        demographic_divergence: DemDiv(S*) from Definition 2
        coalition_distribution: Demographic distribution of coalition
        target_distribution: Target demographic distribution
        epsilon_satisfied: Whether DemDiv ≤ ε threshold
        group_representation: Representation per demographic group
    """

    demographic_divergence: float
    coalition_distribution: dict[str, float] = field(default_factory=dict)
    target_distribution: dict[str, float] = field(default_factory=dict)
    epsilon_satisfied: bool = False
    group_representation: dict[str, float] = field(default_factory=dict)

    def representation_gap(self, group: str) -> float:
        """
        Compute representation gap for a specific group.

        Args:
            group: Demographic group name

        Returns:
            Absolute difference between coalition and target representation
        """
        coalition_rep = self.coalition_distribution.get(group, 0.0)
        target_rep = self.target_distribution.get(group, 0.0)
        return abs(coalition_rep - target_rep)

    def max_representation_gap(self) -> float:
        """Maximum representation gap across all groups."""
        if not self.target_distribution:
            return 0.0

        gaps = [self.representation_gap(group) for group in self.target_distribution]
        return max(gaps) if gaps else 0.0


@dataclass
class OptimizationResult:
    """
    Complete result from FairSwarm optimization.

    Contains the optimized coalition, fitness analysis, convergence
    metrics, and fairness verification.

    Attributes:
        coalition: The selected client indices (S*)
        fitness: Final fitness value F(S*)
        fitness_components: Breakdown of fitness by component
        position: Final position vector (selection probabilities)
        convergence: Convergence metrics
        fairness: Fairness metrics
        config: Configuration used for optimization
        metadata: Additional run information

    Example:
        >>> result = optimizer.optimize(fitness_fn, n_iterations=100)
        >>> print(f"Selected clients: {result.coalition}")
        >>> print(f"Demographic divergence: {result.fairness.demographic_divergence:.4f}")
        >>> if result.fairness.epsilon_satisfied:
        ...     print("Theorem 2 ε-fairness achieved!")
    """

    coalition: Coalition
    fitness: FitnessValue
    fitness_components: dict[str, float] = field(default_factory=dict)
    position: NDArray[np.float64] | None = None
    convergence: ConvergenceMetrics | None = None
    fairness: FairnessMetrics | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def coalition_size(self) -> int:
        """Number of clients in the coalition."""
        return len(self.coalition)

    @property
    def is_converged(self) -> bool:
        """Whether optimization converged."""
        return self.convergence is not None and self.convergence.converged

    @property
    def is_fair(self) -> bool:
        """Whether coalition satisfies ε-fairness."""
        return self.fairness is not None and self.fairness.epsilon_satisfied

    def summary(self) -> str:
        """
        Generate human-readable summary of results.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 50,
            "FairSwarm Optimization Result",
            "=" * 50,
            f"Coalition Size: {self.coalition_size}",
            f"Coalition: {self.coalition}",
            f"Fitness: {self.fitness:.6f}",
        ]

        if self.fitness_components:
            lines.append("\nFitness Components:")
            for name, value in self.fitness_components.items():
                lines.append(f"  {name}: {value:.6f}")

        if self.convergence:
            lines.append("\nConvergence:")
            lines.append(f"  Iterations: {self.convergence.iterations}")
            lines.append(f"  Converged: {self.convergence.converged}")
            lines.append(f"  Final Diversity: {self.convergence.final_diversity:.4f}")

        if self.fairness:
            lines.append("\nFairness (Theorem 2):")
            lines.append(f"  DemDiv(S*): {self.fairness.demographic_divergence:.6f}")
            lines.append(f"  ε-satisfied: {self.fairness.epsilon_satisfied}")
            lines.append(f"  Max Gap: {self.fairness.max_representation_gap():.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        result = {
            "coalition": self.coalition,
            "fitness": self.fitness,
            "fitness_components": self.fitness_components,
            "coalition_size": self.coalition_size,
        }

        if self.position is not None:
            result["position"] = self.position.tolist()

        if self.convergence:
            result["convergence"] = {
                "iterations": self.convergence.iterations,
                "converged": self.convergence.converged,
                "convergence_iteration": self.convergence.convergence_iteration,
                "improvement_rate": self.convergence.improvement_rate,
            }

        if self.fairness:
            result["fairness"] = {
                "demographic_divergence": self.fairness.demographic_divergence,
                "epsilon_satisfied": self.fairness.epsilon_satisfied,
                "coalition_distribution": self.fairness.coalition_distribution,
                "target_distribution": self.fairness.target_distribution,
            }

        result["config"] = self.config
        result["metadata"] = self.metadata

        return result

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(coalition_size={self.coalition_size}, "
            f"fitness={self.fitness:.4f}, converged={self.is_converged}, "
            f"fair={self.is_fair})"
        )
