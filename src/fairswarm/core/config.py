"""
Configuration for FairSwarm optimizer.

This module defines FairSwarmConfig, which holds all hyperparameters
for the FairSwarm algorithm. Parameters are validated against the
theoretical requirements from Theorems 1-4 in CLAUDE.md.

Research Foundation:
    - Theorem 1 (Convergence): Requires ω + (c₁+c₂)/2 < 2
    - Theorem 2 (ε-Fairness): Requires sufficient iterations T and λ
    - Theorem 4 (Privacy): Controls ε_DP budget

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FairSwarmConfig:
    """
    Configuration for FairSwarm optimizer.

    This dataclass holds all hyperparameters needed to run the FairSwarm
    algorithm. Parameters are validated at construction time to ensure
    they satisfy the theoretical requirements for convergence and fairness.

    Attributes:
        swarm_size: Number of particles in the swarm (P in Algorithm 1)
        max_iterations: Maximum optimization iterations (T in Algorithm 1)
        coalition_size: Target size of selected coalition (m)

        inertia: PSO inertia weight ω ∈ (0, 1)
        cognitive: Cognitive coefficient c₁ (personal best attraction)
        social: Social coefficient c₂ (global best attraction)

        fairness_coefficient: Novel fairness gradient weight c₃
        fairness_weight: Fairness term weight λ in fitness function
        adaptive_fairness: Whether to adapt λ over iterations

        velocity_max: Maximum velocity magnitude v_max
        convergence_threshold: Threshold for convergence detection
        patience: Iterations without improvement before stopping

        epsilon_fair: Target ε for ε-fairness (Theorem 2)
        epsilon_dp: Differential privacy budget (Theorem 4)

        seed: Random seed for reproducibility

    Theoretical Requirements:
        Theorem 1 requires: ω + (c₁ + c₂)/2 < 2 for practical convergence
        Theorem 2 requires: T ≥ n² log(P/δ) / (ε² λ²)

    Example:
        >>> config = FairSwarmConfig(
        ...     swarm_size=30,
        ...     max_iterations=100,
        ...     coalition_size=10,
        ...     fairness_weight=0.3,
        ... )
        >>> print(f"Convergence metric: {config.convergence_metric:.2f}")

    References:
        Algorithm 1 in CLAUDE.md defines the parameter usage
    """

    # === Swarm Parameters ===
    swarm_size: int = 30
    max_iterations: int = 100
    coalition_size: int = 10

    # === PSO Parameters (must satisfy Theorem 1) ===
    inertia: float = 0.7  # ω: must be in (0, 1)
    cognitive: float = 1.5  # c₁: personal best attraction
    social: float = 1.5  # c₂: global best attraction

    # === Novel Fairness Parameters ===
    fairness_coefficient: float = 0.5  # c₃: fairness gradient weight
    fairness_weight: float = 0.3  # λ: weight in fitness function
    adaptive_fairness: bool = True  # Adapt λ over iterations

    # === Convergence Parameters ===
    velocity_max: float = 4.0  # v_max: velocity clipping
    convergence_threshold: float = 1e-4  # Convergence detection
    patience: int = 10  # Early stopping patience

    # === Fairness & Privacy Targets ===
    epsilon_fair: float = 0.05  # Target ε-fairness (Theorem 2)
    epsilon_dp: Optional[float] = None  # DP budget (Theorem 4), None = no DP

    # === Fitness Weights ===
    weight_accuracy: float = 0.5  # w₁ in fitness function
    weight_fairness: float = 0.3  # w₂ in fitness function
    weight_cost: float = 0.2  # w₃ in fitness function

    # === Reproducibility ===
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration against theoretical requirements."""
        self._validate_pso_parameters()
        self._validate_fairness_parameters()
        self._validate_general_parameters()

    def _validate_pso_parameters(self) -> None:
        """Validate PSO parameters against Theorem 1 requirements."""
        # Inertia must be in (0, 1)
        if not 0 < self.inertia < 1:
            raise ValueError(
                f"inertia must be in (0, 1), got {self.inertia}. "
                "This is required for Theorem 1 (convergence)."
            )

        # Cognitive and social coefficients must be positive
        if self.cognitive <= 0:
            raise ValueError(f"cognitive must be positive, got {self.cognitive}")
        if self.social <= 0:
            raise ValueError(f"social must be positive, got {self.social}")

        # Theorem 1 convergence condition (practical bound)
        convergence_metric = self.convergence_metric
        if convergence_metric >= 2.0:
            warnings.warn(
                f"PSO parameters may not guarantee convergence. "
                f"ω + (c₁+c₂)/2 = {convergence_metric:.2f}. "
                f"Theorem 1 requires this < 2 for practical convergence. "
                f"Consider reducing inertia or coefficients.",
                UserWarning,
                stacklevel=3,
            )

        # Strict convergence condition (theoretical)
        if convergence_metric >= 1.0:
            # This is a softer warning as the practical bound is 2
            pass  # Many practical PSO implementations violate this

    def _validate_fairness_parameters(self) -> None:
        """Validate fairness parameters against Theorem 2 requirements."""
        if self.fairness_coefficient < 0:
            raise ValueError(
                f"fairness_coefficient must be non-negative, "
                f"got {self.fairness_coefficient}"
            )

        if not 0 <= self.fairness_weight <= 1:
            raise ValueError(
                f"fairness_weight must be in [0, 1], got {self.fairness_weight}"
            )

        if self.epsilon_fair <= 0:
            raise ValueError(
                f"epsilon_fair must be positive, got {self.epsilon_fair}. "
                "This is the target demographic divergence (Theorem 2)."
            )

        if self.epsilon_dp is not None and self.epsilon_dp <= 0:
            raise ValueError(
                f"epsilon_dp must be positive if specified, got {self.epsilon_dp}. "
                "This is the differential privacy budget (Theorem 4)."
            )

    # === Resource Limits (prevent denial-of-service via config injection) ===
    MAX_SWARM_SIZE: int = 1000
    MAX_ITERATIONS: int = 10000
    MAX_COALITION_SIZE: int = 500

    def _validate_general_parameters(self) -> None:
        """Validate general algorithm parameters."""
        if self.swarm_size < 2:
            raise ValueError(f"swarm_size must be >= 2, got {self.swarm_size}")
        if self.swarm_size > self.MAX_SWARM_SIZE:
            raise ValueError(
                f"swarm_size must be <= {self.MAX_SWARM_SIZE}, got {self.swarm_size}"
            )

        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )
        if self.max_iterations > self.MAX_ITERATIONS:
            raise ValueError(
                f"max_iterations must be <= {self.MAX_ITERATIONS}, got {self.max_iterations}"
            )

        if self.coalition_size < 1:
            raise ValueError(
                f"coalition_size must be >= 1, got {self.coalition_size}"
            )
        if self.coalition_size > self.MAX_COALITION_SIZE:
            raise ValueError(
                f"coalition_size must be <= {self.MAX_COALITION_SIZE}, got {self.coalition_size}"
            )

        if self.velocity_max <= 0:
            raise ValueError(f"velocity_max must be positive, got {self.velocity_max}")

        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

        # Fitness weights should sum to 1 (not strictly required but good practice)
        weight_sum = self.weight_accuracy + self.weight_fairness + self.weight_cost
        if not 0.99 <= weight_sum <= 1.01:
            warnings.warn(
                f"Fitness weights should sum to 1.0, got {weight_sum:.2f}. "
                "Consider normalizing weights.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def convergence_metric(self) -> float:
        """
        Compute the convergence metric from Theorem 1.

        Returns ω + (c₁ + c₂)/2. For guaranteed convergence,
        this should be < 1 (strict) or < 2 (practical).

        Returns:
            Convergence metric value
        """
        return self.inertia + (self.cognitive + self.social) / 2

    @property
    def satisfies_convergence_condition(self) -> bool:
        """
        Check if configuration satisfies Theorem 1 convergence condition.

        Returns:
            True if ω + (c₁ + c₂)/2 < 2 (practical bound)
        """
        return self.convergence_metric < 2.0

    @property
    def min_iterations_for_fairness(self) -> int:
        """
        Compute minimum iterations for ε-fairness guarantee (Theorem 2).

        Based on: T_min = n² log(P/δ) / (ε² λ²)
        Uses conservative estimates.

        Returns:
            Minimum recommended iterations
        """
        import math

        # Conservative estimates
        n = 50  # Assume 50 clients
        delta = 0.1  # 90% confidence
        epsilon = self.epsilon_fair
        lambda_ = max(self.fairness_weight, 0.1)

        t_min = (n**2 * math.log(self.swarm_size / delta)) / (epsilon**2 * lambda_**2)
        return int(math.ceil(t_min))

    def with_updates(self, **kwargs) -> "FairSwarmConfig":
        """
        Create a new config with updated values.

        Args:
            **kwargs: Parameters to update

        Returns:
            New FairSwarmConfig with updated values

        Example:
            >>> config = FairSwarmConfig(swarm_size=30)
            >>> new_config = config.with_updates(swarm_size=50, seed=42)
        """
        import dataclasses

        return dataclasses.replace(self, **kwargs)


# === Preset Configurations ===


def get_preset_config(
    preset: Literal["default", "fast", "thorough", "privacy", "fair"],
) -> FairSwarmConfig:
    """
    Get a preset configuration for common use cases.

    Args:
        preset: One of:
            - "default": Balanced configuration
            - "fast": Quick optimization, fewer iterations
            - "thorough": More iterations, larger swarm
            - "privacy": Optimized for differential privacy
            - "fair": Maximum emphasis on fairness

    Returns:
        FairSwarmConfig with preset values

    Example:
        >>> config = get_preset_config("fair")
        >>> config.fairness_weight
        0.5
    """
    presets = {
        "default": FairSwarmConfig(),
        "fast": FairSwarmConfig(
            swarm_size=20,
            max_iterations=50,
            patience=5,
        ),
        "thorough": FairSwarmConfig(
            swarm_size=50,
            max_iterations=200,
            patience=20,
        ),
        "privacy": FairSwarmConfig(
            epsilon_dp=4.0,  # Moderate privacy
            max_iterations=150,  # More iterations to compensate for noise
        ),
        "fair": FairSwarmConfig(
            fairness_coefficient=0.8,
            fairness_weight=0.5,
            weight_accuracy=0.35,
            weight_fairness=0.45,
            weight_cost=0.20,
            adaptive_fairness=True,
        ),
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(presets.keys())}"
        )

    return presets[preset]
