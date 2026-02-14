"""
Particle representation for FairSwarm PSO.

This module implements the Particle class used in the FairSwarm algorithm.
Each particle represents a candidate coalition selection in the search space.

Algorithm Reference (Algorithm 1 in CLAUDE.md):
    Initialize:
        For each particle p ∈ {1, ..., P}:
            xₚ ← random vector in [0,1]ⁿ      // Position (selection probabilities)
            vₚ ← random vector in [-1,1]ⁿ     // Velocity
            Sₚ ← SelectTop(xₚ, m)             // Decode to coalition
            pBestₚ ← xₚ                       // Personal best position
            pBestFitₚ ← Fitness(Sₚ)           // Personal best fitness

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.position import decode_coalition, sigmoid
from fairswarm.types import Coalition


@dataclass
class Particle:
    """
    A particle in the FairSwarm PSO algorithm.

    Each particle represents a point in the search space, where the position
    encodes selection probabilities for each client. The velocity determines
    how the particle moves through the search space.

    Attributes:
        position: Selection probabilities for each client, x_p ∈ [0,1]^n
        velocity: Rate of change for position, v_p ∈ [-v_max, v_max]^n
        p_best: Personal best position found by this particle
        p_best_fitness: Fitness value at personal best position
        p_best_coalition: Coalition decoded from personal best (cached)

    Mathematical Notation (Algorithm 1):
        - position → x_p
        - velocity → v_p
        - p_best → pBest_p
        - p_best_fitness → pBestFit_p

    Example:
        >>> import numpy as np
        >>> from fairswarm.core.particle import Particle
        >>>
        >>> particle = Particle.initialize(n_clients=20, seed=42)
        >>> coalition = particle.decode(coalition_size=5)
        >>> print(f"Selected clients: {coalition}")

    Research Reference:
        Algorithm 1 in CLAUDE.md defines the particle structure and update rules.
    """

    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    p_best: NDArray[np.float64]
    p_best_fitness: float = float("-inf")
    p_best_coalition: Coalition | None = field(default=None)

    @property
    def n_clients(self) -> int:
        """Number of clients (dimension of position/velocity vectors)."""
        return len(self.position)

    def decode(self, coalition_size: int) -> Coalition:
        """
        Decode current position to a coalition.

        Uses SelectTop to select the m clients with highest position values.

        Args:
            coalition_size: Number of clients to select (m)

        Returns:
            List of client indices forming the coalition

        Algorithm Reference:
            S_p ← SelectTop(x_p, m)
        """
        return decode_coalition(self.position, coalition_size)

    def update_personal_best(
        self,
        fitness: float,
        coalition: Coalition | None = None,
    ) -> bool:
        """
        Update personal best if current position is better.

        Args:
            fitness: Fitness value at current position
            coalition: Coalition at current position (optional, for caching)

        Returns:
            True if personal best was updated, False otherwise

        Algorithm Reference:
            If fit_p > pBestFit_p:
                pBest_p ← x_p
                pBestFit_p ← fit_p
        """
        if fitness > self.p_best_fitness:
            self.p_best = self.position.copy()
            self.p_best_fitness = fitness
            self.p_best_coalition = coalition
            return True
        return False

    def apply_velocity_update(
        self,
        inertia: float,
        cognitive: float,
        social: float,
        fairness_coeff: float,
        g_best: NDArray[np.float64] | None,
        fairness_gradient: NDArray[np.float64],
        velocity_max: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Apply the FairSwarm velocity update rule.

        This implements the novel fairness-aware velocity update from Algorithm 1:
            v_p ← ω·v_p + c₁·r₁·(pBest - x) + c₂·r₂·(gBest - x) + c₃·∇_fair

        Args:
            inertia: Inertia weight ω
            cognitive: Cognitive coefficient c₁
            social: Social coefficient c₂
            fairness_coeff: Fairness gradient coefficient c₃
            g_best: Global best position (None if not yet found)
            fairness_gradient: Fairness gradient ∇_fair
            velocity_max: Maximum velocity magnitude v_max
            rng: Random number generator

        Algorithm Reference (Lines 554-567 of Algorithm 1):
            r₁, r₂ ← Uniform(0,1)
            v_cognitive ← c₁ · r₁ · (pBest_p - x_p)
            v_social ← c₂ · r₂ · (gBest - x_p)
            v_fairness ← c₃ · ∇_fair
            v_p ← ω · v_p + v_cognitive + v_social + v_fairness
            v_p ← Clamp(v_p, -v_max, v_max)
        """
        r1, r2 = rng.random(2)

        # Cognitive component: attraction to personal best
        v_cognitive = cognitive * r1 * (self.p_best - self.position)

        # Social component: attraction to global best
        if g_best is not None:
            v_social = social * r2 * (g_best - self.position)
        else:
            v_social = np.zeros_like(self.position)

        # NOVEL: Fairness gradient component
        v_fairness = fairness_coeff * fairness_gradient

        # Combined velocity update
        self.velocity = inertia * self.velocity + v_cognitive + v_social + v_fairness

        # Clamp velocity to [-v_max, v_max]
        self.velocity = np.clip(self.velocity, -velocity_max, velocity_max)

    def apply_position_update(self) -> None:
        """
        Apply position update with sigmoid bounding.

        This updates the position based on velocity and applies sigmoid
        to bound values to [0, 1].

        Algorithm Reference (Lines 569-571 of Algorithm 1):
            x_p ← x_p + v_p
            x_p ← Sigmoid(x_p)
        """
        self.position = self.position + self.velocity
        self.position = sigmoid(self.position)

    @classmethod
    def initialize(
        cls,
        n_clients: int,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> Particle:
        """
        Initialize a particle with random position and velocity.

        Args:
            n_clients: Number of clients (dimension n)
            rng: Random number generator (preferred)
            seed: Random seed (used if rng not provided)

        Returns:
            New Particle with random initialization

        Algorithm Reference (Initialization phase):
            x_p ← random vector in [0,1]^n
            v_p ← random vector in [-1,1]^n
            pBest_p ← x_p
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        # Position: random in [0, 1]^n
        position = rng.random(n_clients)

        # Velocity: random in [-1, 1]^n
        velocity = rng.uniform(-1.0, 1.0, n_clients)

        # Personal best initialized to current position
        p_best = position.copy()

        return cls(
            position=position,
            velocity=velocity,
            p_best=p_best,
            p_best_fitness=float("-inf"),
            p_best_coalition=None,
        )

    @classmethod
    def initialize_with_bias(
        cls,
        n_clients: int,
        bias_indices: list[int],
        bias_strength: float = 0.3,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> Particle:
        """
        Initialize a particle with bias toward certain clients.

        This can be used to seed the swarm with domain knowledge about
        promising clients.

        Args:
            n_clients: Number of clients
            bias_indices: Indices of clients to bias toward
            bias_strength: How much to boost bias indices (added to position)
            rng: Random number generator
            seed: Random seed

        Returns:
            Particle biased toward specified clients
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        # Start with random position
        position = rng.random(n_clients)

        # Add bias to specified indices
        for idx in bias_indices:
            if 0 <= idx < n_clients:
                position[idx] = min(1.0, position[idx] + bias_strength)

        # Velocity still random
        velocity = rng.uniform(-1.0, 1.0, n_clients)

        return cls(
            position=position,
            velocity=velocity,
            p_best=position.copy(),
            p_best_fitness=float("-inf"),
            p_best_coalition=None,
        )

    def copy(self) -> Particle:
        """Create a deep copy of this particle."""
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            p_best=self.p_best.copy(),
            p_best_fitness=self.p_best_fitness,
            p_best_coalition=(
                list(self.p_best_coalition) if self.p_best_coalition else None
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Particle(n_clients={self.n_clients}, "
            f"p_best_fitness={self.p_best_fitness:.4f})"
        )
