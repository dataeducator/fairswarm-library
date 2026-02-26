"""
Swarm manager for FairSwarm PSO.

This module implements the Swarm class that manages a collection of
particles and tracks global best solutions.

Algorithm Reference:
    The swarm manages P particles and tracks the global best position
    gBest that represents the best solution found by any particle.

Author: Tenicka Norwood
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.particle import Particle
from fairswarm.core.position import decode_coalition
from fairswarm.types import Coalition


@dataclass
class Swarm:
    """
    A swarm of particles for FairSwarm optimization.

    The swarm maintains a collection of particles and tracks the global
    best position found by any particle. It provides methods for
    initialization, iteration, and global best management.

    Attributes:
        particles: List of particles in the swarm
        g_best: Global best position found by any particle
        g_best_fitness: Fitness value at global best
        g_best_coalition: Coalition decoded from global best
        iteration: Current iteration number

    Algorithm Reference:
        gBest ← argmax_p pBestFit_p

    Example:
        >>> from fairswarm.core.swarm import Swarm
        >>>
        >>> swarm = Swarm.initialize(
        ...     swarm_size=30,
        ...     n_clients=20,
        ...     seed=42,
        ... )
        >>> print(f"Swarm has {len(swarm)} particles")
        >>> best = swarm.get_global_best_coalition(coalition_size=5)
    """

    particles: list[Particle]
    g_best: NDArray[np.float64] | None = field(default=None)
    g_best_fitness: float = field(default=float("-inf"))
    g_best_coalition: Coalition | None = field(default=None)
    iteration: int = field(default=0)

    @property
    def size(self) -> int:
        """Number of particles in the swarm."""
        return len(self.particles)

    @property
    def n_clients(self) -> int:
        """Number of clients (dimension of position vectors)."""
        if self.particles:
            return self.particles[0].n_clients
        return 0

    def __len__(self) -> int:
        """Number of particles."""
        return self.size

    def __iter__(self) -> Iterator[Particle]:
        """Iterate over particles."""
        return iter(self.particles)

    def __getitem__(self, index: int) -> Particle:
        """Get particle by index."""
        return self.particles[index]

    def update_global_best(
        self,
        coalition_size: int | None = None,
    ) -> bool:
        """
        Update global best from current particle personal bests.

        Finds the particle with the best personal best fitness and
        updates the swarm's global best if it's better.

        Args:
            coalition_size: If provided, decode and cache the coalition

        Returns:
            True if global best was updated, False otherwise

        Algorithm Reference:
            gBest ← argmax_p pBestFit_p
        """
        best_particle = max(self.particles, key=lambda p: p.p_best_fitness)

        if best_particle.p_best_fitness > self.g_best_fitness:
            self.g_best = best_particle.p_best.copy()
            self.g_best_fitness = best_particle.p_best_fitness

            # Cache coalition if size provided
            if coalition_size is not None:
                self.g_best_coalition = decode_coalition(self.g_best, coalition_size)
            elif best_particle.p_best_coalition is not None:
                self.g_best_coalition = list(best_particle.p_best_coalition)

            return True

        return False

    def get_global_best_coalition(self, coalition_size: int) -> Coalition:
        """
        Decode global best position to coalition.

        Args:
            coalition_size: Number of clients to select

        Returns:
            Coalition decoded from global best position

        Raises:
            ValueError: If no global best exists yet

        Algorithm Reference:
            Return SelectTop(gBest, m)
        """
        if self.g_best is None:
            raise ValueError("No global best found yet")

        return decode_coalition(self.g_best, coalition_size)

    def get_statistics(self) -> dict[str, object]:
        """
        Get statistics about the current swarm state.

        Returns:
            Dictionary with swarm statistics
        """
        if not self.particles:
            return {"size": 0}

        fitnesses = [p.p_best_fitness for p in self.particles]
        valid_fitnesses = [f for f in fitnesses if f != float("-inf")]

        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])

        return {
            "size": self.size,
            "iteration": self.iteration,
            "g_best_fitness": self.g_best_fitness,
            "mean_p_best_fitness": (
                np.mean(valid_fitnesses) if valid_fitnesses else float("-inf")
            ),
            "std_p_best_fitness": (
                np.std(valid_fitnesses) if len(valid_fitnesses) > 1 else 0.0
            ),
            "position_mean": float(np.mean(positions)),
            "position_std": float(np.std(positions)),
            "velocity_mean": float(np.mean(np.abs(velocities))),
            "velocity_max": float(np.max(np.abs(velocities))),
        }

    def get_diversity(self) -> float:
        """
        Compute swarm diversity based on position variance.

        Higher diversity means particles are spread across the search space.
        Low diversity may indicate premature convergence.

        Returns:
            Diversity measure (average standard deviation of positions)
        """
        if len(self.particles) < 2:
            return 0.0

        positions = np.array([p.position for p in self.particles])

        # Average std across dimensions
        return float(np.mean(np.std(positions, axis=0)))

    def get_convergence_ratio(self, threshold: float = 0.1) -> float:
        """
        Compute ratio of particles near the global best.

        Args:
            threshold: Maximum distance to be considered "converged"

        Returns:
            Ratio of particles within threshold of global best
        """
        if self.g_best is None or not self.particles:
            return 0.0

        converged = 0
        for particle in self.particles:
            distance = np.linalg.norm(particle.position - self.g_best)
            if distance < threshold:
                converged += 1

        return converged / len(self.particles)

    def reset_velocities(self, scale: float = 0.5) -> None:
        """
        Reset particle velocities to promote exploration.

        Useful when swarm has converged prematurely.

        Args:
            scale: Scale factor for new random velocities
        """
        rng = np.random.default_rng()
        for particle in self.particles:
            particle.velocity = rng.uniform(-scale, scale, size=particle.n_clients)

    def inject_diversity(
        self,
        n_particles: int = 5,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Replace worst particles with new random particles.

        This helps escape local optima by injecting fresh genetic material.

        Args:
            n_particles: Number of particles to replace
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()

        n_particles = min(n_particles, len(self.particles))

        # Sort by fitness (ascending, so worst are first)
        sorted_particles = sorted(
            enumerate(self.particles),
            key=lambda x: x[1].p_best_fitness,
        )

        # Replace worst particles
        for i in range(n_particles):
            idx = sorted_particles[i][0]
            self.particles[idx] = Particle.initialize(
                n_clients=self.n_clients,
                rng=rng,
            )

    @classmethod
    def initialize(
        cls,
        swarm_size: int,
        n_clients: int,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> Swarm:
        """
        Initialize a swarm with random particles.

        Args:
            swarm_size: Number of particles (P)
            n_clients: Number of clients (n)
            rng: Random number generator
            seed: Random seed (used if rng not provided)

        Returns:
            New Swarm with initialized particles

        Algorithm Reference (Initialization):
            For each particle p ∈ {1, ..., P}:
                x_p ← random vector in [0,1]^n
                v_p ← random vector in [-1,1]^n
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        particles = [
            Particle.initialize(n_clients=n_clients, rng=rng) for _ in range(swarm_size)
        ]

        return cls(particles=particles)

    @classmethod
    def initialize_with_seed_coalitions(
        cls,
        swarm_size: int,
        n_clients: int,
        seed_coalitions: list[Coalition],
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> Swarm:
        """
        Initialize a swarm with some particles seeded from known coalitions.

        This can improve convergence by starting with domain knowledge.

        Args:
            swarm_size: Total number of particles
            n_clients: Number of clients
            seed_coalitions: Known good coalitions to seed
            rng: Random number generator
            seed: Random seed

        Returns:
            Swarm with some particles initialized near seed coalitions
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        particles = []

        # Seed particles from known coalitions
        for coalition in seed_coalitions[: swarm_size // 2]:
            particle = Particle.initialize_with_bias(
                n_clients=n_clients,
                bias_indices=coalition,
                bias_strength=0.3,
                rng=rng,
            )
            particles.append(particle)

        # Fill remaining with random particles
        while len(particles) < swarm_size:
            particles.append(Particle.initialize(n_clients=n_clients, rng=rng))

        return cls(particles=particles)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Swarm(size={self.size}, n_clients={self.n_clients}, "
            f"g_best_fitness={self.g_best_fitness:.4f}, "
            f"iteration={self.iteration})"
        )


@dataclass
class SwarmHistory:
    """
    Tracks optimization history for analysis and visualization.

    Attributes:
        fitness_history: Global best fitness at each iteration
        fairness_history: Demographic divergence at each iteration
        diversity_history: Swarm diversity at each iteration
        coalition_history: Global best coalition at each iteration
    """

    fitness_history: list[float] = field(default_factory=list)
    fairness_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    coalition_history: list[Coalition] = field(default_factory=list)

    def record(
        self,
        fitness: float,
        fairness: float,
        diversity: float,
        coalition: Coalition | None = None,
    ) -> None:
        """Record metrics for current iteration."""
        self.fitness_history.append(fitness)
        self.fairness_history.append(fairness)
        self.diversity_history.append(diversity)
        if coalition is not None:
            self.coalition_history.append(coalition)

    @property
    def n_iterations(self) -> int:
        """Number of recorded iterations."""
        return len(self.fitness_history)

    def as_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Convert histories to numpy arrays."""
        return {
            "fitness": np.array(self.fitness_history, dtype=np.float64),
            "fairness": np.array(self.fairness_history, dtype=np.float64),
            "diversity": np.array(self.diversity_history, dtype=np.float64),
        }
