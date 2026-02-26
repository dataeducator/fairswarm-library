"""
Shapley value computation for FairSwarm.

The Shapley value provides a fair allocation of rewards based on
each participant's marginal contribution.

Definition (Shapley Value):
    φ_i(v) = Σ_{S⊆N\\{i}} (|S|!(n-|S|-1)!/n!) [v(S∪{i}) - v(S)]

Properties:
    1. Efficiency: Σ φ_i = v(N)
    2. Symmetry: If v(S∪{i}) = v(S∪{j}) for all S, then φ_i = φ_j
    3. Null player: If v(S∪{i}) = v(S) for all S, then φ_i = 0
    4. Additivity: φ(v+w) = φ(v) + φ(w)

Author: Tenicka Norwood
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client
    from fairswarm.fitness.base import FitnessFunction


@dataclass
class ShapleyResult:
    """
    Result of Shapley value computation.

    Attributes:
        values: Shapley value for each player
        n_samples: Number of samples (for Monte Carlo)
        variance: Variance estimate (for Monte Carlo)
        computation_time: Time taken
    """

    values: NDArray[np.float64]
    n_samples: int = 0
    variance: NDArray[np.float64] | None = None
    computation_time: float = 0.0

    def normalize(self) -> NDArray[np.float64]:
        """Normalize values to sum to 1."""
        total: np.floating[Any] = np.sum(self.values)
        if total > 0:
            return self.values / total
        return self.values

    def get_ranking(self) -> list[int]:
        """Get player indices sorted by contribution (descending)."""
        return [int(x) for x in np.argsort(self.values)[::-1]]


class ShapleyValue(ABC):
    """
    Abstract base class for Shapley value computation.

    Subclasses implement different algorithms:
    - ExactShapley: O(n!) exact computation
    - MonteCarloShapley: O(m·n) approximation
    """

    @abstractmethod
    def compute(
        self,
        coalition: Coalition,
        clients: list[Client],
        value_fn: Callable[[Coalition, list[Client]], float],
    ) -> ShapleyResult:
        """
        Compute Shapley values for coalition members.

        Args:
            coalition: Indices of participating clients
            clients: All clients
            value_fn: Characteristic function v(S)

        Returns:
            ShapleyResult with values for each coalition member
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        pass


class ExactShapley(ShapleyValue):
    """
    Exact Shapley value computation.

    Computes exact Shapley values using the definition:
        φ_i = Σ_{S⊆N\\{i}} (|S|!(n-|S|-1)!/n!) [v(S∪{i}) - v(S)]

    Complexity: O(n!) where n is coalition size (permutation-based formula).
    Only feasible for small coalitions (n ≤ 12).

    Example:
        >>> shapley = ExactShapley()
        >>> result = shapley.compute(coalition, clients, value_fn)
        >>> print(f"Player 0 contribution: {result.values[0]:.4f}")
    """

    def __init__(self, max_size: int = 12):
        """
        Initialize ExactShapley.

        Args:
            max_size: Maximum coalition size (raises error if exceeded)
        """
        self.max_size = max_size

    def compute(
        self,
        coalition: Coalition,
        clients: list[Client],
        value_fn: Callable[[Coalition, list[Client]], float],
    ) -> ShapleyResult:
        """
        Compute exact Shapley values.

        Args:
            coalition: Indices of participating clients
            clients: All clients
            value_fn: Characteristic function v(S)

        Returns:
            ShapleyResult with exact Shapley values
        """
        import time

        start_time = time.time()

        n = len(coalition)

        if n > self.max_size:
            raise ValueError(
                f"Coalition size {n} exceeds max_size {self.max_size}. "
                f"Use MonteCarloShapley for large coalitions."
            )

        if n == 0:
            return ShapleyResult(
                values=np.array([]),
                n_samples=0,
                computation_time=time.time() - start_time,
            )

        # Initialize Shapley values
        shapley_values = np.zeros(n)

        # Compute using permutation formula
        # φ_i = (1/n!) Σ_{π} [v(S_π^i ∪ {i}) - v(S_π^i)]
        # where S_π^i are predecessors of i in permutation π

        n_factorial = factorial(n)
        n_permutations = 0

        for perm in permutations(range(n)):
            n_permutations += 1
            predecessors: set[int] = set()

            for player_local in perm:
                # v(S ∪ {i})
                coalition_with = list(predecessors) + [player_local]
                global_coalition_with = [coalition[j] for j in coalition_with]
                value_with = value_fn(global_coalition_with, clients)

                # v(S)
                if predecessors:
                    global_coalition_without = [coalition[j] for j in predecessors]
                    value_without = value_fn(global_coalition_without, clients)
                else:
                    value_without = 0.0

                # Marginal contribution
                marginal = value_with - value_without
                shapley_values[player_local] += marginal

                predecessors.add(player_local)

        # Average over all permutations
        shapley_values /= n_factorial

        return ShapleyResult(
            values=shapley_values,
            n_samples=n_factorial,
            computation_time=time.time() - start_time,
        )

    @property
    def name(self) -> str:
        return "ExactShapley"


class MonteCarloShapley(ShapleyValue):
    """
    Monte Carlo approximation of Shapley values.

    Approximates Shapley values by sampling random permutations:
        φ̂_i = (1/m) Σ_{k=1}^{m} [v(S_πk^i ∪ {i}) - v(S_πk^i)]

    Complexity: O(m·n) where m is number of samples.
    Suitable for any coalition size.

    Properties:
        - Unbiased estimator: E[φ̂_i] = φ_i
        - Variance decreases as O(1/m)

    Example:
        >>> shapley = MonteCarloShapley(n_samples=1000)
        >>> result = shapley.compute(coalition, clients, value_fn)
        >>> print(f"Estimated values: {result.values}")
        >>> print(f"Variance: {result.variance}")
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int | None = None,
    ):
        """
        Initialize MonteCarloShapley.

        Args:
            n_samples: Number of permutation samples
            seed: Random seed for reproducibility
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute(
        self,
        coalition: Coalition,
        clients: list[Client],
        value_fn: Callable[[Coalition, list[Client]], float],
    ) -> ShapleyResult:
        """
        Compute Monte Carlo approximation of Shapley values.

        Args:
            coalition: Indices of participating clients
            clients: All clients
            value_fn: Characteristic function v(S)

        Returns:
            ShapleyResult with estimated Shapley values and variance
        """
        import time

        # Reset RNG for reproducibility across repeated calls
        self.rng = np.random.default_rng(self.seed)

        start_time = time.time()

        n = len(coalition)

        if n == 0:
            return ShapleyResult(
                values=np.array([]),
                n_samples=0,
                computation_time=time.time() - start_time,
            )

        # Track marginal contributions for variance estimation
        marginals = np.zeros((self.n_samples, n))

        for sample in range(self.n_samples):
            # Random permutation
            perm = self.rng.permutation(n)
            predecessors: set[int] = set()

            for _pos, player_local in enumerate(perm):
                # v(S ∪ {i})
                coalition_with_local = list(predecessors) + [player_local]
                coalition_with = [coalition[j] for j in coalition_with_local]
                value_with = value_fn(coalition_with, clients)

                # v(S)
                if predecessors:
                    coalition_without = [coalition[j] for j in predecessors]
                    value_without = value_fn(coalition_without, clients)
                else:
                    value_without = 0.0

                # Marginal contribution
                marginals[sample, player_local] = value_with - value_without

                predecessors.add(player_local)

        # Estimate Shapley values (mean of marginals)
        shapley_values = np.mean(marginals, axis=0)

        # Estimate variance
        variance = np.var(marginals, axis=0, ddof=1) / self.n_samples

        return ShapleyResult(
            values=shapley_values,
            n_samples=self.n_samples,
            variance=variance,
            computation_time=time.time() - start_time,
        )

    @property
    def name(self) -> str:
        return "MonteCarloShapley"


class StratifiedShapley(ShapleyValue):
    """
    Stratified sampling for Shapley value estimation.

    Uses stratified sampling based on coalition sizes to reduce variance.

    For each size k, samples permutations where player i is at position k.
    """

    def __init__(
        self,
        samples_per_stratum: int = 100,
        seed: int | None = None,
    ):
        """
        Initialize StratifiedShapley.

        Args:
            samples_per_stratum: Samples per coalition size
            seed: Random seed
        """
        self.samples_per_stratum = samples_per_stratum
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute(
        self,
        coalition: Coalition,
        clients: list[Client],
        value_fn: Callable[[Coalition, list[Client]], float],
    ) -> ShapleyResult:
        """
        Compute Shapley values using stratified sampling.

        Args:
            coalition: Indices of participating clients
            clients: All clients
            value_fn: Characteristic function v(S)

        Returns:
            ShapleyResult with estimated values
        """
        import time

        start_time = time.time()

        n = len(coalition)

        if n == 0:
            return ShapleyResult(
                values=np.array([]),
                n_samples=0,
                computation_time=time.time() - start_time,
            )

        shapley_values = np.zeros(n)
        total_samples = 0

        # For each player
        for player_local in range(n):
            player_value = 0.0

            # For each coalition size (stratum)
            for size in range(n):
                stratum_value = 0.0
                others = [i for i in range(n) if i != player_local]

                # Skip invalid strata (can't have more predecessors than
                # available players)
                if size > len(others):
                    continue

                for _ in range(self.samples_per_stratum):
                    # Sample random subset of size `size` from others
                    if size > 0:
                        subset_local = self.rng.choice(
                            others, size=size, replace=False
                        ).tolist()
                    else:
                        subset_local = []

                    # v(S ∪ {i})
                    coalition_with_local = subset_local + [player_local]
                    coalition_with = [coalition[j] for j in coalition_with_local]
                    value_with = value_fn(coalition_with, clients)

                    # v(S)
                    if subset_local:
                        coalition_without = [coalition[j] for j in subset_local]
                        value_without = value_fn(coalition_without, clients)
                    else:
                        value_without = 0.0

                    stratum_value += value_with - value_without
                    total_samples += 1

                stratum_value /= self.samples_per_stratum
                player_value += stratum_value

            shapley_values[player_local] = player_value / n

        return ShapleyResult(
            values=shapley_values,
            n_samples=total_samples,
            computation_time=time.time() - start_time,
        )

    @property
    def name(self) -> str:
        return "StratifiedShapley"


def compute_shapley_values(
    coalition: Coalition,
    clients: list[Client],
    value_fn: Callable[[Coalition, list[Client]], float],
    method: str = "auto",
    n_samples: int = 1000,
    seed: int | None = None,
) -> ShapleyResult:
    """
    Compute Shapley values with automatic method selection.

    Args:
        coalition: Indices of participating clients
        clients: All clients
        value_fn: Characteristic function v(S)
        method: "exact", "monte_carlo", or "auto"
        n_samples: Samples for Monte Carlo
        seed: Random seed

    Returns:
        ShapleyResult

    Example:
        >>> def value_fn(coalition, clients):
        ...     return sum(clients[i].data_quality for i in coalition)
        >>> result = compute_shapley_values(
        ...     coalition=[0, 1, 2],
        ...     clients=clients,
        ...     value_fn=value_fn,
        ... )
    """
    n = len(coalition)

    if method == "auto":
        # Use exact for small coalitions, Monte Carlo for large
        method = "exact" if n <= 10 else "monte_carlo"

    computer: ShapleyValue
    if method == "exact":
        computer = ExactShapley(max_size=max(n, 12))
    else:
        computer = MonteCarloShapley(n_samples=n_samples, seed=seed)

    return computer.compute(coalition, clients, value_fn)


def shapley_from_fitness(
    coalition: Coalition,
    clients: list[Client],
    fitness_fn: FitnessFunction,
) -> ShapleyResult:
    """
    Compute Shapley values using a fitness function.

    Convenience wrapper that uses fitness function as characteristic function.

    Args:
        coalition: Coalition indices
        clients: All clients
        fitness_fn: FitnessFunction instance

    Returns:
        ShapleyResult
    """

    def value_fn(subset: Coalition, all_clients: list[Client]) -> float:
        if not subset:
            return 0.0
        result = fitness_fn.evaluate(subset, all_clients)
        return result.value

    return compute_shapley_values(coalition, clients, value_fn)
