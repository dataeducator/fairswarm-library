"""
Reward allocation mechanisms for FairSwarm.

This module provides mechanisms for distributing rewards/payments
to coalition members based on their contributions.

Key Allocators:
    - EqualAllocator: Equal shares to all members
    - ProportionalAllocator: Proportional to individual metrics
    - ShapleyAllocator: Based on Shapley values

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fairswarm.incentives.shapley import compute_shapley_values
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client
    from fairswarm.demographics.distribution import DemographicDistribution


@dataclass
class ContributionMetrics:
    """
    Metrics for measuring client contributions.

    Attributes:
        data_contribution: Based on data size/quality
        computation_contribution: Based on compute resources
        communication_contribution: Based on communication cost
        fairness_contribution: Based on demographic coverage
        total_contribution: Weighted sum of contributions
    """

    data_contribution: float = 0.0
    computation_contribution: float = 0.0
    communication_contribution: float = 0.0
    fairness_contribution: float = 0.0
    total_contribution: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """
    Result of reward allocation.

    Attributes:
        allocations: Reward for each coalition member
        total_reward: Total reward distributed
        allocation_method: Name of allocation method
        metrics: Per-client contribution metrics
    """

    allocations: dict[int, float]
    total_reward: float
    allocation_method: str
    metrics: dict[int, ContributionMetrics] | None = None

    def get_shares(self) -> dict[int, float]:
        """Get allocation as fractions of total."""
        if self.total_reward <= 0:
            return dict.fromkeys(self.allocations, 0.0)
        return {k: v / self.total_reward for k, v in self.allocations.items()}

    def get_ranking(self) -> list[int]:
        """Get client indices sorted by allocation (descending)."""
        return sorted(self.allocations.keys(), key=lambda k: -self.allocations[k])


class RewardAllocator(ABC):
    """
    Abstract base class for reward allocation.

    Allocators distribute a total reward among coalition members
    based on their contributions.
    """

    @abstractmethod
    def allocate(
        self,
        coalition: Coalition,
        clients: list[Client],
        total_reward: float,
    ) -> AllocationResult:
        """
        Allocate reward among coalition members.

        Args:
            coalition: Indices of participating clients
            clients: All clients
            total_reward: Total reward to distribute

        Returns:
            AllocationResult with per-client allocations
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Allocator name."""
        pass


class EqualAllocator(RewardAllocator):
    """
    Equal allocation to all coalition members.

    Each member receives: reward_i = total_reward / n

    Simple but doesn't account for differing contributions.

    Example:
        >>> allocator = EqualAllocator()
        >>> result = allocator.allocate([0, 1, 2], clients, total_reward=100)
        >>> # Each client gets 33.33
    """

    def allocate(
        self,
        coalition: Coalition,
        clients: list[Client],
        total_reward: float,
    ) -> AllocationResult:
        """
        Allocate equal rewards.

        Args:
            coalition: Participating clients
            clients: All clients
            total_reward: Total reward

        Returns:
            AllocationResult with equal shares
        """
        n = len(coalition)
        if n == 0:
            return AllocationResult(
                allocations={},
                total_reward=0.0,
                allocation_method=self.name,
            )

        share = total_reward / n
        allocations = dict.fromkeys(coalition, share)

        return AllocationResult(
            allocations=allocations,
            total_reward=total_reward,
            allocation_method=self.name,
        )

    @property
    def name(self) -> str:
        return "Equal"


class ProportionalAllocator(RewardAllocator):
    """
    Proportional allocation based on client metrics.

    Reward is proportional to a weighted combination of:
    - Data size
    - Data quality
    - Communication efficiency (inverse cost)

    reward_i = total_reward * (score_i / Σ scores)

    Attributes:
        data_weight: Weight for data size
        quality_weight: Weight for data quality
        efficiency_weight: Weight for communication efficiency

    Example:
        >>> allocator = ProportionalAllocator(
        ...     data_weight=0.5,
        ...     quality_weight=0.3,
        ...     efficiency_weight=0.2,
        ... )
        >>> result = allocator.allocate(coalition, clients, total_reward=100)
    """

    def __init__(
        self,
        data_weight: float = 0.4,
        quality_weight: float = 0.4,
        efficiency_weight: float = 0.2,
    ):
        """
        Initialize ProportionalAllocator.

        Args:
            data_weight: Weight for data size
            quality_weight: Weight for data quality
            efficiency_weight: Weight for communication efficiency
        """
        total = data_weight + quality_weight + efficiency_weight
        self.data_weight = data_weight / total
        self.quality_weight = quality_weight / total
        self.efficiency_weight = efficiency_weight / total

    def allocate(
        self,
        coalition: Coalition,
        clients: list[Client],
        total_reward: float,
    ) -> AllocationResult:
        """
        Allocate rewards proportional to contributions.

        Args:
            coalition: Participating clients
            clients: All clients
            total_reward: Total reward

        Returns:
            AllocationResult with proportional shares
        """
        if not coalition:
            return AllocationResult(
                allocations={},
                total_reward=0.0,
                allocation_method=self.name,
            )

        # Compute scores for each client
        scores: dict[int, float] = {}
        metrics: dict[int, ContributionMetrics] = {}

        # Get normalization factors
        max_samples = max(clients[i].dataset_size for i in coalition)
        max_cost = max(clients[i].communication_cost for i in coalition)

        for idx in coalition:
            client = clients[idx]

            # Normalize contributions
            data_score = client.dataset_size / max_samples if max_samples > 0 else 0
            quality_score = client.data_quality
            efficiency_score = (
                1 - (client.communication_cost / max_cost) if max_cost > 0 else 1
            )

            # Weighted sum
            total_score = (
                self.data_weight * data_score
                + self.quality_weight * quality_score
                + self.efficiency_weight * efficiency_score
            )

            scores[idx] = total_score
            metrics[idx] = ContributionMetrics(
                data_contribution=data_score,
                computation_contribution=quality_score,
                communication_contribution=efficiency_score,
                total_contribution=total_score,
                details={
                    "num_samples": client.dataset_size,
                    "data_quality": client.data_quality,
                    "communication_cost": client.communication_cost,
                },
            )

        # Allocate proportionally
        total_scores = sum(scores.values())
        if total_scores <= 0:
            # Fall back to equal allocation
            share = total_reward / len(coalition)
            allocations = dict.fromkeys(coalition, share)
        else:
            allocations = {
                idx: total_reward * (score / total_scores)
                for idx, score in scores.items()
            }

        return AllocationResult(
            allocations=allocations,
            total_reward=total_reward,
            allocation_method=self.name,
            metrics=metrics,
        )

    @property
    def name(self) -> str:
        return "Proportional"


class ShapleyAllocator(RewardAllocator):
    """
    Allocation based on Shapley values.

    Uses game-theoretic Shapley values to fairly allocate rewards
    based on marginal contributions.

    reward_i = total_reward * (φ_i / Σ φ)

    Attributes:
        value_fn: Characteristic function v(S)
        n_samples: Samples for Monte Carlo Shapley

    Example:
        >>> def value_fn(coalition, clients):
        ...     return sum(clients[i].data_quality for i in coalition)
        >>> allocator = ShapleyAllocator(value_fn=value_fn)
        >>> result = allocator.allocate(coalition, clients, total_reward=100)
    """

    def __init__(
        self,
        value_fn: Callable[[Coalition, list[Client]], float] | None = None,
        n_samples: int = 1000,
        seed: int | None = None,
    ):
        """
        Initialize ShapleyAllocator.

        Args:
            value_fn: Characteristic function (default: sum of data quality)
            n_samples: Samples for Monte Carlo estimation
            seed: Random seed
        """
        self.value_fn = value_fn or self._default_value_fn
        self.n_samples = n_samples
        self.seed = seed

    def _default_value_fn(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> float:
        """Default value function: sum of data qualities."""
        if not coalition:
            return 0.0
        return sum(clients[i].data_quality for i in coalition if 0 <= i < len(clients))

    def allocate(
        self,
        coalition: Coalition,
        clients: list[Client],
        total_reward: float,
    ) -> AllocationResult:
        """
        Allocate rewards based on Shapley values.

        Args:
            coalition: Participating clients
            clients: All clients
            total_reward: Total reward

        Returns:
            AllocationResult with Shapley-based allocations
        """
        if not coalition:
            return AllocationResult(
                allocations={},
                total_reward=0.0,
                allocation_method=self.name,
            )

        # Compute Shapley values
        shapley_result = compute_shapley_values(
            coalition=coalition,
            clients=clients,
            value_fn=self.value_fn,
            n_samples=self.n_samples,
            seed=self.seed,
        )

        # Map back to client indices
        shapley_values = {
            coalition[i]: shapley_result.values[i] for i in range(len(coalition))
        }

        # Normalize and allocate
        total_shapley = sum(max(0, v) for v in shapley_values.values())

        if total_shapley <= 0:
            # Fall back to equal allocation
            share = total_reward / len(coalition)
            allocations = dict.fromkeys(coalition, share)
        else:
            allocations = {
                idx: total_reward * (max(0, v) / total_shapley)
                for idx, v in shapley_values.items()
            }

        # Build metrics
        metrics = {
            coalition[i]: ContributionMetrics(
                total_contribution=shapley_result.values[i],
                details={
                    "shapley_value": shapley_result.values[i],
                    "variance": (
                        shapley_result.variance[i]
                        if shapley_result.variance is not None
                        else None
                    ),
                },
            )
            for i in range(len(coalition))
        }

        return AllocationResult(
            allocations=allocations,
            total_reward=total_reward,
            allocation_method=self.name,
            metrics=metrics,
        )

    @property
    def name(self) -> str:
        return "Shapley"


class FairnessAwareAllocator(RewardAllocator):
    """
    Allocation that rewards demographic fairness contributions.

    Clients who improve coalition fairness receive higher rewards.

    Attributes:
        target_distribution: Target demographics
        fairness_weight: Weight for fairness contribution
        base_allocator: Base allocator for non-fairness component
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution | None = None,
        fairness_weight: float = 0.3,
        base_allocator: RewardAllocator | None = None,
    ):
        """
        Initialize FairnessAwareAllocator.

        Args:
            target_distribution: Target demographic distribution
            fairness_weight: Weight for fairness (0-1)
            base_allocator: Allocator for base rewards
        """
        self.target_distribution = target_distribution
        self.fairness_weight = fairness_weight
        self.base_allocator = base_allocator or ProportionalAllocator()

    def allocate(
        self,
        coalition: Coalition,
        clients: list[Client],
        total_reward: float,
    ) -> AllocationResult:
        """
        Allocate rewards with fairness bonus.

        Args:
            coalition: Participating clients
            clients: All clients
            total_reward: Total reward

        Returns:
            AllocationResult with fairness-adjusted allocations
        """
        if not coalition:
            return AllocationResult(
                allocations={},
                total_reward=0.0,
                allocation_method=self.name,
            )

        # Get base allocations
        base_result = self.base_allocator.allocate(
            coalition, clients, total_reward * (1 - self.fairness_weight)
        )

        # Compute fairness contributions
        fairness_scores: dict[int, float] = {}

        if self.target_distribution is not None:
            target = self.target_distribution.as_array()

            # Compute coalition demographics without each client
            for idx in coalition:
                # Coalition without this client
                others = [i for i in coalition if i != idx]

                if others:
                    # Average demographics without client
                    demos_without = np.mean(
                        [np.asarray(clients[i].demographics) for i in others],
                        axis=0,
                    )
                else:
                    demos_without = np.zeros_like(target)

                # Average demographics with client
                demos_with = np.mean(
                    [np.asarray(clients[i].demographics) for i in coalition],
                    axis=0,
                )

                # Marginal fairness improvement
                dist_without = np.sum((demos_without - target) ** 2)
                dist_with = np.sum((demos_with - target) ** 2)
                improvement = dist_without - dist_with

                fairness_scores[idx] = max(0, improvement)
        else:
            # No target: equal fairness scores
            fairness_scores = dict.fromkeys(coalition, 1.0)

        # Normalize fairness scores
        total_fairness = sum(fairness_scores.values())
        if total_fairness <= 0:
            fairness_allocations = dict.fromkeys(coalition, 0.0)
        else:
            fairness_reward = total_reward * self.fairness_weight
            fairness_allocations = {
                idx: fairness_reward * (score / total_fairness)
                for idx, score in fairness_scores.items()
            }

        # Combine allocations
        final_allocations = {
            idx: base_result.allocations.get(idx, 0) + fairness_allocations.get(idx, 0)
            for idx in coalition
        }

        # Build metrics
        metrics: dict[int, ContributionMetrics] = {}
        for idx in coalition:
            base_metrics = base_result.metrics.get(idx) if base_result.metrics else None
            metrics[idx] = ContributionMetrics(
                data_contribution=(
                    base_metrics.data_contribution if base_metrics else 0.0
                ),
                fairness_contribution=fairness_scores.get(idx, 0.0),
                total_contribution=final_allocations[idx],
                details={
                    "base_allocation": base_result.allocations.get(idx, 0),
                    "fairness_allocation": fairness_allocations.get(idx, 0),
                },
            )

        return AllocationResult(
            allocations=final_allocations,
            total_reward=total_reward,
            allocation_method=self.name,
            metrics=metrics,
        )

    @property
    def name(self) -> str:
        return "FairnessAware"


def allocate_rewards(
    coalition: Coalition,
    clients: list[Client],
    total_reward: float,
    method: str = "proportional",
    **kwargs: Any,
) -> AllocationResult:
    """
    Convenience function for reward allocation.

    Args:
        coalition: Participating clients
        clients: All clients
        total_reward: Total reward
        method: "equal", "proportional", or "shapley"
        **kwargs: Additional arguments for allocator

    Returns:
        AllocationResult

    Example:
        >>> result = allocate_rewards(
        ...     coalition=[0, 1, 2],
        ...     clients=clients,
        ...     total_reward=100,
        ...     method="shapley",
        ... )
    """
    allocator: RewardAllocator
    if method == "equal":
        allocator = EqualAllocator()
    elif method == "shapley":
        allocator = ShapleyAllocator(**kwargs)
    else:
        allocator = ProportionalAllocator(**kwargs)

    return allocator.allocate(coalition, clients, total_reward)
