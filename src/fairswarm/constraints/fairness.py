"""
Fairness constraints for FairSwarm.

These constraints enforce demographic fairness requirements
based on Definition 2 and Theorem 2.

Definition 2: DemDiv(S) = D_KL(δ_S || δ*)
Theorem 2: DemDiv(S*) ≤ ε with probability ≥ 1 - δ

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from fairswarm.constraints.base import Constraint, ConstraintResult
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence, total_variation_distance
from fairswarm.fitness.fairness import (
    compute_coalition_demographics,
    compute_fairness_gradient,
)
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

__all__ = [
    "FairnessConstraint",
    "DivergenceConstraint",
    "RepresentationConstraint",
    "MinorityRepresentationConstraint",
    "TotalVariationConstraint",
]


class FairnessConstraint(Constraint):
    """
    Base class for fairness constraints.

    Provides common functionality for demographic fairness
    constraints.

    Attributes:
        target_distribution: Target demographic distribution δ*
    """

    def __init__(self, target_distribution: DemographicDistribution):
        self.target_distribution = target_distribution

    def _get_coalition_demographics(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> NDArray[np.float64]:
        """Compute coalition demographic distribution."""
        if not coalition:
            return np.zeros(len(self.target_distribution.as_array()))
        return compute_coalition_demographics(coalition, clients)


class DivergenceConstraint(FairnessConstraint):
    """
    Demographic divergence constraint (Theorem 2).

    Enforces: DemDiv(S) = D_KL(δ_S || δ*) ≤ ε

    This is the core fairness constraint from Theorem 2.

    Attributes:
        target_distribution: Target demographic distribution
        epsilon: Maximum allowed divergence

    Example:
        >>> target = CensusTarget.US_2020.as_distribution()
        >>> constraint = DivergenceConstraint(target, epsilon=0.1)
        >>> result = constraint.evaluate(coalition, clients)
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        epsilon: float = 0.1,
    ):
        """
        Initialize DivergenceConstraint.

        Args:
            target_distribution: Target demographics δ*
            epsilon: Maximum allowed KL divergence
        """
        super().__init__(target_distribution)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        """
        Check if demographic divergence is within bounds.

        Implements: DemDiv(S) ≤ ε

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with divergence information
        """
        if not coalition:
            return ConstraintResult(
                satisfied=False,
                violation=float("inf"),
                message="Empty coalition",
            )

        # Compute divergence: D_KL(δ_S || δ*)
        coalition_demo = self._get_coalition_demographics(coalition, clients)
        target = self.target_distribution.as_array()
        divergence = kl_divergence(coalition_demo, target)

        satisfied = divergence <= self.epsilon
        violation = max(0, divergence - self.epsilon)

        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            message=f"DemDiv={divergence:.4f} {'≤' if satisfied else '>'} ε={self.epsilon}",
            details={
                "divergence": divergence,
                "epsilon": self.epsilon,
                "coalition_demographics": coalition_demo.tolist(),
            },
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient toward lower divergence.
        """
        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=self.target_distribution,
            coalition_size=coalition_size,
        )
        return result.gradient

    def is_hard_constraint(self) -> bool:
        """Divergence is typically a soft constraint (penalized)."""
        return False

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "target": self.target_distribution.categories,
        }


class RepresentationConstraint(FairnessConstraint):
    """
    Per-group representation constraint.

    Enforces: |δ_S[g] - δ*[g]| ≤ threshold for all groups g

    Ensures each demographic group is within threshold of target.

    Attributes:
        target_distribution: Target demographic distribution
        threshold: Maximum deviation per group
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        threshold: float = 0.1,
    ):
        """
        Initialize RepresentationConstraint.

        Args:
            target_distribution: Target demographics
            threshold: Maximum deviation from target per group
        """
        super().__init__(target_distribution)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        """
        Check if each group is within threshold of target.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with per-group information
        """
        if not coalition:
            return ConstraintResult(
                satisfied=False,
                violation=float("inf"),
                message="Empty coalition",
            )

        coalition_demo = self._get_coalition_demographics(coalition, clients)
        target = self.target_distribution.as_array()

        # Check each group
        deviations = np.abs(coalition_demo - target)
        max_deviation = np.max(deviations)
        total_violation = np.sum(np.maximum(0, deviations - self.threshold))

        satisfied = max_deviation <= self.threshold

        return ConstraintResult(
            satisfied=satisfied,
            violation=total_violation,
            message=f"Max deviation={max_deviation:.4f} {'≤' if satisfied else '>'} {self.threshold}",
            details={
                "max_deviation": max_deviation,
                "deviations": deviations.tolist(),
                "threshold": self.threshold,
            },
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "target": self.target_distribution.categories,
        }


class MinorityRepresentationConstraint(FairnessConstraint):
    """
    Minimum representation for minority groups.

    Enforces: δ_S[g] ≥ min_representation for specified groups

    Ensures underrepresented groups have minimum presence.

    Attributes:
        target_distribution: Target demographic distribution
        minority_groups: List of group names to protect
        min_representation: Minimum required representation
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        minority_groups: Optional[List[str]] = None,
        min_representation: float = 0.05,
    ):
        """
        Initialize MinorityRepresentationConstraint.

        Args:
            target_distribution: Target demographics
            minority_groups: Groups to protect (default: those < 10% in target)
            min_representation: Minimum required representation
        """
        super().__init__(target_distribution)

        if minority_groups is None:
            # Auto-detect minority groups (< 10% in target)
            minority_groups = [
                group
                for group, value in target_distribution.categories.items()
                if value < 0.10
            ]

        self.minority_groups = minority_groups
        self.min_representation = min_representation
        self._group_indices = self._compute_group_indices()

    def _compute_group_indices(self) -> Dict[str, int]:
        """Map group names to array indices."""
        categories = list(self.target_distribution.categories.keys())
        return {group: i for i, group in enumerate(categories)}

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        """
        Check if minority groups meet minimum representation.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with per-group information
        """
        if not coalition:
            return ConstraintResult(
                satisfied=False,
                violation=float("inf"),
                message="Empty coalition",
            )

        coalition_demo = self._get_coalition_demographics(coalition, clients)

        # Check each minority group
        violations = {}
        total_violation = 0.0

        for group in self.minority_groups:
            if group in self._group_indices:
                idx = self._group_indices[group]
                if idx < len(coalition_demo):
                    representation = coalition_demo[idx]
                    if representation < self.min_representation:
                        shortfall = self.min_representation - representation
                        violations[group] = shortfall
                        total_violation += shortfall

        satisfied = len(violations) == 0

        return ConstraintResult(
            satisfied=satisfied,
            violation=total_violation,
            message=f"{len(violations)} minority groups below threshold"
            if violations
            else "All minority groups represented",
            details={
                "violations": violations,
                "min_representation": self.min_representation,
                "minority_groups": self.minority_groups,
            },
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Gradient encourages clients with underrepresented demographics.
        """
        n_clients = len(clients)
        gradient = np.zeros(n_clients)

        for i, client in enumerate(clients):
            demo = client.demographics.as_array()
            # Boost clients who represent minority groups
            for group in self.minority_groups:
                if group in self._group_indices:
                    idx = self._group_indices[group]
                    if idx < len(demo):
                        gradient[i] += demo[idx]

        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "minority_groups": self.minority_groups,
            "min_representation": self.min_representation,
        }


class TotalVariationConstraint(FairnessConstraint):
    """
    Total variation distance constraint.

    Enforces: TV(δ_S, δ*) ≤ threshold

    Alternative to KL divergence with bounded range [0, 1].

    Attributes:
        target_distribution: Target demographic distribution
        threshold: Maximum total variation distance
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        threshold: float = 0.2,
    ):
        """
        Initialize TotalVariationConstraint.

        Args:
            target_distribution: Target demographics
            threshold: Maximum TV distance
        """
        super().__init__(target_distribution)
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        """
        Check if total variation distance is within bounds.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with TV distance information
        """
        if not coalition:
            return ConstraintResult(
                satisfied=False,
                violation=1.0,
                message="Empty coalition",
            )

        coalition_demo = self._get_coalition_demographics(coalition, clients)
        target = self.target_distribution.as_array()
        tv_distance = total_variation_distance(coalition_demo, target)

        satisfied = tv_distance <= self.threshold
        violation = max(0, tv_distance - self.threshold)

        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            message=f"TV={tv_distance:.4f} {'≤' if satisfied else '>'} {self.threshold}",
            details={
                "tv_distance": tv_distance,
                "threshold": self.threshold,
            },
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "target": self.target_distribution.categories,
        }
