"""
Abstract base class for FairSwarm constraints.

Constraints define feasibility conditions for coalitions during
PSO optimization.

Author: Tenicka Norwood
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

__all__ = [
    "ConstraintResult",
    "Constraint",
    "ConstraintSet",
]


@dataclass(frozen=True)
class ConstraintResult:
    """
    Result of constraint evaluation.

    Attributes:
        satisfied: Whether the constraint is satisfied
        violation: Magnitude of violation (0 if satisfied)
        message: Human-readable description
        details: Additional constraint-specific information
    """

    satisfied: bool
    violation: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using result directly in conditionals."""
        return self.satisfied


class Constraint(ABC):
    """
    Abstract base class for coalition constraints.

    Constraints define conditions that coalitions must satisfy.
    They can be used for:
    - Filtering infeasible coalitions
    - Adding penalty terms to fitness
    - Guiding the optimization toward feasible regions

    Subclasses must implement:
        - evaluate(): Check if coalition satisfies constraint
        - compute_penalty(): Compute penalty for constraint violation

    Example:
        >>> class MinDataConstraint(Constraint):
        ...     def __init__(self, min_samples: int):
        ...         self.min_samples = min_samples
        ...
        ...     def evaluate(self, coalition, clients):
        ...         total = sum(clients[i].num_samples for i in coalition)
        ...         satisfied = total >= self.min_samples
        ...         return ConstraintResult(
        ...             satisfied=satisfied,
        ...             violation=max(0, self.min_samples - total),
        ...         )
    """

    @property
    def name(self) -> str:
        """Constraint name for logging."""
        return self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Evaluate whether the constraint is satisfied.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with satisfaction status and violation magnitude
        """
        pass

    def compute_penalty(
        self,
        coalition: Coalition,
        clients: list[Client],
        penalty_weight: float = 1.0,
    ) -> float:
        """
        Compute penalty for constraint violation.

        Default implementation returns violation * weight.
        Override for custom penalty functions.

        Args:
            coalition: List of client indices
            clients: List of all clients
            penalty_weight: Multiplier for the penalty

        Returns:
            Penalty value (0 if constraint satisfied)
        """
        result = self.evaluate(coalition, clients)
        return penalty_weight * result.violation

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for constraint-aware optimization.

        Default implementation returns zero gradient.
        Override for constraints that should influence particle movement.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Gradient vector
        """
        return np.zeros(len(clients))

    def is_hard_constraint(self) -> bool:
        """
        Whether this is a hard constraint (must be satisfied).

        Hard constraints filter out infeasible coalitions.
        Soft constraints add penalties to fitness.

        Returns:
            True if hard constraint, False if soft
        """
        return True

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {"name": self.name, "hard": self.is_hard_constraint()}


class ConstraintSet:
    """
    Collection of constraints with combined evaluation.

    Manages multiple constraints and provides unified evaluation.

    Attributes:
        constraints: List of constraint objects

    Example:
        >>> constraints = ConstraintSet([
        ...     MinSizeConstraint(min_size=5),
        ...     MaxSizeConstraint(max_size=15),
        ...     DivergenceConstraint(epsilon=0.5),
        ... ])
        >>> result = constraints.evaluate(coalition, clients)
        >>> if result.satisfied:
        ...     print("All constraints satisfied")
    """

    def __init__(self, constraints: list[Constraint] | None = None):
        """
        Initialize ConstraintSet.

        Args:
            constraints: List of constraint objects
        """
        self.constraints = constraints or []

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)

    def remove(self, constraint_name: str) -> bool:
        """
        Remove a constraint by name.

        Args:
            constraint_name: Name of constraint to remove

        Returns:
            True if removed, False if not found
        """
        for i, c in enumerate(self.constraints):
            if c.name == constraint_name:
                del self.constraints[i]
                return True
        return False

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Evaluate all constraints.

        Returns satisfied=True only if all constraints are satisfied.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            Combined ConstraintResult
        """
        if not self.constraints:
            return ConstraintResult(satisfied=True, message="No constraints")

        all_satisfied = True
        total_violation = 0.0
        messages = []
        details = {}

        for constraint in self.constraints:
            result = constraint.evaluate(coalition, clients)

            if not result.satisfied:
                all_satisfied = False
                messages.append(f"{constraint.name}: {result.message}")

            total_violation += result.violation
            details[constraint.name] = {
                "satisfied": result.satisfied,
                "violation": result.violation,
            }

        return ConstraintResult(
            satisfied=all_satisfied,
            violation=total_violation,
            message="; ".join(messages) if messages else "All constraints satisfied",
            details=details,
        )

    def evaluate_hard_only(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Evaluate only hard constraints.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult for hard constraints only
        """
        hard_constraints = [c for c in self.constraints if c.is_hard_constraint()]

        if not hard_constraints:
            return ConstraintResult(satisfied=True)

        temp_set = ConstraintSet(hard_constraints)
        return temp_set.evaluate(coalition, clients)

    def compute_total_penalty(
        self,
        coalition: Coalition,
        clients: list[Client],
        penalty_weight: float = 1.0,
    ) -> float:
        """
        Compute total penalty from all constraints.

        Args:
            coalition: List of client indices
            clients: List of all clients
            penalty_weight: Global penalty weight multiplier

        Returns:
            Sum of all constraint penalties
        """
        return sum(
            c.compute_penalty(coalition, clients, penalty_weight)
            for c in self.constraints
        )

    def compute_combined_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute combined gradient from all constraints.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Sum of constraint gradients
        """
        if not self.constraints:
            return np.zeros(len(clients))

        total_gradient = np.zeros(len(clients))
        for constraint in self.constraints:
            total_gradient += constraint.compute_gradient(
                position, clients, coalition_size
            )

        # Normalize
        norm = np.linalg.norm(total_gradient)
        if norm > 1e-10:
            total_gradient = total_gradient / norm

        return total_gradient

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.constraints)

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "n_constraints": len(self.constraints),
            "constraints": [c.get_config() for c in self.constraints],
        }
