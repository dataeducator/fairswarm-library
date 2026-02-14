"""
Cardinality constraints for FairSwarm.

These constraints enforce limits on coalition size.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from fairswarm.constraints.base import Constraint, ConstraintResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client

__all__ = [
    "CardinalityConstraint",
    "MinSizeConstraint",
    "MaxSizeConstraint",
    "ExactSizeConstraint",
    "MinDataConstraint",
    "MaxCostConstraint",
]


class CardinalityConstraint(Constraint):
    """
    General cardinality constraint on coalition size.

    Enforces: min_size ≤ |S| ≤ max_size

    Attributes:
        min_size: Minimum coalition size (inclusive)
        max_size: Maximum coalition size (inclusive)

    Example:
        >>> constraint = CardinalityConstraint(min_size=5, max_size=15)
        >>> result = constraint.evaluate([0, 1, 2], clients)
        >>> # Fails because |S| = 3 < 5
    """

    def __init__(
        self,
        min_size: int = 1,
        max_size: int = float("inf"),
    ):
        """
        Initialize CardinalityConstraint.

        Args:
            min_size: Minimum coalition size
            max_size: Maximum coalition size
        """
        if min_size < 0:
            raise ValueError("min_size must be non-negative")
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")

        self.min_size = min_size
        self.max_size = max_size

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        """
        Check if coalition size is within bounds.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with satisfaction status
        """
        size = len(coalition)

        if size < self.min_size:
            violation = self.min_size - size
            return ConstraintResult(
                satisfied=False,
                violation=float(violation),
                message=f"Coalition too small: {size} < {self.min_size}",
                details={"size": size, "min_size": self.min_size},
            )

        if size > self.max_size:
            violation = size - self.max_size
            return ConstraintResult(
                satisfied=False,
                violation=float(violation),
                message=f"Coalition too large: {size} > {self.max_size}",
                details={"size": size, "max_size": self.max_size},
            )

        return ConstraintResult(
            satisfied=True,
            violation=0.0,
            message=f"Coalition size {size} within [{self.min_size}, {self.max_size}]",
            details={"size": size},
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


class MinSizeConstraint(Constraint):
    """
    Minimum coalition size constraint.

    Enforces: |S| ≥ min_size

    Attributes:
        min_size: Minimum number of clients required
    """

    def __init__(self, min_size: int):
        if min_size < 1:
            raise ValueError("min_size must be at least 1")
        self.min_size = min_size

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        size = len(coalition)
        satisfied = size >= self.min_size
        violation = max(0, self.min_size - size)

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(violation),
            message=f"Size {size} {'≥' if satisfied else '<'} {self.min_size}",
            details={"size": size, "min_size": self.min_size},
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Gradient encourages selecting more clients if below minimum.
        """
        if coalition_size >= self.min_size:
            return np.zeros(len(clients))

        # Encourage selecting clients with higher data quality
        gradient = np.array([c.data_quality for c in clients])
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "min_size": self.min_size}


class MaxSizeConstraint(Constraint):
    """
    Maximum coalition size constraint.

    Enforces: |S| ≤ max_size

    Attributes:
        max_size: Maximum number of clients allowed
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        size = len(coalition)
        satisfied = size <= self.max_size
        violation = max(0, size - self.max_size)

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(violation),
            message=f"Size {size} {'≤' if satisfied else '>'} {self.max_size}",
            details={"size": size, "max_size": self.max_size},
        )

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "max_size": self.max_size}


class ExactSizeConstraint(Constraint):
    """
    Exact coalition size constraint.

    Enforces: |S| = exact_size

    Attributes:
        exact_size: Required coalition size
    """

    def __init__(self, exact_size: int):
        if exact_size < 1:
            raise ValueError("exact_size must be at least 1")
        self.exact_size = exact_size

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        size = len(coalition)
        satisfied = size == self.exact_size
        violation = abs(size - self.exact_size)

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(violation),
            message=f"Size {size} {'=' if satisfied else '≠'} {self.exact_size}",
            details={"size": size, "exact_size": self.exact_size},
        )

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "exact_size": self.exact_size}


class MinDataConstraint(Constraint):
    """
    Minimum total data samples constraint.

    Enforces: Σ_{i∈S} samples_i ≥ min_samples

    Useful for ensuring sufficient training data.

    Attributes:
        min_samples: Minimum total samples required
    """

    def __init__(self, min_samples: int):
        if min_samples < 0:
            raise ValueError("min_samples must be non-negative")
        self.min_samples = min_samples

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        total_samples = sum(
            clients[i].dataset_size for i in coalition if 0 <= i < len(clients)
        )

        satisfied = total_samples >= self.min_samples
        violation = max(0, self.min_samples - total_samples)

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(violation),
            message=f"Samples {total_samples} {'≥' if satisfied else '<'} {self.min_samples}",
            details={"total_samples": total_samples, "min_samples": self.min_samples},
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Gradient proportional to client sample sizes.
        """
        gradient = np.array([c.dataset_size for c in clients], dtype=np.float64)
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "min_samples": self.min_samples}


class MaxCostConstraint(Constraint):
    """
    Maximum communication cost constraint.

    Enforces: Σ_{i∈S} cost_i ≤ max_cost

    Useful for budget-constrained optimization.

    Attributes:
        max_cost: Maximum total communication cost
    """

    def __init__(self, max_cost: float):
        if max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        self.max_cost = max_cost

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> ConstraintResult:
        total_cost = sum(
            clients[i].communication_cost for i in coalition if 0 <= i < len(clients)
        )

        satisfied = total_cost <= self.max_cost
        violation = max(0, total_cost - self.max_cost)

        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            message=f"Cost {total_cost:.2f} {'≤' if satisfied else '>'} {self.max_cost:.2f}",
            details={"total_cost": total_cost, "max_cost": self.max_cost},
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Negative gradient for high-cost clients.
        """
        gradient = -np.array([c.communication_cost for c in clients])
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "max_cost": self.max_cost}
