"""
Privacy constraints for FairSwarm.

These constraints enforce differential privacy requirements
based on Theorem 4 (Privacy-Fairness Tradeoff).

Theorem 4: UtilityLoss ≥ Ω(√(k·log(1/δ))/(ε_DP·ε_F))

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fairswarm.constraints.base import Constraint, ConstraintResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client
    from fairswarm.privacy.accountant import PrivacyAccountant

__all__ = [
    "PrivacyConstraint",
    "PrivacyBudgetConstraint",
    "LocalPrivacyConstraint",
    "SensitivityConstraint",
    "CompositionConstraint",
]


class PrivacyConstraint(Constraint):
    """
    Base class for privacy constraints.

    Provides common functionality for differential privacy
    constraints in FairSwarm.
    """

    def is_hard_constraint(self) -> bool:
        """Privacy constraints are typically hard constraints."""
        return True


class PrivacyBudgetConstraint(PrivacyConstraint):
    """
    Global privacy budget constraint.

    Enforces: ε_consumed ≤ ε_budget

    Ensures the total privacy expenditure stays within budget.

    Attributes:
        epsilon_budget: Total privacy budget
        delta: Privacy failure probability
        accountant: Privacy accountant (optional, for tracking)
    """

    def __init__(
        self,
        epsilon_budget: float,
        delta: float = 1e-5,
        accountant: PrivacyAccountant | None = None,
        cost_per_round: float = 0.0,
    ):
        """
        Initialize PrivacyBudgetConstraint.

        Args:
            epsilon_budget: Maximum allowed epsilon
            delta: Privacy failure probability
            accountant: Optional privacy accountant for tracking
            cost_per_round: Expected epsilon cost per round of training.
                When > 0, the constraint checks that the remaining budget
                is sufficient to cover at least one more round.
        """
        if epsilon_budget <= 0:
            raise ValueError("epsilon_budget must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")

        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.accountant = accountant
        self.cost_per_round = cost_per_round
        self._consumed_epsilon = 0.0

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Check if privacy budget allows this coalition.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with budget information
        """
        # Get consumed epsilon from accountant if available
        if self.accountant is not None:
            consumed = self.accountant.get_epsilon(self.delta)
        else:
            consumed = self._consumed_epsilon

        remaining = self.epsilon_budget - consumed

        # Check that consumed epsilon hasn't exceeded the budget
        within_budget = consumed <= self.epsilon_budget

        # Check that remaining budget is sufficient for another round
        sufficient_remaining = remaining >= self.cost_per_round

        satisfied = within_budget and sufficient_remaining

        return ConstraintResult(
            satisfied=satisfied,
            violation=max(0.0, consumed - self.epsilon_budget),
            message=f"Budget: {consumed:.4f}/{self.epsilon_budget} consumed, {remaining:.4f} remaining",
            details={
                "consumed": consumed,
                "budget": self.epsilon_budget,
                "remaining": remaining,
                "cost_per_round": self.cost_per_round,
                "delta": self.delta,
            },
        )

    def record_query(self, epsilon: float) -> None:
        """
        Record a privacy expenditure.

        Args:
            epsilon: Epsilon consumed by this query
        """
        if self.accountant is not None:
            self.accountant.step(epsilon, self.delta)
        else:
            self._consumed_epsilon += epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        if self.accountant is not None:
            consumed = self.accountant.get_epsilon(self.delta)
        else:
            consumed = self._consumed_epsilon
        return max(0, self.epsilon_budget - consumed)

    def reset(self) -> None:
        """Reset privacy budget tracking."""
        self._consumed_epsilon = 0.0
        if self.accountant is not None:
            self.accountant.reset()

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epsilon_budget": self.epsilon_budget,
            "delta": self.delta,
            "has_accountant": self.accountant is not None,
        }


class LocalPrivacyConstraint(PrivacyConstraint):
    """
    Local differential privacy constraint per client.

    Enforces minimum privacy level for each participating client.

    Attributes:
        min_epsilon: Minimum epsilon per client
        max_epsilon: Maximum epsilon per client
    """

    def __init__(
        self,
        min_epsilon: float = 0.1,
        max_epsilon: float = 10.0,
    ):
        """
        Initialize LocalPrivacyConstraint.

        Args:
            min_epsilon: Minimum privacy parameter
            max_epsilon: Maximum privacy parameter
        """
        if min_epsilon <= 0:
            raise ValueError("min_epsilon must be positive")
        if max_epsilon < min_epsilon:
            raise ValueError("max_epsilon must be >= min_epsilon")

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Check if all clients meet privacy requirements.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with per-client information
        """
        violations = []

        for idx in coalition:
            if 0 <= idx < len(clients):
                client = clients[idx]
                client_epsilon = client.privacy_epsilon

                if client_epsilon is None:
                    violations.append(
                        f"{client.id}: no privacy_epsilon declared"
                    )
                elif client_epsilon < self.min_epsilon:
                    violations.append(
                        f"{client.id}: ε={client_epsilon:.2f} < {self.min_epsilon}"
                    )
                elif client_epsilon > self.max_epsilon:
                    violations.append(
                        f"{client.id}: ε={client_epsilon:.2f} > {self.max_epsilon}"
                    )

        satisfied = len(violations) == 0

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(len(violations)),
            message=(
                f"{len(violations)} clients violate privacy bounds"
                if violations
                else "All clients within privacy bounds"
            ),
            details={
                "violations": violations,
                "min_epsilon": self.min_epsilon,
                "max_epsilon": self.max_epsilon,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_epsilon": self.min_epsilon,
            "max_epsilon": self.max_epsilon,
        }


class SensitivityConstraint(PrivacyConstraint):
    """
    Data sensitivity constraint.

    Ensures coalition data sensitivity is bounded for privacy.

    Attributes:
        max_sensitivity: Maximum allowed sensitivity
    """

    def __init__(self, max_sensitivity: float = 1.0):
        """
        Initialize SensitivityConstraint.

        Args:
            max_sensitivity: Maximum data sensitivity
        """
        if max_sensitivity <= 0:
            raise ValueError("max_sensitivity must be positive")
        self.max_sensitivity = max_sensitivity

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Check if coalition sensitivity is bounded.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with sensitivity information
        """
        if not coalition:
            return ConstraintResult(
                satisfied=True,
                violation=0.0,
                message="Empty coalition has zero sensitivity",
            )

        # Compute coalition sensitivity
        # Default: L2 sensitivity based on client contributions
        sensitivities = []
        for idx in coalition:
            if 0 <= idx < len(clients):
                client = clients[idx]
                # Get client sensitivity (if defined); default to 1.0
                sensitivity = getattr(client, "sensitivity", 1.0)
                sensitivities.append(sensitivity)

        total_sensitivity = np.sqrt(np.sum(np.array(sensitivities) ** 2))

        satisfied = total_sensitivity <= self.max_sensitivity
        violation = max(0, total_sensitivity - self.max_sensitivity)

        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            message=f"Sensitivity={total_sensitivity:.4f} {'≤' if satisfied else '>'} {self.max_sensitivity}",
            details={
                "total_sensitivity": total_sensitivity,
                "max_sensitivity": self.max_sensitivity,
                "individual_sensitivities": sensitivities,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_sensitivity": self.max_sensitivity,
        }


class CompositionConstraint(PrivacyConstraint):
    """
    Privacy composition constraint.

    Tracks privacy loss under composition of multiple queries.
    Supports both basic and advanced composition theorems.

    Attributes:
        epsilon_per_query: Epsilon per query
        delta_per_query: Delta per query
        max_queries: Maximum number of queries
        max_epsilon: Maximum total composed epsilon budget
        composition_type: "basic" or "advanced"
    """

    def __init__(
        self,
        epsilon_per_query: float,
        delta_per_query: float = 1e-6,
        max_queries: int = 100,
        max_epsilon: float = float("inf"),
        composition_type: str = "advanced",
    ):
        """
        Initialize CompositionConstraint.

        Args:
            epsilon_per_query: Epsilon spent per query
            delta_per_query: Delta per query
            max_queries: Maximum allowed queries
            max_epsilon: Maximum total composed epsilon (privacy budget)
            composition_type: Composition theorem to use
        """
        if epsilon_per_query <= 0:
            raise ValueError("epsilon_per_query must be positive")

        self.epsilon_per_query = epsilon_per_query
        self.delta_per_query = delta_per_query
        self.max_queries = max_queries
        self.max_epsilon = max_epsilon
        self.composition_type = composition_type
        self._query_count = 0

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> ConstraintResult:
        """
        Check if composition budget allows more queries.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            ConstraintResult with composition information
        """
        # Compute composed epsilon based on composition type
        if self.composition_type == "basic":
            # Basic composition: ε_total = k * ε
            composed_epsilon = self._query_count * self.epsilon_per_query
        else:
            # Advanced composition: ε_total = √(2k·ln(1/δ))·ε + k·ε(e^ε - 1)
            k = self._query_count
            eps = self.epsilon_per_query
            delta = self.delta_per_query
            if k > 0 and delta > 0:
                composed_epsilon = np.sqrt(
                    2 * k * np.log(1 / delta)
                ) * eps + k * eps * (np.exp(eps) - 1)
            else:
                composed_epsilon = 0.0

        within_query_limit = self._query_count < self.max_queries
        within_epsilon_budget = bool(composed_epsilon <= self.max_epsilon)
        satisfied = within_query_limit and within_epsilon_budget

        # Violation magnitude: sum of query overshoot and epsilon overshoot
        query_violation = max(0, self._query_count - self.max_queries)
        epsilon_violation = max(0.0, composed_epsilon - self.max_epsilon)

        parts = [f"Queries: {self._query_count}/{self.max_queries}"]
        if self.max_epsilon < float("inf"):
            parts.append(f"Composed ε: {composed_epsilon:.4f}/{self.max_epsilon}")

        return ConstraintResult(
            satisfied=satisfied,
            violation=float(query_violation) + epsilon_violation,
            message=", ".join(parts),
            details={
                "query_count": self._query_count,
                "max_queries": self.max_queries,
                "composed_epsilon": composed_epsilon,
                "max_epsilon": self.max_epsilon,
                "composition_type": self.composition_type,
            },
        )

    def record_query(self) -> None:
        """Record a query."""
        self._query_count += 1

    def reset(self) -> None:
        """Reset query count."""
        self._query_count = 0

    def get_composed_epsilon(self, delta: float | None = None) -> float:
        """
        Get total epsilon under composition.

        Args:
            delta: Delta for advanced composition

        Returns:
            Composed epsilon value
        """
        delta = delta or self.delta_per_query
        k = self._query_count
        eps = self.epsilon_per_query

        if self.composition_type == "basic":
            return float(k * eps)
        else:
            if k > 0 and delta > 0:
                return float(
                    np.sqrt(2 * k * np.log(1 / delta)) * eps
                    + k * eps * (np.exp(eps) - 1)
                )
            return 0.0

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epsilon_per_query": self.epsilon_per_query,
            "delta_per_query": self.delta_per_query,
            "max_queries": self.max_queries,
            "max_epsilon": self.max_epsilon,
            "composition_type": self.composition_type,
        }
