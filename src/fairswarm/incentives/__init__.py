"""
Incentive mechanisms for FairSwarm.

This module provides game-theoretic incentive mechanisms for
fair reward allocation in federated learning coalitions.

Key Components:
    - ShapleyValue: Exact and Monte Carlo Shapley computation
    - RewardAllocator: Distributes rewards based on contributions
    - ContributionMetrics: Measures client contributions

Shapley Value:
    φ_i(v) = Σ_{S⊆N\\{i}} (|S|!(n-|S|-1)!/n!) [v(S∪{i}) - v(S)]

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.incentives.allocation import (
    ContributionMetrics,
    EqualAllocator,
    ProportionalAllocator,
    RewardAllocator,
    ShapleyAllocator,
)
from fairswarm.incentives.shapley import (
    ExactShapley,
    MonteCarloShapley,
    ShapleyValue,
    compute_shapley_values,
)

__all__ = [
    # Shapley
    "ShapleyValue",
    "ExactShapley",
    "MonteCarloShapley",
    "compute_shapley_values",
    # Allocation
    "RewardAllocator",
    "ProportionalAllocator",
    "ShapleyAllocator",
    "EqualAllocator",
    "ContributionMetrics",
]
