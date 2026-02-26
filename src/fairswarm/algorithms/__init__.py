"""
FairSwarm optimization algorithms.

This module provides the core FairSwarm PSO algorithm for fair
coalition selection in federated learning.

Key Classes:
    - FairSwarm: Main optimizer implementing Algorithm 1
    - OptimizationResult: Container for optimization results

Algorithm Reference:
    Algorithm 1 in the paper defines the FairSwarm PSO procedure.

Theoretical Guarantees:
    - Theorem 1: Convergence when ω + (c₁+c₂)/2 < 2
    - Theorem 2: DemDiv(S*) ≤ ε with high probability
    - Theorem 3: (1-1/e-η) approximation for submodular objectives

Author: Tenicka Norwood
"""

from fairswarm.algorithms.fairswarm import FairSwarm, run_fairswarm
from fairswarm.algorithms.fairswarm_dp import DPConfig, DPResult, FairSwarmDP
from fairswarm.algorithms.result import (
    ConvergenceMetrics,
    FairnessMetrics,
    OptimizationResult,
)
from fairswarm.algorithms.sklearn_compat import FairSwarmSelector

__all__ = [
    # Main algorithm
    "FairSwarm",
    "run_fairswarm",
    # DP variant
    "FairSwarmDP",
    "DPConfig",
    "DPResult",
    # Results
    "OptimizationResult",
    "ConvergenceMetrics",
    "FairnessMetrics",
    # Sklearn compatibility
    "FairSwarmSelector",
]
