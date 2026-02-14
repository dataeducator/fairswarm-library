"""
Fitness evaluation for FairSwarm PSO optimization.

This module provides fitness functions that guide the PSO search,
including the novel demographic fairness component.

Key Classes:
    - FitnessFunction: Abstract base class for fitness evaluation
    - DemographicFitness: Fairness-based fitness using Definition 2
    - CompositeFitness: Multi-objective fitness combining multiple components
    - MockFitness: Testing utility for deterministic fitness values

Mathematical Foundation:
    The fitness function F(S) evaluates coalition quality:

    F(S) = ValAcc(S) - λ·DemDiv(S) - μ·CommCost(S)

    Where:
    - ValAcc(S): Validation accuracy of coalition S
    - DemDiv(S): Demographic divergence (Definition 2)
    - CommCost(S): Communication/computation cost

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.composite import CompositeFitness, WeightedFitness
from fairswarm.fitness.fairness import (
    DemographicFitness,
    FairnessGradient,
    compute_fairness_gradient,
)
from fairswarm.fitness.mock import ConstantFitness, MockFitness

__all__ = [
    # Base
    "FitnessFunction",
    "FitnessResult",
    # Fairness
    "DemographicFitness",
    "FairnessGradient",
    "compute_fairness_gradient",
    # Composite
    "CompositeFitness",
    "WeightedFitness",
    # Testing
    "MockFitness",
    "ConstantFitness",
]
