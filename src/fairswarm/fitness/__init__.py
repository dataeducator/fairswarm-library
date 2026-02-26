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
"""

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.composite import (
    CommunicationCostFitness,
    CompositeFitness,
    WeightedComponent,
    WeightedFitness,
)
from fairswarm.fitness.equity import (
    ClientDissimilarityFitness,
    client_dissimilarity,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_gap,
)
from fairswarm.fitness.fairness import (
    AccuracyFairnessFitness,
    DemographicFitness,
    FairnessGradient,
    compute_coalition_demographics,
    compute_fairness_gradient,
)
from fairswarm.fitness.mock import (
    ConstantFitness,
    DataQualityFitness,
    DeterministicFitness,
    MockFitness,
)

__all__ = [
    # Base
    "FitnessFunction",
    "FitnessResult",
    # Fairness
    "DemographicFitness",
    "AccuracyFairnessFitness",
    "FairnessGradient",
    "compute_fairness_gradient",
    "compute_coalition_demographics",
    # Composite
    "CompositeFitness",
    "WeightedFitness",
    "WeightedComponent",
    "CommunicationCostFitness",
    # Equity metrics
    "ClientDissimilarityFitness",
    "client_dissimilarity",
    "equalized_odds_gap",
    "equal_opportunity_difference",
    "demographic_parity_difference",
    # Testing
    "MockFitness",
    "ConstantFitness",
    "DeterministicFitness",
    "DataQualityFitness",
]
