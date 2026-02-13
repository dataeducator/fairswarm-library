"""
Constraint system for FairSwarm optimization.

This module provides constraints that can be applied during coalition
selection to enforce fairness, cardinality, privacy, and other requirements.

Key Classes:
    - Constraint: Abstract base class for constraints
    - FairnessConstraint: Enforces demographic fairness bounds
    - CardinalityConstraint: Enforces coalition size limits
    - PrivacyConstraint: Enforces privacy budget constraints

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.constraints.base import Constraint, ConstraintResult, ConstraintSet
from fairswarm.constraints.cardinality import (
    CardinalityConstraint,
    MinSizeConstraint,
    MaxSizeConstraint,
    ExactSizeConstraint,
)
from fairswarm.constraints.fairness import (
    FairnessConstraint,
    DivergenceConstraint,
    RepresentationConstraint,
    MinorityRepresentationConstraint,
)
from fairswarm.constraints.privacy import (
    PrivacyConstraint,
    PrivacyBudgetConstraint,
    LocalPrivacyConstraint,
)

__all__ = [
    # Base
    "Constraint",
    "ConstraintResult",
    "ConstraintSet",
    # Cardinality
    "CardinalityConstraint",
    "MinSizeConstraint",
    "MaxSizeConstraint",
    "ExactSizeConstraint",
    # Fairness
    "FairnessConstraint",
    "DivergenceConstraint",
    "RepresentationConstraint",
    "MinorityRepresentationConstraint",
    # Privacy
    "PrivacyConstraint",
    "PrivacyBudgetConstraint",
    "LocalPrivacyConstraint",
]
