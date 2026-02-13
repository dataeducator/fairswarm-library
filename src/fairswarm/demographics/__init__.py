"""
Demographics module for FairSwarm.

This module provides tools for handling demographic distributions
and computing divergence metrics used in fair coalition selection.

Key Components:
    - DemographicDistribution: Class for representing and manipulating
      demographic probability distributions
    - kl_divergence: KL divergence matching Definition 2 in CLAUDE.md
    - js_divergence: Jensen-Shannon divergence (symmetric alternative)
    - CensusTarget: Preset demographic targets (e.g., US Census 2020)

Mathematical Foundation:
    Definition 2 (Demographic Divergence):
    DemDiv(S) = D_KL(δ_S || δ*)

    where δ_S is the coalition's demographic distribution and
    δ* is the target distribution.

Example:
    >>> from fairswarm.demographics import (
    ...     DemographicDistribution,
    ...     CensusTarget,
    ...     kl_divergence,
    ... )
    >>>
    >>> hospital = DemographicDistribution.from_dict({
    ...     "white": 0.70, "black": 0.15, "hispanic": 0.10, "other": 0.05
    ... })
    >>> target = CensusTarget.US_2020
    >>> divergence = kl_divergence(hospital.as_array(), target.as_array())
    >>> print(f"Divergence from census: {divergence:.4f}")

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import (
    coalition_demographic_divergence,
    js_divergence,
    kl_divergence,
    total_variation_distance,
    wasserstein_distance,
)
from fairswarm.demographics.targets import CensusTarget

__all__ = [
    # Core class
    "DemographicDistribution",
    # Divergence functions
    "kl_divergence",
    "js_divergence",
    "wasserstein_distance",
    "total_variation_distance",
    "coalition_demographic_divergence",
    # Preset targets
    "CensusTarget",
]
