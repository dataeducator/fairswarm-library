"""
Preset demographic distribution targets.

This module provides standard demographic distribution targets
for use in fair coalition selection, including US Census data.

Research Context:
    Fair federated learning aims to select coalitions whose
    combined demographics match population-level distributions.
    The US Census provides authoritative demographic baselines.

Data Sources:
    - US Census 2020: https://www.census.gov/2020census
    - US Census 2010: https://www.census.gov/2010census

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from fairswarm.demographics.distribution import DemographicDistribution


class CensusTarget(Enum):
    """
    Preset demographic targets from US Census data.

    These targets represent population-level demographic distributions
    that can be used as fairness objectives in FairSwarm optimization.

    Available Targets:
        US_2020: US Census 2020 racial demographics (5 groups)
        US_2020_DETAILED: US Census 2020 with more categories (7 groups)
        US_2010: US Census 2010 racial demographics (5 groups)
        UNIFORM_4: Uniform distribution over 4 groups
        UNIFORM_5: Uniform distribution over 5 groups

    Example:
        >>> from fairswarm.demographics import CensusTarget
        >>> target = CensusTarget.US_2020
        >>> distribution = target.as_distribution()
        >>> print(distribution["white"])
        0.576

    Note:
        These distributions are simplified for demonstration.
        Real applications should use the most current and
        appropriate demographic categories for their context.
    """

    # US Census 2020 - Simplified 5-category racial demographics
    # Source: 2020 Census Redistricting Data (P.L. 94-171)
    # Categories: White, Black, Hispanic/Latino, Asian, Other
    US_2020 = {
        "white": 0.576,  # White alone, not Hispanic
        "black": 0.124,  # Black alone
        "hispanic": 0.187,  # Hispanic or Latino (any race)
        "asian": 0.061,  # Asian alone
        "other": 0.052,  # Other (including multiracial, Native American, Pacific Islander)
    }

    # US Census 2020 - More detailed 7-category breakdown
    US_2020_DETAILED = {
        "white": 0.576,
        "black": 0.124,
        "hispanic": 0.187,
        "asian": 0.061,
        "native_american": 0.013,
        "pacific_islander": 0.003,
        "multiracial": 0.036,
    }

    # US Census 2010 - For comparison/historical analysis
    US_2010 = {
        "white": 0.639,
        "black": 0.126,
        "hispanic": 0.163,
        "asian": 0.048,
        "other": 0.024,
    }

    # Uniform distributions for testing/baselines
    UNIFORM_4 = {
        "group_1": 0.25,
        "group_2": 0.25,
        "group_3": 0.25,
        "group_4": 0.25,
    }

    UNIFORM_5 = {
        "group_1": 0.20,
        "group_2": 0.20,
        "group_3": 0.20,
        "group_4": 0.20,
        "group_5": 0.20,
    }

    def as_distribution(self) -> DemographicDistribution:
        """
        Convert the target to a DemographicDistribution object.

        Returns:
            DemographicDistribution with labeled groups

        Example:
            >>> target = CensusTarget.US_2020
            >>> dist = target.as_distribution()
            >>> dist["hispanic"]
            0.187
        """
        return DemographicDistribution.from_dict(self.value)

    def as_array(self) -> np.ndarray:
        """
        Get the target as a numpy array.

        Returns:
            Numpy array of probability values

        Example:
            >>> CensusTarget.US_2020.as_array()
            array([0.576, 0.124, 0.187, 0.061, 0.052])
        """
        return np.array(list(self.value.values()), dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        """
        Get the target as a dictionary.

        Returns:
            Dictionary mapping group names to proportions
        """
        data: dict[str, float] = self.value
        return dict(data)

    @property
    def labels(self) -> tuple[str, ...]:
        """Get the demographic group labels."""
        data: dict[str, float] = self.value
        return tuple(data.keys())

    @property
    def n_groups(self) -> int:
        """Number of demographic groups."""
        data: dict[str, float] = self.value
        return len(data)


# =============================================================================
# Regional Target Functions
# =============================================================================


def get_regional_target(region: str) -> DemographicDistribution:
    """
    Get demographic target for a specific US region.

    Regional demographics differ significantly from national averages.
    These targets can be used for region-specific fairness optimization.

    Args:
        region: One of "northeast", "southeast", "midwest", "southwest", "west"

    Returns:
        DemographicDistribution for the region

    Raises:
        ValueError: If region is unknown

    Example:
        >>> southeast = get_regional_target("southeast")
        >>> southeast["black"]  # Higher Black population in Southeast
        0.19

    Note:
        These are approximate regional demographics for illustration.
        Production use should employ official census data for specific areas.
    """
    # Approximate regional demographics (simplified)
    regional_data = {
        "northeast": {
            "white": 0.62,
            "black": 0.12,
            "hispanic": 0.15,
            "asian": 0.08,
            "other": 0.03,
        },
        "southeast": {
            "white": 0.55,
            "black": 0.19,
            "hispanic": 0.14,
            "asian": 0.04,
            "other": 0.08,
        },
        "midwest": {
            "white": 0.72,
            "black": 0.10,
            "hispanic": 0.09,
            "asian": 0.04,
            "other": 0.05,
        },
        "southwest": {
            "white": 0.42,
            "black": 0.06,
            "hispanic": 0.40,
            "asian": 0.05,
            "other": 0.07,
        },
        "west": {
            "white": 0.50,
            "black": 0.05,
            "hispanic": 0.28,
            "asian": 0.12,
            "other": 0.05,
        },
    }

    region_lower = region.lower()
    if region_lower not in regional_data:
        available = list(regional_data.keys())
        raise ValueError(f"Unknown region '{region}'. Available: {available}")

    return DemographicDistribution.from_dict(regional_data[region_lower])


def create_custom_target(
    proportions: dict[str, float],
    normalize: bool = True,
) -> DemographicDistribution:
    """
    Create a custom demographic target.

    Args:
        proportions: Dictionary mapping group names to proportions
        normalize: If True, normalize values to sum to 1

    Returns:
        DemographicDistribution representing the target

    Example:
        >>> target = create_custom_target({
        ...     "group_a": 40,
        ...     "group_b": 35,
        ...     "group_c": 25,
        ... })
        >>> target["group_a"]
        0.4
    """
    return DemographicDistribution.from_dict(proportions, normalize=normalize)


# =============================================================================
# Healthcare-Specific Targets
# =============================================================================


class HealthcareTarget(Enum):
    """
    Healthcare-specific demographic targets.

    These targets are designed for healthcare federated learning,
    where patient population demographics may differ from general
    population demographics.

    Note:
        These are illustrative examples. Real healthcare applications
        should use institution-specific or study-specific targets
        developed with clinical and ethical oversight.
    """

    # ICU patient demographics (typically older, different racial mix)
    ICU_TYPICAL = {
        "white": 0.65,
        "black": 0.15,
        "hispanic": 0.12,
        "asian": 0.04,
        "other": 0.04,
    }

    # Diabetes study population (higher prevalence in certain groups)
    DIABETES_STUDY = {
        "white": 0.45,
        "black": 0.22,
        "hispanic": 0.20,
        "asian": 0.08,
        "other": 0.05,
    }

    # Cardiovascular disease study
    CARDIOVASCULAR_STUDY = {
        "white": 0.50,
        "black": 0.25,
        "hispanic": 0.15,
        "asian": 0.05,
        "other": 0.05,
    }

    def as_distribution(self) -> DemographicDistribution:
        """Convert to DemographicDistribution."""
        return DemographicDistribution.from_dict(self.value)

    def as_array(self) -> np.ndarray:
        """Get as numpy array."""
        return np.array(list(self.value.values()), dtype=np.float64)
