"""
Unit tests for FairSwarm demographics module.

Tests for DemographicDistribution, divergence functions, and preset targets.
Validates that KL divergence matches Definition 2 from CLAUDE.md exactly.

Author: Tenicka Norwood
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairswarm.demographics import (
    CensusTarget,
    DemographicDistribution,
    coalition_demographic_divergence,
    js_divergence,
    kl_divergence,
    total_variation_distance,
    wasserstein_distance,
)
from fairswarm.demographics.distribution import combine_distributions
from fairswarm.demographics.divergence import is_epsilon_fair
from fairswarm.demographics.targets import (
    HealthcareTarget,
    create_custom_target,
    get_regional_target,
)


# =============================================================================
# DemographicDistribution Tests
# =============================================================================


class TestDemographicDistribution:
    """Tests for DemographicDistribution class."""

    def test_creation_from_array(self):
        """Test creating distribution from numpy array."""
        values = np.array([0.4, 0.3, 0.2, 0.1])
        dist = DemographicDistribution(values=values)

        assert dist.n_groups == 4
        assert np.allclose(dist.as_array(), values)

    def test_creation_from_dict(self):
        """Test creating distribution from dictionary."""
        dist = DemographicDistribution.from_dict({
            "white": 0.6,
            "black": 0.2,
            "hispanic": 0.15,
            "other": 0.05,
        })

        assert dist.n_groups == 4
        assert dist["white"] == 0.6
        assert dist["hispanic"] == 0.15
        assert dist.labels == ("white", "black", "hispanic", "other")

    def test_creation_from_counts(self):
        """Test creating distribution from counts (auto-normalizes)."""
        dist = DemographicDistribution.from_counts({
            "group_a": 600,
            "group_b": 300,
            "group_c": 100,
        })

        assert np.isclose(dist["group_a"], 0.6)
        assert np.isclose(dist["group_b"], 0.3)
        assert np.isclose(dist["group_c"], 0.1)
        assert np.isclose(np.sum(dist.as_array()), 1.0)

    def test_creation_fails_if_not_sum_to_one(self):
        """Test that values not summing to 1 raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1"):
            DemographicDistribution(values=np.array([0.5, 0.3, 0.1]))

    def test_creation_fails_with_negative_values(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DemographicDistribution(values=np.array([0.5, 0.6, -0.1]))

    def test_creation_with_normalize(self):
        """Test from_dict with normalize=True."""
        dist = DemographicDistribution.from_dict(
            {"a": 60, "b": 40},
            normalize=True,
        )
        assert np.isclose(dist["a"], 0.6)
        assert np.isclose(dist["b"], 0.4)

    def test_uniform_distribution(self):
        """Test creating uniform distribution."""
        dist = DemographicDistribution.uniform(5)

        assert dist.n_groups == 5
        assert np.allclose(dist.as_array(), [0.2, 0.2, 0.2, 0.2, 0.2])

    def test_uniform_with_labels(self):
        """Test creating uniform distribution with labels."""
        dist = DemographicDistribution.uniform(
            3,
            labels=["low", "medium", "high"],
        )

        assert dist.labels == ("low", "medium", "high")
        assert dist["medium"] == pytest.approx(1/3)

    def test_entropy_uniform_is_maximum(self):
        """Test that uniform distribution has maximum entropy."""
        uniform = DemographicDistribution.uniform(4)
        assert np.isclose(uniform.normalized_entropy, 1.0)

    def test_entropy_concentrated_is_low(self):
        """Test that concentrated distribution has low entropy."""
        concentrated = DemographicDistribution(
            values=np.array([0.97, 0.01, 0.01, 0.01])
        )
        assert concentrated.normalized_entropy < 0.3

    def test_as_dict_requires_labels(self):
        """Test that as_dict raises if no labels defined."""
        dist = DemographicDistribution(values=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="no labels"):
            dist.as_dict()

    def test_getitem_by_index(self):
        """Test accessing values by integer index."""
        dist = DemographicDistribution(values=np.array([0.4, 0.3, 0.2, 0.1]))
        assert dist[0] == 0.4
        assert dist[3] == 0.1

    def test_getitem_by_label(self):
        """Test accessing values by string label."""
        dist = DemographicDistribution.from_dict({"a": 0.7, "b": 0.3})
        assert dist["a"] == 0.7
        assert dist["b"] == 0.3

    def test_getitem_invalid_label_raises(self):
        """Test that invalid label raises KeyError."""
        dist = DemographicDistribution.from_dict({"a": 0.7, "b": 0.3})
        with pytest.raises(KeyError, match="not found"):
            _ = dist["c"]

    def test_immutability(self):
        """Test that distribution is immutable."""
        dist = DemographicDistribution(values=np.array([0.5, 0.5]))
        with pytest.raises(Exception):  # FrozenInstanceError
            dist.values = np.array([0.3, 0.7])

    def test_reorder(self):
        """Test reordering distribution labels."""
        dist = DemographicDistribution.from_dict({
            "a": 0.5, "b": 0.3, "c": 0.2
        })
        reordered = dist.reorder(["c", "a", "b"])

        assert reordered.labels == ("c", "a", "b")
        assert reordered["a"] == 0.5
        assert np.allclose(reordered.as_array(), [0.2, 0.5, 0.3])

    def test_items_iteration(self):
        """Test iterating over (label, value) pairs."""
        dist = DemographicDistribution.from_dict({
            "x": 0.6, "y": 0.4
        })
        items = list(dist.items())
        assert items == [("x", 0.6), ("y", 0.4)]


class TestCombineDistributions:
    """Tests for combine_distributions function."""

    def test_uniform_combination(self):
        """Test combining distributions with uniform weights."""
        d1 = DemographicDistribution(values=np.array([0.8, 0.2]))
        d2 = DemographicDistribution(values=np.array([0.2, 0.8]))

        combined = combine_distributions([d1, d2])

        assert np.allclose(combined.as_array(), [0.5, 0.5])

    def test_weighted_combination(self):
        """Test combining distributions with custom weights."""
        d1 = DemographicDistribution(values=np.array([1.0, 0.0]))
        d2 = DemographicDistribution(values=np.array([0.0, 1.0]))

        combined = combine_distributions([d1, d2], weights=[0.7, 0.3])

        assert np.allclose(combined.as_array(), [0.7, 0.3])

    def test_single_distribution(self):
        """Test combining single distribution returns same values."""
        d = DemographicDistribution(values=np.array([0.4, 0.4, 0.2]))
        combined = combine_distributions([d])
        assert np.allclose(combined.as_array(), d.as_array())

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            combine_distributions([])

    def test_mismatched_groups_raises(self):
        """Test that mismatched group counts raise ValueError."""
        d1 = DemographicDistribution(values=np.array([0.5, 0.5]))
        d2 = DemographicDistribution(values=np.array([0.3, 0.3, 0.4]))

        with pytest.raises(ValueError, match="same number of groups"):
            combine_distributions([d1, d2])


# =============================================================================
# KL Divergence Tests (Definition 2)
# =============================================================================


class TestKLDivergence:
    """
    Tests for KL divergence implementation.

    CRITICAL: These tests validate that kl_divergence matches
    Definition 2 from CLAUDE.md exactly.
    """

    def test_identical_distributions_zero_divergence(self):
        """KL divergence of identical distributions is 0."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_kl_is_non_negative(self):
        """KL divergence is always non-negative (Gibbs inequality)."""
        # Test many random distributions
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            assert kl_divergence(p, q) >= 0

    def test_kl_asymmetry(self):
        """KL divergence is asymmetric: D_KL(P||Q) != D_KL(Q||P)."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])

        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)

        assert kl_pq != pytest.approx(kl_qp, rel=0.1)

    def test_kl_known_value(self):
        """Test KL divergence against known analytical result."""
        # D_KL([0.5, 0.5] || [0.5, 0.5]) = 0
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-9)

        # D_KL([1, 0] || [0.5, 0.5]) = log(2) ≈ 0.693
        # (with smoothing, slightly different)
        p2 = np.array([0.99, 0.01])
        q2 = np.array([0.5, 0.5])
        kl = kl_divergence(p2, q2)
        assert 0.5 < kl < 1.0  # Should be close to log(2)

    def test_kl_with_demographic_distributions(self):
        """Test KL divergence with DemographicDistribution objects."""
        p = DemographicDistribution.from_dict({"a": 0.6, "b": 0.4})
        q = DemographicDistribution.from_dict({"a": 0.5, "b": 0.5})

        kl = kl_divergence(p, q)
        assert kl > 0
        assert kl < 1.0  # Should be small for similar distributions

    def test_kl_mismatched_length_raises(self):
        """Test that mismatched lengths raise ValueError."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="same length"):
            kl_divergence(p, q)

    def test_kl_formula_matches_definition_2(self):
        """
        Verify KL implementation matches Definition 2 formula exactly.

        Definition 2: D_KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
        """
        p = np.array([0.4, 0.3, 0.2, 0.1])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        # Manual calculation
        expected = sum(p[i] * math.log(p[i] / q[i]) for i in range(4))

        # Library calculation
        actual = kl_divergence(p, q)

        assert actual == pytest.approx(expected, rel=1e-6)


class TestJSDivergence:
    """Tests for Jensen-Shannon divergence."""

    def test_js_is_symmetric(self):
        """JS divergence is symmetric: JS(P||Q) = JS(Q||P)."""
        p = np.array([0.8, 0.2])
        q = np.array([0.3, 0.7])

        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))

    def test_js_identical_is_zero(self):
        """JS divergence of identical distributions is 0."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_js_bounded_by_log2(self):
        """JS divergence is bounded by log(2)."""
        # Maximum JS is achieved by non-overlapping distributions
        p = np.array([1.0 - 1e-10, 1e-10])
        q = np.array([1e-10, 1.0 - 1e-10])

        js = js_divergence(p, q)
        assert js <= np.log(2) + 1e-6


class TestTotalVariationDistance:
    """Tests for total variation distance."""

    def test_tv_identical_is_zero(self):
        """TV distance of identical distributions is 0."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert total_variation_distance(p, p) == pytest.approx(0.0)

    def test_tv_symmetric(self):
        """TV distance is symmetric."""
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        assert total_variation_distance(p, q) == total_variation_distance(q, p)

    def test_tv_bounded_by_one(self):
        """TV distance is bounded by 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert total_variation_distance(p, q) == pytest.approx(1.0)

    def test_tv_known_value(self):
        """Test TV distance against known value."""
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        # TV = 0.5 * (|0.5-0.9| + |0.5-0.1|) = 0.5 * (0.4 + 0.4) = 0.4
        assert total_variation_distance(p, q) == pytest.approx(0.4)


class TestWassersteinDistance:
    """Tests for Wasserstein distance."""

    def test_wasserstein_identical_is_zero(self):
        """Wasserstein distance of identical distributions is 0."""
        p = np.array([0.3, 0.4, 0.3])
        assert wasserstein_distance(p, p) == pytest.approx(0.0)

    def test_wasserstein_symmetric(self):
        """Wasserstein distance is symmetric."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        assert wasserstein_distance(p, q) == wasserstein_distance(q, p)


# =============================================================================
# Coalition Divergence Tests
# =============================================================================


class TestCoalitionDemographicDivergence:
    """Tests for coalition_demographic_divergence function."""

    def test_single_client_coalition(self):
        """Test divergence for single-client coalition."""
        demographics = [
            np.array([0.6, 0.4]),
            np.array([0.4, 0.6]),
        ]
        target = np.array([0.5, 0.5])

        # Coalition with just client 0
        div = coalition_demographic_divergence(demographics, [0], target)

        # Should equal KL divergence of client 0 from target
        expected = kl_divergence(demographics[0], target)
        assert div == pytest.approx(expected)

    def test_coalition_average_computation(self):
        """Test that coalition demographics are averaged correctly."""
        demographics = [
            np.array([0.8, 0.2]),  # Client 0
            np.array([0.2, 0.8]),  # Client 1
        ]
        target = np.array([0.5, 0.5])

        # Coalition of both clients: average = [0.5, 0.5] = target
        div = coalition_demographic_divergence(demographics, [0, 1], target)

        # Average matches target, so divergence should be ~0
        assert div == pytest.approx(0.0, abs=1e-9)

    def test_empty_coalition_raises(self):
        """Test that empty coalition raises ValueError."""
        demographics = [np.array([0.5, 0.5])]
        target = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="empty"):
            coalition_demographic_divergence(demographics, [], target)

    def test_invalid_index_raises(self):
        """Test that invalid coalition index raises ValueError."""
        demographics = [np.array([0.5, 0.5])]
        target = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="out of range"):
            coalition_demographic_divergence(demographics, [5], target)

    def test_with_demographic_distribution_objects(self):
        """Test with DemographicDistribution objects."""
        demographics = [
            DemographicDistribution(values=np.array([0.7, 0.3])),
            DemographicDistribution(values=np.array([0.3, 0.7])),
        ]
        target = DemographicDistribution(values=np.array([0.5, 0.5]))

        div = coalition_demographic_divergence(demographics, [0, 1], target)
        assert div == pytest.approx(0.0, abs=1e-9)


class TestEpsilonFairness:
    """Tests for is_epsilon_fair function."""

    def test_fair_when_below_epsilon(self):
        """Test that divergence below epsilon is fair."""
        assert is_epsilon_fair(0.03, epsilon=0.05)
        assert is_epsilon_fair(0.05, epsilon=0.05)  # Boundary

    def test_unfair_when_above_epsilon(self):
        """Test that divergence above epsilon is unfair."""
        assert not is_epsilon_fair(0.06, epsilon=0.05)
        assert not is_epsilon_fair(0.10, epsilon=0.05)


# =============================================================================
# Census Target Tests
# =============================================================================


class TestCensusTarget:
    """Tests for CensusTarget enum."""

    def test_us_2020_sums_to_one(self):
        """US 2020 census target sums to 1."""
        target = CensusTarget.US_2020
        assert np.isclose(sum(target.value.values()), 1.0)

    def test_us_2020_as_distribution(self):
        """Test converting US 2020 to DemographicDistribution."""
        dist = CensusTarget.US_2020.as_distribution()

        assert dist.n_groups == 5
        assert dist["white"] == 0.576
        assert dist["hispanic"] == 0.187

    def test_us_2020_as_array(self):
        """Test getting US 2020 as numpy array."""
        arr = CensusTarget.US_2020.as_array()

        assert len(arr) == 5
        assert arr[0] == 0.576  # white

    def test_us_2020_labels(self):
        """Test getting US 2020 labels."""
        labels = CensusTarget.US_2020.labels
        assert "white" in labels
        assert "black" in labels
        assert "hispanic" in labels

    def test_all_targets_valid_distributions(self):
        """Test all census targets are valid distributions."""
        for target in CensusTarget:
            arr = target.as_array()
            assert np.all(arr >= 0)
            assert np.isclose(np.sum(arr), 1.0)

    def test_us_2020_detailed_has_more_groups(self):
        """Test US 2020 detailed has more groups than basic."""
        basic = CensusTarget.US_2020
        detailed = CensusTarget.US_2020_DETAILED

        assert detailed.n_groups > basic.n_groups


class TestRegionalTargets:
    """Tests for regional demographic targets."""

    def test_get_regional_target_valid(self):
        """Test getting valid regional targets."""
        for region in ["northeast", "southeast", "midwest", "southwest", "west"]:
            target = get_regional_target(region)
            assert isinstance(target, DemographicDistribution)
            assert np.isclose(np.sum(target.as_array()), 1.0)

    def test_get_regional_target_invalid_raises(self):
        """Test that invalid region raises ValueError."""
        with pytest.raises(ValueError, match="Unknown region"):
            get_regional_target("nonexistent")

    def test_regional_demographics_differ(self):
        """Test that regional demographics differ from each other."""
        southeast = get_regional_target("southeast")
        midwest = get_regional_target("midwest")

        # Black population higher in Southeast
        assert southeast["black"] > midwest["black"]


class TestHealthcareTarget:
    """Tests for healthcare-specific targets."""

    def test_healthcare_targets_valid(self):
        """Test all healthcare targets are valid distributions."""
        for target in HealthcareTarget:
            arr = target.as_array()
            assert np.all(arr >= 0)
            assert np.isclose(np.sum(arr), 1.0)


class TestCustomTarget:
    """Tests for custom target creation."""

    def test_create_custom_target(self):
        """Test creating a custom target."""
        target = create_custom_target({
            "majority": 70,
            "minority_a": 20,
            "minority_b": 10,
        })

        assert target["majority"] == pytest.approx(0.7)
        assert target["minority_a"] == pytest.approx(0.2)
        assert target["minority_b"] == pytest.approx(0.1)
