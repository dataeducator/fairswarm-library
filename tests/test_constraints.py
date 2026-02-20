"""
Tests for FairSwarm constraints module.

Tests constraint evaluation, violation tracking, and constraint sets.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from fairswarm.constraints.base import ConstraintResult, ConstraintSet
from fairswarm.constraints.cardinality import (
    CardinalityConstraint,
    ExactSizeConstraint,
    MaxCostConstraint,
    MaxSizeConstraint,
    MinDataConstraint,
    MinSizeConstraint,
)
from fairswarm.constraints.fairness import (
    DivergenceConstraint,
    MinorityRepresentationConstraint,
    RepresentationConstraint,
    TotalVariationConstraint,
)
from fairswarm.constraints.privacy import (
    CompositionConstraint,
    LocalPrivacyConstraint,
    PrivacyBudgetConstraint,
    SensitivityConstraint,
)
from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.types import Demographics

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_clients():
    """Create sample clients for testing."""
    clients = []
    demographics_list = [
        Demographics(age=0.3, gender=0.5, race=0.2),
        Demographics(age=0.5, gender=0.5, race=0.3),
        Demographics(age=0.7, gender=0.6, race=0.4),
        Demographics(age=0.4, gender=0.4, race=0.5),
        Demographics(age=0.6, gender=0.5, race=0.6),
    ]

    for i, demo in enumerate(demographics_list):
        client = Client(
            id=f"client_{i}",
            num_samples=100 * (i + 1),
            demographics=demo,
            data_quality=0.7 + 0.05 * i,
            communication_cost=0.1 * (i + 1),
        )
        clients.append(client)

    return clients


@pytest.fixture
def target_demographics():
    """Target demographic distribution as DemographicDistribution."""
    return DemographicDistribution.from_demographics(
        Demographics(age=0.5, gender=0.5, race=0.4)
    )


# =============================================================================
# ConstraintResult Tests
# =============================================================================


class TestConstraintResult:
    """Tests for ConstraintResult dataclass."""

    def test_satisfied_result(self):
        """Test creating a satisfied constraint result."""
        result = ConstraintResult(satisfied=True, violation=0.0)
        assert result.satisfied is True
        assert result.violation == 0.0
        assert result.message == ""
        assert result.details == {}

    def test_unsatisfied_result(self):
        """Test creating an unsatisfied constraint result."""
        result = ConstraintResult(
            satisfied=False,
            violation=0.5,
            message="Constraint violated",
            details={"reason": "too small"},
        )
        assert result.satisfied is False
        assert result.violation == 0.5
        assert result.message == "Constraint violated"
        assert result.details["reason"] == "too small"

    def test_result_immutability(self):
        """Test that ConstraintResult is immutable (frozen dataclass)."""
        result = ConstraintResult(satisfied=True, violation=0.0)
        with pytest.raises(AttributeError):
            result.satisfied = False


# =============================================================================
# Cardinality Constraint Tests
# =============================================================================


class TestCardinalityConstraint:
    """Tests for CardinalityConstraint."""

    def test_within_bounds(self, sample_clients):
        """Test coalition within cardinality bounds."""
        constraint = CardinalityConstraint(min_size=2, max_size=4)
        coalition = [0, 1, 2]  # 3 clients

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True
        assert result.violation == 0.0

    def test_below_minimum(self, sample_clients):
        """Test coalition below minimum size."""
        constraint = CardinalityConstraint(min_size=3, max_size=5)
        coalition = [0, 1]  # 2 clients

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert result.violation > 0

    def test_above_maximum(self, sample_clients):
        """Test coalition above maximum size."""
        constraint = CardinalityConstraint(min_size=1, max_size=2)
        coalition = [0, 1, 2, 3]  # 4 clients

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert result.violation > 0

    def test_empty_coalition(self, sample_clients):
        """Test empty coalition with non-zero minimum."""
        constraint = CardinalityConstraint(min_size=1, max_size=5)
        coalition = []

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False

    @given(
        min_size=st.integers(min_value=1, max_value=10),
        max_size=st.integers(min_value=1, max_value=10),
    )
    def test_invalid_bounds_raises(self, min_size, max_size):
        """Test that invalid bounds raise ValueError."""
        assume(min_size > max_size)
        with pytest.raises(ValueError):
            CardinalityConstraint(min_size=min_size, max_size=max_size)


class TestMinSizeConstraint:
    """Tests for MinSizeConstraint."""

    def test_meets_minimum(self, sample_clients):
        """Test coalition meets minimum size."""
        constraint = MinSizeConstraint(min_size=2)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_below_minimum(self, sample_clients):
        """Test coalition below minimum size."""
        constraint = MinSizeConstraint(min_size=4)
        coalition = [0, 1]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert result.violation == 2  # 4 - 2

    def test_is_hard_constraint(self):
        """Test that MinSizeConstraint is a hard constraint."""
        constraint = MinSizeConstraint(min_size=2)
        assert constraint.is_hard_constraint() is True


class TestMaxSizeConstraint:
    """Tests for MaxSizeConstraint."""

    def test_within_maximum(self, sample_clients):
        """Test coalition within maximum size."""
        constraint = MaxSizeConstraint(max_size=4)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_exceeds_maximum(self, sample_clients):
        """Test coalition exceeds maximum size."""
        constraint = MaxSizeConstraint(max_size=2)
        coalition = [0, 1, 2, 3]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert result.violation == 2  # 4 - 2


class TestExactSizeConstraint:
    """Tests for ExactSizeConstraint."""

    def test_exact_match(self, sample_clients):
        """Test coalition with exact size."""
        constraint = ExactSizeConstraint(exact_size=3)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True
        assert result.violation == 0

    def test_wrong_size(self, sample_clients):
        """Test coalition with wrong size."""
        constraint = ExactSizeConstraint(exact_size=3)
        coalition = [0, 1]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert result.violation == 1  # |3 - 2| = 1


class TestMinDataConstraint:
    """Tests for MinDataConstraint."""

    def test_meets_data_requirement(self, sample_clients):
        """Test coalition meets minimum data requirement."""
        # Clients have 100, 200, 300, 400, 500 samples
        constraint = MinDataConstraint(min_samples=500)
        coalition = [0, 1, 2]  # 100 + 200 + 300 = 600

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_below_data_requirement(self, sample_clients):
        """Test coalition below minimum data requirement."""
        constraint = MinDataConstraint(min_samples=1000)
        coalition = [0, 1]  # 100 + 200 = 300

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False


class TestMaxCostConstraint:
    """Tests for MaxCostConstraint."""

    def test_within_budget(self, sample_clients):
        """Test coalition within communication budget."""
        # Clients have costs 0.1, 0.2, 0.3, 0.4, 0.5
        constraint = MaxCostConstraint(max_cost=1.0)
        coalition = [0, 1, 2]  # 0.1 + 0.2 + 0.3 = 0.6

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_exceeds_budget(self, sample_clients):
        """Test coalition exceeds communication budget."""
        constraint = MaxCostConstraint(max_cost=0.5)
        coalition = [0, 1, 2, 3]  # 0.1 + 0.2 + 0.3 + 0.4 = 1.0

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False


# =============================================================================
# Fairness Constraint Tests
# =============================================================================


class TestDivergenceConstraint:
    """Tests for DivergenceConstraint (Theorem 2)."""

    def test_low_divergence(self, sample_clients, target_demographics):
        """Test coalition with low demographic divergence."""
        constraint = DivergenceConstraint(
            target_distribution=target_demographics,
            epsilon=1.0,  # Lenient threshold
        )
        coalition = [0, 1, 2, 3, 4]  # All clients

        result = constraint.evaluate(coalition, sample_clients)
        # With all clients, divergence should be reasonable
        assert result.violation >= 0  # Violation is non-negative

    def test_empty_coalition_handling(self, sample_clients, target_demographics):
        """Test empty coalition handling."""
        constraint = DivergenceConstraint(
            target_distribution=target_demographics,
            epsilon=0.1,
        )
        coalition = []

        result = constraint.evaluate(coalition, sample_clients)
        # Source code returns satisfied=False for empty coalition
        assert result.satisfied is False

    def test_epsilon_threshold(self, sample_clients, target_demographics):
        """Test that tighter epsilon leads to more violations."""
        coalition = [0, 1]

        lenient = DivergenceConstraint(
            target_distribution=target_demographics, epsilon=10.0
        )
        strict = DivergenceConstraint(
            target_distribution=target_demographics, epsilon=0.001
        )

        lenient_result = lenient.evaluate(coalition, sample_clients)
        strict_result = strict.evaluate(coalition, sample_clients)

        # Strict constraint should have same or higher violation
        assert lenient_result.violation <= strict_result.violation or (
            lenient_result.satisfied and not strict_result.satisfied
        )

    def test_is_soft_constraint(self, target_demographics):
        """Test that DivergenceConstraint is soft by default."""
        constraint = DivergenceConstraint(
            target_distribution=target_demographics,
            epsilon=0.1,
        )
        assert constraint.is_hard_constraint() is False


class TestRepresentationConstraint:
    """Tests for RepresentationConstraint."""

    def test_lenient_threshold(self, sample_clients, target_demographics):
        """Test coalition with lenient threshold."""
        constraint = RepresentationConstraint(
            target_distribution=target_demographics,
            threshold=1.0,  # Very lenient
        )
        coalition = [0, 1, 2, 3, 4]  # All clients

        result = constraint.evaluate(coalition, sample_clients)
        # With lenient threshold, should be satisfied
        assert result.satisfied

    def test_strict_threshold(self, sample_clients, target_demographics):
        """Test coalition with strict threshold."""
        constraint = RepresentationConstraint(
            target_distribution=target_demographics,
            threshold=0.001,  # Very strict
        )
        coalition = [0]  # Single client

        result = constraint.evaluate(coalition, sample_clients)
        # Very strict threshold likely violated
        assert result.violation >= 0


class TestMinorityRepresentationConstraint:
    """Tests for MinorityRepresentationConstraint."""

    def test_auto_detect_minority_groups(self, target_demographics):
        """Test auto-detection of minority groups."""
        constraint = MinorityRepresentationConstraint(
            target_distribution=target_demographics,
            min_representation=0.05,
        )
        # Should auto-detect minority groups from target distribution
        assert isinstance(constraint.minority_groups, list)

    def test_explicit_minority_groups(self, sample_clients, target_demographics):
        """Test with explicitly specified minority groups."""
        constraint = MinorityRepresentationConstraint(
            target_distribution=target_demographics,
            minority_groups=["age"],
            min_representation=0.01,
        )
        coalition = [0, 1, 2, 3, 4]

        result = constraint.evaluate(coalition, sample_clients)
        # Check that result has expected structure
        assert result.violation >= 0
        assert "violations" in result.details


class TestTotalVariationConstraint:
    """Tests for TotalVariationConstraint."""

    def test_low_tv_distance(self, sample_clients, target_demographics):
        """Test coalition with low total variation distance."""
        constraint = TotalVariationConstraint(
            target_distribution=target_demographics,
            threshold=1.0,  # Lenient
        )
        coalition = [0, 1, 2, 3, 4]

        result = constraint.evaluate(coalition, sample_clients)
        # TV distance should be bounded
        assert result.details.get("tv_distance", 0) >= 0


# =============================================================================
# Privacy Constraint Tests
# =============================================================================


class TestPrivacyBudgetConstraint:
    """Tests for PrivacyBudgetConstraint."""

    def test_within_budget(self, sample_clients):
        """Test privacy budget not exhausted."""
        constraint = PrivacyBudgetConstraint(epsilon_budget=10.0, delta=1e-5)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True
        assert result.details["remaining"] == 10.0

    def test_record_and_check(self, sample_clients):
        """Test recording queries and checking budget."""
        constraint = PrivacyBudgetConstraint(epsilon_budget=1.0, delta=1e-5)

        # Record some queries
        constraint.record_query(0.3)
        constraint.record_query(0.3)
        constraint.record_query(0.3)

        result = constraint.evaluate([], sample_clients)
        assert result.details["consumed"] == pytest.approx(0.9, rel=1e-5)
        assert result.satisfied is True

        # Strict mode (default): exceeding budget raises PrivacyBudgetExhausted
        from fairswarm.constraints.privacy import PrivacyBudgetExhausted

        with pytest.raises(PrivacyBudgetExhausted):
            constraint.record_query(0.2)

    def test_record_and_check_non_strict(self, sample_clients):
        """Test that strict=False allows exceeding budget (soft check)."""
        constraint = PrivacyBudgetConstraint(
            epsilon_budget=1.0, delta=1e-5, strict=False
        )
        constraint.record_query(0.3)
        constraint.record_query(0.3)
        constraint.record_query(0.3)
        constraint.record_query(0.2)  # Exceeds budget — no exception

        result = constraint.evaluate([], sample_clients)
        assert result.satisfied is False  # 1.1 > 1.0

    def test_reset(self, sample_clients):
        """Test resetting privacy budget."""
        constraint = PrivacyBudgetConstraint(epsilon_budget=1.0, delta=1e-5)
        constraint.record_query(0.5)
        constraint.record_query(0.5)

        constraint.reset()

        result = constraint.evaluate([], sample_clients)
        assert result.details["consumed"] == 0.0
        assert result.details["remaining"] == 1.0

    def test_invalid_epsilon_raises(self):
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError):
            PrivacyBudgetConstraint(epsilon_budget=0.0)
        with pytest.raises(ValueError):
            PrivacyBudgetConstraint(epsilon_budget=-1.0)

    def test_invalid_delta_raises(self):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError):
            PrivacyBudgetConstraint(epsilon_budget=1.0, delta=0.0)
        with pytest.raises(ValueError):
            PrivacyBudgetConstraint(epsilon_budget=1.0, delta=1.0)


class TestLocalPrivacyConstraint:
    """Tests for LocalPrivacyConstraint."""

    def test_all_clients_within_bounds(self, sample_clients):
        """Test all clients within privacy bounds."""
        # Add privacy_epsilon to clients
        for client in sample_clients:
            object.__setattr__(client, "privacy_epsilon", 5.0)

        constraint = LocalPrivacyConstraint(min_epsilon=1.0, max_epsilon=10.0)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_client_below_minimum(self, sample_clients):
        """Test client below minimum epsilon."""
        for i, client in enumerate(sample_clients):
            object.__setattr__(client, "privacy_epsilon", 0.5 if i == 0 else 5.0)

        constraint = LocalPrivacyConstraint(min_epsilon=1.0, max_epsilon=10.0)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False
        assert len(result.details["violations"]) == 1

    def test_client_above_maximum(self, sample_clients):
        """Test client above maximum epsilon."""
        for i, client in enumerate(sample_clients):
            object.__setattr__(client, "privacy_epsilon", 15.0 if i == 1 else 5.0)

        constraint = LocalPrivacyConstraint(min_epsilon=1.0, max_epsilon=10.0)
        coalition = [0, 1, 2]

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied is False


class TestSensitivityConstraint:
    """Tests for SensitivityConstraint."""

    def test_within_sensitivity_bound(self, sample_clients):
        """Test coalition within sensitivity bound."""
        for client in sample_clients:
            object.__setattr__(client, "sensitivity", 0.2)

        constraint = SensitivityConstraint(max_sensitivity=1.0)
        coalition = [0, 1, 2]  # sqrt(3 * 0.04) ~ 0.35

        result = constraint.evaluate(coalition, sample_clients)
        assert result.satisfied

    def test_exceeds_sensitivity_bound(self, sample_clients):
        """Test coalition exceeds sensitivity bound."""
        for client in sample_clients:
            object.__setattr__(client, "sensitivity", 1.0)

        constraint = SensitivityConstraint(max_sensitivity=1.0)
        coalition = [0, 1, 2, 3, 4]  # sqrt(5) ~ 2.24

        result = constraint.evaluate(coalition, sample_clients)
        assert not result.satisfied

    def test_empty_coalition(self, sample_clients):
        """Test empty coalition has zero sensitivity."""
        constraint = SensitivityConstraint(max_sensitivity=1.0)
        result = constraint.evaluate([], sample_clients)
        assert result.satisfied is True


class TestCompositionConstraint:
    """Tests for CompositionConstraint."""

    def test_within_query_limit(self, sample_clients):
        """Test within maximum queries."""
        constraint = CompositionConstraint(
            epsilon_per_query=0.1,
            max_queries=100,
        )

        for _ in range(50):
            constraint.record_query()

        result = constraint.evaluate([], sample_clients)
        assert result.satisfied is True
        assert result.details["query_count"] == 50

    def test_exceeds_query_limit(self, sample_clients):
        """Test exceeds maximum queries."""
        constraint = CompositionConstraint(
            epsilon_per_query=0.1,
            max_queries=10,
        )

        for _ in range(15):
            constraint.record_query()

        result = constraint.evaluate([], sample_clients)
        assert result.satisfied is False

    def test_basic_composition(self, sample_clients):
        """Test basic composition theorem."""
        constraint = CompositionConstraint(
            epsilon_per_query=0.1,
            max_queries=100,
            composition_type="basic",
        )

        for _ in range(10):
            constraint.record_query()

        composed = constraint.get_composed_epsilon()
        assert composed == pytest.approx(1.0, rel=1e-5)  # 10 * 0.1

    def test_advanced_composition(self, sample_clients):
        """Test advanced composition theorem gives different result."""
        constraint = CompositionConstraint(
            epsilon_per_query=0.1,
            delta_per_query=1e-5,
            max_queries=100,
            composition_type="advanced",
        )

        for _ in range(10):
            constraint.record_query()

        composed = constraint.get_composed_epsilon()
        basic_composed = 10 * 0.1

        # Advanced composition should be different from basic
        # (typically tighter for small epsilon)
        assert composed != pytest.approx(basic_composed, rel=0.01)

    def test_reset(self, sample_clients):
        """Test resetting query count."""
        constraint = CompositionConstraint(epsilon_per_query=0.1, max_queries=10)

        for _ in range(5):
            constraint.record_query()

        constraint.reset()

        result = constraint.evaluate([], sample_clients)
        assert result.details["query_count"] == 0


# =============================================================================
# ConstraintSet Tests
# =============================================================================


class TestConstraintSet:
    """Tests for ConstraintSet."""

    def test_all_satisfied(self, sample_clients):
        """Test all constraints satisfied."""
        constraints = ConstraintSet(
            [
                MinSizeConstraint(min_size=2),
                MaxSizeConstraint(max_size=5),
            ]
        )
        coalition = [0, 1, 2]

        result = constraints.evaluate(coalition, sample_clients)
        assert result.satisfied is True

    def test_one_violated(self, sample_clients):
        """Test one constraint violated."""
        constraints = ConstraintSet(
            [
                MinSizeConstraint(min_size=2),
                MaxSizeConstraint(max_size=2),  # Will be violated
            ]
        )
        coalition = [0, 1, 2]  # 3 clients

        result = constraints.evaluate(coalition, sample_clients)
        assert result.satisfied is False

    def test_hard_constraints_checked(self, sample_clients):
        """Test hard constraints are checked."""
        constraints = ConstraintSet(
            [
                MinSizeConstraint(min_size=5),  # Hard, will fail
            ]
        )
        coalition = [0, 1, 2]

        result = constraints.evaluate_hard_only(coalition, sample_clients)
        assert result.satisfied is False

    def test_total_violation(self, sample_clients):
        """Test total violation computation."""
        constraints = ConstraintSet(
            [
                MinSizeConstraint(min_size=5),  # Violation = 2
                MaxCostConstraint(max_cost=0.1),  # Will have some violation
            ]
        )
        coalition = [0, 1, 2]  # Size 3, cost 0.6

        result = constraints.evaluate(coalition, sample_clients)
        assert result.violation >= 2  # At least the cardinality violation

    def test_add_constraint(self, sample_clients):
        """Test adding constraint dynamically."""
        constraints = ConstraintSet([MinSizeConstraint(min_size=2)])
        constraints.add(MaxSizeConstraint(max_size=4))

        assert len(constraints.constraints) == 2

        coalition = [0, 1, 2]
        result = constraints.evaluate(coalition, sample_clients)
        # evaluate returns a single ConstraintResult, check it
        assert result.satisfied is True

    def test_compute_total_penalty(self, sample_clients):
        """Test total penalty computation."""
        min_constraint = MinSizeConstraint(min_size=5)
        max_constraint = MaxSizeConstraint(max_size=10)

        constraints = ConstraintSet([min_constraint, max_constraint])
        coalition = [0, 1, 2]

        penalty = constraints.compute_total_penalty(coalition, sample_clients)
        assert penalty >= 2  # MinSize violation: 5-3=2


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestConstraintProperties:
    """Property-based tests for constraints."""

    @given(
        coalition_size=st.integers(min_value=0, max_value=5),
        min_size=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_min_size_violation_nonnegative(
        self, coalition_size, min_size, sample_clients
    ):
        """Violation is always non-negative."""
        constraint = MinSizeConstraint(min_size=min_size)
        coalition = list(range(min(coalition_size, len(sample_clients))))

        result = constraint.evaluate(coalition, sample_clients)
        assert result.violation >= 0

    @given(
        epsilon=st.floats(min_value=0.01, max_value=10.0),
        n_queries=st.integers(min_value=0, max_value=20),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_privacy_budget_monotonic(self, epsilon, n_queries, sample_clients):
        """Privacy consumption increases monotonically."""
        constraint = PrivacyBudgetConstraint(
            epsilon_budget=100.0,
            strict=False,
        )

        previous_remaining = constraint.get_remaining_budget()

        for _ in range(n_queries):
            constraint.record_query(epsilon)
            current_remaining = constraint.get_remaining_budget()
            assert current_remaining <= previous_remaining
            previous_remaining = current_remaining

    @given(
        eps_per_query=st.floats(min_value=0.01, max_value=1.0),
        n_queries=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_basic_composition_linear(self, eps_per_query, n_queries, sample_clients):
        """Basic composition is linear in number of queries."""
        constraint = CompositionConstraint(
            epsilon_per_query=eps_per_query,
            max_queries=1000,
            composition_type="basic",
        )

        for _ in range(n_queries):
            constraint.record_query()

        composed = constraint.get_composed_epsilon()
        expected = n_queries * eps_per_query

        assert composed == pytest.approx(expected, rel=1e-5)
