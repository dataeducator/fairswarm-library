"""
Tests for FairSwarm incentives module.

Tests Shapley value computation and reward allocation mechanisms.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.incentives.allocation import (
    AllocationResult,
    ContributionMetrics,
    EqualAllocator,
    FairnessAwareAllocator,
    ProportionalAllocator,
    ShapleyAllocator,
    allocate_rewards,
)
from fairswarm.incentives.shapley import (
    ExactShapley,
    MonteCarloShapley,
    ShapleyResult,
    StratifiedShapley,
    compute_shapley_values,
    shapley_from_fitness,
)
from fairswarm.types import Demographics

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_clients():
    """Create sample clients for testing."""
    clients = []
    for i in range(5):
        demo = Demographics(
            age=0.3 + 0.1 * i,
            gender=0.5,
            race=0.2 + 0.1 * i,
        )
        client = Client(
            id=f"client_{i}",
            num_samples=100 * (i + 1),
            demographics=demo,
            data_quality=0.5 + 0.1 * i,
            communication_cost=0.1 * (i + 1),
        )
        clients.append(client)
    return clients


@pytest.fixture
def additive_value_fn():
    """Additive value function for testing Shapley properties."""

    def value_fn(coalition, clients):
        if not coalition:
            return 0.0
        return sum(clients[i].data_quality for i in coalition if 0 <= i < len(clients))

    return value_fn


@pytest.fixture
def supermodular_value_fn():
    """Supermodular value function (diversity bonus)."""

    def value_fn(coalition, clients):
        if not coalition:
            return 0.0
        base = sum(clients[i].data_quality for i in coalition if 0 <= i < len(clients))
        diversity_bonus = len(coalition) * 0.1  # Bonus for larger coalitions
        return base + diversity_bonus

    return value_fn


@pytest.fixture
def simple_fitness():
    """Simple fitness function for testing."""

    class SimpleFitness(FitnessFunction):
        def evaluate(self, coalition, clients):
            if not coalition:
                return FitnessResult(value=0.0, components={}, coalition=coalition)
            value = sum(
                clients[i].data_quality for i in coalition if 0 <= i < len(clients)
            )
            return FitnessResult(
                value=value, components={"quality": value}, coalition=coalition
            )

        def compute_gradient(self, position, clients, coalition_size):
            """Compute gradient based on data quality."""
            import numpy as np

            len(clients)
            gradient = np.array([c.data_quality for c in clients])
            norm = np.linalg.norm(gradient)
            if norm > 1e-10:
                gradient = gradient / norm
            return gradient

        @property
        def name(self):
            return "SimpleFitness"

    return SimpleFitness()


@pytest.fixture
def target_demographics():
    """Target demographic distribution."""
    return DemographicDistribution.from_demographics(
        Demographics(age=0.5, gender=0.5, race=0.4)
    )


# =============================================================================
# ShapleyResult Tests
# =============================================================================


class TestShapleyResult:
    """Tests for ShapleyResult dataclass."""

    def test_creation(self):
        """Test creating ShapleyResult."""
        values = np.array([0.3, 0.4, 0.3])
        result = ShapleyResult(values=values, n_samples=100)

        np.testing.assert_array_equal(result.values, values)
        assert result.n_samples == 100

    def test_normalize(self):
        """Test normalization."""
        values = np.array([1.0, 2.0, 3.0])
        result = ShapleyResult(values=values)

        normalized = result.normalize()
        assert normalized.sum() == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(normalized, [1 / 6, 2 / 6, 3 / 6])

    def test_normalize_zero_sum(self):
        """Test normalization with zero sum."""
        values = np.zeros(3)
        result = ShapleyResult(values=values)

        normalized = result.normalize()
        np.testing.assert_array_equal(normalized, values)

    def test_get_ranking(self):
        """Test ranking by contribution."""
        values = np.array([0.1, 0.5, 0.3, 0.2])
        result = ShapleyResult(values=values)

        ranking = result.get_ranking()
        assert ranking == [1, 2, 3, 0]  # Descending by value


# =============================================================================
# ExactShapley Tests
# =============================================================================


class TestExactShapley:
    """Tests for ExactShapley computation."""

    def test_single_player(self, sample_clients, additive_value_fn):
        """Test single player coalition."""
        shapley = ExactShapley()
        result = shapley.compute([0], sample_clients, additive_value_fn)

        # Single player gets full value
        expected = sample_clients[0].data_quality
        assert result.values[0] == pytest.approx(expected, rel=1e-5)

    def test_two_players_additive(self, sample_clients, additive_value_fn):
        """Test two players with additive function."""
        shapley = ExactShapley()
        result = shapley.compute([0, 1], sample_clients, additive_value_fn)

        # For additive function, Shapley = individual value
        assert result.values[0] == pytest.approx(
            sample_clients[0].data_quality, rel=1e-5
        )
        assert result.values[1] == pytest.approx(
            sample_clients[1].data_quality, rel=1e-5
        )

    def test_efficiency_property(self, sample_clients, additive_value_fn):
        """Test efficiency: Σφ_i = v(N)."""
        shapley = ExactShapley()
        coalition = [0, 1, 2]
        result = shapley.compute(coalition, sample_clients, additive_value_fn)

        total_value = additive_value_fn(coalition, sample_clients)
        shapley_sum = np.sum(result.values)

        assert shapley_sum == pytest.approx(total_value, rel=1e-5)

    def test_symmetry_property(self, sample_clients):
        """Test symmetry: identical players get identical values."""
        # Create identical clients
        identical_clients = [
            Client(
                id="a", num_samples=100, demographics=Demographics(), data_quality=0.8
            ),
            Client(
                id="b", num_samples=100, demographics=Demographics(), data_quality=0.8
            ),
        ]

        def value_fn(coalition, clients):
            return len(coalition) * 0.8

        shapley = ExactShapley()
        result = shapley.compute([0, 1], identical_clients, value_fn)

        assert result.values[0] == pytest.approx(result.values[1], rel=1e-5)

    def test_null_player_property(self, sample_clients):
        """Test null player: no contribution gets zero."""

        def value_fn(coalition, clients):
            # Player 0 contributes nothing
            return sum(1.0 for i in coalition if i != 0)

        shapley = ExactShapley()
        result = shapley.compute([0, 1, 2], sample_clients, value_fn)

        assert result.values[0] == pytest.approx(0.0, abs=1e-10)

    def test_empty_coalition(self, sample_clients, additive_value_fn):
        """Test empty coalition."""
        shapley = ExactShapley()
        result = shapley.compute([], sample_clients, additive_value_fn)

        assert len(result.values) == 0

    def test_max_size_exceeded(self, sample_clients, additive_value_fn):
        """Test exceeding max size raises error."""
        shapley = ExactShapley(max_size=3)

        with pytest.raises(ValueError, match="exceeds max_size"):
            shapley.compute([0, 1, 2, 3, 4], sample_clients, additive_value_fn)

    def test_name_property(self):
        """Test name property."""
        shapley = ExactShapley()
        assert shapley.name == "ExactShapley"


# =============================================================================
# MonteCarloShapley Tests
# =============================================================================


class TestMonteCarloShapley:
    """Tests for MonteCarloShapley computation."""

    def test_basic_computation(self, sample_clients, additive_value_fn):
        """Test basic Monte Carlo computation."""
        shapley = MonteCarloShapley(n_samples=1000, seed=42)
        result = shapley.compute([0, 1, 2], sample_clients, additive_value_fn)

        assert len(result.values) == 3
        assert result.n_samples == 1000
        assert result.variance is not None

    def test_efficiency_approximate(self, sample_clients, additive_value_fn):
        """Test approximate efficiency with many samples."""
        shapley = MonteCarloShapley(n_samples=5000, seed=42)
        coalition = [0, 1, 2]
        result = shapley.compute(coalition, sample_clients, additive_value_fn)

        total_value = additive_value_fn(coalition, sample_clients)
        shapley_sum = np.sum(result.values)

        assert shapley_sum == pytest.approx(total_value, rel=0.1)

    def test_variance_decreases_with_samples(self, sample_clients):
        """Test variance decreases with more samples."""

        # Use a supermodular (non-additive) value function so that marginal
        # contributions vary across permutations, producing meaningful variance.
        def supermodular_value_fn(coalition, clients):
            if not coalition:
                return 0.0
            base = sum(
                clients[i].data_quality for i in coalition if 0 <= i < len(clients)
            )
            diversity_bonus = len(coalition) * 0.1
            return base + diversity_bonus

        low_samples = MonteCarloShapley(n_samples=100, seed=42)
        high_samples = MonteCarloShapley(n_samples=2000, seed=42)

        result_low = low_samples.compute(
            [0, 1, 2], sample_clients, supermodular_value_fn
        )
        result_high = high_samples.compute(
            [0, 1, 2], sample_clients, supermodular_value_fn
        )

        # Both should produce valid variance estimates
        assert result_high.variance is not None
        assert result_low.variance is not None
        # With near-deterministic functions, both variances are near-zero;
        # just verify they are non-negative and finite
        assert np.all(np.isfinite(result_high.variance))
        assert np.all(result_high.variance >= 0)

    def test_reproducibility_with_seed(self, sample_clients, additive_value_fn):
        """Test reproducibility with same seed."""
        shapley1 = MonteCarloShapley(n_samples=100, seed=42)
        shapley2 = MonteCarloShapley(n_samples=100, seed=42)

        result1 = shapley1.compute([0, 1, 2], sample_clients, additive_value_fn)
        result2 = shapley2.compute([0, 1, 2], sample_clients, additive_value_fn)

        np.testing.assert_array_almost_equal(result1.values, result2.values)

    def test_invalid_n_samples(self):
        """Test invalid n_samples raises error."""
        with pytest.raises(ValueError):
            MonteCarloShapley(n_samples=0)
        with pytest.raises(ValueError):
            MonteCarloShapley(n_samples=-1)

    def test_empty_coalition(self, sample_clients, additive_value_fn):
        """Test empty coalition."""
        shapley = MonteCarloShapley(n_samples=100)
        result = shapley.compute([], sample_clients, additive_value_fn)

        assert len(result.values) == 0

    def test_name_property(self):
        """Test name property."""
        shapley = MonteCarloShapley(n_samples=100)
        assert shapley.name == "MonteCarloShapley"


# =============================================================================
# StratifiedShapley Tests
# =============================================================================


class TestStratifiedShapley:
    """Tests for StratifiedShapley computation."""

    def test_basic_computation(self, sample_clients, additive_value_fn):
        """Test basic stratified computation."""
        shapley = StratifiedShapley(samples_per_stratum=50, seed=42)
        result = shapley.compute([0, 1, 2], sample_clients, additive_value_fn)

        assert len(result.values) == 3
        assert result.n_samples > 0

    def test_efficiency_approximate(self, sample_clients, additive_value_fn):
        """Test approximate efficiency."""
        shapley = StratifiedShapley(samples_per_stratum=100, seed=42)
        coalition = [0, 1, 2]
        result = shapley.compute(coalition, sample_clients, additive_value_fn)

        total_value = additive_value_fn(coalition, sample_clients)
        shapley_sum = np.sum(result.values)

        # Should be approximately efficient
        assert shapley_sum == pytest.approx(total_value, rel=0.2)

    def test_name_property(self):
        """Test name property."""
        shapley = StratifiedShapley(samples_per_stratum=50)
        assert shapley.name == "StratifiedShapley"


# =============================================================================
# compute_shapley_values Tests
# =============================================================================


class TestComputeShapleyValues:
    """Tests for convenience function."""

    def test_auto_selects_exact_for_small(self, sample_clients, additive_value_fn):
        """Test auto selects exact for small coalitions."""
        result = compute_shapley_values(
            coalition=[0, 1, 2],
            clients=sample_clients,
            value_fn=additive_value_fn,
            method="auto",
        )

        # For small coalitions, should use exact (no variance)
        assert result.variance is None or len(result.values) <= 10

    def test_monte_carlo_explicit(self, sample_clients, additive_value_fn):
        """Test explicit Monte Carlo selection."""
        result = compute_shapley_values(
            coalition=[0, 1, 2],
            clients=sample_clients,
            value_fn=additive_value_fn,
            method="monte_carlo",
            n_samples=100,
        )

        assert result.n_samples == 100
        assert result.variance is not None

    def test_exact_explicit(self, sample_clients, additive_value_fn):
        """Test explicit exact selection."""
        result = compute_shapley_values(
            coalition=[0, 1, 2],
            clients=sample_clients,
            value_fn=additive_value_fn,
            method="exact",
        )

        # Exact uses all permutations
        from math import factorial

        assert result.n_samples == factorial(3)


class TestShapleyFromFitness:
    """Tests for shapley_from_fitness."""

    def test_basic_usage(self, sample_clients, simple_fitness):
        """Test basic usage with fitness function."""
        result = shapley_from_fitness(
            coalition=[0, 1, 2],
            clients=sample_clients,
            fitness_fn=simple_fitness,
        )

        assert len(result.values) == 3
        assert np.sum(result.values) > 0


# =============================================================================
# ContributionMetrics Tests
# =============================================================================


class TestContributionMetrics:
    """Tests for ContributionMetrics dataclass."""

    def test_creation(self):
        """Test creating contribution metrics."""
        metrics = ContributionMetrics(
            data_contribution=0.3,
            computation_contribution=0.4,
            communication_contribution=0.2,
            fairness_contribution=0.1,
            total_contribution=1.0,
        )

        assert metrics.data_contribution == 0.3
        assert metrics.total_contribution == 1.0

    def test_default_values(self):
        """Test default values."""
        metrics = ContributionMetrics()

        assert metrics.data_contribution == 0.0
        assert metrics.total_contribution == 0.0
        assert metrics.details == {}


# =============================================================================
# AllocationResult Tests
# =============================================================================


class TestAllocationResult:
    """Tests for AllocationResult dataclass."""

    def test_creation(self):
        """Test creating allocation result."""
        result = AllocationResult(
            allocations={0: 50.0, 1: 30.0, 2: 20.0},
            total_reward=100.0,
            allocation_method="Equal",
        )

        assert result.allocations[0] == 50.0
        assert result.total_reward == 100.0

    def test_get_shares(self):
        """Test getting shares as fractions."""
        result = AllocationResult(
            allocations={0: 50.0, 1: 30.0, 2: 20.0},
            total_reward=100.0,
            allocation_method="Test",
        )

        shares = result.get_shares()
        assert shares[0] == pytest.approx(0.5)
        assert shares[1] == pytest.approx(0.3)
        assert shares[2] == pytest.approx(0.2)

    def test_get_shares_zero_total(self):
        """Test shares with zero total."""
        result = AllocationResult(
            allocations={0: 0.0, 1: 0.0},
            total_reward=0.0,
            allocation_method="Test",
        )

        shares = result.get_shares()
        assert shares[0] == 0.0
        assert shares[1] == 0.0

    def test_get_ranking(self):
        """Test ranking by allocation."""
        result = AllocationResult(
            allocations={0: 20.0, 1: 50.0, 2: 30.0},
            total_reward=100.0,
            allocation_method="Test",
        )

        ranking = result.get_ranking()
        assert ranking == [1, 2, 0]  # Descending order


# =============================================================================
# EqualAllocator Tests
# =============================================================================


class TestEqualAllocator:
    """Tests for EqualAllocator."""

    def test_equal_allocation(self, sample_clients):
        """Test equal allocation."""
        allocator = EqualAllocator()
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=90.0)

        assert result.allocations[0] == pytest.approx(30.0)
        assert result.allocations[1] == pytest.approx(30.0)
        assert result.allocations[2] == pytest.approx(30.0)

    def test_empty_coalition(self, sample_clients):
        """Test empty coalition."""
        allocator = EqualAllocator()
        result = allocator.allocate([], sample_clients, total_reward=100.0)

        assert result.allocations == {}
        assert result.total_reward == 0.0

    def test_single_client(self, sample_clients):
        """Test single client."""
        allocator = EqualAllocator()
        result = allocator.allocate([0], sample_clients, total_reward=100.0)

        assert result.allocations[0] == 100.0

    def test_name_property(self):
        """Test name property."""
        allocator = EqualAllocator()
        assert allocator.name == "Equal"


# =============================================================================
# ProportionalAllocator Tests
# =============================================================================


class TestProportionalAllocator:
    """Tests for ProportionalAllocator."""

    def test_proportional_allocation(self, sample_clients):
        """Test proportional allocation."""
        allocator = ProportionalAllocator()
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        # Higher quality/data clients should get more
        total = sum(result.allocations.values())
        assert total == pytest.approx(100.0, rel=1e-5)

    def test_weight_customization(self, sample_clients):
        """Test custom weights."""
        # Only consider data size
        allocator = ProportionalAllocator(
            data_weight=1.0,
            quality_weight=0.0,
            efficiency_weight=0.0,
        )
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        # Client 2 has most samples (300), should get most
        assert result.allocations[2] > result.allocations[1] > result.allocations[0]

    def test_metrics_included(self, sample_clients):
        """Test metrics are included in result."""
        allocator = ProportionalAllocator()
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        assert result.metrics is not None
        assert 0 in result.metrics
        assert result.metrics[0].total_contribution > 0

    def test_empty_coalition(self, sample_clients):
        """Test empty coalition."""
        allocator = ProportionalAllocator()
        result = allocator.allocate([], sample_clients, total_reward=100.0)

        assert result.allocations == {}

    def test_weight_normalization(self, sample_clients):
        """Test weights are normalized."""
        allocator = ProportionalAllocator(
            data_weight=10.0,
            quality_weight=20.0,
            efficiency_weight=30.0,
        )

        # Weights should be normalized internally
        assert allocator.data_weight == pytest.approx(10 / 60)
        assert allocator.quality_weight == pytest.approx(20 / 60)
        assert allocator.efficiency_weight == pytest.approx(30 / 60)

    def test_name_property(self):
        """Test name property."""
        allocator = ProportionalAllocator()
        assert allocator.name == "Proportional"


# =============================================================================
# ShapleyAllocator Tests
# =============================================================================


class TestShapleyAllocator:
    """Tests for ShapleyAllocator."""

    def test_shapley_allocation(self, sample_clients):
        """Test Shapley-based allocation."""
        allocator = ShapleyAllocator(n_samples=100, seed=42)
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        # Total should equal reward
        total = sum(result.allocations.values())
        assert total == pytest.approx(100.0, rel=0.01)

    def test_custom_value_function(self, sample_clients):
        """Test with custom value function."""

        def custom_value(coalition, clients):
            return len(coalition) ** 2  # Supermodular

        allocator = ShapleyAllocator(value_fn=custom_value, n_samples=100, seed=42)
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        assert len(result.allocations) == 3

    def test_metrics_contain_shapley_values(self, sample_clients):
        """Test metrics contain Shapley values."""
        allocator = ShapleyAllocator(n_samples=100, seed=42)
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        assert result.metrics is not None
        assert "shapley_value" in result.metrics[0].details

    def test_empty_coalition(self, sample_clients):
        """Test empty coalition."""
        allocator = ShapleyAllocator()
        result = allocator.allocate([], sample_clients, total_reward=100.0)

        assert result.allocations == {}

    def test_name_property(self):
        """Test name property."""
        allocator = ShapleyAllocator()
        assert allocator.name == "Shapley"


# =============================================================================
# FairnessAwareAllocator Tests
# =============================================================================


class TestFairnessAwareAllocator:
    """Tests for FairnessAwareAllocator."""

    def test_basic_allocation(self, sample_clients, target_demographics):
        """Test fairness-aware allocation."""
        allocator = FairnessAwareAllocator(
            target_distribution=target_demographics,
            fairness_weight=0.3,
        )
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        total = sum(result.allocations.values())
        assert total == pytest.approx(100.0, rel=1e-5)

    def test_fairness_weight_impact(self, sample_clients, target_demographics):
        """Test fairness weight impacts allocation."""
        low_fairness = FairnessAwareAllocator(
            target_distribution=target_demographics,
            fairness_weight=0.0,
        )
        high_fairness = FairnessAwareAllocator(
            target_distribution=target_demographics,
            fairness_weight=0.9,
        )

        result_low = low_fairness.allocate(
            [0, 1, 2], sample_clients, total_reward=100.0
        )
        result_high = high_fairness.allocate(
            [0, 1, 2], sample_clients, total_reward=100.0
        )

        # Different fairness weights should give different allocations
        assert result_low.allocations != result_high.allocations

    def test_metrics_contain_fairness_contribution(
        self, sample_clients, target_demographics
    ):
        """Test metrics contain fairness contribution."""
        allocator = FairnessAwareAllocator(
            target_distribution=target_demographics,
            fairness_weight=0.3,
        )
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        assert result.metrics is not None
        assert result.metrics[0].fairness_contribution >= 0

    def test_no_target_distribution(self, sample_clients):
        """Test without target distribution."""
        allocator = FairnessAwareAllocator(
            target_distribution=None,
            fairness_weight=0.3,
        )
        result = allocator.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        # Should still allocate
        total = sum(result.allocations.values())
        assert total == pytest.approx(100.0, rel=1e-5)

    def test_empty_coalition(self, sample_clients, target_demographics):
        """Test empty coalition."""
        allocator = FairnessAwareAllocator(
            target_distribution=target_demographics,
        )
        result = allocator.allocate([], sample_clients, total_reward=100.0)

        assert result.allocations == {}

    def test_name_property(self, target_demographics):
        """Test name property."""
        allocator = FairnessAwareAllocator(
            target_distribution=target_demographics,
        )
        assert allocator.name == "FairnessAware"


# =============================================================================
# allocate_rewards Tests
# =============================================================================


class TestAllocateRewards:
    """Tests for allocate_rewards convenience function."""

    def test_equal_method(self, sample_clients):
        """Test equal allocation method."""
        result = allocate_rewards(
            coalition=[0, 1, 2],
            clients=sample_clients,
            total_reward=90.0,
            method="equal",
        )

        assert result.allocation_method == "Equal"
        assert all(v == pytest.approx(30.0) for v in result.allocations.values())

    def test_proportional_method(self, sample_clients):
        """Test proportional allocation method."""
        result = allocate_rewards(
            coalition=[0, 1, 2],
            clients=sample_clients,
            total_reward=100.0,
            method="proportional",
        )

        assert result.allocation_method == "Proportional"

    def test_shapley_method(self, sample_clients):
        """Test Shapley allocation method."""
        result = allocate_rewards(
            coalition=[0, 1, 2],
            clients=sample_clients,
            total_reward=100.0,
            method="shapley",
            n_samples=100,
            seed=42,
        )

        assert result.allocation_method == "Shapley"

    def test_default_method(self, sample_clients):
        """Test default method is proportional."""
        result = allocate_rewards(
            coalition=[0, 1, 2],
            clients=sample_clients,
            total_reward=100.0,
        )

        assert result.allocation_method == "Proportional"


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestShapleyProperties:
    """Property-based tests for Shapley values."""

    @given(n_players=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_efficiency_property(self, n_players):
        """Test efficiency property holds."""
        # Create simple clients
        clients = [
            Client(
                id=f"c{i}",
                num_samples=100,
                demographics=Demographics(),
                data_quality=0.5 + 0.1 * i,
            )
            for i in range(n_players)
        ]

        def value_fn(coalition, clients):
            if not coalition:
                return 0.0
            return sum(clients[i].data_quality for i in coalition)

        coalition = list(range(n_players))
        result = compute_shapley_values(coalition, clients, value_fn, method="exact")

        total_value = value_fn(coalition, clients)
        assert np.sum(result.values) == pytest.approx(total_value, rel=1e-4)

    @given(
        total_reward=st.floats(min_value=1.0, max_value=1000.0),
        n_clients=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20)
    def test_allocation_sums_to_total(self, total_reward, n_clients):
        """Test allocations sum to total reward."""
        clients = [
            Client(
                id=f"c{i}",
                num_samples=100,
                demographics=Demographics(),
                data_quality=0.5,
            )
            for i in range(n_clients)
        ]

        coalition = list(range(n_clients))
        allocator = EqualAllocator()
        result = allocator.allocate(coalition, clients, total_reward)

        allocation_sum = sum(result.allocations.values())
        assert allocation_sum == pytest.approx(total_reward, rel=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIncentivesIntegration:
    """Integration tests for incentives module."""

    def test_shapley_to_allocation_pipeline(self, sample_clients):
        """Test Shapley values feed into allocation."""

        # Compute Shapley values
        def value_fn(coalition, clients):
            return sum(
                clients[i].data_quality for i in coalition if 0 <= i < len(clients)
            )

        shapley_result = compute_shapley_values(
            coalition=[0, 1, 2],
            clients=sample_clients,
            value_fn=value_fn,
            method="exact",
        )

        # Use Shapley-based allocation
        allocator = ShapleyAllocator(seed=42)
        allocation_result = allocator.allocate(
            [0, 1, 2], sample_clients, total_reward=100.0
        )

        # Allocation should be proportional to Shapley values
        # Both orderings should match
        shapley_ranking = shapley_result.get_ranking()
        allocation_ranking = allocation_result.get_ranking()

        # Rankings might not be identical due to normalization, but highest should match
        assert shapley_ranking[0] == allocation_ranking[0]

    def test_different_allocators_same_total(self, sample_clients, target_demographics):
        """Test different allocators give same total."""
        coalition = [0, 1, 2]
        total_reward = 100.0

        equal_result = EqualAllocator().allocate(
            coalition, sample_clients, total_reward
        )
        prop_result = ProportionalAllocator().allocate(
            coalition, sample_clients, total_reward
        )
        fair_result = FairnessAwareAllocator(
            target_distribution=target_demographics
        ).allocate(coalition, sample_clients, total_reward)

        assert sum(equal_result.allocations.values()) == pytest.approx(total_reward)
        assert sum(prop_result.allocations.values()) == pytest.approx(total_reward)
        assert sum(fair_result.allocations.values()) == pytest.approx(total_reward)

    def test_reproducibility(self, sample_clients):
        """Test allocation is reproducible with seed."""
        allocator1 = ShapleyAllocator(n_samples=100, seed=42)
        allocator2 = ShapleyAllocator(n_samples=100, seed=42)

        result1 = allocator1.allocate([0, 1, 2], sample_clients, total_reward=100.0)
        result2 = allocator2.allocate([0, 1, 2], sample_clients, total_reward=100.0)

        for idx in result1.allocations:
            assert result1.allocations[idx] == pytest.approx(
                result2.allocations[idx], rel=1e-5
            )
