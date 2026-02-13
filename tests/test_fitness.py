"""
Tests for FairSwarm fitness functions.

Tests cover:
- FitnessResult dataclass
- DemographicFitness with Definition 2 KL divergence
- CompositeFitness multi-objective combinations
- MockFitness for deterministic testing
- Gradient computation for fairness-aware updates

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import numpy as np
import pytest

from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.demographics.targets import CensusTarget
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.composite import (
    CompositeFitness,
    CommunicationCostFitness,
    WeightedFitness,
)
from fairswarm.fitness.fairness import (
    AccuracyFairnessFitness,
    DemographicFitness,
    compute_coalition_demographics,
    compute_fairness_gradient,
)
from fairswarm.fitness.mock import (
    ConstantFitness,
    DataQualityFitness,
    DeterministicFitness,
    MockFitness,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_clients():
    """Create sample clients for testing."""
    return create_synthetic_clients(
        n_clients=10,
        seed=42,
    )


@pytest.fixture
def target_distribution():
    """Get US Census 2020 target distribution."""
    return CensusTarget.US_2020.as_distribution()


@pytest.fixture
def diverse_clients():
    """Create clients with diverse demographics."""
    clients = []
    demographics_list = [
        {"white": 0.9, "black": 0.1},
        {"white": 0.1, "black": 0.9},
        {"white": 0.5, "hispanic": 0.5},
        {"asian": 0.8, "other": 0.2},
        {"white": 0.3, "black": 0.3, "hispanic": 0.4},
    ]

    for i, demo in enumerate(demographics_list):
        clients.append(
            Client(
                id=f"client_{i}",
                demographics=DemographicDistribution.from_dict(demo),
                num_samples=1000 * (i + 1),
                data_quality=0.5 + 0.1 * i,
                communication_cost=0.1 * (i + 1),
            )
        )

    return clients


# =============================================================================
# FitnessResult Tests
# =============================================================================


class TestFitnessResult:
    """Tests for FitnessResult dataclass."""

    def test_basic_creation(self):
        """Test creating a FitnessResult."""
        result = FitnessResult(
            value=0.85,
            components={"accuracy": 0.9, "fairness": -0.05},
            coalition=[0, 1, 2],
        )

        assert result.value == 0.85
        assert result.components["accuracy"] == 0.9
        assert result.coalition == [0, 1, 2]

    def test_frozen_immutability(self):
        """Test that FitnessResult is immutable."""
        result = FitnessResult(value=0.5, coalition=[0, 1])

        # FitnessResult is frozen dataclass
        with pytest.raises(AttributeError):
            result.value = 0.9

    def test_repr(self):
        """Test string representation."""
        result = FitnessResult(
            value=0.75,
            components={"accuracy": 0.8, "fairness": -0.05},
            coalition=[0, 1],
        )

        repr_str = repr(result)
        assert "0.75" in repr_str
        assert "accuracy" in repr_str or "components" in repr_str


# =============================================================================
# Coalition Demographics Tests
# =============================================================================


class TestCoalitionDemographics:
    """Tests for coalition demographic computation."""

    def test_single_client_coalition(self, diverse_clients):
        """Test demographics with single client."""
        demo = compute_coalition_demographics([0], diverse_clients)

        # Should match client 0's demographics
        expected = diverse_clients[0].demographics.as_array()
        np.testing.assert_array_almost_equal(demo, expected)

    def test_average_demographics(self, diverse_clients):
        """Test that coalition demographics are averaged."""
        # Coalition of clients 0 and 1 with opposite demographics
        demo = compute_coalition_demographics([0, 1], diverse_clients)

        # Average of [white: 0.9, black: 0.1] and [white: 0.1, black: 0.9]
        # Should be approximately [white: 0.5, black: 0.5]
        assert demo.sum() == pytest.approx(1.0, rel=1e-5)

    def test_empty_coalition_raises(self, diverse_clients):
        """Test that empty coalition raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_coalition_demographics([], diverse_clients)


# =============================================================================
# DemographicFitness Tests
# =============================================================================


class TestDemographicFitness:
    """Tests for DemographicFitness (Definition 2 implementation)."""

    def test_initialization(self, target_distribution):
        """Test DemographicFitness initialization."""
        fitness = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )

        assert fitness.target_distribution == target_distribution
        assert fitness.divergence_weight == 1.0

    def test_evaluate_returns_fitness_result(
        self, sample_clients, target_distribution
    ):
        """Test that evaluate returns FitnessResult."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        result = fitness.evaluate([0, 1, 2], sample_clients)

        assert isinstance(result, FitnessResult)
        assert "divergence" in result.components

    def test_fitness_is_negative_divergence(
        self, diverse_clients, target_distribution
    ):
        """Test that fitness equals negative weighted divergence."""
        fitness = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )

        coalition = [0, 1, 2]
        result = fitness.evaluate(coalition, diverse_clients)

        # Compute expected divergence
        coalition_demo = compute_coalition_demographics(coalition, diverse_clients)
        expected_div = kl_divergence(
            coalition_demo, target_distribution.as_array()
        )

        assert result.value == pytest.approx(-expected_div, rel=1e-5)

    def test_weight_affects_fitness(self, diverse_clients, target_distribution):
        """Test that divergence weight scales the fitness."""
        fitness_w1 = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )
        fitness_w2 = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=2.0,
        )

        coalition = [0, 1]
        result_w1 = fitness_w1.evaluate(coalition, diverse_clients)
        result_w2 = fitness_w2.evaluate(coalition, diverse_clients)

        # With weight 2, fitness should be twice as negative
        assert result_w2.value == pytest.approx(2 * result_w1.value, rel=1e-5)

    def test_empty_coalition_returns_negative_inf(
        self, diverse_clients, target_distribution
    ):
        """Test that empty coalition returns -inf fitness."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        result = fitness.evaluate([], diverse_clients)

        assert result.value == float("-inf")

    def test_gradient_computation(self, diverse_clients, target_distribution):
        """Test fairness gradient computation."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        position = np.array([0.8, 0.2, 0.5, 0.3, 0.7])
        gradient = fitness.compute_gradient(position, diverse_clients, coalition_size=3)

        assert len(gradient) == len(diverse_clients)
        assert np.isfinite(gradient).all()
        # Gradient should be normalized
        assert np.linalg.norm(gradient) == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# Fairness Gradient Tests
# =============================================================================


class TestFairnessGradient:
    """Tests for fairness gradient computation (Algorithm 1)."""

    def test_gradient_returns_correct_structure(
        self, diverse_clients, target_distribution
    ):
        """Test that compute_fairness_gradient returns correct structure."""
        position = np.ones(len(diverse_clients)) * 0.5

        result = compute_fairness_gradient(
            position=position,
            clients=diverse_clients,
            target_distribution=target_distribution,
            coalition_size=3,
        )

        assert len(result.gradient) == len(diverse_clients)
        assert isinstance(result.divergence, float)
        assert len(result.coalition_distribution) > 0

    def test_gradient_is_normalized(self, diverse_clients, target_distribution):
        """Test that gradient is normalized to unit length."""
        position = np.random.default_rng(42).random(len(diverse_clients))

        result = compute_fairness_gradient(
            position=position,
            clients=diverse_clients,
            target_distribution=target_distribution,
            coalition_size=3,
        )

        norm = np.linalg.norm(result.gradient)
        assert norm == pytest.approx(1.0, rel=1e-5)

    def test_gradient_favors_underrepresented_groups(
        self, target_distribution
    ):
        """Test that gradient pushes toward underrepresented demographics."""
        # Create two clients: one matching target, one not
        target = target_distribution.categories

        # Client 0: matches target well
        # Client 1: very different from target
        clients = [
            Client(
                id="matching",
                demographics=target_distribution,
                num_samples=1000,
                data_quality=0.8,
            ),
            Client(
                id="different",
                demographics=DemographicDistribution.from_dict({"white": 1.0}),
                num_samples=1000,
                data_quality=0.8,
            ),
        ]

        # Start with position favoring the non-matching client
        position = np.array([0.2, 0.8])

        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=target_distribution,
            coalition_size=1,
        )

        # Gradient should favor the matching client (index 0)
        assert result.gradient[0] > result.gradient[1]


# =============================================================================
# MockFitness Tests
# =============================================================================


class TestMockFitness:
    """Tests for MockFitness."""

    def test_size_mode(self, sample_clients):
        """Test size mode returns coalition size."""
        fitness = MockFitness(mode="size")

        result = fitness.evaluate([0, 1, 2], sample_clients)

        assert result.value == 3.0

    def test_sum_mode(self, sample_clients):
        """Test sum mode returns sum of indices."""
        fitness = MockFitness(mode="sum")

        result = fitness.evaluate([0, 2, 4], sample_clients)

        assert result.value == 6.0  # 0 + 2 + 4

    def test_mean_quality_mode(self, diverse_clients):
        """Test mean_quality mode returns average quality."""
        fitness = MockFitness(mode="mean_quality")

        # Clients have qualities 0.5, 0.6, 0.7, 0.8, 0.9
        result = fitness.evaluate([0, 1], diverse_clients)

        expected = (0.5 + 0.6) / 2
        assert result.value == pytest.approx(expected)

    def test_custom_mode(self, sample_clients):
        """Test custom mode with user function."""
        custom_fn = lambda coalition, clients: len(coalition) * 10.0
        fitness = MockFitness(mode="custom", custom_fn=custom_fn)

        result = fitness.evaluate([0, 1], sample_clients)

        assert result.value == 20.0

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            MockFitness(mode="invalid")

    def test_custom_without_fn_raises(self):
        """Test that custom mode without function raises."""
        with pytest.raises(ValueError, match="custom_fn required"):
            MockFitness(mode="custom")


class TestConstantFitness:
    """Tests for ConstantFitness."""

    def test_returns_constant(self, sample_clients):
        """Test that ConstantFitness returns the constant value."""
        fitness = ConstantFitness(value=0.42)

        result1 = fitness.evaluate([0], sample_clients)
        result2 = fitness.evaluate([0, 1, 2, 3], sample_clients)

        assert result1.value == 0.42
        assert result2.value == 0.42

    def test_zero_gradient(self, sample_clients):
        """Test that gradient is zero for constant fitness."""
        fitness = ConstantFitness(value=1.0)

        gradient = fitness.compute_gradient(
            np.ones(len(sample_clients)), sample_clients, coalition_size=3
        )

        np.testing.assert_array_equal(gradient, np.zeros(len(sample_clients)))


class TestDeterministicFitness:
    """Tests for DeterministicFitness."""

    def test_returns_predetermined_value(self, sample_clients):
        """Test that predetermined values are returned."""
        values = {
            frozenset([0, 1]): 0.9,
            frozenset([1, 2]): 0.7,
        }
        fitness = DeterministicFitness(values=values, default_value=0.5)

        result1 = fitness.evaluate([0, 1], sample_clients)
        result2 = fitness.evaluate([1, 2], sample_clients)
        result3 = fitness.evaluate([0, 2], sample_clients)  # Not in values

        assert result1.value == 0.9
        assert result2.value == 0.7
        assert result3.value == 0.5  # Default


# =============================================================================
# CompositeFitness Tests
# =============================================================================


class TestWeightedFitness:
    """Tests for WeightedFitness."""

    def test_weighted_sum(self, diverse_clients, target_distribution):
        """Test weighted sum of multiple fitness functions."""
        fairness = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )
        cost = CommunicationCostFitness()

        composite = WeightedFitness([
            ("fairness", fairness, 0.7),
            ("cost", cost, 0.3),
        ])

        result = composite.evaluate([0, 1], diverse_clients)

        # Verify components are tracked
        assert "fairness" in result.components
        assert "cost" in result.components

    def test_components_tracked_separately(self, sample_clients):
        """Test that components are tracked with raw and weighted values."""
        fitness1 = MockFitness(mode="size")
        fitness2 = ConstantFitness(value=1.0)

        composite = WeightedFitness([
            ("size", fitness1, 0.5),
            ("constant", fitness2, 0.5),
        ])

        result = composite.evaluate([0, 1, 2], sample_clients)

        assert result.components["size"] == 3.0  # Raw value
        assert result.components["size_weighted"] == 1.5  # Weighted
        assert result.components["constant"] == 1.0
        assert result.components["constant_weighted"] == 0.5


class TestCompositeFitness:
    """Tests for CompositeFitness with different aggregation methods."""

    def test_weighted_sum_aggregation(self, sample_clients):
        """Test weighted_sum aggregation."""
        fitness1 = MockFitness(mode="size")
        fitness2 = ConstantFitness(value=1.0)

        composite = CompositeFitness(
            components=[
                ("size", fitness1, 1.0),
                ("constant", fitness2, 1.0),
            ],
            aggregation="weighted_sum",
        )

        result = composite.evaluate([0, 1], sample_clients)

        # size=2, constant=1, sum=3
        assert result.value == pytest.approx(3.0)

    def test_min_aggregation(self, sample_clients):
        """Test min (Rawlsian) aggregation."""
        fitness1 = ConstantFitness(value=0.8)
        fitness2 = ConstantFitness(value=0.5)

        composite = CompositeFitness(
            components=[
                ("high", fitness1, 1.0),
                ("low", fitness2, 1.0),
            ],
            aggregation="min",
        )

        result = composite.evaluate([0], sample_clients)

        assert result.value == pytest.approx(0.5)

    def test_invalid_aggregation_raises(self):
        """Test that invalid aggregation method raises."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            CompositeFitness(
                components=[("test", ConstantFitness(1.0), 1.0)],
                aggregation="invalid",
            )


class TestCommunicationCostFitness:
    """Tests for CommunicationCostFitness."""

    def test_cost_is_negative_fitness(self, diverse_clients):
        """Test that cost is returned as negative fitness."""
        fitness = CommunicationCostFitness()

        # Client 0 has cost 0.1, client 1 has cost 0.2
        result = fitness.evaluate([0, 1], diverse_clients)

        expected_cost = 0.1 + 0.2
        assert result.value == pytest.approx(-expected_cost)

    def test_normalized_cost(self, diverse_clients):
        """Test normalized cost by coalition size."""
        fitness = CommunicationCostFitness(normalize=True)

        result = fitness.evaluate([0, 1], diverse_clients)

        expected_cost = (0.1 + 0.2) / 2
        assert result.value == pytest.approx(-expected_cost)

    def test_empty_coalition_zero_cost(self, diverse_clients):
        """Test that empty coalition has zero cost."""
        fitness = CommunicationCostFitness()

        result = fitness.evaluate([], diverse_clients)

        assert result.value == 0.0


# =============================================================================
# AccuracyFairnessFitness Tests
# =============================================================================


class TestAccuracyFairnessFitness:
    """Tests for combined accuracy and fairness fitness."""

    def test_with_accuracy_function(self, diverse_clients, target_distribution):
        """Test with custom accuracy function."""
        def mock_accuracy(coalition, clients):
            return 0.85

        fitness = AccuracyFairnessFitness(
            target_distribution=target_distribution,
            accuracy_fn=mock_accuracy,
            fairness_weight=0.3,
        )

        result = fitness.evaluate([0, 1, 2], diverse_clients)

        assert result.components["accuracy"] == 0.85
        assert "divergence" in result.components
        # Fitness = accuracy - weight * divergence
        assert result.value < 0.85  # Fairness penalty reduces fitness

    def test_without_accuracy_function_uses_quality(
        self, diverse_clients, target_distribution
    ):
        """Test that without accuracy_fn, data quality is used."""
        fitness = AccuracyFairnessFitness(
            target_distribution=target_distribution,
            accuracy_fn=None,
            fairness_weight=0.3,
        )

        result = fitness.evaluate([0, 1], diverse_clients)

        # Should use average of client data qualities
        expected_quality = (0.5 + 0.6) / 2
        assert result.components["accuracy"] == pytest.approx(expected_quality)


# =============================================================================
# DataQualityFitness Tests
# =============================================================================


class TestDataQualityFitness:
    """Tests for DataQualityFitness."""

    def test_quality_weight(self, diverse_clients):
        """Test that quality weight affects fitness."""
        fitness = DataQualityFitness(
            quality_weight=1.0,
            size_weight=0.0,
            staleness_penalty=0.0,
        )

        result = fitness.evaluate([0, 1], diverse_clients)

        # Average quality of clients 0,1 is (0.5 + 0.6) / 2 = 0.55
        expected = 0.55
        assert result.value == pytest.approx(expected, rel=1e-2)

    def test_gradient_favors_high_quality(self, diverse_clients):
        """Test that gradient favors high-quality clients."""
        fitness = DataQualityFitness(quality_weight=1.0)

        position = np.ones(len(diverse_clients)) * 0.5
        gradient = fitness.compute_gradient(position, diverse_clients, coalition_size=3)

        # Higher index clients have higher quality
        # So gradient should increase with index
        for i in range(len(gradient) - 1):
            assert gradient[i] < gradient[i + 1]


# =============================================================================
# Integration Tests
# =============================================================================


class TestFitnessIntegration:
    """Integration tests for fitness functions."""

    def test_full_evaluation_pipeline(
        self, diverse_clients, target_distribution
    ):
        """Test complete fitness evaluation with all components."""
        # Create composite fitness matching paper formulation
        # F(S) = ValAcc(S) - λ·DemDiv(S) - μ·CommCost(S)
        fairness = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )
        cost = CommunicationCostFitness()
        quality = MockFitness(mode="mean_quality")

        composite = WeightedFitness([
            ("accuracy", quality, 0.5),
            ("fairness", fairness, 0.3),
            ("cost", cost, 0.2),
        ])

        # Evaluate a coalition
        coalition = [0, 2, 4]
        result = composite.evaluate(coalition, diverse_clients)

        # Verify all components present
        assert "accuracy" in result.components
        assert "fairness" in result.components
        assert "cost" in result.components
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_gradient_integration_with_particle(
        self, diverse_clients, target_distribution
    ):
        """Test gradient computation for particle velocity update."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        # Simulate particle position
        position = np.array([0.7, 0.3, 0.5, 0.4, 0.6])

        # Compute gradient
        gradient = fitness.compute_gradient(
            position, diverse_clients, coalition_size=3
        )

        # Simulate velocity update (Algorithm 1)
        fairness_coeff = 0.2
        v_fairness = fairness_coeff * gradient

        # Verify the update is reasonable
        assert len(v_fairness) == len(position)
        assert np.isfinite(v_fairness).all()
        assert np.abs(v_fairness).max() <= fairness_coeff  # Bounded by coeff
