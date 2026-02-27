"""
Tests for FairSwarm optimization algorithm.

Tests cover:
- FairSwarm initialization and configuration
- Optimization execution (Algorithm 1)
- Convergence behavior (Theorem 1)
- Fairness guarantees (Theorem 2)
- Result structure and metrics

Author: Tenicka Norwood
"""

import warnings

import numpy as np
import pytest

from fairswarm.algorithms.fairswarm import FairSwarm, run_fairswarm
from fairswarm.algorithms.result import (
    ConvergenceMetrics,
    FairnessMetrics,
    OptimizationResult,
)
from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.targets import CensusTarget
from fairswarm.fitness.fairness import DemographicFitness
from fairswarm.fitness.mock import ConstantFitness, MockFitness

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_clients():
    """Create sample clients for testing."""
    return create_synthetic_clients(n_clients=20, n_demographic_groups=5, seed=42)


@pytest.fixture
def target_distribution():
    """Get US Census 2020 target distribution."""
    return CensusTarget.US_2020.as_distribution()


@pytest.fixture
def default_config():
    """Create default FairSwarm configuration."""
    return FairSwarmConfig(
        swarm_size=10,
        max_iterations=50,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.2,
    )


@pytest.fixture
def mock_fitness():
    """Create mock fitness function."""
    return MockFitness(mode="mean_quality")


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestConvergenceMetrics:
    """Tests for ConvergenceMetrics."""

    def test_basic_creation(self):
        """Test creating ConvergenceMetrics."""
        metrics = ConvergenceMetrics(
            iterations=100,
            fitness_history=[0.1, 0.2, 0.3],
            converged=True,
            convergence_iteration=95,
        )

        assert metrics.iterations == 100
        assert metrics.converged is True
        assert len(metrics.fitness_history) == 3

    def test_improvement_rate(self):
        """Test improvement rate calculation."""
        metrics = ConvergenceMetrics(
            iterations=100,
            global_best_updates=[0, 5, 10, 50],
        )

        assert metrics.improvement_rate == 4 / 100

    def test_final_diversity(self):
        """Test final diversity property."""
        metrics = ConvergenceMetrics(
            iterations=10,
            diversity_history=[0.9, 0.8, 0.7, 0.5],
        )

        assert metrics.final_diversity == 0.5


class TestFairnessMetrics:
    """Tests for FairnessMetrics."""

    def test_representation_gap(self):
        """Test representation gap calculation."""
        metrics = FairnessMetrics(
            demographic_divergence=0.1,
            coalition_distribution={"white": 0.5, "black": 0.3},
            target_distribution={"white": 0.6, "black": 0.2},
        )

        assert metrics.representation_gap("white") == pytest.approx(0.1)
        assert metrics.representation_gap("black") == pytest.approx(0.1)

    def test_max_representation_gap(self):
        """Test maximum representation gap."""
        metrics = FairnessMetrics(
            demographic_divergence=0.1,
            coalition_distribution={"a": 0.5, "b": 0.3},
            target_distribution={"a": 0.6, "b": 0.1},
        )

        assert metrics.max_representation_gap() == pytest.approx(0.2)


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_basic_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            coalition=[0, 1, 2],
            fitness=0.85,
            fitness_components={"accuracy": 0.9, "fairness": -0.05},
        )

        assert result.coalition == [0, 1, 2]
        assert result.fitness == 0.85
        assert result.coalition_size == 3

    def test_is_converged_property(self):
        """Test is_converged property."""
        # Without convergence metrics
        result1 = OptimizationResult(coalition=[0], fitness=0.5)
        assert result1.is_converged is False

        # With converged=True
        result2 = OptimizationResult(
            coalition=[0],
            fitness=0.5,
            convergence=ConvergenceMetrics(iterations=100, converged=True),
        )
        assert result2.is_converged is True

    def test_is_fair_property(self):
        """Test is_fair property."""
        # Without fairness metrics
        result1 = OptimizationResult(coalition=[0], fitness=0.5)
        assert result1.is_fair is False

        # With epsilon satisfied
        result2 = OptimizationResult(
            coalition=[0],
            fitness=0.5,
            fairness=FairnessMetrics(
                demographic_divergence=0.05,
                epsilon_satisfied=True,
            ),
        )
        assert result2.is_fair is True

    def test_summary_generation(self):
        """Test summary string generation."""
        result = OptimizationResult(
            coalition=[0, 1, 2],
            fitness=0.85,
            fitness_components={"accuracy": 0.9},
        )

        summary = result.summary()

        assert "Coalition" in summary
        assert "0.85" in summary or "Fitness" in summary

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = OptimizationResult(
            coalition=[0, 1, 2],
            fitness=0.85,
        )

        d = result.to_dict()

        assert d["coalition"] == [0, 1, 2]
        assert d["fitness"] == 0.85
        assert d["coalition_size"] == 3


# =============================================================================
# FairSwarm Initialization Tests
# =============================================================================


class TestFairSwarmInitialization:
    """Tests for FairSwarm initialization."""

    def test_basic_initialization(self, sample_clients):
        """Test basic FairSwarm initialization."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
        )

        assert optimizer.n_clients == 20
        assert optimizer.coalition_size == 5
        assert optimizer.swarm is None  # Not initialized until optimize()

    def test_with_config(self, sample_clients, default_config):
        """Test initialization with custom config."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            config=default_config,
        )

        assert optimizer.config.swarm_size == 10
        assert optimizer.config.inertia == 0.7

    def test_with_target_distribution(self, sample_clients, target_distribution):
        """Test initialization with target demographics."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            target_distribution=target_distribution,
        )

        assert optimizer.target_distribution == target_distribution

    def test_invalid_coalition_size_too_large(self, sample_clients):
        """Test that coalition_size > n_clients raises error."""
        with pytest.raises(ValueError, match="cannot exceed"):
            FairSwarm(
                clients=sample_clients,
                coalition_size=100,  # More than 20 clients
            )

    def test_invalid_coalition_size_zero(self, sample_clients):
        """Test that coalition_size < 1 raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            FairSwarm(
                clients=sample_clients,
                coalition_size=0,
            )

    def test_reproducibility_with_seed(self, sample_clients, mock_fitness):
        """Test that same seed produces same results."""
        optimizer1 = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )
        result1 = optimizer1.optimize(mock_fitness, n_iterations=10)

        optimizer2 = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )
        result2 = optimizer2.optimize(mock_fitness, n_iterations=10)

        assert result1.coalition == result2.coalition
        assert result1.fitness == pytest.approx(result2.fitness)


# =============================================================================
# FairSwarm Optimization Tests
# =============================================================================


class TestFairSwarmOptimization:
    """Tests for FairSwarm.optimize() method (Algorithm 1)."""

    def test_optimize_returns_result(self, sample_clients, mock_fitness):
        """Test that optimize returns OptimizationResult."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=20)

        assert isinstance(result, OptimizationResult)
        assert len(result.coalition) == 5
        assert result.fitness is not None

    def test_swarm_initialized_after_optimize(self, sample_clients, mock_fitness):
        """Test that swarm is initialized during optimize."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
        )

        assert optimizer.swarm is None

        optimizer.optimize(mock_fitness, n_iterations=10)

        assert optimizer.swarm is not None
        assert len(optimizer.swarm.particles) > 0

    def test_fitness_improves_or_stable(self, sample_clients, mock_fitness):
        """Test that fitness never decreases (global best is tracked)."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=50)
        history = result.convergence.fitness_history

        # Global best should never decrease
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10

    def test_convergence_detection(self, sample_clients):
        """Test that convergence is detected when fitness stabilizes."""
        # Use constant fitness to guarantee fast convergence
        fitness = ConstantFitness(value=1.0)

        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(
            fitness,
            n_iterations=100,
            convergence_threshold=1e-6,
            convergence_window=10,
        )

        # Should converge with constant fitness
        assert result.convergence.converged is True
        assert result.convergence.convergence_iteration < 100

    def test_callback_invoked(self, sample_clients, mock_fitness):
        """Test that callback is invoked during optimization."""
        callback_calls = []

        def callback(iteration, swarm, result):
            callback_calls.append((iteration, result.value))

        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
        )

        optimizer.optimize(mock_fitness, n_iterations=20, callback=callback)

        assert len(callback_calls) == 20

    def test_coalition_contains_valid_indices(self, sample_clients, mock_fitness):
        """Test that coalition contains valid client indices."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=20)

        # All indices should be valid
        for idx in result.coalition:
            assert 0 <= idx < len(sample_clients)

        # No duplicates
        assert len(result.coalition) == len(set(result.coalition))


# =============================================================================
# Fairness Tests (Theorem 2)
# =============================================================================


class TestFairSwarmFairness:
    """Tests for fairness behavior (Theorem 2)."""

    def test_fairness_metrics_computed(self, sample_clients, target_distribution):
        """Test that fairness metrics are computed when target provided."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=10,
            target_distribution=target_distribution,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        assert result.fairness is not None
        assert result.fairness.demographic_divergence >= 0
        assert len(result.fairness.coalition_distribution) > 0

    def test_no_fairness_without_target(self, sample_clients, mock_fitness):
        """Test that no fairness metrics without target distribution."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            target_distribution=None,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=20)

        assert result.fairness is None

    def test_fairness_gradient_reduces_divergence(self, target_distribution):
        """Test that fairness gradient helps reduce divergence."""
        # Create clients with diverse demographics
        # Build an "all white" distribution with the same groups as the target
        all_white = dict.fromkeys(target_distribution.labels, 0.0)
        all_white["white"] = 1.0

        clients = []
        for i in range(20):
            if i < 10:
                # Half match target approximately
                demo = target_distribution
            else:
                # Half are very different (all white)
                demo = DemographicDistribution.from_dict(all_white)

            clients.append(
                Client(
                    id=f"client_{i}",
                    demographics=demo,
                    num_samples=1000,
                    data_quality=0.8,
                )
            )

        fitness = DemographicFitness(target_distribution=target_distribution)

        # With fairness coefficient
        optimizer_fair = FairSwarm(
            clients=clients,
            coalition_size=10,
            target_distribution=target_distribution,
            config=FairSwarmConfig(fairness_coefficient=0.5),
            seed=42,
        )

        # Without fairness coefficient
        optimizer_unfair = FairSwarm(
            clients=clients,
            coalition_size=10,
            target_distribution=target_distribution,
            config=FairSwarmConfig(fairness_coefficient=0.0),
            seed=42,
        )

        result_fair = optimizer_fair.optimize(fitness, n_iterations=100)
        result_unfair = optimizer_unfair.optimize(fitness, n_iterations=100)

        # Both should have fairness metrics
        assert result_fair.fairness is not None
        assert result_unfair.fairness is not None

        # Fair version should have lower divergence than unfair version
        assert (
            result_fair.fairness.demographic_divergence
            < result_unfair.fairness.demographic_divergence
        ), (
            f"Fair optimizer ({result_fair.fairness.demographic_divergence:.4f}) "
            f"should have lower divergence than unfair ({result_unfair.fairness.demographic_divergence:.4f})"
        )

        # Final divergence from fair optimizer should be below a reasonable threshold
        assert result_fair.fairness.demographic_divergence < 0.5, (
            f"Fair optimizer divergence ({result_fair.fairness.demographic_divergence:.4f}) "
            f"should be below 0.5"
        )


# =============================================================================
# Convergence Tests (Theorem 1)
# =============================================================================


class TestFairSwarmConvergence:
    """Tests for convergence behavior (Theorem 1)."""

    def test_theorem1_stability_condition(self, sample_clients, mock_fitness):
        """Test that Theorem 1 stability condition is enforced."""
        # ω + (c₁+c₂)/2 < 2
        config = FairSwarmConfig(
            inertia=0.7,
            cognitive=1.0,
            social=1.0,
        )

        # Verify condition: ω + (c₁+c₂)/2 < 2
        stability = config.inertia + (config.cognitive + config.social) / 2
        assert stability < 2, "Theorem 1 stability condition"

        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            config=config,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=100)

        # Should converge without diverging
        assert np.isfinite(result.fitness)

    def test_diversity_decreases_over_time(self, sample_clients, mock_fitness):
        """Test that swarm diversity generally decreases (convergence)."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(mock_fitness, n_iterations=100)
        diversity = result.convergence.diversity_history

        # First quarter should have higher diversity than last quarter (on average)
        first_quarter = np.mean(diversity[: len(diversity) // 4])
        last_quarter = np.mean(diversity[-len(diversity) // 4 :])

        assert last_quarter <= first_quarter + 0.1


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunFairswarm:
    """Tests for run_fairswarm convenience function."""

    def test_run_fairswarm_basic(self, sample_clients, mock_fitness):
        """Test run_fairswarm convenience function."""
        result = run_fairswarm(
            clients=sample_clients,
            coalition_size=5,
            fitness_fn=mock_fitness,
            n_iterations=20,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.coalition) == 5

    def test_run_fairswarm_with_target(self, sample_clients, target_distribution):
        """Test run_fairswarm with target distribution."""
        fitness = DemographicFitness(target_distribution=target_distribution)

        result = run_fairswarm(
            clients=sample_clients,
            coalition_size=10,
            fitness_fn=fitness,
            target_distribution=target_distribution,
            n_iterations=50,
        )

        assert result.fairness is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestFairSwarmIntegration:
    """Integration tests for complete optimization pipeline."""

    def test_full_optimization_pipeline(self, target_distribution):
        """Test complete optimization with all components."""
        # Create clients with 5 groups to match US_2020 target distribution
        clients = create_synthetic_clients(
            n_clients=30, n_demographic_groups=5, seed=42
        )

        # Configure optimizer
        config = FairSwarmConfig(
            swarm_size=20,
            max_iterations=100,
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coefficient=0.3,
            epsilon_fair=0.5,  # ε-fairness threshold
        )

        # Create fitness function
        fitness = DemographicFitness(
            target_distribution=target_distribution,
            divergence_weight=1.0,
        )

        # Run optimization
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=15,
            config=config,
            target_distribution=target_distribution,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=100)

        # Verify result structure
        assert len(result.coalition) == 15
        assert result.fitness is not None
        assert result.convergence is not None
        assert result.fairness is not None

        # Verify fairness metrics computed
        assert result.fairness.demographic_divergence >= 0
        assert len(result.fairness.coalition_distribution) > 0

        # Verify we can get summary
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_reset_and_rerun(self, sample_clients, mock_fitness):
        """Test reset and rerun with different seed."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            seed=42,
        )

        result1 = optimizer.optimize(mock_fitness, n_iterations=20)

        # Reset with new seed
        optimizer.reset(seed=123)
        result2 = optimizer.optimize(mock_fitness, n_iterations=20)

        # Results should potentially differ (different seed)
        # Just verify both are valid
        assert len(result1.coalition) == 5
        assert len(result2.coalition) == 5

    def test_get_swarm_state(self, sample_clients, mock_fitness):
        """Test getting swarm state during optimization."""
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
        )

        # Before optimization
        state1 = optimizer.get_swarm_state()
        assert state1["initialized"] is False

        # After optimization
        optimizer.optimize(mock_fitness, n_iterations=20)
        state2 = optimizer.get_swarm_state()

        assert state2["initialized"] is True
        assert state2["n_particles"] > 0
        assert state2["g_best_coalition"] is not None


# =============================================================================
# Theorem Condition Warning Tests
# =============================================================================


class TestTheoremWarnings:
    """Tests for pre-flight warnings about theorem conditions."""

    def test_low_iterations_warns(self, sample_clients, target_distribution):
        """Warn when T < T_min (Theorem 2)."""
        config = FairSwarmConfig(
            swarm_size=10,
            fairness_weight=0.3,
            fairness_coefficient=0.5,
            epsilon_fair=0.05,
        )
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            config=config,
            target_distribution=target_distribution,
            seed=42,
        )
        fitness = MockFitness(mode="mean_quality")

        with pytest.warns(UserWarning, match="Theorem 2"):
            optimizer.optimize(fitness, n_iterations=5)

    def test_zero_fairness_warns(self, sample_clients, target_distribution):
        """Warn when λ=0 and c3=0 with a target distribution."""
        config = FairSwarmConfig(
            swarm_size=10,
            fairness_weight=0.0,
            fairness_coefficient=0.0,
        )
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            config=config,
            target_distribution=target_distribution,
            seed=42,
        )
        fitness = MockFitness(mode="mean_quality")

        with pytest.warns(UserWarning, match="Theorem 2"):
            optimizer.optimize(fitness, n_iterations=10)

    def test_no_target_no_warning(self, sample_clients):
        """No warnings when target_distribution is None (e.g. StandardPSO)."""
        config = FairSwarmConfig(
            swarm_size=10,
            fairness_weight=0.0,
            fairness_coefficient=0.0,
        )
        optimizer = FairSwarm(
            clients=sample_clients,
            coalition_size=5,
            config=config,
            target_distribution=None,
            seed=42,
        )
        fitness = MockFitness(mode="mean_quality")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            optimizer.optimize(fitness, n_iterations=10)
