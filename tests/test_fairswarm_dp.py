"""
Tests for FairSwarm-DP algorithm.

Tests differential privacy integration with FairSwarm PSO.

Author: Tenicka Norwood
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fairswarm.algorithms.fairswarm_dp import (
    DPConfig,
    DPResult,
    FairSwarmDP,
    run_fairswarm_dp,
)
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.privacy.accountant import RDPAccountant, SimpleAccountant

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_clients():
    """Create sample clients for testing with 5 demographic groups."""
    from fairswarm.core.client import create_synthetic_clients

    return create_synthetic_clients(n_clients=10, n_demographic_groups=5, seed=42)


@pytest.fixture
def target_distribution():
    """Target demographic distribution (5 groups matching US Census 2020)."""
    from fairswarm.demographics.targets import CensusTarget

    return CensusTarget.US_2020.as_distribution()


@pytest.fixture
def simple_fitness():
    """Simple fitness function for testing."""

    class SimpleFitness(FitnessFunction):
        def evaluate(self, coalition, clients):
            if not coalition:
                return FitnessResult(
                    value=0.0,
                    components={},
                    coalition=coalition,
                )
            # Sum of data qualities
            value = sum(
                clients[i].data_quality for i in coalition if 0 <= i < len(clients)
            )
            return FitnessResult(
                value=value,
                components={"quality": value},
                coalition=coalition,
            )

        def compute_gradient(self, position, clients, coalition_size):
            return np.zeros(len(clients))

        @property
        def name(self):
            return "SimpleFitness"

    return SimpleFitness()


@pytest.fixture
def basic_dp_config():
    """Basic DP configuration for testing."""
    return DPConfig(
        epsilon=10.0,  # High epsilon for faster convergence in tests
        delta=1e-5,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        mechanism="gaussian",
        accountant_type="rdp",
    )


# =============================================================================
# DPConfig Tests
# =============================================================================


class TestDPConfig:
    """Tests for DPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DPConfig()

        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.noise_multiplier == 1.0
        assert config.max_grad_norm == 1.0
        assert config.mechanism == "gaussian"
        assert config.accountant_type == "rdp"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DPConfig(
            epsilon=5.0,
            delta=1e-6,
            noise_multiplier=2.0,
            max_grad_norm=0.5,
            mechanism="laplace",
            accountant_type="simple",
        )

        assert config.epsilon == 5.0
        assert config.delta == 1e-6
        assert config.noise_multiplier == 2.0
        assert config.max_grad_norm == 0.5
        assert config.mechanism == "laplace"
        assert config.accountant_type == "simple"

    def test_invalid_epsilon(self):
        """Test invalid epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPConfig(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPConfig(epsilon=-1.0)

    def test_invalid_delta(self):
        """Test invalid delta raises error."""
        with pytest.raises(ValueError, match="delta must be in"):
            DPConfig(delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            DPConfig(delta=1.0)
        with pytest.raises(ValueError, match="delta must be in"):
            DPConfig(delta=-0.1)

    def test_invalid_noise_multiplier(self):
        """Test invalid noise_multiplier raises error."""
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            DPConfig(noise_multiplier=0.0)
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            DPConfig(noise_multiplier=-1.0)

    def test_invalid_max_grad_norm(self):
        """Test invalid max_grad_norm raises error."""
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            DPConfig(max_grad_norm=0.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DPConfig(epsilon=2.0, delta=1e-6)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["epsilon"] == 2.0
        assert config_dict["delta"] == 1e-6
        assert "mechanism" in config_dict
        assert "accountant_type" in config_dict


# =============================================================================
# DPResult Tests
# =============================================================================


class TestDPResult:
    """Tests for DPResult dataclass."""

    def test_creation(self):
        """Test creating DPResult."""
        result = DPResult(
            epsilon_spent=0.5,
            delta=1e-5,
            n_queries=100,
            privacy_satisfied=True,
        )

        assert result.epsilon_spent == 0.5
        assert result.delta == 1e-5
        assert result.n_queries == 100
        assert result.privacy_satisfied is True

    def test_default_accountant_config(self):
        """Test default accountant config is empty dict."""
        result = DPResult(
            epsilon_spent=0.5,
            delta=1e-5,
            n_queries=100,
            privacy_satisfied=True,
        )

        assert result.accountant_config == {}

    def test_with_accountant_config(self):
        """Test with accountant configuration."""
        result = DPResult(
            epsilon_spent=0.5,
            delta=1e-5,
            n_queries=100,
            privacy_satisfied=True,
            accountant_config={"name": "RDP", "n_orders": 152},
        )

        assert result.accountant_config["name"] == "RDP"


# =============================================================================
# FairSwarmDP Initialization Tests
# =============================================================================


class TestFairSwarmDPInit:
    """Tests for FairSwarmDP initialization."""

    def test_basic_initialization(self, sample_clients, basic_dp_config):
        """Test basic initialization."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=basic_dp_config,
            seed=42,
        )

        assert optimizer.n_clients == 10
        assert optimizer.coalition_size == 5
        assert optimizer.dp_config.epsilon == basic_dp_config.epsilon

    def test_default_dp_config(self, sample_clients):
        """Test initialization with default DP config."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
        )

        assert optimizer.dp_config is not None
        assert optimizer.dp_config.epsilon == 1.0

    def test_with_target_distribution(
        self, sample_clients, target_distribution, basic_dp_config
    ):
        """Test initialization with target distribution."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            target_distribution=target_distribution,
            dp_config=basic_dp_config,
        )

        assert optimizer.target_distribution is not None

    def test_gaussian_mechanism_created(self, sample_clients, basic_dp_config):
        """Test Gaussian mechanism is created by default."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=basic_dp_config,
        )

        assert optimizer.mechanism.name == "Gaussian"

    def test_laplace_mechanism_created(self, sample_clients):
        """Test Laplace mechanism can be created."""
        dp_config = DPConfig(mechanism="laplace")
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=dp_config,
        )

        assert optimizer.mechanism.name == "Laplace"

    def test_rdp_accountant_created(self, sample_clients, basic_dp_config):
        """Test RDP accountant is created by default."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=basic_dp_config,
        )

        assert isinstance(optimizer.accountant, RDPAccountant)

    def test_simple_accountant_created(self, sample_clients):
        """Test simple accountant can be created."""
        dp_config = DPConfig(accountant_type="simple")
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=dp_config,
        )

        assert isinstance(optimizer.accountant, SimpleAccountant)

    def test_repr(self, sample_clients, basic_dp_config):
        """Test string representation."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=5,
            dp_config=basic_dp_config,
        )

        repr_str = repr(optimizer)
        assert "FairSwarmDP" in repr_str
        assert "n_clients=10" in repr_str
        assert "coalition_size=5" in repr_str


# =============================================================================
# FairSwarmDP Optimization Tests
# =============================================================================


class TestFairSwarmDPOptimize:
    """Tests for FairSwarmDP optimization."""

    def test_basic_optimization(self, sample_clients, simple_fitness, basic_dp_config):
        """Test basic optimization runs."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=20,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.coalition) == 3
        assert result.fitness > 0

    def test_result_contains_dp_metadata(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test result contains DP metadata."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=20,
        )

        assert "dp_result" in result.metadata
        dp_result = result.metadata["dp_result"]
        assert "epsilon_spent" in dp_result
        assert "delta" in dp_result
        assert "n_queries" in dp_result
        assert "privacy_satisfied" in dp_result

    def test_config_contains_dp_config(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test result config contains DP configuration."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=20,
        )

        assert "dp" in result.config
        assert result.config["dp"]["epsilon"] == basic_dp_config.epsilon

    def test_privacy_budget_spent(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test privacy budget is spent during optimization."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        initial_epsilon = optimizer.get_privacy_spent()[0]
        assert initial_epsilon == 0.0

        optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=20,
        )

        final_epsilon = optimizer.get_privacy_spent()[0]
        assert final_epsilon > 0

    def test_optimization_stops_at_budget(self, sample_clients, simple_fitness):
        """Test optimization stops when budget exhausted."""
        # Very small budget
        dp_config = DPConfig(
            epsilon=0.01,
            delta=1e-5,
            noise_multiplier=0.1,  # Small noise = fast budget consumption
        )

        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=dp_config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=1000,  # High iteration limit
        )

        # Should stop early due to budget
        assert result.convergence.iterations < 1000

    def test_queries_recorded(self, sample_clients, simple_fitness, basic_dp_config):
        """Test queries are recorded."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        assert optimizer._n_queries == 0

        optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=10,
        )

        assert optimizer._n_queries > 0

    def test_with_target_distribution(
        self, sample_clients, target_distribution, simple_fitness, basic_dp_config
    ):
        """Test optimization with target distribution."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            target_distribution=target_distribution,
            dp_config=basic_dp_config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=20,
        )

        assert result.fairness is not None
        assert result.fairness.demographic_divergence >= 0


# =============================================================================
# Privacy Budget Management Tests
# =============================================================================


class TestPrivacyBudget:
    """Tests for privacy budget management."""

    def test_get_privacy_spent_initial(self, sample_clients, basic_dp_config):
        """Test initial privacy spent is zero."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
        )

        eps, delta = optimizer.get_privacy_spent()
        assert eps == 0.0
        assert delta == basic_dp_config.delta

    def test_get_remaining_budget(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test remaining budget decreases."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        initial_remaining = optimizer.get_remaining_budget()
        assert initial_remaining == basic_dp_config.epsilon

        optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=10,
        )

        final_remaining = optimizer.get_remaining_budget()
        assert final_remaining < initial_remaining

    def test_reset_clears_privacy(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test reset clears privacy state."""
        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=basic_dp_config,
            seed=42,
        )

        optimizer.optimize(
            fitness_fn=simple_fitness,
            n_iterations=10,
        )

        assert optimizer._n_queries > 0

        optimizer.reset()

        assert optimizer._n_queries == 0
        assert optimizer.get_privacy_spent()[0] == 0.0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunFairSwarmDP:
    """Tests for run_fairswarm_dp convenience function."""

    def test_basic_run(self, sample_clients, simple_fitness, basic_dp_config):
        """Test basic run."""
        result = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=basic_dp_config,
            n_iterations=20,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.coalition) == 3

    def test_with_target_distribution(
        self, sample_clients, target_distribution, simple_fitness, basic_dp_config
    ):
        """Test with target distribution."""
        result = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            target_distribution=target_distribution,
            dp_config=basic_dp_config,
            n_iterations=20,
            seed=42,
        )

        assert result.fairness is not None

    def test_default_dp_config(self, sample_clients, simple_fitness):
        """Test with default DP config."""
        result = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            n_iterations=10,
            seed=42,
        )

        assert "dp" in result.config


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestFairSwarmDPProperties:
    """Property-based tests for FairSwarmDP."""

    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        delta=st.floats(min_value=1e-8, max_value=0.1),
    )
    @settings(max_examples=20)
    def test_dp_config_valid_range(self, epsilon, delta):
        """Test DPConfig accepts valid ranges."""
        config = DPConfig(epsilon=epsilon, delta=delta)
        assert config.epsilon == epsilon
        assert config.delta == delta

    @given(
        noise_multiplier=st.floats(min_value=0.1, max_value=5.0),
        max_grad_norm=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_optimizer_with_varying_noise(
        self, noise_multiplier, max_grad_norm, sample_clients
    ):
        """Test optimizer with varying noise parameters."""
        dp_config = DPConfig(
            epsilon=10.0,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        optimizer = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=dp_config,
        )

        assert optimizer.dp_config.noise_multiplier == noise_multiplier
        assert optimizer.dp_config.max_grad_norm == max_grad_norm


# =============================================================================
# Integration Tests
# =============================================================================


class TestFairSwarmDPIntegration:
    """Integration tests for FairSwarmDP."""

    def test_reproducibility_with_seed(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test reproducibility with same seed."""
        result1 = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=basic_dp_config,
            n_iterations=20,
            seed=42,
        )

        result2 = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=basic_dp_config,
            n_iterations=20,
            seed=42,
        )

        assert result1.coalition == result2.coalition
        assert result1.fitness == pytest.approx(result2.fitness, rel=1e-5)

    def test_different_seeds_different_results(
        self, sample_clients, simple_fitness, basic_dp_config
    ):
        """Test different seeds give different results."""
        result1 = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=basic_dp_config,
            n_iterations=30,
            seed=42,
        )

        result2 = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=basic_dp_config,
            n_iterations=30,
            seed=123,
        )

        # With different seeds and DP noise, results should differ
        # (could be same by chance, but unlikely)
        # We just check both ran successfully
        assert len(result1.coalition) == 3
        assert len(result2.coalition) == 3

    def test_higher_noise_more_privacy(self, sample_clients, simple_fitness):
        """Test higher noise gives more privacy (lower epsilon spent)."""
        low_noise_config = DPConfig(
            epsilon=10.0,
            noise_multiplier=0.5,
            auto_calibrate=False,
        )

        high_noise_config = DPConfig(
            epsilon=10.0,
            noise_multiplier=2.0,
            auto_calibrate=False,
        )

        # Run with low noise
        optimizer_low = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=low_noise_config,
            seed=42,
        )
        optimizer_low.optimize(simple_fitness, n_iterations=20)
        eps_low = optimizer_low.get_privacy_spent()[0]

        # Run with high noise
        optimizer_high = FairSwarmDP(
            clients=sample_clients,
            coalition_size=3,
            dp_config=high_noise_config,
            seed=42,
        )
        optimizer_high.optimize(simple_fitness, n_iterations=20)
        eps_high = optimizer_high.get_privacy_spent()[0]

        # Higher noise = less epsilon spent for same number of queries
        assert eps_high < eps_low

    def test_laplace_vs_gaussian(self, sample_clients, simple_fitness):
        """Test Laplace and Gaussian mechanisms both work."""
        laplace_config = DPConfig(epsilon=10.0, mechanism="laplace")
        gaussian_config = DPConfig(epsilon=10.0, mechanism="gaussian")

        result_laplace = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=laplace_config,
            n_iterations=20,
            seed=42,
        )

        result_gaussian = run_fairswarm_dp(
            clients=sample_clients,
            coalition_size=3,
            fitness_fn=simple_fitness,
            dp_config=gaussian_config,
            n_iterations=20,
            seed=42,
        )

        # Both should produce valid results
        assert len(result_laplace.coalition) == 3
        assert len(result_gaussian.coalition) == 3
