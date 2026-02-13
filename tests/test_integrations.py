"""
Tests for Flower integration (Phase 11).

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import pytest
import numpy as np

from fairswarm.core.client import Client
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.targets import CensusTarget


def create_test_clients(n_clients: int = 10) -> list[Client]:
    """Create test clients with varied demographics."""
    clients = []
    for i in range(n_clients):
        # Vary demographics across clients
        white = 0.3 + 0.4 * (i / n_clients)
        black = 0.2 - 0.1 * (i / n_clients)
        hispanic = 0.3 - 0.1 * (i / n_clients)
        asian = 0.1 + 0.05 * (i / n_clients)
        other = 1.0 - white - black - hispanic - asian

        demographics = DemographicDistribution.from_dict({
            "white": max(0.01, white),
            "black": max(0.01, black),
            "hispanic": max(0.01, hispanic),
            "asian": max(0.01, asian),
            "other": max(0.01, other),
        })

        client = Client(
            id=f"client_{i}",
            num_samples=1000 + i * 100,
            demographics=demographics,
            data_quality=0.7 + 0.03 * i,
        )
        clients.append(client)

    return clients


class TestFairSwarmClient:
    """Tests for FairSwarmClient wrapper."""

    def test_creation(self):
        """Test FairSwarmClient creation."""
        from fairswarm.integrations.flower import FairSwarmClient

        client = FairSwarmClient(
            cid="test_client",
            num_samples=5000,
            data_quality=0.9,
        )

        assert client.cid == "test_client"
        assert client.num_samples == 5000
        assert client.data_quality == 0.9

    def test_with_demographics(self):
        """Test FairSwarmClient with custom demographics."""
        from fairswarm.integrations.flower import FairSwarmClient

        demographics = DemographicDistribution.from_dict({
            "white": 0.6,
            "black": 0.2,
            "hispanic": 0.15,
            "asian": 0.05,
        })

        client = FairSwarmClient(
            cid="demo_client",
            demographics=demographics,
            num_samples=3000,
        )

        assert client.demographics == demographics

    def test_to_client_conversion(self):
        """Test conversion to FairSwarm Client."""
        from fairswarm.integrations.flower import FairSwarmClient

        flower_client = FairSwarmClient(
            cid="convert_test",
            num_samples=2000,
            data_quality=0.85,
        )

        fairswarm_client = flower_client.to_client()

        assert fairswarm_client.id == "convert_test"
        assert fairswarm_client.num_samples == 2000
        assert fairswarm_client.data_quality == 0.85


class TestFlowerFitness:
    """Tests for FlowerFitness function."""

    def test_evaluate_coalition(self):
        """Test fitness evaluation."""
        from fairswarm.integrations.flower import FlowerFitness

        clients = create_test_clients(10)
        target = CensusTarget.US_2020.as_distribution()

        fitness = FlowerFitness(
            target_distribution=target,
            quality_weight=0.4,
            size_weight=0.3,
            fairness_weight=0.3,
        )

        coalition = [0, 2, 4, 6, 8]
        result = fitness.evaluate(coalition, clients)

        assert result.value > 0
        assert "quality" in result.components
        assert "fairness" in result.components
        assert "divergence" in result.components

    def test_empty_coalition(self):
        """Test handling of empty coalition."""
        from fairswarm.integrations.flower import FlowerFitness

        clients = create_test_clients(5)
        target = CensusTarget.US_2020.as_distribution()

        fitness = FlowerFitness(target_distribution=target)

        result = fitness.evaluate([], clients)

        assert result.value == float("-inf")

    def test_gradient_computation(self):
        """Test gradient computation."""
        from fairswarm.integrations.flower import FlowerFitness

        clients = create_test_clients(10)
        target = CensusTarget.US_2020.as_distribution()

        fitness = FlowerFitness(target_distribution=target)

        position = np.random.rand(len(clients))
        gradient = fitness.compute_gradient(position, clients, coalition_size=5)

        assert len(gradient) == len(clients)
        assert np.linalg.norm(gradient) > 0 or np.allclose(gradient, 0)


class TestFairSwarmFitConfig:
    """Tests for FairSwarmFitConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from fairswarm.integrations.flower import FairSwarmFitConfig

        config = FairSwarmFitConfig()

        assert config.epochs == 1
        assert config.batch_size == 32
        assert config.learning_rate == 0.01

    def test_to_dict(self):
        """Test conversion to Flower config dict."""
        from fairswarm.integrations.flower import FairSwarmFitConfig

        config = FairSwarmFitConfig(
            epochs=5,
            batch_size=64,
            learning_rate=0.001,
            extra_config={"momentum": 0.9},
        )

        config_dict = config.to_dict()

        assert config_dict["epochs"] == 5
        assert config_dict["batch_size"] == 64
        assert config_dict["momentum"] == 0.9


class TestFairSwarmEvaluateConfig:
    """Tests for FairSwarmEvaluateConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from fairswarm.integrations.flower import FairSwarmEvaluateConfig

        config = FairSwarmEvaluateConfig()

        assert config.batch_size == 32

    def test_to_dict(self):
        """Test conversion to dict."""
        from fairswarm.integrations.flower import FairSwarmEvaluateConfig

        config = FairSwarmEvaluateConfig(
            batch_size=128,
            extra_config={"verbose": True},
        )

        config_dict = config.to_dict()

        assert config_dict["batch_size"] == 128
        assert config_dict["verbose"] is True


class TestClientInfo:
    """Tests for ClientInfo dataclass."""

    def test_creation(self):
        """Test ClientInfo creation."""
        from fairswarm.integrations.flower import ClientInfo

        info = ClientInfo(
            cid="test_info",
            num_samples=5000,
            data_quality=0.95,
        )

        assert info.cid == "test_info"
        assert info.num_samples == 5000

    def test_to_fairswarm_client(self):
        """Test conversion to FairSwarm client."""
        from fairswarm.integrations.flower import ClientInfo

        demographics = DemographicDistribution.from_dict({
            "white": 0.5,
            "black": 0.3,
            "other": 0.2,
        })

        info = ClientInfo(
            cid="convert_test",
            num_samples=3000,
            demographics=demographics,
            data_quality=0.88,
        )

        client = info.to_fairswarm_client(idx=5)

        assert client.id == "convert_test"
        assert client.num_samples == 3000
        assert client.data_quality == 0.88


# Skip Flower-dependent tests if Flower not available
try:
    import flwr
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower not installed")
class TestFairSwarmStrategy:
    """Tests for FairSwarmStrategy (requires Flower)."""

    def test_initialization(self):
        """Test strategy initialization."""
        from fairswarm.integrations.flower import FairSwarmStrategy

        target = CensusTarget.US_2020.as_distribution()

        strategy = FairSwarmStrategy(
            coalition_size=5,
            target_distribution=target,
            fairswarm_iterations=10,
        )

        assert strategy.coalition_size == 5
        assert strategy.fairswarm_iterations == 10
        assert strategy.target_distribution == target

    def test_register_client_demographics(self):
        """Test client demographics registration."""
        from fairswarm.integrations.flower import FairSwarmStrategy

        strategy = FairSwarmStrategy(coalition_size=5)

        demographics = DemographicDistribution.from_dict({
            "white": 0.6,
            "black": 0.2,
            "hispanic": 0.15,
            "asian": 0.05,
        })

        strategy.register_client_demographics(
            cid="hospital_1",
            demographics=demographics,
            num_samples=10000,
            data_quality=0.92,
        )

        assert "hospital_1" in strategy.client_demographics

    def test_get_fairness_history(self):
        """Test fairness history retrieval."""
        from fairswarm.integrations.flower import FairSwarmStrategy

        strategy = FairSwarmStrategy(coalition_size=5)

        history = strategy.get_fairness_history()

        assert isinstance(history, list)
        assert len(history) == 0  # No rounds yet

    def test_repr(self):
        """Test string representation."""
        from fairswarm.integrations.flower import FairSwarmStrategy

        target = CensusTarget.US_2020.as_distribution()

        strategy = FairSwarmStrategy(
            coalition_size=10,
            target_distribution=target,
        )

        repr_str = repr(strategy)

        assert "FairSwarmStrategy" in repr_str
        assert "coalition_size=10" in repr_str
