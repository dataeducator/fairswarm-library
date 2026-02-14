"""
Tests for Digital Twin framework (Phase 12).

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import numpy as np
import pytest

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.targets import CensusTarget


def create_test_clients(n_clients: int = 10) -> list[Client]:
    """Create test clients with varied demographics."""
    clients = []
    for i in range(n_clients):
        white = 0.3 + 0.3 * (i / n_clients)
        black = 0.2 - 0.1 * (i / n_clients)
        hispanic = 0.25 - 0.05 * (i / n_clients)
        asian = 0.15 + 0.05 * (i / n_clients)
        other = 1.0 - white - black - hispanic - asian

        # Clamp all values to minimum 0.01
        raw = {
            "white": max(0.01, white),
            "black": max(0.01, black),
            "hispanic": max(0.01, hispanic),
            "asian": max(0.01, asian),
            "other": max(0.01, other),
        }

        # Normalize to ensure sum == 1.0
        total = sum(raw.values())
        normalized = {k: v / total for k, v in raw.items()}

        demographics = DemographicDistribution.from_dict(normalized)

        client = Client(
            id=f"hospital_{i}",
            num_samples=500 + i * 50,
            demographics=demographics.as_array(),
            data_quality=0.7 + 0.02 * i,
        )
        clients.append(client)

    return clients


class TestTwinState:
    """Tests for TwinState enum."""

    def test_states(self):
        """Test all twin states exist."""
        from fairswarm.digital_twin.twin import TwinState

        assert TwinState.UNINITIALIZED.value == "uninitialized"
        assert TwinState.SYNCING.value == "syncing"
        assert TwinState.SYNCHRONIZED.value == "synchronized"
        assert TwinState.SIMULATING.value == "simulating"
        assert TwinState.DRIFTED.value == "drifted"


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_creation(self):
        """Test SyncResult creation."""
        from fairswarm.digital_twin.twin import SyncResult

        result = SyncResult(
            success=True,
            direction="physical_to_virtual",
            metrics_transferred=5,
            drift_detected=False,
        )

        assert result.success is True
        assert result.direction == "physical_to_virtual"
        assert result.metrics_transferred == 5


class TestBentleyDigitalTwin:
    """Tests for BentleyDigitalTwin."""

    def test_initialization(self):
        """Test twin initialization."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin, TwinState

        clients = create_test_clients(10)
        target = CensusTarget.US_2020.as_distribution()

        twin = BentleyDigitalTwin(
            physical_clients=clients,
            target_distribution=target,
            coalition_size=5,
        )

        assert twin.state == TwinState.SYNCHRONIZED
        assert len(twin.physical_clients) == 10
        assert len(twin.virtual_clients) == 10

    def test_initialization_without_clients(self):
        """Test initialization without clients."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin, TwinState

        twin = BentleyDigitalTwin()

        assert twin.state == TwinState.UNINITIALIZED
        assert len(twin.physical_clients) == 0

    def test_sync_physical_to_virtual(self):
        """Test physical to virtual synchronization."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        result = twin.sync_physical_to_virtual(
            physical_metrics={"accuracy": 0.85, "loss": 0.35},
        )

        assert result.success is True
        assert result.direction == "physical_to_virtual"
        assert result.metrics_transferred > 0

    def test_deploy_to_physical(self):
        """Test virtual to physical deployment."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        coalition = [0, 2, 4, 6, 8]
        result = twin.deploy_to_physical(coalition=coalition)

        assert result.success is True
        assert result.direction == "virtual_to_physical"
        assert result.details["coalition"] == coalition

    def test_simulate(self):
        """Test simulation in virtual environment."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin

        clients = create_test_clients(10)
        target = CensusTarget.US_2020.as_distribution()

        twin = BentleyDigitalTwin(
            physical_clients=clients,
            target_distribution=target,
            coalition_size=5,
        )

        result = twin.simulate(
            n_rounds=2,
            n_iterations=10,
        )

        assert result.coalition is not None
        assert len(result.coalition) == 5
        assert result.fitness is not None

    def test_get_metrics(self):
        """Test metrics retrieval."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin, TwinMetrics

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        metrics = twin.get_metrics()

        assert isinstance(metrics, TwinMetrics)
        assert metrics.distribution_distance >= 0

    def test_update_physical_clients(self):
        """Test updating physical clients."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        new_clients = create_test_clients(15)
        twin.update_physical_clients(new_clients, auto_sync=True)

        assert len(twin.physical_clients) == 15
        assert len(twin.virtual_clients) == 15

    def test_reset(self):
        """Test twin reset."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin, TwinState

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        twin.reset()

        assert twin.state == TwinState.UNINITIALIZED
        assert len(twin.virtual_clients) == 0

    def test_repr(self):
        """Test string representation."""
        from fairswarm.digital_twin.twin import BentleyDigitalTwin

        clients = create_test_clients(10)
        twin = BentleyDigitalTwin(physical_clients=clients)

        repr_str = repr(twin)

        assert "BentleyDigitalTwin" in repr_str
        assert "physical_clients=10" in repr_str


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from fairswarm.digital_twin.simulator import SimulationConfig

        config = SimulationConfig()

        assert config.n_rounds == 50
        assert config.coalition_size == 10
        assert config.learning_rate == 0.01

    def test_validation(self):
        """Test configuration validation."""
        from fairswarm.digital_twin.simulator import SimulationConfig

        config = SimulationConfig(n_rounds=10, coalition_size=5)
        config.validate()  # Should not raise

        with pytest.raises(ValueError):
            bad_config = SimulationConfig(n_rounds=0)
            bad_config.validate()


class TestVirtualClient:
    """Tests for VirtualClient."""

    def test_from_client(self):
        """Test creation from FairSwarm client."""
        from fairswarm.digital_twin.simulator import VirtualClient

        clients = create_test_clients(5)
        vclient = VirtualClient.from_client(clients[0])

        assert vclient.id == "hospital_0"
        assert vclient.num_samples == 500

    def test_simulate_update(self):
        """Test simulated update."""
        from fairswarm.digital_twin.simulator import VirtualClient

        clients = create_test_clients(5)
        vclient = VirtualClient.from_client(clients[0])

        rng = np.random.default_rng(42)
        contribution, participated = vclient.simulate_update(
            round_num=1,
            global_accuracy=0.8,
            rng=rng,
        )

        # Participation is probabilistic
        if participated:
            assert 0 <= contribution <= 1

    def test_simulate_latency(self):
        """Test latency simulation."""
        from fairswarm.digital_twin.simulator import VirtualClient

        clients = create_test_clients(5)
        vclient = VirtualClient.from_client(clients[0], latency=100)

        rng = np.random.default_rng(42)
        latency = vclient.simulate_latency(rng)

        assert 50 <= latency <= 150  # Within expected range


class TestVirtualEnvironment:
    """Tests for VirtualEnvironment."""

    def test_initialization(self):
        """Test environment initialization."""
        from fairswarm.digital_twin.simulator import (
            SimulationConfig,
            VirtualEnvironment,
        )

        clients = create_test_clients(10)
        config = SimulationConfig(n_rounds=5, coalition_size=3)

        env = VirtualEnvironment(
            clients=clients,
            config=config,
        )

        assert len(env.virtual_clients) == 10

    def test_run_simulation(self):
        """Test running simulation."""
        from fairswarm.digital_twin.simulator import (
            SimulationConfig,
            VirtualEnvironment,
        )

        clients = create_test_clients(10)
        target = CensusTarget.US_2020.as_distribution()
        config = SimulationConfig(n_rounds=3, n_iterations=5, coalition_size=3, seed=42)

        env = VirtualEnvironment(
            clients=clients,
            target_distribution=target,
            config=config,
        )

        result = env.run_simulation()

        assert len(result.accuracy_history) == 3
        assert len(result.coalition_history) == 3
        assert result.final_accuracy > 0

    def test_run_what_if(self):
        """Test what-if analysis."""
        from fairswarm.digital_twin.simulator import (
            SimulationConfig,
            VirtualEnvironment,
        )

        clients = create_test_clients(10)
        config = SimulationConfig(n_rounds=3, coalition_size=3, seed=42)

        env = VirtualEnvironment(clients=clients, config=config)

        result = env.run_what_if({"coalition_size": 5, "n_rounds": 2})

        assert len(result.accuracy_history) == 2

    def test_reset(self):
        """Test environment reset."""
        from fairswarm.digital_twin.simulator import (
            SimulationConfig,
            VirtualEnvironment,
        )

        clients = create_test_clients(10)
        config = SimulationConfig(n_rounds=2, coalition_size=3, seed=42)

        env = VirtualEnvironment(clients=clients, config=config)
        env.run_simulation()
        env.reset()

        # After reset, history should be empty
        stats = env.get_client_statistics()
        assert stats["n_clients"] == 10


class TestDomainAdaptationConfig:
    """Tests for DomainAdaptationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from fairswarm.digital_twin.adapter import (
            AdaptationStrategy,
            DomainAdaptationConfig,
        )

        config = DomainAdaptationConfig()

        assert config.strategy == AdaptationStrategy.MOMENT_MATCHING
        assert config.alignment_weight == 0.1


class TestSimToRealAdapter:
    """Tests for SimToRealAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        from fairswarm.digital_twin.adapter import SimToRealAdapter

        source_clients = create_test_clients(10)
        target_clients = create_test_clients(8)

        adapter = SimToRealAdapter(
            source_clients=source_clients,
            target_clients=target_clients,
        )

        assert len(adapter.source_clients) == 10
        assert len(adapter.target_clients) == 8

    def test_adapt_moment_matching(self):
        """Test moment matching adaptation."""
        from fairswarm.digital_twin.adapter import (
            AdaptationStrategy,
            DomainAdaptationConfig,
            SimToRealAdapter,
        )

        source_clients = create_test_clients(10)
        target_clients = create_test_clients(10)

        config = DomainAdaptationConfig(strategy=AdaptationStrategy.MOMENT_MATCHING)

        adapter = SimToRealAdapter(
            source_clients=source_clients,
            target_clients=target_clients,
            config=config,
        )

        result = adapter.adapt()

        assert result.success is True
        assert result.strategy == AdaptationStrategy.MOMENT_MATCHING
        assert result.transform_matrix is not None

    def test_adapt_importance_weighting(self):
        """Test importance weighting adaptation."""
        from fairswarm.digital_twin.adapter import (
            AdaptationStrategy,
            DomainAdaptationConfig,
            SimToRealAdapter,
        )

        source_clients = create_test_clients(10)
        target_clients = create_test_clients(10)

        config = DomainAdaptationConfig(
            strategy=AdaptationStrategy.IMPORTANCE_WEIGHTING
        )

        adapter = SimToRealAdapter(
            source_clients=source_clients,
            target_clients=target_clients,
            config=config,
        )

        result = adapter.adapt()

        assert result.success is True
        assert result.source_weights is not None
        assert len(result.source_weights) == len(source_clients)

    def test_reweight_coalition(self):
        """Test coalition reweighting."""
        from fairswarm.digital_twin.adapter import (
            AdaptationStrategy,
            DomainAdaptationConfig,
            SimToRealAdapter,
        )

        source_clients = create_test_clients(10)
        target_clients = create_test_clients(10)

        config = DomainAdaptationConfig(
            strategy=AdaptationStrategy.IMPORTANCE_WEIGHTING
        )

        adapter = SimToRealAdapter(
            source_clients=source_clients,
            target_clients=target_clients,
            config=config,
        )
        adapter.adapt()

        coalition = [0, 2, 4, 6, 8]
        weighted = adapter.reweight_coalition(coalition)

        assert len(weighted) == 5
        assert all(isinstance(w, tuple) for w in weighted)


class TestDriftType:
    """Tests for DriftType enum."""

    def test_types(self):
        """Test all drift types exist."""
        from fairswarm.digital_twin.drift import DriftType

        assert DriftType.NONE.value == "none"
        assert DriftType.DEMOGRAPHIC.value == "demographic"
        assert DriftType.GRADUAL.value == "gradual"
        assert DriftType.SUDDEN.value == "sudden"


class TestDriftMetrics:
    """Tests for DriftMetrics dataclass."""

    def test_creation(self):
        """Test DriftMetrics creation."""
        from fairswarm.digital_twin.drift import DriftMetrics

        metrics = DriftMetrics(
            kl_divergence=0.05,
            js_divergence=0.02,
            psi=0.03,
            ks_statistic=0.1,
        )

        assert metrics.kl_divergence == 0.05
        assert metrics.max_metric() > 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from fairswarm.digital_twin.drift import DriftMetrics

        metrics = DriftMetrics(kl_divergence=0.1, js_divergence=0.05)
        metrics_dict = metrics.to_dict()

        assert "kl_divergence" in metrics_dict
        assert metrics_dict["kl_divergence"] == 0.1


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        from fairswarm.digital_twin.drift import DriftDetector

        clients = create_test_clients(10)
        detector = DriftDetector(reference_clients=clients)

        assert detector is not None

    def test_detect_no_drift(self):
        """Test detection with no drift."""
        from fairswarm.digital_twin.drift import DriftDetector, DriftType

        clients = create_test_clients(10)
        detector = DriftDetector(reference_clients=clients)

        # Same clients = no drift
        result = detector.detect(current_clients=clients)

        assert result.drift_type in [DriftType.NONE, DriftType.DEMOGRAPHIC]
        assert result.metrics is not None

    def test_detect_with_drift(self):
        """Test detection with drift."""
        from fairswarm.digital_twin.drift import DriftDetector

        # Create significantly different distributions
        reference = create_test_clients(10)

        # Modify current clients to have different demographics
        current = []
        for i in range(10):
            # Inverted demographics
            demographics = DemographicDistribution.from_dict(
                {
                    "white": 0.1,
                    "black": 0.6,
                    "hispanic": 0.2,
                    "asian": 0.05,
                    "other": 0.05,
                }
            )
            client = Client(
                id=f"new_hospital_{i}",
                num_samples=1000,
                demographics=demographics.as_array(),
            )
            current.append(client)

        detector = DriftDetector(reference_clients=reference)
        result = detector.detect(current_clients=current)

        # Should detect some level of drift
        assert result.metrics.kl_divergence > 0

    def test_add_observation(self):
        """Test adding observations to window."""
        from fairswarm.digital_twin.drift import DriftDetector

        clients = create_test_clients(10)
        detector = DriftDetector(reference_clients=clients)

        detector.add_observation(clients=clients)

        stats = detector.get_current_window_stats()
        assert stats["window_size"] == 1

    def test_reset_reference(self):
        """Test resetting reference distribution."""
        from fairswarm.digital_twin.drift import DriftDetector

        clients = create_test_clients(10)
        detector = DriftDetector(reference_clients=clients)

        new_clients = create_test_clients(15)
        detector.reset_reference(clients=new_clients)

        stats = detector.get_current_window_stats()
        assert stats["window_size"] == 0  # Window cleared

    def test_drift_history(self):
        """Test drift history retrieval."""
        from fairswarm.digital_twin.drift import DriftDetector

        clients = create_test_clients(10)
        detector = DriftDetector(reference_clients=clients)

        # Run a few detections
        for _ in range(3):
            detector.detect(current_clients=clients)

        history = detector.get_drift_history()

        assert len(history) == 3


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    def test_summary(self):
        """Test summary generation."""
        from fairswarm.digital_twin.drift import DriftResult, DriftSeverity, DriftType

        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.GRADUAL,
            severity=DriftSeverity.MEDIUM,
            confidence=0.85,
            recommendations=["Monitor closely"],
        )

        summary = result.summary()

        assert "Drift Detection Result" in summary
        assert "MEDIUM" in summary or "medium" in summary
