"""
Bentley Digital Twin for Federated Learning.

This module implements the core digital twin architecture with
bidirectional synchronization between physical and virtual environments.

The Bentley Framework:
    Digital twins create a virtual representation of the federated
    learning system that can be used for:
    1. Pre-deployment testing and validation
    2. What-if analysis of coalition selection strategies
    3. Continuous monitoring via virtual shadow execution
    4. Transfer learning from simulation to production

Bidirectional Sync:
    - Physical → Virtual: Real client metrics update the simulation
    - Virtual → Physical: Optimized policies deploy to production

Research Attribution:
    - Digital Twin Architecture: Dr. Elizabeth Bentley (Computer Networks 2023)
    - FSL-SAGE: Dr. Elizabeth Bentley (ICML 2025)
    - FairSwarm: Novel contribution (this thesis)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.core.client import Client
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.base import FitnessFunction
from fairswarm.types import ClientId, Coalition

logger = logging.getLogger(__name__)


class TwinState(Enum):
    """State of the digital twin."""

    UNINITIALIZED = "uninitialized"
    SYNCING = "syncing"
    SYNCHRONIZED = "synchronized"
    SIMULATING = "simulating"
    DEPLOYING = "deploying"
    DRIFTED = "drifted"
    ERROR = "error"


@dataclass
class SyncResult:
    """
    Result of a synchronization operation.

    Attributes:
        success: Whether sync completed successfully
        direction: "physical_to_virtual" or "virtual_to_physical"
        timestamp: When sync occurred
        metrics_transferred: Number of metrics synced
        drift_detected: Whether distribution drift was detected
        drift_magnitude: Magnitude of detected drift (if any)
        details: Additional sync details
    """

    success: bool
    direction: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics_transferred: int = 0
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TwinMetrics:
    """
    Metrics comparing physical and virtual environments.

    Attributes:
        accuracy_gap: Difference in model accuracy
        fairness_gap: Difference in demographic divergence
        latency_gap: Difference in communication latency
        distribution_distance: Distance between data distributions
        sync_lag: Time since last sync (seconds)
    """

    accuracy_gap: float = 0.0
    fairness_gap: float = 0.0
    latency_gap: float = 0.0
    distribution_distance: float = 0.0
    sync_lag: float = 0.0

    def total_gap(self) -> float:
        """Total gap across all metrics."""
        return (
            abs(self.accuracy_gap)
            + abs(self.fairness_gap)
            + abs(self.latency_gap)
            + abs(self.distribution_distance)
        )

    def is_aligned(self, threshold: float = 0.1) -> bool:
        """Check if twin is aligned with physical environment."""
        return self.total_gap() < threshold


@dataclass
class PhysicalState:
    """
    State captured from physical federated learning system.

    Attributes:
        clients: Physical client information
        model_parameters: Current model weights
        performance_metrics: Recent performance history
        demographic_distribution: Current coalition demographics
        round_number: Current training round
        timestamp: When state was captured
    """

    clients: list[Client] = field(default_factory=list)
    model_parameters: NDArray[np.float64] | None = None
    performance_metrics: dict[str, list[float]] = field(default_factory=dict)
    demographic_distribution: NDArray[np.float64] | None = None
    round_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VirtualState:
    """
    State of the virtual (simulated) environment.

    Attributes:
        clients: Virtual client representations
        model_parameters: Simulated model weights
        performance_metrics: Simulated performance
        demographic_distribution: Simulated demographics
        simulation_round: Current simulation round
        timestamp: When state was updated
    """

    clients: list[Client] = field(default_factory=list)
    model_parameters: NDArray[np.float64] | None = None
    performance_metrics: dict[str, list[float]] = field(default_factory=dict)
    demographic_distribution: NDArray[np.float64] | None = None
    simulation_round: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class BentleyDigitalTwin:
    """
    Digital Twin with bidirectional sim-to-real transfer.

    Implements the Bentley Framework for federated learning:
    - Maintains synchronized virtual representation of FL system
    - Enables policy optimization in simulation
    - Supports deployment of optimized policies to production
    - Monitors for distribution drift and model staleness

    Architecture:
        Physical System ←→ Digital Twin ←→ Virtual Environment
              ↓                 ↑                    ↓
        Real Metrics      Sync Engine         Simulation
              ↓                 ↑                    ↓
        Production FL    Domain Adapter      FairSwarm Optimizer

    Key Features:
        1. sync_physical_to_virtual(): Update twin from reality
        2. simulate(): Run FairSwarm optimization in virtual env
        3. prepare_deployment(): Generate deployment configuration
        4. deploy_to_physical(): Override in subclass for real deployment
        5. monitor_drift(): Detect distribution shift

    Theoretical Connection:
        The twin enables Theorem 2 (ε-fairness) verification
        before deploying coalition selection policies.

    Example:
        >>> from fairswarm.digital_twin import BentleyDigitalTwin
        >>> from fairswarm.demographics import CensusTarget
        >>>
        >>> # Create digital twin
        >>> twin = BentleyDigitalTwin(
        ...     physical_clients=production_clients,
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ... )
        >>>
        >>> # Sync from production
        >>> sync_result = twin.sync_physical_to_virtual()
        >>>
        >>> # Optimize in simulation
        >>> opt_result = twin.simulate(n_rounds=10)
        >>>
        >>> # Prepare deployment config if satisfactory
        >>> if opt_result.is_fair:
        ...     deploy_config = twin.prepare_deployment()

    Research Attribution:
        - Dr. Elizabeth Bentley (Computer Networks 2023)
        - Dr. Elizabeth Bentley (FSL-SAGE, ICML 2025)

    Author: Tenicka Norwood
    Advisor: Dr. Uttam Ghosh
    """

    def __init__(
        self,
        physical_clients: list[Client] | None = None,
        target_distribution: DemographicDistribution | None = None,
        fairswarm_config: FairSwarmConfig | None = None,
        coalition_size: int = 10,
        sync_threshold: float = 0.1,
        drift_threshold: float = 0.05,
        on_sync: Callable[[SyncResult], None] | None = None,
        on_drift: Callable[[float], None] | None = None,
    ):
        """
        Initialize BentleyDigitalTwin.

        Args:
            physical_clients: Initial physical client list
            target_distribution: Target demographic distribution
            fairswarm_config: FairSwarm PSO configuration
            coalition_size: Target coalition size
            sync_threshold: Threshold for sync triggering
            drift_threshold: Threshold for drift detection
            on_sync: Callback on sync events
            on_drift: Callback on drift detection
        """
        self.target_distribution = target_distribution
        self.fairswarm_config = fairswarm_config or FairSwarmConfig()
        self.coalition_size = coalition_size
        self.sync_threshold = sync_threshold
        self.drift_threshold = drift_threshold
        self.on_sync = on_sync
        self.on_drift = on_drift

        # State management
        self._state = TwinState.UNINITIALIZED
        self._physical_state = PhysicalState(clients=physical_clients or [])
        self._virtual_state = VirtualState()

        # History tracking
        self._sync_history: list[SyncResult] = []
        self._metrics_history: list[TwinMetrics] = []
        self._optimization_history: list[OptimizationResult] = []

        # Cached optimizer
        self._optimizer: FairSwarm | None = None
        self._last_sync: datetime | None = None

        # Initialize virtual state if physical clients provided
        if physical_clients:
            self._initialize_virtual_environment()

        logger.info(
            f"Initialized BentleyDigitalTwin with {len(physical_clients or [])} clients"
        )

    @property
    def state(self) -> TwinState:
        """Current twin state."""
        return self._state

    @property
    def is_synchronized(self) -> bool:
        """Check if twin is synchronized with physical system."""
        return self._state == TwinState.SYNCHRONIZED

    @property
    def physical_clients(self) -> list[Client]:
        """Physical client list."""
        return self._physical_state.clients

    @property
    def virtual_clients(self) -> list[Client]:
        """Virtual client representations."""
        return self._virtual_state.clients

    def _initialize_virtual_environment(self) -> None:
        """
        Initialize virtual environment from physical state.

        Creates virtual representations of physical clients
        with the same demographics and data characteristics.
        """
        self._virtual_state.clients = []

        for client in self._physical_state.clients:
            # Create virtual copy of client
            virtual_client = Client(
                id=ClientId(f"virtual_{client.id}"),
                num_samples=client.dataset_size,
                demographics=client.demographics,
                data_quality=client.data_quality,
            )
            self._virtual_state.clients.append(virtual_client)

        self._virtual_state.timestamp = datetime.now()
        self._state = TwinState.SYNCHRONIZED

        logger.debug(f"Initialized {len(self._virtual_state.clients)} virtual clients")

    def sync_physical_to_virtual(
        self,
        physical_metrics: dict[str, Any] | None = None,
        model_parameters: NDArray[np.float64] | None = None,
    ) -> SyncResult:
        """
        Synchronize virtual environment from physical system.

        Updates the virtual representation with current state
        from the production federated learning system.

        Args:
            physical_metrics: Current metrics from physical system
            model_parameters: Current model weights

        Returns:
            SyncResult with sync status and metrics

        Example:
            >>> result = twin.sync_physical_to_virtual(
            ...     physical_metrics={
            ...         "accuracy": 0.85,
            ...         "loss": 0.35,
            ...         "round": 50,
            ...     },
            ...     model_parameters=model.get_weights(),
            ... )
        """
        self._state = TwinState.SYNCING
        metrics_transferred = 0

        try:
            # Update model parameters
            if model_parameters is not None:
                self._physical_state.model_parameters = model_parameters.copy()
                self._virtual_state.model_parameters = model_parameters.copy()
                metrics_transferred += 1

            # Update performance metrics
            if physical_metrics:
                for key, value in physical_metrics.items():
                    if key not in self._physical_state.performance_metrics:
                        self._physical_state.performance_metrics[key] = []
                    if isinstance(value, (int, float)):
                        self._physical_state.performance_metrics[key].append(
                            float(value)
                        )
                        metrics_transferred += 1

                    # Copy to virtual state
                    self._virtual_state.performance_metrics = (
                        self._physical_state.performance_metrics.copy()
                    )

            # Check for drift
            drift_magnitude = self._compute_drift()
            drift_detected = drift_magnitude > self.drift_threshold

            if drift_detected:
                self._state = TwinState.DRIFTED
                if self.on_drift:
                    self.on_drift(drift_magnitude)
            else:
                self._state = TwinState.SYNCHRONIZED

            # Update timestamps
            self._physical_state.timestamp = datetime.now()
            self._virtual_state.timestamp = datetime.now()
            self._last_sync = datetime.now()

            # Create result
            result = SyncResult(
                success=True,
                direction="physical_to_virtual",
                metrics_transferred=metrics_transferred,
                drift_detected=drift_detected,
                drift_magnitude=drift_magnitude,
                details={
                    "physical_round": (
                        physical_metrics.get("round", 0) if physical_metrics else 0
                    )
                },
            )

            self._sync_history.append(result)

            if self.on_sync:
                self.on_sync(result)

            logger.info(
                f"Sync physical→virtual: {metrics_transferred} metrics, "
                f"drift={drift_magnitude:.4f}"
            )

            return result

        except Exception as e:
            self._state = TwinState.ERROR
            logger.error(f"Sync failed: {e}")
            return SyncResult(
                success=False,
                direction="physical_to_virtual",
                details={"error": str(e)},
            )

    def deploy_to_physical(
        self,
        coalition: Coalition | None = None,
        policy_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Deploy optimized policy to physical system.

        This method is not implemented in the base digital twin because
        physical deployment is platform-specific. Subclass BentleyDigitalTwin
        and override this method with your deployment logic, or use
        prepare_deployment() to generate a deployment configuration.

        Args:
            coalition: Optimized coalition to deploy
            policy_parameters: Additional policy parameters

        Raises:
            NotImplementedError: Always. Physical deployment requires
                platform-specific implementation.
        """
        raise NotImplementedError(
            "Physical deployment requires platform-specific implementation. "
            "Use prepare_deployment() to generate deployment configuration, "
            "or subclass BentleyDigitalTwin and override deploy_to_physical() "
            "with your platform-specific deployment logic."
        )

    def prepare_deployment(
        self,
        coalition: Coalition | None = None,
        policy_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare deployment configuration for physical system.

        Generates a deployment configuration dictionary containing the
        coalition, policy parameters, and virtual model metadata without
        actually performing a deployment. The returned configuration can
        be used by platform-specific deployment tooling.

        Args:
            coalition: Optimized coalition to deploy
            policy_parameters: Additional policy parameters

        Returns:
            Deployment configuration dictionary with keys:
                - coalition: List of client IDs (if provided)
                - coalition_size: Number of clients (if provided)
                - policy_parameters: Policy parameters (if provided)
                - has_model: Whether virtual model parameters are available
                - virtual_simulation_round: Current simulation round
                - timestamp: When configuration was generated

        Example:
            >>> # After optimization
            >>> opt_result = twin.simulate(n_rounds=10)
            >>>
            >>> # Generate deployment config
            >>> config = twin.prepare_deployment(
            ...     coalition=opt_result.coalition,
            ... )
            >>> # Pass config to your platform-specific deployer
            >>> my_deployer.deploy(config)
        """
        config: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "virtual_simulation_round": self._virtual_state.simulation_round,
        }

        if coalition is not None:
            config["coalition"] = list(coalition)
            config["coalition_size"] = len(coalition)

        if policy_parameters:
            config["policy_parameters"] = policy_parameters

        config["has_model"] = self._virtual_state.model_parameters is not None

        logger.info(
            f"Prepared deployment config: "
            f"coalition={len(coalition) if coalition else 0} clients"
        )

        return config

    def simulate(
        self,
        fitness_fn: FitnessFunction | None = None,
        n_rounds: int = 1,
        n_iterations: int = 50,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run FairSwarm optimization in virtual environment.

        Uses the digital twin to test coalition selection
        strategies without affecting the physical system.

        Args:
            fitness_fn: Fitness function for optimization
            n_rounds: Number of FL rounds to simulate
            n_iterations: FairSwarm iterations per round
            verbose: Print progress

        Returns:
            OptimizationResult from FairSwarm

        Example:
            >>> result = twin.simulate(
            ...     fitness_fn=DemographicFitness(target),
            ...     n_rounds=5,
            ...     n_iterations=100,
            ... )
            >>> print(f"Fairness: {result.fairness.demographic_divergence:.4f}")
        """
        self._state = TwinState.SIMULATING

        if not self._virtual_state.clients:
            raise ValueError("No virtual clients. Call sync_physical_to_virtual first.")

        # Create optimizer if needed
        if self._optimizer is None:
            self._optimizer = FairSwarm(
                clients=self._virtual_state.clients,
                coalition_size=min(
                    self.coalition_size, len(self._virtual_state.clients)
                ),
                config=self.fairswarm_config,
                target_distribution=self.target_distribution,
            )
        optimizer = self._optimizer

        # Use default fitness if none provided
        if fitness_fn is None:
            from fairswarm.fitness.fairness import DemographicFitness

            if self.target_distribution:
                fitness_fn = DemographicFitness(
                    target_distribution=self.target_distribution
                )
            else:
                from fairswarm.fitness.mock import MockFitness

                fitness_fn = MockFitness(mode="mean_quality")

        # Run optimization
        best_result: OptimizationResult | None = None

        for round_num in range(n_rounds):
            if verbose:
                logger.info(f"Simulation round {round_num + 1}/{n_rounds}")

            result = optimizer.optimize(
                fitness_fn=fitness_fn,
                n_iterations=n_iterations,
                verbose=verbose,
            )

            if best_result is None or result.fitness > best_result.fitness:
                best_result = result

            # Reset optimizer for next round
            optimizer.reset()

            # Update virtual state
            self._virtual_state.simulation_round = round_num + 1

        assert best_result is not None, "No optimization rounds were executed"
        self._optimization_history.append(best_result)
        self._state = TwinState.SYNCHRONIZED

        logger.info(
            f"Simulation complete: fitness={best_result.fitness:.4f}, "
            f"converged={best_result.is_converged}"
        )

        return best_result

    def _compute_drift(self) -> float:
        """
        Compute drift between physical and virtual states.

        Returns:
            Drift magnitude (0 = perfectly aligned)
        """
        if not self._physical_state.clients or not self._virtual_state.clients:
            return 0.0

        # Compute demographic drift
        physical_demos = np.array(
            [np.asarray(c.demographics) for c in self._physical_state.clients]
        )
        virtual_demos = np.array(
            [np.asarray(c.demographics) for c in self._virtual_state.clients]
        )

        # Average demographics
        physical_mean = np.mean(physical_demos, axis=0)
        virtual_mean = np.mean(virtual_demos, axis=0)

        # L2 distance
        drift = np.linalg.norm(physical_mean - virtual_mean)

        return float(drift)

    def get_metrics(self) -> TwinMetrics:
        """
        Get current twin alignment metrics.

        Returns:
            TwinMetrics comparing physical and virtual states
        """
        # Compute accuracy gap
        accuracy_gap = 0.0
        if (
            "accuracy" in self._physical_state.performance_metrics
            and "accuracy" in self._virtual_state.performance_metrics
        ):
            physical_acc = self._physical_state.performance_metrics["accuracy"]
            virtual_acc = self._virtual_state.performance_metrics["accuracy"]
            if physical_acc and virtual_acc:
                accuracy_gap = abs(physical_acc[-1] - virtual_acc[-1])

        # Compute distribution distance
        distribution_distance = self._compute_drift()

        # Compute sync lag
        sync_lag = 0.0
        if self._last_sync:
            sync_lag = (datetime.now() - self._last_sync).total_seconds()

        metrics = TwinMetrics(
            accuracy_gap=accuracy_gap,
            distribution_distance=distribution_distance,
            sync_lag=sync_lag,
        )

        self._metrics_history.append(metrics)
        return metrics

    def update_physical_clients(
        self,
        clients: list[Client],
        auto_sync: bool = True,
    ) -> None:
        """
        Update physical client list.

        Args:
            clients: New physical client list
            auto_sync: Automatically sync to virtual
        """
        self._physical_state.clients = clients

        if auto_sync:
            self._initialize_virtual_environment()

        logger.debug(f"Updated {len(clients)} physical clients")

    def get_sync_history(self) -> list[SyncResult]:
        """Get history of sync operations."""
        return self._sync_history.copy()

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of optimization results."""
        return self._optimization_history.copy()

    def reset(self) -> None:
        """Reset twin to uninitialized state."""
        self._state = TwinState.UNINITIALIZED
        self._virtual_state = VirtualState()
        self._optimizer = None
        self._last_sync = None
        self._sync_history.clear()
        self._metrics_history.clear()
        self._optimization_history.clear()

        logger.info("Digital twin reset")

    def __repr__(self) -> str:
        return (
            f"BentleyDigitalTwin(state={self._state.value}, "
            f"physical_clients={len(self._physical_state.clients)}, "
            f"virtual_clients={len(self._virtual_state.clients)})"
        )
