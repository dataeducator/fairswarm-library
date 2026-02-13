"""
Virtual Environment for Digital Twin Simulation.

This module implements the simulated federated learning environment
for the Bentley Digital Twin framework.

Simulation Architecture:
    The VirtualEnvironment creates a simulated FL system that mirrors
    the physical system, enabling:
    - Coalition selection strategy testing
    - What-if analysis
    - Performance prediction
    - Fairness verification before deployment

Key Features:
    - VirtualClient: Simulated FL client with configurable behavior
    - VirtualEnvironment: Full FL simulation with FairSwarm
    - SimulationConfig: Configuration for simulation runs
    - SimulationResult: Results with performance metrics

Research Attribution:
    - Digital Twin Architecture: Dr. Elizabeth Bentley (Computer Networks 2023)
    - FairSwarm Algorithm: Novel contribution (this thesis)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.core.client import Client
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """
    Configuration for virtual environment simulation.

    Attributes:
        n_rounds: Number of FL rounds to simulate
        n_iterations: FairSwarm iterations per round
        coalition_size: Target coalition size
        learning_rate: Simulated learning rate
        aggregation_strategy: "fedavg" or "weighted"
        noise_level: Noise added to simulate real-world variance
        dropout_prob: Probability of client dropout per round
        latency_model: Model for simulating communication latency
        seed: Random seed for reproducibility
    """

    n_rounds: int = 50
    n_iterations: int = 50
    coalition_size: int = 10
    learning_rate: float = 0.01
    aggregation_strategy: str = "fedavg"
    noise_level: float = 0.01
    dropout_prob: float = 0.1
    latency_model: str = "uniform"
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if self.coalition_size < 1:
            raise ValueError("coalition_size must be >= 1")
        if not 0 <= self.dropout_prob <= 1:
            raise ValueError("dropout_prob must be in [0, 1]")


@dataclass
class SimulationResult:
    """
    Results from virtual environment simulation.

    Attributes:
        final_accuracy: Final simulated accuracy
        final_divergence: Final demographic divergence
        accuracy_history: Accuracy per round
        divergence_history: Divergence per round
        coalition_history: Selected coalitions per round
        convergence_round: Round where convergence detected
        total_time: Total simulation time (simulated)
        optimization_results: FairSwarm results per round
        metadata: Additional simulation metadata
    """

    final_accuracy: float = 0.0
    final_divergence: float = 0.0
    accuracy_history: List[float] = field(default_factory=list)
    divergence_history: List[float] = field(default_factory=list)
    coalition_history: List[Coalition] = field(default_factory=list)
    convergence_round: Optional[int] = None
    total_time: float = 0.0
    optimization_results: List[OptimizationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_converged(self) -> bool:
        """Whether simulation converged."""
        return self.convergence_round is not None

    @property
    def average_divergence(self) -> float:
        """Average demographic divergence across rounds."""
        if not self.divergence_history:
            return 0.0
        return float(np.mean(self.divergence_history))

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Virtual Environment Simulation Results",
            "=" * 50,
            f"Final Accuracy: {self.final_accuracy:.4f}",
            f"Final Divergence: {self.final_divergence:.4f}",
            f"Rounds: {len(self.accuracy_history)}",
            f"Converged: {self.is_converged}",
            f"Avg Divergence: {self.average_divergence:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


@dataclass
class VirtualClient:
    """
    Virtual representation of an FL client for simulation.

    Extends Client with simulation-specific attributes like
    latency models, dropout behavior, and update gradients.

    Attributes:
        client: Underlying FairSwarm client
        latency: Simulated network latency (ms)
        reliability: Probability of successful participation
        local_accuracy: Simulated local model accuracy
        gradient_variance: Variance in local updates
        last_participation: Last round participated
    """

    client: Client
    latency: float = 100.0  # ms
    reliability: float = 0.9
    local_accuracy: float = 0.8
    gradient_variance: float = 0.01
    last_participation: int = -1

    @property
    def id(self) -> str:
        """Client ID."""
        return self.client.id

    @property
    def demographics(self) -> DemographicDistribution:
        """Client demographics."""
        return self.client.demographics

    @property
    def num_samples(self) -> int:
        """Number of local samples."""
        return self.client.num_samples

    def simulate_update(
        self,
        round_num: int,
        global_accuracy: float,
        rng: np.random.Generator,
    ) -> Tuple[float, bool]:
        """
        Simulate local training update.

        Args:
            round_num: Current round number
            global_accuracy: Current global model accuracy
            rng: Random number generator

        Returns:
            (contribution, participated) tuple
        """
        # Check participation
        if rng.random() > self.reliability:
            return 0.0, False

        # Simulate local training contribution
        base_contribution = self.local_accuracy * self.client.data_quality
        noise = rng.normal(0, self.gradient_variance)
        contribution = np.clip(base_contribution + noise, 0, 1)

        self.last_participation = round_num
        return float(contribution), True

    def simulate_latency(self, rng: np.random.Generator) -> float:
        """
        Simulate network latency for this client.

        Args:
            rng: Random number generator

        Returns:
            Simulated latency in ms
        """
        # Add random variation to base latency
        return self.latency * (0.5 + rng.random())

    @classmethod
    def from_client(
        cls,
        client: Client,
        latency: float = 100.0,
        reliability: float = 0.9,
    ) -> VirtualClient:
        """
        Create VirtualClient from FairSwarm Client.

        Args:
            client: FairSwarm client
            latency: Base network latency
            reliability: Participation probability

        Returns:
            VirtualClient instance
        """
        return cls(
            client=client,
            latency=latency,
            reliability=reliability,
            local_accuracy=client.data_quality,
        )


class VirtualEnvironment:
    """
    Virtual federated learning environment for simulation.

    Creates a complete simulated FL system that mirrors physical
    deployments, enabling coalition strategy testing without
    affecting production systems.

    Key Features:
        - Simulated client behavior with configurable noise
        - FairSwarm coalition selection integration
        - Performance prediction and fairness verification
        - What-if analysis for different strategies

    Integration with Digital Twin:
        The VirtualEnvironment is used by BentleyDigitalTwin for:
        1. Testing coalition strategies before deployment
        2. Predicting performance of FairSwarm configurations
        3. Verifying Theorem 2 (ε-fairness) guarantees

    Example:
        >>> from fairswarm.digital_twin import VirtualEnvironment, SimulationConfig
        >>> from fairswarm.demographics import CensusTarget
        >>>
        >>> # Create virtual clients
        >>> clients = [create_virtual_client(i) for i in range(20)]
        >>>
        >>> # Configure simulation
        >>> config = SimulationConfig(
        ...     n_rounds=50,
        ...     coalition_size=10,
        ... )
        >>>
        >>> # Run simulation
        >>> env = VirtualEnvironment(
        ...     clients=clients,
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ...     config=config,
        ... )
        >>> result = env.run_simulation()
        >>> print(result.summary())

    Author: Tenicka Norwood
    Advisor: Dr. Uttam Ghosh
    """

    def __init__(
        self,
        clients: List[Client],
        target_distribution: Optional[DemographicDistribution] = None,
        config: Optional[SimulationConfig] = None,
        fairswarm_config: Optional[FairSwarmConfig] = None,
        fitness_fn: Optional[FitnessFunction] = None,
    ):
        """
        Initialize VirtualEnvironment.

        Args:
            clients: List of FairSwarm clients
            target_distribution: Target demographic distribution
            config: Simulation configuration
            fairswarm_config: FairSwarm PSO configuration
            fitness_fn: Custom fitness function
        """
        self.config = config or SimulationConfig()
        self.config.validate()

        self.target_distribution = target_distribution
        self.fairswarm_config = fairswarm_config or FairSwarmConfig()
        self.fitness_fn = fitness_fn

        # Create virtual clients
        self.virtual_clients = [
            VirtualClient.from_client(c) for c in clients
        ]
        self.clients = clients

        # Random state
        self.rng = np.random.default_rng(self.config.seed)

        # Simulation state
        self._current_round = 0
        self._global_accuracy = 0.5  # Starting accuracy
        self._accuracy_history: List[float] = []
        self._divergence_history: List[float] = []
        self._coalition_history: List[Coalition] = []
        self._optimization_results: List[OptimizationResult] = []

        # FairSwarm optimizer
        coalition_size = min(self.config.coalition_size, len(clients))
        self._optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=self.fairswarm_config,
            target_distribution=target_distribution,
            seed=self.config.seed,
        )

        logger.info(
            f"Initialized VirtualEnvironment with {len(clients)} clients, "
            f"coalition_size={coalition_size}"
        )

    def run_simulation(
        self,
        callback: Optional[Callable[[int, float, float], None]] = None,
        verbose: bool = False,
    ) -> SimulationResult:
        """
        Run full FL simulation with FairSwarm coalition selection.

        Args:
            callback: Optional callback(round, accuracy, divergence)
            verbose: Print progress

        Returns:
            SimulationResult with full simulation results
        """
        start_time = datetime.now()
        convergence_round = None

        for round_num in range(self.config.n_rounds):
            self._current_round = round_num

            # Run FairSwarm for coalition selection
            fitness = self._get_fitness_fn()
            opt_result = self._optimizer.optimize(
                fitness_fn=fitness,
                n_iterations=self.config.n_iterations,
            )
            self._optimization_results.append(opt_result)
            self._coalition_history.append(opt_result.coalition)

            # Simulate federated round
            round_accuracy = self._simulate_round(opt_result.coalition)
            self._accuracy_history.append(round_accuracy)
            self._global_accuracy = round_accuracy

            # Compute divergence
            divergence = self._compute_divergence(opt_result.coalition)
            self._divergence_history.append(divergence)

            # Callback
            if callback:
                callback(round_num, round_accuracy, divergence)

            if verbose and round_num % 10 == 0:
                logger.info(
                    f"Round {round_num}: accuracy={round_accuracy:.4f}, "
                    f"divergence={divergence:.4f}"
                )

            # Check convergence
            if self._check_convergence() and convergence_round is None:
                convergence_round = round_num
                if verbose:
                    logger.info(f"Converged at round {round_num}")

            # Reset optimizer for next round
            self._optimizer.reset(seed=self.config.seed + round_num + 1)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        result = SimulationResult(
            final_accuracy=self._accuracy_history[-1] if self._accuracy_history else 0.0,
            final_divergence=self._divergence_history[-1] if self._divergence_history else 0.0,
            accuracy_history=self._accuracy_history.copy(),
            divergence_history=self._divergence_history.copy(),
            coalition_history=self._coalition_history.copy(),
            convergence_round=convergence_round,
            total_time=elapsed_time,
            optimization_results=self._optimization_results.copy(),
            metadata={
                "n_clients": len(self.clients),
                "coalition_size": self.config.coalition_size,
                "n_rounds": self.config.n_rounds,
            },
        )

        logger.info(
            f"Simulation complete: {self.config.n_rounds} rounds, "
            f"final_accuracy={result.final_accuracy:.4f}"
        )

        return result

    def _get_fitness_fn(self) -> FitnessFunction:
        """Get or create fitness function."""
        if self.fitness_fn:
            return self.fitness_fn

        from fairswarm.fitness.fairness import DemographicFitness
        from fairswarm.fitness.mock import MockFitness

        if self.target_distribution:
            return DemographicFitness(
                target_distribution=self.target_distribution
            )
        else:
            return MockFitness(mode="mean_quality")

    def _simulate_round(self, coalition: Coalition) -> float:
        """
        Simulate a federated learning round.

        Args:
            coalition: Selected client indices

        Returns:
            Simulated round accuracy
        """
        contributions = []
        total_samples = 0

        for idx in coalition:
            if 0 <= idx < len(self.virtual_clients):
                vclient = self.virtual_clients[idx]

                # Simulate client update
                contribution, participated = vclient.simulate_update(
                    self._current_round,
                    self._global_accuracy,
                    self.rng,
                )

                if participated:
                    weight = vclient.num_samples
                    contributions.append(contribution * weight)
                    total_samples += weight

        if total_samples == 0:
            return self._global_accuracy

        # Weighted average of contributions
        avg_contribution = sum(contributions) / total_samples

        # Update global accuracy with learning rate
        new_accuracy = (
            (1 - self.config.learning_rate) * self._global_accuracy
            + self.config.learning_rate * avg_contribution
        )

        # Add noise
        noise = self.rng.normal(0, self.config.noise_level)
        new_accuracy = np.clip(new_accuracy + noise, 0, 1)

        return float(new_accuracy)

    def _compute_divergence(self, coalition: Coalition) -> float:
        """
        Compute demographic divergence for coalition.

        Args:
            coalition: Selected client indices

        Returns:
            KL divergence from target distribution
        """
        if not self.target_distribution:
            return 0.0

        if not coalition:
            return float("inf")

        # Compute coalition demographics
        demo_vectors = []
        for idx in coalition:
            if 0 <= idx < len(self.clients):
                demo_vectors.append(self.clients[idx].demographics.as_array())

        if not demo_vectors:
            return float("inf")

        coalition_demo = np.mean(demo_vectors, axis=0)
        target = self.target_distribution.as_array()

        return float(kl_divergence(coalition_demo, target))

    def _check_convergence(self, window: int = 10, threshold: float = 0.01) -> bool:
        """
        Check if simulation has converged.

        Args:
            window: Number of rounds to check
            threshold: Improvement threshold

        Returns:
            True if converged
        """
        if len(self._accuracy_history) < window:
            return False

        recent = self._accuracy_history[-window:]
        improvement = max(recent) - min(recent)
        return improvement < threshold

    def run_what_if(
        self,
        parameter_changes: Dict[str, Any],
    ) -> SimulationResult:
        """
        Run what-if analysis with modified parameters.

        Args:
            parameter_changes: Dictionary of parameter changes

        Returns:
            SimulationResult with modified configuration
        """
        # Create modified config
        modified_config = SimulationConfig(
            n_rounds=parameter_changes.get("n_rounds", self.config.n_rounds),
            n_iterations=parameter_changes.get("n_iterations", self.config.n_iterations),
            coalition_size=parameter_changes.get("coalition_size", self.config.coalition_size),
            learning_rate=parameter_changes.get("learning_rate", self.config.learning_rate),
            noise_level=parameter_changes.get("noise_level", self.config.noise_level),
            dropout_prob=parameter_changes.get("dropout_prob", self.config.dropout_prob),
            seed=parameter_changes.get("seed", self.config.seed),
        )

        # Create new environment with modified config
        what_if_env = VirtualEnvironment(
            clients=self.clients,
            target_distribution=self.target_distribution,
            config=modified_config,
            fairswarm_config=self.fairswarm_config,
            fitness_fn=self.fitness_fn,
        )

        return what_if_env.run_simulation()

    def reset(self) -> None:
        """Reset simulation state."""
        self._current_round = 0
        self._global_accuracy = 0.5
        self._accuracy_history.clear()
        self._divergence_history.clear()
        self._coalition_history.clear()
        self._optimization_results.clear()

        # Reset random state
        self.rng = np.random.default_rng(self.config.seed)

        # Reset optimizer
        self._optimizer.reset(seed=self.config.seed)

        logger.debug("VirtualEnvironment reset")

    def get_client_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about virtual clients.

        Returns:
            Dictionary with client statistics
        """
        latencies = [vc.latency for vc in self.virtual_clients]
        reliabilities = [vc.reliability for vc in self.virtual_clients]
        accuracies = [vc.local_accuracy for vc in self.virtual_clients]

        return {
            "n_clients": len(self.virtual_clients),
            "avg_latency": np.mean(latencies),
            "avg_reliability": np.mean(reliabilities),
            "avg_local_accuracy": np.mean(accuracies),
            "latency_std": np.std(latencies),
        }

    def __repr__(self) -> str:
        return (
            f"VirtualEnvironment(n_clients={len(self.virtual_clients)}, "
            f"coalition_size={self.config.coalition_size}, "
            f"n_rounds={self.config.n_rounds})"
        )
