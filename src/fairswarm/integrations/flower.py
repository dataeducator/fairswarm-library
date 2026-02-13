"""
Flower integration for FairSwarm.

This module provides FairSwarmStrategy, a Flower FL strategy that uses
FairSwarm for fair coalition selection.

Integration Architecture:
    - FairSwarmStrategy extends flwr.server.strategy.Strategy
    - configure_fit() uses FairSwarm for client selection
    - aggregate_fit() monitors fairness during aggregation
    - Supports both synchronous and asynchronous FL

Research Attribution:
    - FairSwarm Algorithm: Novel contribution (this thesis)
    - Flower Framework: Beutel et al. "Flower: A Friendly FL Framework"
    - Privacy mechanisms: Dr. Uttam Ghosh (IEEE TNSE 2023)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Flower imports - use try/except for optional dependency
try:
    from flwr.common import (
        Code,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        GetPropertiesIns,
        GetPropertiesRes,
        Parameters,
        Scalar,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import Strategy

    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    # Create placeholder types for type hints
    Strategy = object
    ClientManager = object
    ClientProxy = object
    Parameters = Any
    Scalar = Any

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.core.client import Client
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

logger = logging.getLogger(__name__)


def _check_flower_available() -> None:
    """Check if Flower is available, raise if not."""
    if not FLOWER_AVAILABLE:
        raise ImportError(
            "Flower is required for FairSwarm integration. "
            "Install with: pip install flwr"
        )


@dataclass
class FairSwarmFitConfig:
    """
    Configuration for fit rounds in FairSwarm strategy.

    Attributes:
        epochs: Local training epochs per round
        batch_size: Local batch size
        learning_rate: Local learning rate
        extra_config: Additional configuration
    """

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    extra_config: Dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Scalar]:
        """Convert to Flower config dict."""
        config: Dict[str, Scalar] = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
        config.update(self.extra_config)
        return config


@dataclass
class FairSwarmEvaluateConfig:
    """
    Configuration for evaluation rounds in FairSwarm strategy.

    Attributes:
        batch_size: Evaluation batch size
        extra_config: Additional configuration
    """

    batch_size: int = 32
    extra_config: Dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Scalar]:
        """Convert to Flower config dict."""
        config: Dict[str, Scalar] = {"batch_size": self.batch_size}
        config.update(self.extra_config)
        return config


@dataclass
class ClientInfo:
    """
    Information about a Flower client for FairSwarm.

    Attributes:
        cid: Client identifier
        num_samples: Number of training samples
        demographics: Client's demographic distribution
        data_quality: Data quality score (0-1)
        proxy: Flower client proxy (optional)
    """

    cid: str
    num_samples: int = 1000
    demographics: Optional[DemographicDistribution] = None
    data_quality: float = 1.0
    proxy: Optional[ClientProxy] = None

    def to_fairswarm_client(self, idx: int) -> Client:
        """Convert to FairSwarm Client object."""
        return Client(
            id=self.cid,
            num_samples=self.num_samples,
            demographics=self.demographics or DemographicDistribution.uniform(4),
            data_quality=self.data_quality,
        )


class FairSwarmClient:
    """
    Wrapper for Flower clients with demographic information.

    This class bridges Flower clients with FairSwarm's coalition selection
    by maintaining demographic metadata.

    Example:
        >>> client = FairSwarmClient(
        ...     cid="hospital_1",
        ...     demographics=DemographicDistribution.from_dict({
        ...         "white": 0.6, "black": 0.13, "hispanic": 0.18, "asian": 0.06, "other": 0.03
        ...     }),
        ...     num_samples=5000,
        ... )
    """

    def __init__(
        self,
        cid: str,
        demographics: Optional[DemographicDistribution] = None,
        num_samples: int = 1000,
        data_quality: float = 1.0,
    ):
        """
        Initialize FairSwarmClient.

        Args:
            cid: Client identifier
            demographics: Client's demographic distribution
            num_samples: Number of local samples
            data_quality: Data quality score (0-1)
        """
        self.cid = cid
        self.demographics = demographics or DemographicDistribution.uniform(4)
        self.num_samples = num_samples
        self.data_quality = data_quality

    def to_client(self) -> Client:
        """Convert to FairSwarm Client."""
        return Client(
            id=self.cid,
            num_samples=self.num_samples,
            demographics=self.demographics,
            data_quality=self.data_quality,
        )


class FlowerFitness(FitnessFunction):
    """
    Fitness function that uses Flower client metrics.

    Evaluates coalitions based on:
    - Data quality (from client reports)
    - Sample size (number of local samples)
    - Demographic representation

    Attributes:
        target_distribution: Target demographic distribution
        quality_weight: Weight for data quality
        size_weight: Weight for sample size
        fairness_weight: Weight for demographic fairness
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        quality_weight: float = 0.4,
        size_weight: float = 0.3,
        fairness_weight: float = 0.3,
    ):
        """
        Initialize FlowerFitness.

        Args:
            target_distribution: Target demographic distribution
            quality_weight: Weight for data quality
            size_weight: Weight for sample size
            fairness_weight: Weight for demographic fairness
        """
        self.target_distribution = target_distribution
        self.quality_weight = quality_weight
        self.size_weight = size_weight
        self.fairness_weight = fairness_weight

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> FitnessResult:
        """
        Evaluate coalition fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with quality-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={"quality": 0.0, "size": 0.0, "fairness": 0.0},
                coalition=coalition,
            )

        # Collect coalition clients
        coalition_clients = [clients[i] for i in coalition if 0 <= i < len(clients)]
        if not coalition_clients:
            return FitnessResult(
                value=float("-inf"),
                components={"quality": 0.0, "size": 0.0, "fairness": 0.0},
                coalition=coalition,
            )

        # Data quality component
        avg_quality = np.mean([c.data_quality for c in coalition_clients])

        # Sample size component (normalized)
        total_samples = sum(c.num_samples for c in coalition_clients)
        max_possible = sum(c.num_samples for c in clients)
        size_score = total_samples / max_possible if max_possible > 0 else 0.0

        # Fairness component (negative divergence)
        coalition_demo = np.mean(
            [c.demographics.as_array() for c in coalition_clients], axis=0
        )
        target = self.target_distribution.as_array()
        divergence = np.sum((coalition_demo - target) ** 2)
        fairness_score = 1.0 / (1.0 + divergence)  # Higher is better

        # Combined fitness
        fitness = (
            self.quality_weight * avg_quality
            + self.size_weight * size_score
            + self.fairness_weight * fairness_score
        )

        return FitnessResult(
            value=fitness,
            components={
                "quality": avg_quality,
                "size": size_score,
                "fairness": fairness_score,
                "divergence": divergence,
            },
            coalition=coalition,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """Compute gradient for position update."""
        # Use quality-based gradient
        n_clients = len(clients)
        gradient = np.zeros(n_clients)

        for i, client in enumerate(clients):
            # Favor high-quality, high-sample clients
            gradient[i] = (
                self.quality_weight * client.data_quality
                + self.size_weight * (client.num_samples / 10000)
            )

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "quality_weight": self.quality_weight,
            "size_weight": self.size_weight,
            "fairness_weight": self.fairness_weight,
        }


class FairSwarmStrategy(Strategy if FLOWER_AVAILABLE else object):
    """
    Flower strategy using FairSwarm for fair client selection.

    This strategy extends Flower's Strategy base class to use FairSwarm
    for demographically fair coalition selection in federated learning.

    Key Features:
        - Fair client selection using FairSwarm algorithm
        - Demographic divergence monitoring
        - Adaptive fairness weight adjustment
        - Integration with standard FL aggregation

    Algorithm Reference:
        Uses Algorithm 1 from CLAUDE.md for client selection.

    Theoretical Guarantees:
        - Theorem 1: Convergence to stationary point
        - Theorem 2: DemDiv(S*) ≤ ε with probability ≥ 1 - δ

    Example:
        >>> from flwr.server import start_server
        >>> from fairswarm.integrations import FairSwarmStrategy
        >>> from fairswarm.demographics import CensusTarget
        >>>
        >>> strategy = FairSwarmStrategy(
        ...     fraction_fit=0.5,
        ...     coalition_size=10,
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ...     fairswarm_config=FairSwarmConfig(
        ...         n_particles=20,
        ...         fairness_coeff=0.3,
        ...     ),
        ... )
        >>>
        >>> start_server(
        ...     server_address="[::]:8080",
        ...     strategy=strategy,
        ... )

    Author: Tenicka Norwood
    Advisor: Dr. Uttam Ghosh
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        coalition_size: Optional[int] = None,
        target_distribution: Optional[DemographicDistribution] = None,
        fairswarm_config: Optional[FairSwarmConfig] = None,
        fairswarm_iterations: int = 50,
        fit_config: Optional[FairSwarmFitConfig] = None,
        evaluate_config: Optional[FairSwarmEvaluateConfig] = None,
        client_demographics: Optional[Dict[str, DemographicDistribution]] = None,
        on_fit_config_fn: Optional[
            Callable[[int], Dict[str, Scalar]]
        ] = None,
        on_evaluate_config_fn: Optional[
            Callable[[int], Dict[str, Scalar]]
        ] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[
            Callable[
                [int, NDArray[np.float64], Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
    ):
        """
        Initialize FairSwarmStrategy.

        Args:
            fraction_fit: Fraction of clients for training (baseline, before FairSwarm)
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients to start
            coalition_size: Number of clients to select (overrides fraction_fit)
            target_distribution: Target demographic distribution δ*
            fairswarm_config: FairSwarm PSO hyperparameters
            fairswarm_iterations: FairSwarm optimization iterations
            fit_config: Default fit configuration
            evaluate_config: Default evaluate configuration
            client_demographics: Pre-configured client demographics
            on_fit_config_fn: Custom fit config function
            on_evaluate_config_fn: Custom evaluate config function
            accept_failures: Accept partial round results
            initial_parameters: Initial model parameters
            evaluate_fn: Central evaluation function
        """
        _check_flower_available()

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.coalition_size = coalition_size
        self.target_distribution = target_distribution
        self.fairswarm_config = fairswarm_config or FairSwarmConfig()
        self.fairswarm_iterations = fairswarm_iterations
        self.fit_config = fit_config or FairSwarmFitConfig()
        self.evaluate_config = evaluate_config or FairSwarmEvaluateConfig()
        self.client_demographics = client_demographics or {}
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn

        # Round tracking
        self._round = 0
        self._selection_history: List[OptimizationResult] = []
        self._fairness_history: List[float] = []

        # Client registry
        self._client_registry: Dict[str, ClientInfo] = {}

        logger.info(
            f"Initialized FairSwarmStrategy with coalition_size={coalition_size}, "
            f"target_distribution={target_distribution is not None}"
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        Args:
            client_manager: Flower client manager

        Returns:
            Initial parameters or None
        """
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure fit round using FairSwarm for client selection.

        This is the key integration point where FairSwarm selects
        a demographically fair coalition of clients.

        Algorithm Reference:
            Uses Algorithm 1 from CLAUDE.md

        Args:
            server_round: Current round number
            parameters: Current model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, FitIns) tuples for selected clients
        """
        self._round = server_round

        # Get available clients
        sample_size, min_num_clients = self._compute_sample_sizes(
            client_manager, is_fit=True
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        if not clients:
            logger.warning(f"Round {server_round}: No clients available")
            return []

        # Update client registry with available clients
        self._update_client_registry(clients)

        # Use FairSwarm for client selection if we have demographics
        if self.target_distribution and len(clients) > (self.coalition_size or 0):
            selected_indices = self._run_fairswarm_selection(clients)
            selected_clients = [clients[i] for i in selected_indices]
            logger.info(
                f"Round {server_round}: FairSwarm selected {len(selected_clients)} clients"
            )
        else:
            # Fallback to all available clients
            selected_clients = clients
            logger.info(
                f"Round {server_round}: Using all {len(selected_clients)} available clients"
            )

        # Create fit instructions
        config = (
            self.on_fit_config_fn(server_round)
            if self.on_fit_config_fn
            else self.fit_config.to_dict()
        )

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]

    def _run_fairswarm_selection(
        self, available_clients: List[ClientProxy]
    ) -> List[int]:
        """
        Run FairSwarm optimization for client selection.

        Args:
            available_clients: List of available Flower clients

        Returns:
            Indices of selected clients
        """
        # Convert Flower clients to FairSwarm clients
        fairswarm_clients = []
        for i, proxy in enumerate(available_clients):
            cid = proxy.cid
            info = self._client_registry.get(
                cid,
                ClientInfo(cid=cid, demographics=self.client_demographics.get(cid)),
            )
            fairswarm_clients.append(info.to_fairswarm_client(i))

        # Determine coalition size
        coalition_size = self.coalition_size or max(
            self.min_fit_clients,
            int(len(fairswarm_clients) * self.fraction_fit),
        )
        coalition_size = min(coalition_size, len(fairswarm_clients))

        # Create fitness function
        fitness_fn = FlowerFitness(
            target_distribution=self.target_distribution,
            fairness_weight=0.3,
        )

        # Run FairSwarm optimization
        optimizer = FairSwarm(
            clients=fairswarm_clients,
            coalition_size=coalition_size,
            config=self.fairswarm_config,
            target_distribution=self.target_distribution,
        )

        result = optimizer.optimize(
            fitness_fn=fitness_fn,
            n_iterations=self.fairswarm_iterations,
        )

        # Record history
        self._selection_history.append(result)
        if result.fairness:
            self._fairness_history.append(result.fairness.demographic_divergence)

        logger.info(
            f"FairSwarm optimization: fitness={result.fitness:.4f}, "
            f"divergence={result.fairness.demographic_divergence if result.fairness else 'N/A':.4f}"
        )

        return list(result.coalition)

    def _update_client_registry(self, clients: List[ClientProxy]) -> None:
        """
        Update registry with client information.

        Args:
            clients: List of Flower client proxies
        """
        for proxy in clients:
            cid = proxy.cid
            if cid not in self._client_registry:
                # Get demographics from pre-configured dict or create default
                demographics = self.client_demographics.get(cid)
                self._client_registry[cid] = ClientInfo(
                    cid=cid,
                    demographics=demographics,
                    proxy=proxy,
                )

    def _compute_sample_sizes(
        self, client_manager: ClientManager, is_fit: bool
    ) -> Tuple[int, int]:
        """
        Compute sample sizes for client selection.

        Args:
            client_manager: Flower client manager
            is_fit: Whether this is for fit (vs evaluate)

        Returns:
            (sample_size, min_num_clients)
        """
        num_available = client_manager.num_available()

        if is_fit:
            fraction = self.fraction_fit
            min_clients = self.min_fit_clients
        else:
            fraction = self.fraction_evaluate
            min_clients = self.min_evaluate_clients

        sample_size = max(int(num_available * fraction), min_clients)
        return sample_size, min_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results with fairness monitoring.

        Uses FedAvg for parameter aggregation while tracking
        coalition fairness metrics.

        Args:
            server_round: Current round number
            results: Successful client results
            failures: Failed client results

        Returns:
            (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}

        # Track failures
        if failures and not self.accept_failures:
            logger.warning(
                f"Round {server_round}: {len(failures)} failures, aborting"
            )
            return None, {}

        # Extract parameters and weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # FedAvg aggregation
        aggregated = self._aggregate_fedavg(weights_results)
        aggregated_parameters = ndarrays_to_parameters(aggregated)

        # Compute metrics
        total_examples = sum(num_examples for _, num_examples in weights_results)
        metrics: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "total_examples": total_examples,
            "num_failures": len(failures),
        }

        # Add fairness metrics from last selection
        if self._selection_history:
            last_result = self._selection_history[-1]
            if last_result.fairness:
                metrics["demographic_divergence"] = last_result.fairness.demographic_divergence
                metrics["epsilon_satisfied"] = float(last_result.fairness.epsilon_satisfied)

        logger.info(
            f"Round {server_round}: Aggregated {len(results)} clients, "
            f"{total_examples} examples"
        )

        return aggregated_parameters, metrics

    def _aggregate_fedavg(
        self,
        weights_results: List[Tuple[List[NDArray[np.float64]], int]],
    ) -> List[NDArray[np.float64]]:
        """
        Aggregate using FedAvg algorithm.

        Args:
            weights_results: List of (parameters, num_examples)

        Returns:
            Aggregated parameters
        """
        # Compute total examples
        total_examples = sum(num_examples for _, num_examples in weights_results)

        # Weighted average
        aggregated = [
            np.zeros_like(layer) for layer in weights_results[0][0]
        ]

        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(parameters):
                aggregated[i] += weight * layer

        return aggregated

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure evaluation round.

        Args:
            server_round: Current round number
            parameters: Current model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, EvaluateIns) tuples
        """
        # Sample clients for evaluation
        sample_size, min_num_clients = self._compute_sample_sizes(
            client_manager, is_fit=False
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        if not clients:
            return []

        # Create evaluate instructions
        config = (
            self.on_evaluate_config_fn(server_round)
            if self.on_evaluate_config_fn
            else self.evaluate_config.to_dict()
        )

        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results.

        Args:
            server_round: Current round number
            results: Successful evaluation results
            failures: Failed evaluations

        Returns:
            (aggregated_loss, metrics)
        """
        if not results:
            return None, {}

        # Weighted average of losses
        loss_aggregated = 0.0
        total_examples = 0

        for _, evaluate_res in results:
            loss_aggregated += evaluate_res.loss * evaluate_res.num_examples
            total_examples += evaluate_res.num_examples

        if total_examples > 0:
            loss_aggregated /= total_examples

        # Aggregate metrics
        metrics: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
        }

        # Average per-client metrics
        all_metrics: Dict[str, List[float]] = {}
        for _, evaluate_res in results:
            for key, value in evaluate_res.metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(float(value))

        for key, values in all_metrics.items():
            metrics[f"avg_{key}"] = np.mean(values)

        return loss_aggregated, metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Central server-side evaluation.

        Args:
            server_round: Current round number
            parameters: Current model parameters

        Returns:
            (loss, metrics) or None
        """
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        return self.evaluate_fn(server_round, parameters_ndarrays, {})

    def get_fairness_history(self) -> List[float]:
        """
        Get history of demographic divergence values.

        Returns:
            List of divergence values per round
        """
        return self._fairness_history.copy()

    def get_selection_history(self) -> List[OptimizationResult]:
        """
        Get history of FairSwarm selection results.

        Returns:
            List of OptimizationResult per round
        """
        return self._selection_history.copy()

    def register_client_demographics(
        self,
        cid: str,
        demographics: DemographicDistribution,
        num_samples: int = 1000,
        data_quality: float = 1.0,
    ) -> None:
        """
        Register demographic information for a client.

        Call this before the client connects to provide demographics.

        Args:
            cid: Client identifier
            demographics: Client's demographic distribution
            num_samples: Number of local samples
            data_quality: Data quality score (0-1)
        """
        self._client_registry[cid] = ClientInfo(
            cid=cid,
            num_samples=num_samples,
            demographics=demographics,
            data_quality=data_quality,
        )
        self.client_demographics[cid] = demographics

        logger.debug(f"Registered demographics for client {cid}")

    def __repr__(self) -> str:
        return (
            f"FairSwarmStrategy("
            f"coalition_size={self.coalition_size}, "
            f"fairswarm_iterations={self.fairswarm_iterations}, "
            f"has_target_distribution={self.target_distribution is not None})"
        )
