"""
Domain Adaptation for Sim-to-Real Transfer.

This module implements domain adaptation techniques for transferring
policies and models between simulated and real federated learning
environments.

Adaptation Strategies:
    1. Feature Alignment: Align simulation statistics to real data
    2. Importance Weighting: Reweight simulation samples
    3. Adversarial Adaptation: Learn domain-invariant representations
    4. Calibration: Adjust model confidence for domain shift

Research Attribution:
    - Digital Twin Framework: Dr. Elizabeth Bentley (Computer Networks 2023)
    - Domain Adaptation Theory: Ben-David et al. (2010)
    - FairSwarm Algorithm: Novel contribution (this thesis)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Domain adaptation strategy."""

    NONE = "none"
    FEATURE_ALIGNMENT = "feature_alignment"
    IMPORTANCE_WEIGHTING = "importance_weighting"
    MOMENT_MATCHING = "moment_matching"
    OPTIMAL_TRANSPORT = "optimal_transport"


@dataclass
class DomainAdaptationConfig:
    """
    Configuration for domain adaptation.

    Attributes:
        strategy: Adaptation strategy to use
        alignment_weight: Weight for feature alignment loss
        n_moments: Number of moments to match (for moment matching)
        regularization: Regularization strength
        max_iterations: Maximum adaptation iterations
        tolerance: Convergence tolerance
    """

    strategy: AdaptationStrategy = AdaptationStrategy.MOMENT_MATCHING
    alignment_weight: float = 0.1
    n_moments: int = 2
    regularization: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-4


@dataclass
class AdaptationResult:
    """
    Result of domain adaptation.

    Attributes:
        success: Whether adaptation succeeded
        strategy: Strategy used
        domain_distance_before: Domain distance before adaptation
        domain_distance_after: Domain distance after adaptation
        adaptation_loss: Final adaptation loss
        iterations: Number of iterations used
        source_weights: Importance weights for source samples
        transform_matrix: Feature transformation matrix
        metadata: Additional adaptation metadata
    """

    success: bool = True
    strategy: AdaptationStrategy = AdaptationStrategy.NONE
    domain_distance_before: float = 0.0
    domain_distance_after: float = 0.0
    adaptation_loss: float = 0.0
    iterations: int = 0
    source_weights: NDArray[np.float64] | None = None
    transform_matrix: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def distance_reduction(self) -> float:
        """Reduction in domain distance."""
        if self.domain_distance_before == 0:
            return 0.0
        return (
            self.domain_distance_before - self.domain_distance_after
        ) / self.domain_distance_before


class SimToRealAdapter:
    """
    Domain adapter for sim-to-real transfer in federated learning.

    Adapts policies, models, and statistics from the virtual
    environment to match the physical system distribution.

    Key Methods:
        - adapt_demographics(): Align demographic distributions
        - compute_importance_weights(): Compute sample weights
        - align_features(): Learn feature transformation
        - calibrate_predictions(): Adjust model confidence

    Integration with Digital Twin:
        Used by BentleyDigitalTwin to ensure that policies
        optimized in simulation transfer effectively to production.

    Example:
        >>> from fairswarm.digital_twin import SimToRealAdapter
        >>>
        >>> # Create adapter
        >>> adapter = SimToRealAdapter(
        ...     source_clients=virtual_clients,
        ...     target_clients=physical_clients,
        ... )
        >>>
        >>> # Compute adaptation
        >>> result = adapter.adapt()
        >>>
        >>> # Apply weights to simulation results
        >>> weighted_coalition = adapter.reweight_coalition(coalition)

    Author: Tenicka Norwood
    Advisor: Dr. Uttam Ghosh
    """

    def __init__(
        self,
        source_clients: list[Client],
        target_clients: list[Client],
        config: DomainAdaptationConfig | None = None,
    ):
        """
        Initialize SimToRealAdapter.

        Args:
            source_clients: Simulated (virtual) clients
            target_clients: Physical (real) clients
            config: Adaptation configuration
        """
        self.source_clients = source_clients
        self.target_clients = target_clients
        self.config = config or DomainAdaptationConfig()

        # Compute feature representations
        self._source_features = self._compute_features(source_clients)
        self._target_features = self._compute_features(target_clients)

        # Adaptation state
        self._importance_weights: NDArray[np.float64] | None = None
        self._transform_matrix: NDArray[np.float64] | None = None
        self._adapted = False

        logger.info(
            f"Initialized SimToRealAdapter: "
            f"{len(source_clients)} source, {len(target_clients)} target clients"
        )

    def _compute_features(self, clients: list[Client]) -> NDArray[np.float64]:
        """
        Compute feature matrix from clients.

        Features include:
        - Demographics (k dimensions)
        - Data quality (1 dimension)
        - Sample size (1 dimension, normalized)

        Args:
            clients: List of clients

        Returns:
            Feature matrix (n_clients x n_features)
        """
        if not clients:
            return np.array([]).reshape(0, 0)

        features = []
        max_samples = max(c.dataset_size for c in clients) if clients else 1

        for client in clients:
            demo = np.asarray(client.demographics)
            quality = np.array([client.data_quality])
            size = np.array([client.dataset_size / max_samples])
            client_features = np.concatenate([demo, quality, size])
            features.append(client_features)

        return np.array(features)

    def adapt(self) -> AdaptationResult:
        """
        Run domain adaptation.

        Returns:
            AdaptationResult with adaptation metrics
        """
        if self.config.strategy == AdaptationStrategy.NONE:
            return AdaptationResult(
                success=True,
                strategy=AdaptationStrategy.NONE,
            )

        # Compute initial domain distance
        distance_before = self._compute_domain_distance()

        # Apply selected strategy
        if self.config.strategy == AdaptationStrategy.IMPORTANCE_WEIGHTING:
            result = self._adapt_importance_weighting()
        elif self.config.strategy == AdaptationStrategy.MOMENT_MATCHING:
            result = self._adapt_moment_matching()
        elif self.config.strategy == AdaptationStrategy.FEATURE_ALIGNMENT:
            result = self._adapt_feature_alignment()
        elif self.config.strategy == AdaptationStrategy.OPTIMAL_TRANSPORT:
            result = self._adapt_optimal_transport()
        else:
            return AdaptationResult(
                success=False,
                strategy=self.config.strategy,
                metadata={"error": "Unknown strategy"},
            )

        # Compute final domain distance
        distance_after = self._compute_domain_distance(
            weights=result.source_weights,
            transform=result.transform_matrix,
        )

        result.domain_distance_before = distance_before
        result.domain_distance_after = distance_after
        self._adapted = True

        logger.info(
            f"Adaptation complete: distance {distance_before:.4f} -> {distance_after:.4f}"
        )

        return result

    def _compute_domain_distance(
        self,
        weights: NDArray[np.float64] | None = None,
        transform: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Compute domain distance between source and target.

        Uses Maximum Mean Discrepancy (MMD) as the distance metric.

        Args:
            weights: Importance weights for source samples
            transform: Feature transformation matrix

        Returns:
            Domain distance (MMD)
        """
        if len(self._source_features) == 0 or len(self._target_features) == 0:
            return 0.0

        source = self._source_features.copy()
        target = self._target_features.copy()

        # Apply transformation if provided
        if transform is not None:
            source = source @ transform

        # Apply weights if provided
        if weights is not None:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(source)) / len(source)

        # Compute MMD
        source_mean = np.average(source, axis=0, weights=weights)
        target_mean = np.mean(target, axis=0)

        mmd = np.linalg.norm(source_mean - target_mean)
        return float(mmd)

    def _adapt_importance_weighting(self) -> AdaptationResult:
        """
        Adapt using importance weighting.

        Computes weights to reweight source samples to match
        target distribution.

        Returns:
            AdaptationResult with importance weights
        """
        if len(self._source_features) == 0:
            return AdaptationResult(
                success=False,
                strategy=AdaptationStrategy.IMPORTANCE_WEIGHTING,
            )

        # Compute density ratio estimation using kernel mean matching
        n_source = len(self._source_features)

        # Simple approach: weight by demographic similarity to target mean
        target_mean = np.mean(self._target_features, axis=0)

        weights = np.zeros(n_source)
        for i, source_feat in enumerate(self._source_features):
            # Inverse distance weighting
            distance = np.linalg.norm(source_feat - target_mean)
            weights[i] = 1.0 / (1.0 + distance)

        # Normalize weights
        weights = weights / weights.sum() * n_source

        self._importance_weights = weights

        return AdaptationResult(
            success=True,
            strategy=AdaptationStrategy.IMPORTANCE_WEIGHTING,
            source_weights=weights,
            iterations=1,
        )

    def _adapt_moment_matching(self) -> AdaptationResult:
        """
        Adapt by matching moments of distributions.

        Matches first n_moments moments between source and target.

        Returns:
            AdaptationResult with transformation
        """
        if len(self._source_features) == 0 or len(self._target_features) == 0:
            return AdaptationResult(
                success=False,
                strategy=AdaptationStrategy.MOMENT_MATCHING,
            )

        # Compute moments
        source_mean = np.mean(self._source_features, axis=0)
        target_mean = np.mean(self._target_features, axis=0)

        source_std = np.std(self._source_features, axis=0) + 1e-8
        target_std = np.std(self._target_features, axis=0) + 1e-8

        # Compute transformation: (x - source_mean) / source_std * target_std + target_mean
        # This is equivalent to a diagonal transform + shift
        transform = np.diag(target_std / source_std)

        # Compute mean shift to align source mean to target mean after scaling
        shift = target_mean - source_mean * (target_std / source_std)

        # Apply the moment-matching transformation to source features:
        # transformed = source @ diag(target_std/source_std) + shift
        self._source_features = self._source_features @ transform + shift

        self._transform_matrix = transform

        return AdaptationResult(
            success=True,
            strategy=AdaptationStrategy.MOMENT_MATCHING,
            transform_matrix=transform,
            iterations=1,
            metadata={"mean_shift": shift.tolist()},
        )

    def _adapt_feature_alignment(self) -> AdaptationResult:
        """
        Adapt by learning feature transformation.

        Uses gradient descent to minimize domain distance.

        Returns:
            AdaptationResult with learned transformation
        """
        if len(self._source_features) == 0:
            return AdaptationResult(
                success=False,
                strategy=AdaptationStrategy.FEATURE_ALIGNMENT,
            )

        n_features = self._source_features.shape[1]

        # Initialize transformation as identity
        W = np.eye(n_features)
        learning_rate = 0.01

        losses = []
        for _iteration in range(self.config.max_iterations):
            # Compute gradient
            transformed_source = self._source_features @ W
            source_mean = np.mean(transformed_source, axis=0)
            target_mean = np.mean(self._target_features, axis=0)

            # Gradient of MMD w.r.t. W
            diff = source_mean - target_mean
            grad = (
                self._source_features.T @ np.ones((len(self._source_features), 1))
            ) @ diff.reshape(1, -1)
            grad = grad / len(self._source_features)

            # Add regularization
            grad += self.config.regularization * W

            # Update
            W = W - learning_rate * grad

            # Compute loss
            loss = np.linalg.norm(diff) + 0.5 * self.config.regularization * np.sum(
                W**2
            )
            losses.append(loss)

            # Check convergence
            if len(losses) > 1 and abs(losses[-1] - losses[-2]) < self.config.tolerance:
                break

        self._transform_matrix = W

        return AdaptationResult(
            success=True,
            strategy=AdaptationStrategy.FEATURE_ALIGNMENT,
            transform_matrix=W,
            adaptation_loss=losses[-1] if losses else 0.0,
            iterations=len(losses),
        )

    def _adapt_optimal_transport(self) -> AdaptationResult:
        """
        Adapt using optimal transport.

        Computes transport plan between source and target distributions.

        Returns:
            AdaptationResult with transport weights
        """
        if len(self._source_features) == 0 or len(self._target_features) == 0:
            return AdaptationResult(
                success=False,
                strategy=AdaptationStrategy.OPTIMAL_TRANSPORT,
            )

        n_source = len(self._source_features)
        n_target = len(self._target_features)

        # Compute cost matrix
        cost_matrix = np.zeros((n_source, n_target))
        for i in range(n_source):
            for j in range(n_target):
                cost_matrix[i, j] = np.linalg.norm(
                    self._source_features[i] - self._target_features[j]
                )

        # Simple Sinkhorn-like approximation
        # Initialize uniform transport
        T = np.ones((n_source, n_target)) / (n_source * n_target)

        # Sinkhorn iterations
        for _ in range(50):
            T = T * np.exp(-cost_matrix / self.config.regularization)
            T = T / T.sum(axis=1, keepdims=True)
            T = T / T.sum(axis=0, keepdims=True) * (1.0 / n_target)
            T = np.nan_to_num(T, nan=1.0 / (n_source * n_target))

        # Compute marginal weights
        weights = T.sum(axis=1) * n_source

        self._importance_weights = weights

        return AdaptationResult(
            success=True,
            strategy=AdaptationStrategy.OPTIMAL_TRANSPORT,
            source_weights=weights,
            iterations=50,
            metadata={"transport_cost": np.sum(cost_matrix * T)},
        )

    def get_importance_weights(self) -> NDArray[np.float64] | None:
        """
        Get importance weights for source samples.

        Returns:
            Importance weights or None if not adapted
        """
        return self._importance_weights

    def get_transform_matrix(self) -> NDArray[np.float64] | None:
        """
        Get feature transformation matrix.

        Returns:
            Transform matrix or None if not adapted
        """
        return self._transform_matrix

    def reweight_coalition(
        self,
        coalition: list[int],
    ) -> list[tuple[int, float]]:
        """
        Reweight coalition members using adaptation weights.

        Args:
            coalition: Coalition indices

        Returns:
            List of (index, weight) tuples
        """
        if self._importance_weights is None:
            return [(idx, 1.0) for idx in coalition]

        return [
            (idx, float(self._importance_weights[idx]))
            for idx in coalition
            if 0 <= idx < len(self._importance_weights)
        ]

    def transform_features(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Transform features using learned transformation.

        Args:
            features: Feature matrix to transform

        Returns:
            Transformed features
        """
        if self._transform_matrix is None:
            return features

        return features @ self._transform_matrix

    def adapt_demographics(
        self,
        source_distribution: DemographicDistribution,
    ) -> DemographicDistribution:
        """
        Adapt a demographic distribution from source to target domain.

        Args:
            source_distribution: Source domain demographics

        Returns:
            Adapted demographics
        """
        if not self._adapted:
            return source_distribution

        # Compute average target demographics
        if not self.target_clients:
            return source_distribution

        target_demos = [np.asarray(c.demographics) for c in self.target_clients]
        target_mean = np.mean(target_demos, axis=0)

        source_array = source_distribution.as_array()

        # Blend toward target
        alpha = 0.3  # Blending factor
        adapted_array = (1 - alpha) * source_array + alpha * target_mean

        # Renormalize
        adapted_array = adapted_array / adapted_array.sum()

        # Create new distribution
        if source_distribution.labels:
            labels = source_distribution.labels
        else:
            labels = None

        return DemographicDistribution(values=adapted_array, labels=labels)

    def __repr__(self) -> str:
        return (
            f"SimToRealAdapter("
            f"source={len(self.source_clients)}, "
            f"target={len(self.target_clients)}, "
            f"strategy={self.config.strategy.value}, "
            f"adapted={self._adapted})"
        )
