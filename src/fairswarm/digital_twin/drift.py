"""
Distribution Drift Detection for Digital Twins.

This module implements drift detection mechanisms for monitoring
distribution shift between physical and virtual environments.

Drift Types:
    1. Covariate Drift: Input distribution changes
    2. Concept Drift: Relationship between inputs and outputs changes
    3. Prior Drift: Label distribution changes
    4. Demographic Drift: Client demographics change

Detection Methods:
    - Statistical tests (KS, Chi-square, PSI)
    - Distribution distance metrics (KL, JS, Wasserstein)
    - Sliding window comparison
    - CUSUM and ADWIN algorithms

Research Attribution:
    - Digital Twin Framework: Dr. Elizabeth Bentley (Computer Networks 2023)
    - Drift Detection: Gama et al. (2014) "A Survey on Concept Drift"
    - FairSwarm Algorithm: Novel contribution (this thesis)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Type of distribution drift."""

    NONE = "none"
    COVARIATE = "covariate"
    CONCEPT = "concept"
    PRIOR = "prior"
    DEMOGRAPHIC = "demographic"
    GRADUAL = "gradual"
    SUDDEN = "sudden"


class DriftSeverity(Enum):
    """Severity level of detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftMetrics:
    """
    Metrics for drift detection.

    Attributes:
        kl_divergence: KL divergence between distributions
        js_divergence: Jensen-Shannon divergence
        psi: Population Stability Index
        ks_statistic: Kolmogorov-Smirnov statistic
        chi_square: Chi-square test statistic
        wasserstein: Wasserstein distance (Earth Mover's Distance)
    """

    kl_divergence: float = 0.0
    js_divergence: float = 0.0
    psi: float = 0.0
    ks_statistic: float = 0.0
    chi_square: float = 0.0
    wasserstein: float = 0.0

    def max_metric(self) -> float:
        """Get maximum normalized metric value."""
        # Normalize metrics to comparable scale
        normalized = [
            min(self.kl_divergence, 10) / 10,  # Cap at 10
            self.js_divergence,  # Already in [0, 1]
            min(self.psi, 1),  # Cap at 1
            self.ks_statistic,  # Already in [0, 1]
        ]
        return max(normalized)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "chi_square": self.chi_square,
            "wasserstein": self.wasserstein,
        }


@dataclass
class DriftResult:
    """
    Result of drift detection.

    Attributes:
        drift_detected: Whether significant drift was detected
        drift_type: Type of drift detected
        severity: Severity level
        metrics: Detailed drift metrics
        confidence: Confidence in detection (0-1)
        affected_dimensions: Which features drifted most
        timestamp: When drift was detected
        recommendations: Suggested actions
    """

    drift_detected: bool = False
    drift_type: DriftType = DriftType.NONE
    severity: DriftSeverity = DriftSeverity.NONE
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    confidence: float = 0.0
    affected_dimensions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Drift Detection Result",
            "=" * 50,
            f"Drift Detected: {self.drift_detected}",
            f"Type: {self.drift_type.value}",
            f"Severity: {self.severity.value}",
            f"Confidence: {self.confidence:.2%}",
        ]

        if self.affected_dimensions:
            lines.append(f"Affected: {', '.join(self.affected_dimensions)}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class DriftDetectorConfig:
    """
    Configuration for drift detection.

    Attributes:
        threshold_low: Threshold for low severity
        threshold_medium: Threshold for medium severity
        threshold_high: Threshold for high severity
        threshold_critical: Threshold for critical severity
        window_size: Size of sliding window
        min_samples: Minimum samples for detection
        significance_level: Statistical significance level
        use_bonferroni: Apply Bonferroni correction
    """

    threshold_low: float = 0.05
    threshold_medium: float = 0.1
    threshold_high: float = 0.2
    threshold_critical: float = 0.5
    window_size: int = 100
    min_samples: int = 30
    significance_level: float = 0.05
    use_bonferroni: bool = True


class DriftDetector:
    """
    Drift detector for federated learning environments.

    Monitors distribution shift between physical and virtual
    environments in the digital twin framework.

    Detection Approach:
        1. Maintain sliding windows of recent observations
        2. Compare reference and current distributions
        3. Apply multiple statistical tests
        4. Aggregate evidence for final decision

    Key Methods:
        - detect(): Run drift detection
        - add_observation(): Add new data point
        - get_drift_history(): Get history of detections
        - reset_reference(): Reset reference distribution

    Integration with Digital Twin:
        The DriftDetector monitors for distribution shift that
        may invalidate the digital twin's virtual representation,
        triggering re-synchronization.

    Example:
        >>> from fairswarm.digital_twin import DriftDetector
        >>>
        >>> # Create detector
        >>> detector = DriftDetector(
        ...     reference_clients=initial_clients,
        ... )
        >>>
        >>> # Monitor for drift
        >>> for round_num in range(100):
        ...     current_clients = get_current_clients()
        ...     result = detector.detect(current_clients)
        ...     if result.drift_detected:
        ...         handle_drift(result)

    Author: Tenicka Norwood
    Advisor: Dr. Uttam Ghosh
    """

    def __init__(
        self,
        reference_clients: Optional[List[Client]] = None,
        reference_distribution: Optional[DemographicDistribution] = None,
        config: Optional[DriftDetectorConfig] = None,
        on_drift: Optional[Callable[[DriftResult], None]] = None,
    ):
        """
        Initialize DriftDetector.

        Args:
            reference_clients: Reference client list
            reference_distribution: Reference demographics
            config: Detection configuration
            on_drift: Callback on drift detection
        """
        self.config = config or DriftDetectorConfig()
        self.on_drift = on_drift

        # Reference distribution
        if reference_clients:
            self._reference_features = self._extract_features(reference_clients)
            self._reference_distribution = self._compute_aggregate_demographics(
                reference_clients
            )
        elif reference_distribution:
            self._reference_distribution = reference_distribution.as_array()
            self._reference_features = None
        else:
            self._reference_distribution = None
            self._reference_features = None

        # Sliding window
        self._observation_window: List[NDArray[np.float64]] = []
        self._demographic_window: List[NDArray[np.float64]] = []

        # History
        self._drift_history: List[DriftResult] = []
        self._metric_history: List[DriftMetrics] = []

        logger.info(
            f"Initialized DriftDetector with window_size={self.config.window_size}"
        )

    def _extract_features(self, clients: List[Client]) -> NDArray[np.float64]:
        """
        Extract feature matrix from clients.

        Args:
            clients: List of clients

        Returns:
            Feature matrix
        """
        if not clients:
            return np.array([]).reshape(0, 0)

        features = []
        for client in clients:
            demo = np.asarray(client.demographics)
            features.append(demo)

        return np.array(features)

    def _compute_aggregate_demographics(
        self, clients: List[Client]
    ) -> NDArray[np.float64]:
        """
        Compute aggregate demographics from clients.

        Args:
            clients: List of clients

        Returns:
            Aggregate demographic distribution
        """
        if not clients:
            return np.array([])

        demos = [np.asarray(c.demographics) for c in clients]
        return np.mean(demos, axis=0)

    def detect(
        self,
        current_clients: Optional[List[Client]] = None,
        current_distribution: Optional[NDArray[np.float64]] = None,
    ) -> DriftResult:
        """
        Detect drift between reference and current distributions.

        Args:
            current_clients: Current client list
            current_distribution: Current demographic distribution

        Returns:
            DriftResult with detection outcome
        """
        if self._reference_distribution is None:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                recommendations=["Set reference distribution first"],
            )

        # Get current distribution
        if current_clients:
            current = self._compute_aggregate_demographics(current_clients)
            current_features = self._extract_features(current_clients)
        elif current_distribution is not None:
            current = current_distribution
            current_features = None
        else:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                recommendations=["Provide current clients or distribution"],
            )

        # Add to window
        self._demographic_window.append(current)
        if len(self._demographic_window) > self.config.window_size:
            self._demographic_window.pop(0)

        # Compute drift metrics
        metrics = self._compute_drift_metrics(self._reference_distribution, current)
        self._metric_history.append(metrics)

        # Determine if drift detected
        drift_detected, drift_type, severity = self._evaluate_drift(metrics)

        # Identify affected dimensions
        affected = self._identify_affected_dimensions(
            self._reference_distribution, current
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            drift_detected, drift_type, severity, affected
        )

        # Compute confidence
        confidence = self._compute_confidence(metrics)

        result = DriftResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            metrics=metrics,
            confidence=confidence,
            affected_dimensions=affected,
            recommendations=recommendations,
        )

        self._drift_history.append(result)

        # Callback
        if drift_detected and self.on_drift:
            self.on_drift(result)

        if drift_detected:
            logger.warning(
                f"Drift detected: {drift_type.value}, severity={severity.value}"
            )

        return result

    def _compute_drift_metrics(
        self,
        reference: NDArray[np.float64],
        current: NDArray[np.float64],
    ) -> DriftMetrics:
        """
        Compute all drift metrics.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            DriftMetrics with all computed values
        """
        # Ensure same shape
        if len(reference) != len(current):
            min_len = min(len(reference), len(current))
            reference = reference[:min_len]
            current = current[:min_len]

        # Add smoothing
        eps = 1e-10
        reference = np.clip(reference, eps, 1)
        current = np.clip(current, eps, 1)

        # Normalize
        reference = reference / reference.sum()
        current = current / current.sum()

        # KL Divergence
        kl = float(kl_divergence(current, reference))

        # Jensen-Shannon Divergence
        m = 0.5 * (reference + current)
        js = 0.5 * kl_divergence(reference, m) + 0.5 * kl_divergence(current, m)

        # Population Stability Index (PSI)
        psi = float(np.sum((current - reference) * np.log(current / reference)))

        # Kolmogorov-Smirnov statistic (approximation for discrete)
        ref_cdf = np.cumsum(reference)
        cur_cdf = np.cumsum(current)
        ks = float(np.max(np.abs(ref_cdf - cur_cdf)))

        # Chi-square statistic
        chi_sq = float(np.sum((current - reference) ** 2 / reference))

        # Wasserstein distance (1D Earth Mover's Distance)
        wasserstein = float(np.sum(np.abs(ref_cdf - cur_cdf)))

        return DriftMetrics(
            kl_divergence=kl,
            js_divergence=js,
            psi=abs(psi),
            ks_statistic=ks,
            chi_square=chi_sq,
            wasserstein=wasserstein,
        )

    def _evaluate_drift(
        self,
        metrics: DriftMetrics,
    ) -> Tuple[bool, DriftType, DriftSeverity]:
        """
        Evaluate if drift is significant.

        Args:
            metrics: Computed drift metrics

        Returns:
            (drift_detected, drift_type, severity)
        """
        max_metric = metrics.max_metric()

        # Determine severity
        if max_metric >= self.config.threshold_critical:
            severity = DriftSeverity.CRITICAL
        elif max_metric >= self.config.threshold_high:
            severity = DriftSeverity.HIGH
        elif max_metric >= self.config.threshold_medium:
            severity = DriftSeverity.MEDIUM
        elif max_metric >= self.config.threshold_low:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        drift_detected = severity != DriftSeverity.NONE

        # Determine drift type based on pattern
        drift_type = DriftType.NONE
        if drift_detected:
            # Check for gradual vs sudden drift
            if len(self._metric_history) > 5:
                recent_metrics = [m.max_metric() for m in self._metric_history[-5:]]
                if np.std(recent_metrics) < 0.05:
                    drift_type = DriftType.GRADUAL
                else:
                    drift_type = DriftType.SUDDEN
            else:
                drift_type = DriftType.DEMOGRAPHIC

        return drift_detected, drift_type, severity

    def _identify_affected_dimensions(
        self,
        reference: NDArray[np.float64],
        current: NDArray[np.float64],
    ) -> List[str]:
        """
        Identify which dimensions have drifted most.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            List of affected dimension names
        """
        if len(reference) != len(current):
            return []

        # Compute per-dimension drift
        diffs = np.abs(current - reference)

        # Get top drifting dimensions
        threshold = np.mean(diffs) + np.std(diffs)
        affected_indices = np.where(diffs > threshold)[0]

        # Generate names (generic if no category names available)
        return [f"dimension_{i}" for i in affected_indices]

    def _generate_recommendations(
        self,
        drift_detected: bool,
        drift_type: DriftType,
        severity: DriftSeverity,
        affected: List[str],
    ) -> List[str]:
        """
        Generate action recommendations.

        Args:
            drift_detected: Whether drift was detected
            drift_type: Type of drift
            severity: Drift severity
            affected: Affected dimensions

        Returns:
            List of recommendations
        """
        recommendations = []

        if not drift_detected:
            return ["No action required - distributions are stable"]

        if severity == DriftSeverity.CRITICAL:
            recommendations.extend([
                "URGENT: Re-synchronize digital twin immediately",
                "Halt deployment of current policies",
                "Investigate root cause of distribution shift",
            ])
        elif severity == DriftSeverity.HIGH:
            recommendations.extend([
                "Schedule digital twin re-synchronization",
                "Review coalition selection policies",
                "Increase monitoring frequency",
            ])
        elif severity == DriftSeverity.MEDIUM:
            recommendations.extend([
                "Monitor drift trend over next iterations",
                "Consider adapting fairness weights",
            ])
        else:
            recommendations.extend([
                "Continue monitoring",
                "Log drift for trend analysis",
            ])

        if drift_type == DriftType.GRADUAL:
            recommendations.append("Consider continuous adaptation strategy")
        elif drift_type == DriftType.SUDDEN:
            recommendations.append("Investigate sudden change in client population")

        return recommendations

    def _compute_confidence(self, metrics: DriftMetrics) -> float:
        """
        Compute confidence in drift detection.

        Args:
            metrics: Drift metrics

        Returns:
            Confidence score (0-1)
        """
        # Use agreement between metrics as confidence indicator
        normalized_metrics = [
            min(metrics.kl_divergence / 1, 1),
            metrics.js_divergence,
            min(metrics.psi, 1),
            metrics.ks_statistic,
        ]

        # Higher agreement = higher confidence
        std = np.std(normalized_metrics)
        agreement = 1 - min(std * 2, 1)

        # Also factor in magnitude
        magnitude = np.mean(normalized_metrics)

        return float(0.5 * agreement + 0.5 * magnitude)

    def add_observation(
        self,
        clients: Optional[List[Client]] = None,
        distribution: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Add observation to sliding window.

        Args:
            clients: New client observation
            distribution: New distribution observation
        """
        if clients:
            features = self._extract_features(clients)
            demographics = self._compute_aggregate_demographics(clients)
            self._observation_window.append(features)
            self._demographic_window.append(demographics)
        elif distribution is not None:
            self._demographic_window.append(distribution)

        # Maintain window size
        while len(self._observation_window) > self.config.window_size:
            self._observation_window.pop(0)
        while len(self._demographic_window) > self.config.window_size:
            self._demographic_window.pop(0)

    def reset_reference(
        self,
        clients: Optional[List[Client]] = None,
        distribution: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Reset reference distribution.

        Args:
            clients: New reference clients
            distribution: New reference distribution
        """
        if clients:
            self._reference_features = self._extract_features(clients)
            self._reference_distribution = self._compute_aggregate_demographics(clients)
        elif distribution is not None:
            self._reference_distribution = distribution
            self._reference_features = None

        # Clear windows
        self._observation_window.clear()
        self._demographic_window.clear()

        logger.info("Reference distribution reset")

    def get_drift_history(self) -> List[DriftResult]:
        """Get history of drift detections."""
        return self._drift_history.copy()

    def get_metric_history(self) -> List[DriftMetrics]:
        """Get history of drift metrics."""
        return self._metric_history.copy()

    def get_current_window_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current observation window.

        Returns:
            Dictionary with window statistics
        """
        if not self._demographic_window:
            return {"window_size": 0}

        window_array = np.array(self._demographic_window)
        return {
            "window_size": len(self._demographic_window),
            "mean": window_array.mean(axis=0).tolist(),
            "std": window_array.std(axis=0).tolist(),
            "min": window_array.min(axis=0).tolist(),
            "max": window_array.max(axis=0).tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"DriftDetector("
            f"window_size={self.config.window_size}, "
            f"observations={len(self._demographic_window)}, "
            f"detections={len(self._drift_history)})"
        )
