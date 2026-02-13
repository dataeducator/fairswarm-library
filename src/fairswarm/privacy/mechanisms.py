"""
Differential privacy noise mechanisms.

This module provides noise mechanisms for achieving differential
privacy in FairSwarm optimization.

Key Mechanisms:
    - Laplace: For ε-differential privacy (pure DP)
    - Gaussian: For (ε,δ)-differential privacy (approximate DP)
    - Exponential: For private selection

Mathematical Foundation:
    Laplace Mechanism: M(x) = f(x) + Lap(Δf/ε)
    Gaussian Mechanism: M(x) = f(x) + N(0, σ²) where σ = Δf·√(2ln(1.25/δ))/ε

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PrivacyParams",
    "NoiseMechanism",
    "LaplaceMechanism",
    "GaussianMechanism",
    "ExponentialMechanism",
    "SubsampledMechanism",
    "clip_gradient",
    "add_noise_to_gradient",
]


@dataclass
class PrivacyParams:
    """
    Privacy parameters for differential privacy.

    Attributes:
        epsilon: Privacy parameter (smaller = more private)
        delta: Failure probability (0 for pure DP)
        sensitivity: Query sensitivity (L1 or L2)
    """

    epsilon: float
    delta: float = 0.0
    sensitivity: float = 1.0

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta < 0 or self.delta >= 1:
            raise ValueError("delta must be in [0, 1)")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be positive")


class NoiseMechanism(ABC):
    """
    Abstract base class for noise mechanisms.

    Noise mechanisms add calibrated noise to achieve differential
    privacy guarantees.
    """

    @abstractmethod
    def add_noise(
        self,
        value: Union[float, NDArray[np.float64]],
        sensitivity: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Add noise to a value or array.

        Args:
            value: Value(s) to privatize
            sensitivity: Query sensitivity
            rng: Random number generator

        Returns:
            Noisy value(s)
        """
        pass

    @abstractmethod
    def get_epsilon(self, sensitivity: float) -> float:
        """
        Get effective epsilon for given sensitivity.

        Args:
            sensitivity: Query sensitivity

        Returns:
            Effective epsilon
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Mechanism name."""
        pass


class LaplaceMechanism(NoiseMechanism):
    """
    Laplace mechanism for ε-differential privacy.

    Adds Laplace noise calibrated to the sensitivity and epsilon:
        M(x) = f(x) + Lap(0, Δf/ε)

    Properties:
        - Achieves pure ε-differential privacy (δ = 0)
        - Optimal for L1 sensitivity
        - Unbounded noise (can be large)

    Attributes:
        epsilon: Privacy parameter

    Example:
        >>> mechanism = LaplaceMechanism(epsilon=1.0)
        >>> private_value = mechanism.add_noise(true_value, sensitivity=1.0)
    """

    def __init__(self, epsilon: float):
        """
        Initialize LaplaceMechanism.

        Args:
            epsilon: Privacy parameter
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def add_noise(
        self,
        value: Union[float, NDArray[np.float64]],
        sensitivity: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Add Laplace noise to achieve ε-DP.

        Args:
            value: Value(s) to privatize
            sensitivity: L1 sensitivity of the query
            rng: Random number generator

        Returns:
            Value with Laplace noise added
        """
        if rng is None:
            rng = np.random.default_rng()

        scale = sensitivity / self.epsilon

        if isinstance(value, np.ndarray):
            noise = rng.laplace(0, scale, value.shape)
            return value + noise
        else:
            noise = rng.laplace(0, scale)
            return value + noise

    def get_epsilon(self, sensitivity: float) -> float:
        """Get effective epsilon (same as configured for Laplace)."""
        return self.epsilon

    def get_scale(self, sensitivity: float) -> float:
        """Get Laplace scale parameter."""
        return sensitivity / self.epsilon

    @property
    def name(self) -> str:
        return "Laplace"

    def get_config(self) -> Dict[str, Any]:
        return {"mechanism": self.name, "epsilon": self.epsilon}


class GaussianMechanism(NoiseMechanism):
    """
    Gaussian mechanism for (ε,δ)-differential privacy.

    Adds Gaussian noise calibrated for approximate DP:
        M(x) = f(x) + N(0, σ²)
        where σ = Δf · √(2ln(1.25/δ)) / ε

    Properties:
        - Achieves (ε,δ)-differential privacy
        - Optimal for L2 sensitivity
        - Bounded noise (with high probability)

    Attributes:
        epsilon: Privacy parameter
        delta: Failure probability

    Example:
        >>> mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        >>> private_value = mechanism.add_noise(true_value, sensitivity=1.0)
    """

    def __init__(self, epsilon: float, delta: float = 1e-5):
        """
        Initialize GaussianMechanism.

        Args:
            epsilon: Privacy parameter
            delta: Failure probability
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta

    def add_noise(
        self,
        value: Union[float, NDArray[np.float64]],
        sensitivity: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Add Gaussian noise to achieve (ε,δ)-DP.

        Args:
            value: Value(s) to privatize
            sensitivity: L2 sensitivity of the query
            rng: Random number generator

        Returns:
            Value with Gaussian noise added
        """
        if rng is None:
            rng = np.random.default_rng()

        sigma = self.get_sigma(sensitivity)

        if isinstance(value, np.ndarray):
            noise = rng.normal(0, sigma, value.shape)
            return value + noise
        else:
            noise = rng.normal(0, sigma)
            return value + noise

    def get_sigma(self, sensitivity: float) -> float:
        """
        Compute Gaussian noise standard deviation.

        σ = Δf · √(2ln(1.25/δ)) / ε

        Args:
            sensitivity: L2 sensitivity

        Returns:
            Standard deviation for Gaussian noise
        """
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def get_epsilon(self, sensitivity: float) -> float:
        """Get effective epsilon."""
        return self.epsilon

    @property
    def name(self) -> str:
        return "Gaussian"

    def get_config(self) -> Dict[str, Any]:
        return {
            "mechanism": self.name,
            "epsilon": self.epsilon,
            "delta": self.delta,
        }


class ExponentialMechanism(NoiseMechanism):
    """
    Exponential mechanism for private selection.

    Selects from a set of options with probability proportional to:
        P(option) ∝ exp(ε · utility(option) / (2Δu))

    Useful for selecting coalitions or discrete choices privately.

    Attributes:
        epsilon: Privacy parameter
        utility_fn: Function computing utility of each option
        sensitivity: Sensitivity of utility function
    """

    def __init__(
        self,
        epsilon: float,
        utility_fn: Optional[Callable[[Any], float]] = None,
        sensitivity: float = 1.0,
    ):
        """
        Initialize ExponentialMechanism.

        Args:
            epsilon: Privacy parameter
            utility_fn: Utility function for options
            sensitivity: Sensitivity of utility function
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.epsilon = epsilon
        self.utility_fn = utility_fn or (lambda x: 0.0)
        self.sensitivity = sensitivity

    def select(
        self,
        options: List[Any],
        utilities: Optional[List[float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Any:
        """
        Select an option using exponential mechanism.

        Args:
            options: List of options to choose from
            utilities: Pre-computed utilities (optional)
            rng: Random number generator

        Returns:
            Selected option
        """
        if rng is None:
            rng = np.random.default_rng()

        if not options:
            raise ValueError("Options list cannot be empty")

        # Compute utilities if not provided
        if utilities is None:
            utilities = [self.utility_fn(opt) for opt in options]

        utilities = np.array(utilities)

        # Compute selection probabilities
        # P(option) ∝ exp(ε · u(option) / (2Δu))
        scores = self.epsilon * utilities / (2 * self.sensitivity)
        scores = scores - np.max(scores)  # Numerical stability
        probs = np.exp(scores)
        probs = probs / np.sum(probs)

        # Sample
        idx = rng.choice(len(options), p=probs)
        return options[idx]

    def add_noise(
        self,
        value: Union[float, NDArray[np.float64]],
        sensitivity: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Not applicable for exponential mechanism.

        Use select() instead.
        """
        raise NotImplementedError(
            "ExponentialMechanism uses select() instead of add_noise()"
        )

    def get_epsilon(self, sensitivity: float) -> float:
        """Get effective epsilon."""
        return self.epsilon

    @property
    def name(self) -> str:
        return "Exponential"

    def get_config(self) -> Dict[str, Any]:
        return {
            "mechanism": self.name,
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
        }


def clip_gradient(
    gradient: NDArray[np.float64],
    max_norm: float,
) -> Tuple[NDArray[np.float64], float]:
    """
    Clip gradient to maximum L2 norm.

    Used for gradient clipping in DP-SGD and FairSwarm-DP.

    Args:
        gradient: Gradient vector to clip
        max_norm: Maximum L2 norm

    Returns:
        Tuple of (clipped_gradient, scaling_factor)

    Example:
        >>> clipped, scale = clip_gradient(gradient, max_norm=1.0)
    """
    norm = np.linalg.norm(gradient)

    if norm <= max_norm:
        return gradient, 1.0
    else:
        scale = max_norm / norm
        return gradient * scale, scale


def add_noise_to_gradient(
    gradient: NDArray[np.float64],
    noise_multiplier: float,
    max_norm: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Add calibrated Gaussian noise to gradient (DP-SGD style).

    Implements:
        1. Clip gradient to max_norm
        2. Add Gaussian noise with σ = noise_multiplier * max_norm

    Args:
        gradient: Gradient vector
        noise_multiplier: Noise scale multiplier (σ/max_norm)
        max_norm: Maximum gradient norm
        rng: Random number generator

    Returns:
        Private gradient

    Example:
        >>> private_grad = add_noise_to_gradient(
        ...     gradient=grad,
        ...     noise_multiplier=1.0,
        ...     max_norm=1.0,
        ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clip gradient
    clipped, _ = clip_gradient(gradient, max_norm)

    # Add Gaussian noise
    sigma = noise_multiplier * max_norm
    noise = rng.normal(0, sigma, gradient.shape)

    return clipped + noise


class SubsampledMechanism:
    """
    Subsampled mechanism for privacy amplification.

    Applies privacy amplification by subsampling before applying
    a base mechanism.

    Privacy amplification: If we subsample with probability q,
    then (ε,δ)-DP becomes (O(qε), qδ)-DP approximately.

    Attributes:
        base_mechanism: Underlying noise mechanism
        sampling_rate: Probability of including each sample
    """

    def __init__(
        self,
        base_mechanism: NoiseMechanism,
        sampling_rate: float,
    ):
        """
        Initialize SubsampledMechanism.

        Args:
            base_mechanism: Base noise mechanism
            sampling_rate: Subsampling probability
        """
        if not 0 < sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")

        self.base_mechanism = base_mechanism
        self.sampling_rate = sampling_rate

    def add_noise(
        self,
        value: Union[float, NDArray[np.float64]],
        sensitivity: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Add noise with subsampling amplification.

        Args:
            value: Value(s) to privatize
            sensitivity: Query sensitivity
            rng: Random number generator

        Returns:
            Private value with amplified privacy
        """
        # Apply base mechanism
        # Note: The amplification is accounted for in get_epsilon
        return self.base_mechanism.add_noise(value, sensitivity, rng)

    def get_epsilon(self, sensitivity: float, delta: float = 1e-5) -> float:
        """
        Get amplified epsilon.

        Uses privacy amplification by subsampling.

        Args:
            sensitivity: Query sensitivity
            delta: Target delta

        Returns:
            Amplified epsilon (smaller than base)
        """
        base_eps = self.base_mechanism.get_epsilon(sensitivity)

        # Simplified amplification bound
        # More precise bounds available via RDP
        amplified = 2 * self.sampling_rate * base_eps
        return min(base_eps, amplified)

    @property
    def name(self) -> str:
        return f"Subsampled_{self.base_mechanism.name}"

    def get_config(self) -> Dict[str, Any]:
        return {
            "mechanism": self.name,
            "sampling_rate": self.sampling_rate,
            "base": self.base_mechanism.get_config(),
        }
