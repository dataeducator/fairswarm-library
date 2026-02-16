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
from typing import Any, Callable

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

    def __post_init__(self) -> None:
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
        value: float | NDArray[np.float64],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> float | NDArray[np.float64]:
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

    def get_config(self) -> dict[str, Any]:
        """Get mechanism configuration."""
        return {"mechanism": self.name}


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
        value: float | NDArray[np.float64],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> float | NDArray[np.float64]:
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
            arr_noise = rng.laplace(0, scale, value.shape)
            return value + arr_noise
        else:
            scalar_noise: float = float(rng.laplace(0, scale))
            return float(value + scalar_noise)

    def get_epsilon(self, sensitivity: float) -> float:
        """Get effective epsilon (same as configured for Laplace)."""
        return self.epsilon

    def get_scale(self, sensitivity: float) -> float:
        """Get Laplace scale parameter."""
        return sensitivity / self.epsilon

    @property
    def name(self) -> str:
        return "Laplace"

    def get_config(self) -> dict[str, Any]:
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
        value: float | NDArray[np.float64],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> float | NDArray[np.float64]:
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
            arr_noise = rng.normal(0, sigma, value.shape)
            return value + arr_noise
        else:
            scalar_noise: float = float(rng.normal(0, sigma))
            return float(value + scalar_noise)

    def get_sigma(self, sensitivity: float) -> float:
        """
        Compute Gaussian noise standard deviation.

        σ = Δf · √(2ln(1.25/δ)) / ε

        Args:
            sensitivity: L2 sensitivity

        Returns:
            Standard deviation for Gaussian noise
        """
        return float(
            sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        )

    def get_epsilon(self, sensitivity: float) -> float:
        """Get effective epsilon."""
        return self.epsilon

    @property
    def name(self) -> str:
        return "Gaussian"

    def get_config(self) -> dict[str, Any]:
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
        utility_fn: Callable[[Any], float] | None = None,
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
        options: list[Any],
        utilities: list[float] | None = None,
        rng: np.random.Generator | None = None,
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

        utilities_arr = np.array(utilities)

        # Compute selection probabilities
        # P(option) ∝ exp(ε · u(option) / (2Δu))
        scores = self.epsilon * utilities_arr / (2 * self.sensitivity)
        scores = scores - np.max(scores)  # Numerical stability
        probs = np.exp(scores)
        prob_sum = float(np.sum(probs))
        if prob_sum <= 0:
            # Uniform fallback if all probabilities collapse
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / prob_sum

        # Sample
        idx = rng.choice(len(options), p=probs)
        return options[idx]

    def add_noise(
        self,
        value: float | NDArray[np.float64],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> float | NDArray[np.float64]:
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

    def get_config(self) -> dict[str, Any]:
        return {
            "mechanism": self.name,
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
        }


def clip_gradient(
    gradient: NDArray[np.float64],
    max_norm: float,
) -> tuple[NDArray[np.float64], float]:
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
    norm = float(np.linalg.norm(gradient))

    if norm <= max_norm:
        return gradient, 1.0
    else:
        scale = max_norm / norm
        return gradient * scale, float(scale)


def add_noise_to_gradient(
    gradient: NDArray[np.float64],
    noise_multiplier: float,
    max_norm: float,
    rng: np.random.Generator | None = None,
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
        self._subsampling_verified = False

    def add_noise(
        self,
        value: float | NDArray[np.float64],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> float | NDArray[np.float64]:
        """
        Add noise via the base mechanism.

        NOTE: This method does NOT perform Poisson subsampling on the data.
        Actual subsampling (randomly selecting a subset of records) must be
        performed at the FL round level BEFORE calling this method. If you
        call add_noise() on the full dataset without prior subsampling, the
        privacy guarantee is that of the base mechanism alone -- no
        amplification applies.

        Args:
            value: Value(s) to privatize
            sensitivity: Query sensitivity
            rng: Random number generator

        Returns:
            Noisy value (without subsampling amplification)
        """
        # Delegate to the base mechanism.  Privacy amplification by
        # subsampling requires the *caller* to subsample the data before
        # invoking this method.  We intentionally do NOT claim amplified
        # privacy here because we cannot verify that subsampling occurred.
        return self.base_mechanism.add_noise(value, sensitivity, rng)

    def get_epsilon(self, sensitivity: float, delta: float = 1e-5) -> float:
        """
        Get effective epsilon (WITHOUT privacy amplification).

        Privacy amplification by subsampling (Balle et al., 2018) only
        holds when the data is actually Poisson-subsampled before the
        noise mechanism is applied.  Since this class cannot enforce or
        verify that subsampling occurred at the data level, we
        conservatively return the base mechanism's epsilon to avoid
        claiming stronger privacy than is actually provided.

        If you need the amplified bound, perform Poisson subsampling on
        the data yourself and call ``get_amplified_epsilon()`` which
        documents the assumption explicitly.

        Args:
            sensitivity: Query sensitivity
            delta: Target delta

        Returns:
            Base mechanism epsilon (no amplification)
        """
        return self.base_mechanism.get_epsilon(sensitivity)

    def confirm_subsampling(self) -> None:
        """
        Confirm that Poisson subsampling was actually performed.

        Callers MUST invoke this method after performing Poisson
        subsampling on the data and before calling
        ``get_amplified_epsilon()``.  This prevents falsely claiming
        privacy amplification when subsampling was not done.
        """
        self._subsampling_verified = True

    def get_amplified_epsilon(self, sensitivity: float, delta: float = 1e-5) -> float:
        """
        Get amplified epsilon assuming Poisson subsampling was performed.

        IMPORTANT: This bound is ONLY valid if the caller actually
        performed Poisson subsampling (each record included independently
        with probability ``self.sampling_rate``) before applying the base
        noise mechanism.  Calling this without true subsampling yields a
        falsely optimistic privacy guarantee.

        Uses the tight amplification bound from Balle et al. (2018):
            amplified eps <= log(1 + q * (exp(base_eps) - 1))

        Args:
            sensitivity: Query sensitivity
            delta: Target delta

        Returns:
            Amplified epsilon (smaller than base), valid only under
            actual Poisson subsampling.

        Raises:
            RuntimeError: If confirm_subsampling() was not called first.
        """
        if not self._subsampling_verified:
            raise RuntimeError(
                "get_amplified_epsilon() requires confirm_subsampling() "
                "to be called first, certifying that Poisson subsampling "
                "was performed on the data before noise was applied. "
                "Without actual subsampling, use get_epsilon() instead."
            )
        base_eps = self.base_mechanism.get_epsilon(sensitivity)
        q = self.sampling_rate

        amplified = float(np.log(1 + q * (np.exp(base_eps) - 1)))
        return min(base_eps, amplified)

    @property
    def name(self) -> str:
        return f"Subsampled_{self.base_mechanism.name}"

    def get_config(self) -> dict[str, Any]:
        return {
            "mechanism": self.name,
            "sampling_rate": self.sampling_rate,
            "base": self.base_mechanism.get_config(),
        }
