"""
Privacy accountants for tracking privacy budget.

This module provides privacy accounting for tracking cumulative
privacy loss under composition.

Key Accountants:
    - SimpleAccountant: Basic composition
    - MomentsAccountant: Moments-based accounting
    - RDPAccountant: Rényi Differential Privacy accounting

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PrivacySpent",
    "PrivacyAccountant",
    "SimpleAccountant",
    "MomentsAccountant",
    "RDPAccountant",
    "AdvancedCompositionAccountant",
]


@dataclass
class PrivacySpent:
    """
    Record of privacy expenditure.

    Attributes:
        epsilon: Epsilon spent
        delta: Delta for this expenditure
        mechanism: Name of mechanism used
        iteration: Iteration number (if applicable)
    """

    epsilon: float
    delta: float = 0.0
    mechanism: str = "unknown"
    iteration: int = 0


class PrivacyAccountant(ABC):
    """
    Abstract base class for privacy accountants.

    Privacy accountants track cumulative privacy loss under
    composition of multiple queries.
    """

    @abstractmethod
    def step(self, epsilon: float, delta: float = 0.0) -> None:
        """
        Record a privacy expenditure.

        Args:
            epsilon: Epsilon spent in this step
            delta: Delta for this step
        """
        pass

    @abstractmethod
    def get_epsilon(self, delta: float) -> float:
        """
        Get total epsilon spent for given delta.

        Args:
            delta: Target delta

        Returns:
            Total epsilon
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the accountant."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Accountant name."""
        pass

    def get_privacy_spent(self, delta: float) -> tuple[float, float]:
        """
        Get (epsilon, delta) tuple.

        Args:
            delta: Target delta

        Returns:
            Tuple of (epsilon, delta)
        """
        return (self.get_epsilon(delta), delta)


class SimpleAccountant(PrivacyAccountant):
    """
    Simple accountant using basic composition.

    Basic composition theorem:
        ε_total = Σ ε_i
        δ_total = Σ δ_i

    This is the loosest bound but simplest to compute.

    Attributes:
        history: List of privacy expenditures
    """

    def __init__(self):
        self.history: list[PrivacySpent] = []
        self._total_epsilon = 0.0
        self._total_delta = 0.0
        self._step_count = 0

    def step(self, epsilon: float, delta: float = 0.0) -> None:
        """
        Record a privacy expenditure.

        Args:
            epsilon: Epsilon spent
            delta: Delta for this step
        """
        self.history.append(
            PrivacySpent(
                epsilon=epsilon,
                delta=delta,
                mechanism="basic",
                iteration=self._step_count,
            )
        )
        self._total_epsilon += epsilon
        self._total_delta += delta
        self._step_count += 1

    def get_epsilon(self, delta: float) -> float:
        """
        Get total epsilon under basic composition.

        Args:
            delta: Target delta (not used in basic composition)

        Returns:
            Sum of all epsilons
        """
        return self._total_epsilon

    def get_delta(self) -> float:
        """Get total delta."""
        return self._total_delta

    def reset(self) -> None:
        """Reset the accountant."""
        self.history.clear()
        self._total_epsilon = 0.0
        self._total_delta = 0.0
        self._step_count = 0

    @property
    def name(self) -> str:
        return "Simple"

    @property
    def step_count(self) -> int:
        """Number of steps recorded."""
        return self._step_count

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_epsilon": self._total_epsilon,
            "total_delta": self._total_delta,
            "step_count": self._step_count,
        }


class MomentsAccountant(PrivacyAccountant):
    """
    Moments accountant for tighter privacy bounds.

    Uses the moments bound (log of MGF) for composition:
        α(λ) = log E[exp(λ · privacy_loss)]

    Provides tighter bounds than basic composition for
    Gaussian mechanisms.

    Reference:
        Abadi et al., "Deep Learning with Differential Privacy" (2016)

    Attributes:
        noise_multiplier: Noise multiplier σ/Δ
        sampling_rate: Batch sampling rate
        orders: Orders λ for RDP computation
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        sampling_rate: float = 1.0,
        orders: list[float] | None = None,
    ):
        """
        Initialize MomentsAccountant.

        Args:
            noise_multiplier: Ratio of noise std to sensitivity
            sampling_rate: Subsampling rate
            orders: Orders for RDP (default: range of values)
        """
        self.noise_multiplier = noise_multiplier
        self.sampling_rate = sampling_rate
        self.orders = orders or [1 + x / 10.0 for x in range(1, 100)] + list(
            range(12, 64)
        )
        self._steps = 0
        self._step_epsilons: list[float] = []

    def step(self, epsilon: float = 0.0, delta: float = 0.0) -> None:
        """
        Record a step of optimization.

        For MomentsAccountant, epsilon/delta are not directly used.
        Instead, we track steps and compute from noise_multiplier.

        Args:
            epsilon: Not used (kept for interface compatibility)
            delta: Not used
        """
        self._steps += 1
        self._step_epsilons.append(epsilon)

    def _compute_rdp(self, order: float) -> float:
        """
        Compute RDP for given order.

        Args:
            order: Rényi order λ

        Returns:
            RDP value α_λ
        """
        if self.sampling_rate == 1.0:
            # No subsampling: standard Gaussian RDP
            return order / (2 * self.noise_multiplier ** 2)
        else:
            # Subsampled Gaussian mechanism
            # Using simplified bound
            q = self.sampling_rate
            sigma = self.noise_multiplier

            if order <= 1:
                return 0.0

            # Simplified RDP bound for subsampled Gaussian
            rdp = (q ** 2 * order) / (2 * sigma ** 2)
            return rdp

    def _rdp_to_dp(self, rdp: float, order: float, delta: float) -> float:
        """
        Convert RDP to (ε, δ)-DP.

        Using: ε = α - log(δ)/(λ-1)

        Args:
            rdp: RDP value
            order: Rényi order
            delta: Target delta

        Returns:
            Epsilon for (ε, δ)-DP
        """
        if order <= 1:
            return float("inf")
        return rdp + np.log(1 / delta) / (order - 1)

    def get_epsilon(self, delta: float) -> float:
        """
        Get total epsilon for given delta.

        Computes optimal epsilon over all orders.

        Args:
            delta: Target delta

        Returns:
            Minimum epsilon across orders
        """
        if self._steps == 0:
            return 0.0

        min_epsilon = float("inf")

        for order in self.orders:
            # Compute RDP for this order
            rdp_per_step = self._compute_rdp(order)
            total_rdp = self._steps * rdp_per_step

            # Convert to (ε, δ)-DP
            epsilon = self._rdp_to_dp(total_rdp, order, delta)
            min_epsilon = min(min_epsilon, epsilon)

        return min_epsilon

    def reset(self) -> None:
        """Reset the accountant."""
        self._steps = 0
        self._step_epsilons.clear()

    @property
    def name(self) -> str:
        return "Moments"

    @property
    def step_count(self) -> int:
        return self._steps

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate": self.sampling_rate,
            "steps": self._steps,
        }


class RDPAccountant(PrivacyAccountant):
    """
    Rényi Differential Privacy accountant.

    Tracks privacy using Rényi divergence for tighter composition.

    RDP Composition:
        α(λ) = Σ α_i(λ)  (RDPs add directly)

    Conversion to (ε, δ)-DP:
        ε = min_λ { α(λ) + log(1/δ)/(λ-1) }

    Attributes:
        orders: Orders for RDP computation
        rdp_values: RDP values for each order
    """

    def __init__(self, orders: list[float] | None = None):
        """
        Initialize RDPAccountant.

        Args:
            orders: Orders λ for RDP (default: auto-selected range)
        """
        self.orders = orders or (
            [1 + x / 10.0 for x in range(1, 100)]
            + list(range(12, 64))
        )
        self._rdp_values = dict.fromkeys(self.orders, 0.0)
        self._steps = 0

    def step(
        self,
        epsilon: float = 0.0,
        delta: float = 0.0,
        noise_multiplier: float | None = None,
        sampling_rate: float = 1.0,
    ) -> None:
        """
        Record a privacy expenditure.

        Args:
            epsilon: Direct epsilon (if not using noise_multiplier)
            delta: Delta for this step
            noise_multiplier: σ/Δ for Gaussian mechanism
            sampling_rate: Subsampling rate
        """
        self._steps += 1

        for order in self.orders:
            if noise_multiplier is not None:
                # Compute RDP for Gaussian mechanism
                if sampling_rate < 1.0:
                    # Subsampled Gaussian
                    rdp = self._compute_subsampled_gaussian_rdp(
                        order, noise_multiplier, sampling_rate
                    )
                else:
                    # Standard Gaussian
                    rdp = order / (2 * noise_multiplier ** 2)
            else:
                # Convert (ε, δ) to RDP (approximate)
                if order > 1:
                    rdp = epsilon + np.log(1 / max(delta, 1e-10)) / (order - 1)
                else:
                    rdp = epsilon

            self._rdp_values[order] += rdp

    def _compute_subsampled_gaussian_rdp(
        self,
        order: float,
        noise_multiplier: float,
        sampling_rate: float,
    ) -> float:
        """
        Compute RDP for subsampled Gaussian mechanism.

        Args:
            order: Rényi order
            noise_multiplier: σ/Δ
            sampling_rate: Subsampling rate q

        Returns:
            RDP value
        """
        if order <= 1:
            return 0.0

        q = sampling_rate
        sigma = noise_multiplier

        # Simplified bound (more precise bounds available)
        return (q ** 2 * order) / (2 * sigma ** 2)

    def get_epsilon(self, delta: float) -> float:
        """
        Get total epsilon for given delta.

        Finds optimal order for tightest bound.

        Args:
            delta: Target delta

        Returns:
            Minimum epsilon across orders
        """
        if self._steps == 0:
            return 0.0

        min_epsilon = float("inf")

        for order, rdp in self._rdp_values.items():
            if order <= 1:
                continue
            epsilon = rdp + np.log(1 / delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)

        return min_epsilon if min_epsilon < float("inf") else 0.0

    def reset(self) -> None:
        """Reset the accountant."""
        self._rdp_values = dict.fromkeys(self.orders, 0.0)
        self._steps = 0

    @property
    def name(self) -> str:
        return "RDP"

    @property
    def step_count(self) -> int:
        return self._steps

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_orders": len(self.orders),
            "steps": self._steps,
        }


class AdvancedCompositionAccountant(PrivacyAccountant):
    """
    Accountant using advanced composition theorem.

    Advanced composition (Dwork et al., 2010):
        k compositions of (ε, δ)-DP mechanisms give
        (ε', kδ + δ')-DP where:
        ε' = √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1)

    Attributes:
        epsilon_per_step: Epsilon per step
        delta_per_step: Delta per step
    """

    def __init__(
        self,
        epsilon_per_step: float,
        delta_per_step: float = 0.0,
    ):
        """
        Initialize AdvancedCompositionAccountant.

        Args:
            epsilon_per_step: Epsilon per query
            delta_per_step: Delta per query
        """
        if epsilon_per_step <= 0:
            raise ValueError("epsilon_per_step must be positive")

        self.epsilon_per_step = epsilon_per_step
        self.delta_per_step = delta_per_step
        self._steps = 0

    def step(self, epsilon: float = 0.0, delta: float = 0.0) -> None:
        """
        Record a step.

        Note: Uses configured epsilon_per_step, not the provided epsilon.
        """
        self._steps += 1

    def get_epsilon(self, delta: float) -> float:
        """
        Get total epsilon using advanced composition.

        Args:
            delta: Target delta (additional failure probability)

        Returns:
            Composed epsilon
        """
        if self._steps == 0:
            return 0.0

        k = self._steps
        eps = self.epsilon_per_step

        if delta <= 0:
            # Fall back to basic composition
            return k * eps

        # Advanced composition
        # ε' = √(2k·ln(1/δ))·ε + k·ε·(e^ε - 1)
        composed = (
            np.sqrt(2 * k * np.log(1 / delta)) * eps
            + k * eps * (np.exp(eps) - 1)
        )

        return composed

    def get_delta(self, target_epsilon: float, base_delta: float = 1e-5) -> float:
        """
        Get delta for target epsilon.

        Args:
            target_epsilon: Target total epsilon
            base_delta: Additional failure probability

        Returns:
            Total delta
        """
        # Total delta = k·δ_step + δ_additional
        return self._steps * self.delta_per_step + base_delta

    def reset(self) -> None:
        """Reset the accountant."""
        self._steps = 0

    @property
    def name(self) -> str:
        return "AdvancedComposition"

    @property
    def step_count(self) -> int:
        return self._steps

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epsilon_per_step": self.epsilon_per_step,
            "delta_per_step": self.delta_per_step,
            "steps": self._steps,
        }
