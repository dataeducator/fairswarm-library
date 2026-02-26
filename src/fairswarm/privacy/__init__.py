"""
Privacy mechanisms for FairSwarm.

This module provides differential privacy primitives for
privacy-preserving federated learning.

Key Components:
    - NoiseMechanism: Abstract base for noise mechanisms
    - LaplaceMechanism: Laplace noise for ε-DP
    - GaussianMechanism: Gaussian noise for (ε,δ)-DP
    - PrivacyAccountant: Budget tracking with composition

Theorem 4 Connection:
    UtilityLoss ≥ Ω(√(k·log(1/δ))/(ε_DP·ε_F))

Author: Tenicka Norwood
"""

from fairswarm.privacy.accountant import (
    MomentsAccountant,
    PrivacyAccountant,
    RDPAccountant,
    SimpleAccountant,
)
from fairswarm.privacy.mechanisms import (
    ExponentialMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    NoiseMechanism,
    add_noise_to_gradient,
    clip_gradient,
)

__all__ = [
    # Mechanisms
    "NoiseMechanism",
    "LaplaceMechanism",
    "GaussianMechanism",
    "ExponentialMechanism",
    "clip_gradient",
    "add_noise_to_gradient",
    # Accountants
    "PrivacyAccountant",
    "RDPAccountant",
    "MomentsAccountant",
    "SimpleAccountant",
]
