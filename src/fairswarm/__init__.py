"""
FairSwarm: Provably Fair Particle Swarm Optimization for Federated Learning.

This library implements the FairSwarm algorithm for fair coalition selection
in federated learning, with provable theoretical guarantees.

Theoretical Guarantees:
    - Theorem 1: Convergence to stationary point with probability 1
    - Theorem 2: ε-fairness bound on demographic divergence
    - Theorem 3: (1-1/e-η) approximation for submodular objectives
    - Theorem 4: Privacy-fairness tradeoff bounds

Example:
    >>> from fairswarm import FairSwarm, FairSwarmConfig, Client
    >>> from fairswarm.demographics import DemographicDistribution, CensusTarget
    >>>
    >>> clients = [Client(id=f"hospital_{i}", ...) for i in range(20)]
    >>> optimizer = FairSwarm(
    ...     clients=clients,
    ...     coalition_size=10,
    ...     target_demographics=CensusTarget.US_2020,
    ... )
    >>> result = optimizer.optimize(fitness_fn)
    >>> print(f"Selected: {result.coalition}")

References:
    - Norwood, T. "FairSwarm: Provably Fair PSO for FL" (PhD Thesis)
    - Ghosh, U. "Zero Trust Federated Learning" (Springer, 2025)
    - Bentley, E.S. "FSL-SAGE" (ICML, 2025)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
Institution: Meharry Medical College
"""

from fairswarm.__version__ import __version__
from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.core.client import Client
from fairswarm.core.config import FairSwarmConfig
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.fairness import DemographicFitness
from fairswarm.types import (
    ClientId,
    Coalition,
    DemographicVector,
    FitnessValue,
)

# Lazy imports for optional modules
def __getattr__(name: str):
    """Lazy loading for optional integration and digital twin modules."""
    if name == "FairSwarmStrategy":
        from fairswarm.integrations.flower import FairSwarmStrategy
        return FairSwarmStrategy
    elif name == "BentleyDigitalTwin":
        from fairswarm.digital_twin.twin import BentleyDigitalTwin
        return BentleyDigitalTwin
    elif name == "VirtualEnvironment":
        from fairswarm.digital_twin.simulator import VirtualEnvironment
        return VirtualEnvironment
    elif name == "DriftDetector":
        from fairswarm.digital_twin.drift import DriftDetector
        return DriftDetector
    elif name == "SimToRealAdapter":
        from fairswarm.digital_twin.adapter import SimToRealAdapter
        return SimToRealAdapter
    raise AttributeError(f"module 'fairswarm' has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Main optimizer
    "FairSwarm",
    "OptimizationResult",
    # Core classes
    "Client",
    "FairSwarmConfig",
    # Fitness
    "FitnessFunction",
    "FitnessResult",
    "DemographicFitness",
    # Type aliases
    "ClientId",
    "Coalition",
    "DemographicVector",
    "FitnessValue",
    # Integration (Phase 11)
    "FairSwarmStrategy",
    # Digital Twin (Phase 12)
    "BentleyDigitalTwin",
    "VirtualEnvironment",
    "DriftDetector",
    "SimToRealAdapter",
]
