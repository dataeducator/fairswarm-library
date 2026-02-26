"""
Digital Twin framework for FairSwarm federated learning.

This module implements the Digital Twin architecture for
bidirectional sim-to-real transfer in federated learning:

Architecture Components:
    - DigitalTwin: Main digital twin with bidirectional sync
    - VirtualEnvironment: Simulated federated learning environment
    - SimToRealAdapter: Domain adaptation between simulation and reality
    - DriftDetector: Distribution shift monitoring

Author: Tenicka Norwood
"""

from fairswarm.digital_twin.adapter import (
    AdaptationResult,
    DomainAdaptationConfig,
    SimToRealAdapter,
)
from fairswarm.digital_twin.drift import (
    DriftDetector,
    DriftMetrics,
    DriftResult,
    DriftType,
)
from fairswarm.digital_twin.simulator import (
    SimulationConfig,
    SimulationResult,
    VirtualClient,
    VirtualEnvironment,
)
from fairswarm.digital_twin.twin import (
    DigitalTwin,
    SyncResult,
    TwinState,
)

__all__ = [
    # Main digital twin
    "DigitalTwin",
    "TwinState",
    "SyncResult",
    # Simulator
    "VirtualEnvironment",
    "VirtualClient",
    "SimulationConfig",
    "SimulationResult",
    # Adapter
    "SimToRealAdapter",
    "DomainAdaptationConfig",
    "AdaptationResult",
    # Drift detection
    "DriftDetector",
    "DriftResult",
    "DriftType",
    "DriftMetrics",
]
