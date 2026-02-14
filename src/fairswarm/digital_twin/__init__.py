"""
Digital Twin framework for FairSwarm federated learning.

This module implements the Bentley Digital Twin architecture for
bidirectional sim-to-real transfer in federated learning:

Architecture Components:
    - BentleyDigitalTwin: Main digital twin with bidirectional sync
    - VirtualEnvironment: Simulated federated learning environment
    - SimToRealAdapter: Domain adaptation between simulation and reality
    - DriftDetector: Distribution shift monitoring

Research Attribution:
    - Digital Twin Architecture: Dr. Elizabeth Bentley (Computer Networks 2023)
    - FSL-SAGE Split Learning: Dr. Elizabeth Bentley (ICML 2025)
    - Privacy Mechanisms: Dr. Uttam Ghosh (IEEE TNSE 2023)
    - FairSwarm Algorithm: Novel contribution (this thesis)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
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
    BentleyDigitalTwin,
    SyncResult,
    TwinState,
)

__all__ = [
    # Main digital twin
    "BentleyDigitalTwin",
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
