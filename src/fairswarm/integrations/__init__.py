"""
FairSwarm integrations with federated learning frameworks.

This module provides integration with popular FL frameworks:
- Flower (FairSwarmStrategy)
- Future: TensorFlow Federated, PySyft

Author: Tenicka Norwood
"""

from fairswarm.integrations.flower import (
    FairSwarmClient,
    FairSwarmEvaluateConfig,
    FairSwarmFitConfig,
    FairSwarmStrategy,
)

__all__ = [
    "FairSwarmStrategy",
    "FairSwarmClient",
    "FairSwarmFitConfig",
    "FairSwarmEvaluateConfig",
]
