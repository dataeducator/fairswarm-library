"""
FairSwarm integrations with federated learning frameworks.

This module provides integration with popular FL frameworks:
- Flower (FairSwarmStrategy)
- Future: TensorFlow Federated, PySyft

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.integrations.flower import (
    FairSwarmStrategy,
    FairSwarmClient,
    FairSwarmFitConfig,
    FairSwarmEvaluateConfig,
)

__all__ = [
    "FairSwarmStrategy",
    "FairSwarmClient",
    "FairSwarmFitConfig",
    "FairSwarmEvaluateConfig",
]
