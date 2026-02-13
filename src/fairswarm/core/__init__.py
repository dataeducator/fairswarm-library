"""
Core primitives for FairSwarm PSO optimization.

This module provides the fundamental building blocks:
- Client: Represents a federated learning participant
- FairSwarmConfig: Configuration for the optimizer
- Particle: PSO particle with position and velocity
- Swarm: Collection of particles
- Position utilities: sigmoid, decode_coalition

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig, get_preset_config
from fairswarm.core.numerical import (
    check_distribution,
    check_gradient,
    clip_gradient,
    repair_distribution,
    safe_divide,
    safe_log,
    safe_normalize,
)
from fairswarm.core.particle import Particle
from fairswarm.core.position import (
    coalition_overlap,
    decode_coalition,
    encode_coalition,
    inverse_sigmoid,
    position_similarity,
    sigmoid,
    soft_decode_coalition,
)
from fairswarm.core.swarm import Swarm, SwarmHistory

__all__ = [
    # Client
    "Client",
    "create_synthetic_clients",
    # Config
    "FairSwarmConfig",
    "get_preset_config",
    # Particle
    "Particle",
    # Swarm
    "Swarm",
    "SwarmHistory",
    # Position utilities
    "sigmoid",
    "inverse_sigmoid",
    "decode_coalition",
    "encode_coalition",
    "soft_decode_coalition",
    "position_similarity",
    "coalition_overlap",
    # Numerical utilities
    "safe_normalize",
    "safe_log",
    "safe_divide",
    "check_distribution",
    "repair_distribution",
    "check_gradient",
    "clip_gradient",
]
