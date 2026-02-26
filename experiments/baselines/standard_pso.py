"""
Standard PSO baseline (without fairness gradient).

Ablation baseline: standard PSO for coalition selection without the
fairness-aware velocity term (c3 = 0). This isolates the contribution
of the fairness gradient (the key novel contribution of FairSwarm).

Reference: Section 4.2 (Baselines), Algorithm 1 ablation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.core.config import FairSwarmConfig

if TYPE_CHECKING:
    from fairswarm.core.client import Client


class StandardPSO(FairSwarm):
    """
    Ablation baseline: PSO without fairness gradient.

    Identical to FairSwarm except c3 (fairness_coefficient) = 0
    and adaptive_fairness is disabled. This removes the novel
    fairness-aware velocity term to measure its contribution.

    Expected behavior:
        - Converges to high-accuracy coalitions (Theorem 1 still holds)
        - No demographic fairness guarantees (Theorem 2 does NOT apply)
        - Useful for quantifying the fairness-accuracy tradeoff
    """

    def __init__(
        self,
        clients: list[Client],
        coalition_size: int,
        config: FairSwarmConfig | None = None,
        seed: int | None = None,
    ):
        # Override config to disable fairness
        if config is None:
            config = FairSwarmConfig()
        config = FairSwarmConfig(
            swarm_size=config.swarm_size,
            max_iterations=config.max_iterations,
            inertia=config.inertia,
            cognitive=config.cognitive,
            social=config.social,
            fairness_coefficient=0.0,  # Key: no fairness gradient
            velocity_max=config.velocity_max,
            fairness_weight=0.0,
            adaptive_fairness=False,
            epsilon_fair=config.epsilon_fair,
        )

        super().__init__(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            target_distribution=None,  # No target needed
            seed=seed,
        )
