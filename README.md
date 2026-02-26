# FairSwarm

> Provably fair particle swarm optimization for federated learning coalition selection

[![License](https://img.shields.io/badge/License-PolyForm_Noncommercial-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

**FairSwarm** is a novel particle swarm optimization algorithm designed for fair client selection in federated learning. It provides provable guarantees on both convergence and demographic fairness.

### Key Innovation

FairSwarm introduces a **fairness-aware velocity update** that steers optimization toward demographically balanced coalitions:

```
v = ω·v + c₁·r₁·(pBest - x) + c₂·r₂·(gBest - x) + c₃·∇_fair
                                                    ^^^^^^^^
                                                    Novel fairness gradient
```

### Theoretical Guarantees

| Theorem | Guarantee |
|---------|-----------|
| **Theorem 1** | Convergence to stationary point with probability 1 |
| **Theorem 2** | ε-fairness: DemDiv(S*) ≤ ε with high probability |
| **Theorem 3** | (1-1/e-η) approximation for submodular objectives |
| **Theorem 4** | Privacy-fairness tradeoff lower bound |

## Installation

```bash
pip install fairswarm
```

For development:
```bash
pip install fairswarm[dev]
```

## Quick Start

```python
from fairswarm import FairSwarm, FairSwarmConfig, Client
from fairswarm.demographics import DemographicDistribution, CensusTarget
import numpy as np

# Create clients (hospitals) with demographic information
clients = [
    Client(
        id=f"hospital_{i}",
        demographics=np.random.dirichlet([2, 2, 2, 2]),
        dataset_size=1000 + i * 100,
    )
    for i in range(20)
]

# Configure the optimizer
config = FairSwarmConfig(
    swarm_size=30,
    max_iterations=100,
    coalition_size=10,
    fairness_weight=0.3,  # λ in fitness function
    seed=42,
)

# Create target demographics (e.g., US Census 2020)
target = DemographicDistribution.from_dict({
    "white": 0.576,
    "black": 0.124,
    "hispanic": 0.187,
    "asian": 0.061,
    "other": 0.052,
})

# Run optimization
optimizer = FairSwarm(
    clients=clients,
    coalition_size=10,
    target_demographics=target,
    config=config,
)
# Use built-in demographic fitness (or supply your own callable)
from fairswarm.fitness import DemographicFitness
fitness_fn = DemographicFitness(target=target)

result = optimizer.optimize(fitness_fn)

# Check results
print(f"Selected coalition: {result.coalition}")
print(f"Fitness: {result.fitness:.4f}")
print(f"ε-fair: {result.is_epsilon_fair(0.05)}")
```

## Documentation

Full API documentation is available in the source code docstrings. For the formal algorithm specification, theoretical proofs, and experimental methodology, please refer to:

> T. Norwood, "FairSwarm: Provably Fair Particle Swarm Optimization for Federated Learning Coalition Selection," *IEEE Trans. Neural Netw. Learn. Syst.*, 2026.

## Citation

```bibtex
@phdthesis{norwood2026fairswarm,
  title={FairSwarm: A Provably Fair Particle Swarm Optimization Algorithm
         for Federated Learning Coalition Selection with Applications in Healthcare},
  author={Norwood, Tenicka},
  year={2026},
  school={Meharry Medical College}
}
```

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). You are free to use, modify, and distribute it for any noncommercial purpose, including academic research, education, and personal projects.

**Commercial licensing is available.** For commercial use inquiries, please contact [Tenicka Norwood](mailto:tenicka.norwood@gmail.com).

See [LICENSE](LICENSE) for full terms.
