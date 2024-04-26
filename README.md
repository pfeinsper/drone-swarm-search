[![Tests Status 🧪](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/env.yml/badge.svg)](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/env.yml)
[![Docs Deployment 📝](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/deploy.yml/badge.svg?branch=vitepress_docs)](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/deploy.yml)
[![PyPI Release 🚀](https://badge.fury.io/py/DSSE.svg)](https://badge.fury.io/py/DSSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat)](https://github.com/pfeinsper/drone-swarm-search/blob/main/LICENSE)
[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.22.3-blue)]()
![GitHub stars](https://img.shields.io/github/stars/pfeinsper/drone-swarm-search)

# <img src="https://github.com/pfeinsper/drone-swarm-search/blob/main/docs/public/pics/drone.svg" alt="DSSE Icon" width="45" height="25"> Drone Swarm Search Environment (DSSE)

Welcome to the official GitHub repository for the Drone Swarm Search Environment (DSSE). This project is dedicated to providing a comprehensive simulation environment designed to develop, test, and refine drone swarm search strategies. Here, researchers and developers can access a versatile platform that supports a wide range of drone swarm simulations, enabling the exploration of complex behaviors and interactions within dynamic, real-world scenarios.


## 📚 Documentation Links

- **[Documentation Site](https://pfeinsper.github.io/drone-swarm-search/)**: Access comprehensive documentation including tutorials, and usage examples for the Drone Swarm Search Environment (DSSE). Ideal for users seeking detailed information about the project's capabilities and how to integrate them into their own applications.

- **[Algorithm Details](https://github.com/pfeinsper/drone-swarm-search-algorithms)**: Explore in-depth discussions and source code for the algorithms powering the DSSE. This section is perfect for developers interested in the technical underpinnings and enhancements of the search algorithms.

- **[PyPI Repository](https://pypi.org/project/DSSE/)**: Visit the PyPI page for DSSE to download the latest release, view release histories, and read additional installation instructions.

## 🎥 Visual Demonstrations
<p align="center">
    <img src="https://raw.github.com/PFE-Embraer/drone-swarm-search/env-cleanup/docs/gifs/render_with_grid_gradient.gif" width="400" height="400" align="center">
    <br>
    <em>Above: A simulation showing how drones adjust their search pattern over a grid.</em>
</p>

## ⚡Quick Start

### ⚙️ Installation
Install DSSE quickly with pip:
```bash
pip install DSSE
````

## 🎯 Outcome

| If target is found       | If target is not found   |
:-------------------------:|:-------------------------:
| ![](https://raw.githubusercontent.com/PFE-Embraer/drone-swarm-search/main/docs/public/pics/victory_render.png)     | ![](https://raw.github.com/PFE-Embraer/drone-swarm-search/main/docs/public/pics/fail_render.png) |


## 🛠️ Basic Env Usage
```python
from DSSE import DroneSwarmSearch

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(1, 1),
    timestep_limit=300,
    person_amount=4,
    dispersion_inc=0.05,
    person_initial_position=(15, 15),
    drone_amount=2,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time=0,
)


def random_policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = env.action_space(agent).sample()
    return actions


opt = {
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
}
observations, info = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = random_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
```

## 🛠️ Basic Coverage Usage
```python
from DSSE import CoverageDroneSwarmSearch

env = CoverageDroneSwarmSearch(
    grid_size=40,
    drone_amount=3,
    dispersion_inc=0.1,
    vector=(1, 1),
    render_mode="human",
)

opt = {
    "drones_positions": [(0, 10), (10, 10), (20, 10)],
}
obs, info = env.reset(options=opt)

step = 0
while env.agents:
    step += 1
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print(infos["drone0"])
```

## 🆘 Support

If you encounter any issues or have questions, please file an issue on our [GitHub issues page](https://github.com/pfeinsper/drone-swarm-search/issues).

## 📖 How to cite this work

If you use this package, please consider citing it with this piece of BibTeX:

```
@misc{castanares2023dsse,
      title={DSSE: a drone swarm search environment}, 
      author={Manuel Castanares, Luis F. S. Carrete, Enrico F. Damiani, Leonardo D. M. de Abreu, José Fernando B. Brancalion and Fabrício J. Barth},
      year={2024},
      eprint={2307.06240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={https://doi.org/10.48550/arXiv.2307.06240}
}
```
