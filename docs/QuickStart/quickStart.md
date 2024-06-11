# Quick Start Guide
Welcome to the Quick Start guide for the DSSE (Drone Swarm Search Environment). This guide provides step-by-step instructions on how to install and use DSSE for both the Search and Coverage environments.

::: warning Warning
The DSSE project requires Python version 3.10.5 or higher.
:::

## Search Environment

### Installation
Install DSSE for the search environment using pip:

```bash
pip install DSSE
```

### Basic Usage
The following code snippet demonstrates how to set up and run the `DroneSwarmSearch` environment. Expand the details to view the code.

::: details Click me to view the code <a href="https://github.com/pfeinsper/drone-swarm-search/blob/main/basic_env.py" target="blank" style="float:right"><Badge type="tip" text="basic_env.py &boxbox;" /></a>
```python
from DSSE import DroneSwarmSearch

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(1, 1),
    dispersion_inc=0.05,
    timestep_limit=300,
    person_amount=4,
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
    "vector": (0.3, 0.3),


}
observations, info = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = random_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
```
:::

::: details About the agent policy
The "random_policy" function is designed to abstract the concept of a model or function that chooses actions within the environment's action space. In the example below, it samples a random action from the action space and returns a dictionary mapping agents to the actions they should perform next, based on the given observations and number of agents.

```python
def random_policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = env.action_space(agent).sample()
    return actions

actions = random_policy(observations, env.get_agents())
```
Users can adapt this function by incorporating a trained model, such as one implemented with RLlib. For a demonstration, refer to the [Algorithms](../Documentation/docsAlgorithms) section, which shows how to use a model to select actions based on the received observations.
:::

## Coverage Environment

### Install
Install DSSE with coverage environment support using pip:

```bash
pip install DSSE[coverage]
```

### Basic Usage
The following example shows how to initiate and interact with the `CoverageDroneSwarmSearch` environment. Expand the details to see the code.

::: details Click me to view the code <a href="https://github.com/pfeinsper/drone-swarm-search/blob/main/basic_coverage.py" target="blank" style="float:right"><Badge type="tip" text="basic_coverage.py &boxbox;" /></a>
```python
from DSSE import CoverageDroneSwarmSearch

env = CoverageDroneSwarmSearch(
    drone_amount=3,
    render_mode="human",
    disaster_position=(-24.04, -46.17),  # (lat, long)
    pre_render_time=10, # hours to simulate
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
:::
