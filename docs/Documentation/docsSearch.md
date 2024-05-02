# Drone Swarm Search

## About

The Drone Swarm Search project is an environment, based on PettingZoo, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target, and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs. A visual representation of the environment is displayed below. To test the environment (without an algorithm), run `basic_env.py`.

<p align="center">
    <img src="https://raw.github.com/PFE-Embraer/drone-swarm-search/env-cleanup/docs/gifs/render_with_grid_gradient.gif" width="400" height="400" align="center">
</p>


### Outcome

| If target is found       | If target is not found   |
:-------------------------:|:-------------------------:
| ![](https://raw.githubusercontent.com/PFE-Embraer/drone-swarm-search/main/docs/public/pics/victory_render.png)     | ![](https://raw.github.com/PFE-Embraer/drone-swarm-search/main/docs/public/pics/fail_render.png) |


## Quick Start

::: warning Warning
The DSSE project requires Python version 3.10.5 or higher.
:::

#### Install
`pip install DSSE`

#### Use
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


## General Info

| Import             | `from DSSE import DroneSwarmSearch` |
| ------------------ | -------------------------------------------------- |
| Action Space       | Discrete (9)                                      |
| Action Values      | [0, 1, 2, 3, 4, 5, 6, 7, 8]                       |  
| Agents             | N                                                |
| Observation Space  | `{droneN: ((x, y), probability_matrix)}` |

### Action Space

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Move Left              |
| 1     | Move Right             |
| 2     | Move Up                |
| 3     | Move Down              |
| 4     | Diagonal Up Left       |
| 5     | Diagonal Up Right      |
| 6     | Diagonal Down Left     |
| 7     | Diagonal Down Right    |
| 8     | Search Cell            |

### Inputs
| Inputs                    | Possible Values       | Default Values            |
| -------------             | -------------         | -------------             |
| `grid_size`               | `int(N)`              | `20`                      |
| `render_mode`             | `"ansi" or "human"`   | `"ansi"`                  |
| `render_grid`             | `bool`                | `False`                   |
| `render_gradient`         | `bool`                | `True`                    |
| `vector`                  | `[float(x), float(y)` | `(1.1, 1)`                |
| `dispersion_inc`          | `float`               | `0.1`                     |
| `dispersion_start`        | `float`               | `0.5`                     |
| `timestep_limit`          | `int`                 | `100`                     |
| `person_amount`           | `int`                 | `1`                       |
| `person_initial_position` | `(int(x), int(y))`    | `(0, 0)`                  |
| `drone_amount`            | `int`                 | `1`                       |
| `drone_speed`             | `int`                 | `10`                      |
| `probability_of_detection`| `float`               | `1.0`                     |
| `pre_render_time`         | `int`                 | `0`                       |

- **`grid_size`**: Defines the area in which the search will happen. It should always be an integer greater than one.

- **`render_mode`**:

    - **`ansi`**: This mode presents no visualization and is intended to train the reinforcement learning algorithm.
    - **`human`**: This mode presents a visualization of the drones actively searching the target, as well as the visualization of the person moving according to the input vector. 

- **`render_grid`**: If set to **True** along with `render_mode = "human"`, the visualization will be rendered with a grid. If set to **False**, there will be no grid when rendering.

- **`render_gradient`**: If set to **True** along with `render_mode = "human"`, the colors in the visualization will be interpolated according to the probability of the cell. Otherwise, the color of the cell will be solid according to the following values, considering the values of the matrix are normalized between 0 and 1: `1 > value >= 0.75` the cell will be <span style="color:lightgreen; text-shadow: -0.8px 0 black, 0 0.8px black, 0.8px 0 black, 0 -0.8px black;">green</span> | `0.75 > value >= 0.25` the cell will be <span style="color:#FFD700; text-shadow: -0.8px 0 black, 0 0.8px black, 0.8px 0 black, 0 -0.8px black;">yellow</span> | `0.25 > value` the cell will be <span style="color:red; text-shadow: -0.8px 0 black, 0 0.8px black, 0.8px 0 black, 0 -0.8px black;">red</span>.


- **`vector`**: This parameter is a two-element list that specifies the direction and rate of drift over time for a person within the environment. Each element in the list corresponds to displacement along the `x-axis` and `y-axis` respectively:
  - **X-axis Displacement**: A positive value causes a rightward drift, while a negative value results in a leftward drift.
  - **Y-axis Displacement**: A positive value causes a downward drift, while a negative value results in an upward drift.

::: tip Tip
For more realistic movement speeds in `vector`, we recommend varying the values of `x` and `y` between `-1 and 1`.
:::

- **`dispersion_inc`**: It's a float that defines the dispersion of the probability matrix. It must be a float `greater` or `equal` to zero. The greater the number, the quicker the probability matrix will disperse.

- **`dispersion_start`**: Defines the starting value for the dispersion matrix size. It must be a float `greater` or `equal` to zero.

- **`timestep_limit`**: It's an integer that defines the length of an episode. This means that the `timestep_limit` is essentially the number of steps that can be done without resetting or ending the environment.

- **`person_amount`**: Defines the number of `persons` in water. It must be an **integer** `greater` or `equal` to 1.

- **`person_initial_position`**: Specifies the initial coordinates of the target in the form of a tuple `(x, y)`. The `x` value represents the horizontal position, while the `y` value, which increases downward, represents the vertical position. Both coordinates must be **integers**.

- **`drone_amount`**: Specifies the number of drones to be used in the simulation. This **integer** parameter can be adjusted to simulate scenarios with different drone counts.


- **`drone_speed`**: An **integer** parameter that sets the drones' speed in the simulation, measured in meters per second `(m/s)`. Adjust this value to simulate drones operating at various speeds.

- `probability_of_detection`: This **float** parameter. It signifies the probability of a drone detecting an object of interest. Changing this value allows the user to simulate different detection probabilities.

- `pre_render_time`: This **int** parameter. It specifies the amount of time `(minutes)` to pre-render the simulation before starting. Adjusting this value lets the user control the pre-rendering time of the simulation.

## Built in Functions

### `env.reset`:

`env.reset()` reinitializes the environment to its initial state. To customize the starting conditions, such as drone positions, the probability of detection for each person (PIW), or the movement vector, you can pass an `options` dictionary to the method. Here’s how to structure this dictionary and use the `reset()` method:

```python
opt = {
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
    "vector": (0.3, 0.3)
}
observations, info = env.reset(options=opt, vector)
```
#### Parameters in the options Dictionary:

- **`drones_positions`**: Specifies the initial `[(x, y), (x, y), ...]` coordinates for each drone. Ensure this list **contains** an entry for each drone in the environment.

- **`person_pod_multipliers`**: Specifies the detection probability for each person in the environment. The list length **must** match the number of PIWs (Persons in Water) and each value **must be positive**.

- **`vector`**: Sets a **tuple** representing the movement direction and rate for the person within the environment. This parameter alters how the person moves according to the specified vector.

#### Default Behavior:

Without any arguments, `env.reset()` will place drones sequentially from left to right in adjacent cells. When there are no more available cells in a row, it moves to the next row and continues from left to right. If no vector or POD values are specified, they will remain unchanged from their previous states.

#### Return Values:

The method returns an `observations` dictionary containing observations for all drones, which provides insights into the environment's state immediately after the reset.

### `env.step`:

The `env.step()` method defines the drone's next movement. It requires a dictionary input where each key is a drone's name and its corresponding value is the action to be taken. For instance, in an environment initialized with 10 drones, the method call would look like this:

```python
env.step({
    'drone0': 2, 'drone1': 3, 'drone2': 2, 'drone3': 5, 'drone4': 1,
    'drone5': 0, 'drone6': 2, 'drone7': 5, 'drone8': 0, 'drone9': 1
})
```

::: warning Warning
Every drone listed in the dictionary `must` have an associated action. If any drone is omitted or if an action is not specified for a drone, the method will raise an **error**.
:::

#### The method returns a tuple containing the following elements in order:

- **`Observation`**: The new state of the environment after the step.
- **`Reward`**: The immediate reward obtained after the action.
- **`Termination`**: Indicates whether the episode has ended (e.g., find all castway, limit exceeded).
- **`Truncation`**: Indicates whether the episode was truncated (e.g., through a timeout).
- **`Info`**: A dictionary containing auxiliary diagnostic information.

### Person Movement:

The movement of the person in the environment is governed by both the probability matrix and the vector. The vector specifically shifts the probabilities, which then determines the potential positions of the person. Here's how it works:

- **`Probability Matrix`**: Each cell's probability indicates the probabilities of the person being present there.
- **`Vector Movement`**: The vector applies a displacement to these probabilities, effectively moving the person's likely position within the grid.

Moreover, the person is restricted to moving only one cell per timestep. This means they can move to any adjacent cell-up, down, left, or right-but no further, in a single step. This constraint is designed to simulate more realistic movement patterns for a shipwrecked individual.

### Observation:

The observation is a dictionary with all the drones as keys, identified by names such as `drone0`, `drone1`, etc. Each key is associated with a tuple that contains the drone's current position and its perception of the environment, represented as a probability matrix.

- **Tuple Structure**: `((x_position, y_position), probability_matrix)`
  - **`x_position`**, **`y_position`**: The current coordinates of the drone on the grid.
  - **`probability_matrix`**: A matrix representing the drone's view of the probability distribution of the target's location across the grid. 

An output example can be seen below.

```python
{
    'drone0': 
        ((5, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]])
        ),

    'drone1': 
        ((25, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]])
        ),
    'drone2': 
        ((45, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]])
        ),

       
       .................................
       
    'droneN': 
        ((33, 45), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]])
        ),
        
}
```

### Reward:

The reward returns a dictionary with the drones names as keys and their respectful rewards as values. For example `{'drone0': 1, 'drone1': 89.0, 'drone2': 1}`

The rewards values goes as follows:

- **`Default Action`**: Every action receives a baseline reward of `0.1`.
- **`Leaving the Grid`**: A penalty of `-200` is applied if a drone leaves the grid boundaries.
- **`Exceeding Time Limit`**: A penalty of `-200` is imposed if the drone does not locate the person before the timestep_limit is exceeded.
- **`Collision`**: If drones collide, each involved drone receives a penalty of `-200`.
- **`Searching a Cell`**: The reward for searching a cell is proportional to the probability p of the cell being searched, denoted as `[0:p]`.
- **`Finding the Person`**: If a drone successfully locates the person within a cell, the reward is `200 + 200 * ((1 - timestep) / timestep_limit)`, encouraging faster discovery.

### Termination & Truncation

The termination and truncation variables return a dictionary with all drones as keys and boolean as values. By default, these values are set to `False` and will switch to `True` under any of the following conditions:

- **`Collision`**: If two or more drones collide.
- **`Time Limit Exceeded`**: If the simulation's timestep exceeds the `timestep_limit`.
- **`All PIWs Found`**: If all Persons in Water (PIWs) have been successfully located.

#### For example, the dictionary might look like this:

```python
{'drone0': False, 'drone1': False, 'drone2': False}
```

### Info:

the `Info` is a dictionary of dictionaries, with each drone serves as a key, with its value being another dictionary that contains a key called ***`Found`*** that contains a boolean value. The value begins as `False`, and is only changed to `True` once any drone finds the shipwrecked person. The `info` section is to be used as an indicator to see if the person was found.

For example, the dictionary will appear as follows before any drone has found the shipwrecked person:

```python
{'drone0': {'Found': False}, 'drone1': {'Found': False}}
```

After a drone successfully locates the person, the dictionary updates to reflect this:

```python
{'drone0': {'Found': True}, 'drone1': {'Found': True}}
```

This mechanism ensures that users can easily monitor and verify the success of the search operation at any point during the simulation.

### `env.get_agents`:

The `env.get_agents()` method will return a list of all the possible agents (drones) currently initialized in the simulation, you can use it to confirm that all the drones exist in the environment. For example, in an environment configured with 10 drones, the method would return:

```python
['drone0', 'drone1', 'drone2', 'drone3', 'drone4', 'drone5', 'drone6', 'drone7', 'drone8', 'drone9']
```

### `env.close`:

`env.close()` will simply close the render window. Not a necessary function but may be used.

## How to Cite This Work

If you use this package, please consider citing it with this piece of BibTeX:

```bibtex
@misc{castanares2023dsse,
      title={DSSE: A Drone Swarm Search Environment}, 
      author={Manuel Castanares, Luis F. S. Carrete, Enrico F. Damiani, Leonardo D. M. de Abreu, José Fernando B. Brancalion, and Fabrício J. Barth},
      year={2024},
      eprint={2307.06240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={10.48550/arXiv.2307.06240}
}
```

## License
This documentation is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for more details.
