# Drone Swarm Search

## Workflow Status: Automated Testing with Pytest

[![Run Pytest](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/env.yml/badge.svg)](https://github.com/pfeinsper/drone-swarm-search/actions/workflows/env.yml)

## Quick Start

#### Install
`pip install DSSE`

#### Use
`from DSSE.env import DroneSwarmSearch`

#### PyPi Package Page

https://pypi.org/project/DSSE/

## About

The Drone Swarm Search project is an environment, based on PettingZoo, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target, and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs. A visual representation of the environment is displayed below. To test the environment (without an algorithm), run `basic_env.py`.

<p align="center">
    <img src="https://raw.github.com/PFE-Embraer/drone-swarm-search/env-cleanup/docs/gifs/render_with_grid_gradient.gif" width="400" height="400" align="center">
</p>


## Outcome

| If drone is found            | If drone is not found  |
:-------------------------:|:-------------------------:
| ![](https://raw.githubusercontent.com/PFE-Embraer/drone-swarm-search/main/docs/pics/victory_render.png)     | ![](https://raw.github.com/PFE-Embraer/drone-swarm-search/main/docs/pics/fail_render.png) |


## Basic Usage
```python
from DSSE import DroneSwarmSearch

env = DroneSwarmSearch(
    grid_size=50, 
    render_mode="human", 
    render_grid = True,
    render_gradient = True,
    n_drones=4, 
    vector=[0.2, 0.2],
    person_initial_position = [8, 8],
    disperse_constant = 3)

def policy(obs, agent):
    actions = {
        "drone0": 1, # Right
        "drone1": 3, # Down
        "drone2": 0, # Left
        "drone3": 2, # Up
    }
    
    return actions


observations = env.reset(drones_positions=[[5, 5], [43, 5], [43, 43], [5, 43]])
rewards = 0
done = False

while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False

print(rewards)
```

### Installing Dependencies

Using Python version above or equal to 3.10.5.

In order to use the environment download the dependencies using the following command `pip install -r requirements.txt`.

### General Info

| Import             | `from core.environment.env import DroneSwarmSearch` |
| ------------------ | -------------------------------------------------- |
| Action Space       | Discrete (9)                                      |
| Action Values      | [0, 1, 2, 3, 4, 5, 6, 7, 8]                       |  
| Agents             | N                                                |
| Observation Space  | `{droneN: {observation: ((x, y), probability_matrix)}}` |

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
| `grid_size`               | `int(N)`              | `7`                       |
| `render_mode`             | `"ansi" or "human"`   | `"ansi"`                  |
| `render_grid`             | `bool`                | `False`                   |
| `render_gradient`         | `bool`                | `True`                    |
| `n_drones`                | `int(N)`              | `1`                       |
| `vector`                  | `[float(x), float(y)` | `(-0.5, -0.5)`            |
| `person_initial_position` | `[int(x), int(y)]`    | `[0, 0]`                  |
| `disperse_constant`       | `float`               | `10`                      |
| `timestep_limit`          | `int`                 | `100`                     |

### `grid_size`:

The grid size defines the area in which the search will happen. It should always be an integer greater than one.

### `render_mode`:

There are two available render modes, *ansi*  and *human*.

**Ansi**: This mode presents no visualization and is intended to train the reinforcement learning algorithm.

**Human**: This mode presents a visualization of the drones actively searching the target, as well as the visualization of the person moving according to the input vector. 

### `render_grid`:

The *render_grid* variable is a simple boolean that if set to **True** along with the `render_mode = “human”` the visualization will be rendered with a grid, if it is set to **False** there will be no grid when rendering.   

### `render_gradient`:

The *render_gradient* variable is a simple boolean that if set to **True** along with the `render_mode = “human”` the colors in the visualization will be interpolated according to the probability of the cell. Otherwise the color of the cell will be solid according to the following values, considering the values of the matrix are normalized between 0 and 1: `1 > value >= 0.75` the cell will be *green* |` 0.75 > value >= 0.25` the cell will be *yellow* | `0.25 > value` the cell will be *red*.

### `n_drones`:

The `n_drones` input defines the number of drones that will be involved in the search. It needs to be an integer greater than one.

### `vector`:

The `vector` is a list with two values that defines the direction in which the person will drift over time. It is a list with two components where the first value of the list is the displacement in the `x axis` and the second value is the displacement in the `y axis`. A positive x value will result in a displacement to the right and vice versa, and a positive y value will result in a displacement downward. A value equal to 1 will result in a displacement of 1 cell per timestamp, a value of 0.5 will result in a displacement of 1 cell every 2 timesteps, and so on. 

### `person_initial_position`:

The `person_initial_position` defines the starting point of the target, it should be a list with two values where the first component is the `x axis` and the second component is the `y axis`. The `y axis` is directed downward. The values have to be integers.

### `disperse_constant`:

The `disperse_constant` is a float that defines the dispersion of the probability matrix. The greater the number the quicker the probability matrix will disperse.

### `timestep_limit`:

The `timestep_limit` is an integer that defines the length of an episode. This means that the `timestep_limit` is essentially the amount of steps that can be done without resetting or ending the environment.

## Built in Functions:

### `env.reset`:

`env.reset()` will reset the environment to the initial position. If you wish to choose the initial positions of the drones an argument can be sent to the method. To do so, the following syntax should be considered. `env.reset(drones_positions=[[5, 5], [25, 5], [45, 5], [5, 15], [25, 15], [45, 15], [10, 35], [30, 35], [45, 25], [33, 45]])`

Each value of the list represents the `[x, y]` initial position of each drone. Make sure that the list has the same number of positions as the number of drones defined in the environment. 

Additionally, to change the vector, a tuple (representing the vector) can be sent as an argument. This can be done using the following syntax: `env.reset(vector=(0.3, 0.3))`. This way, the person's movement will change according to the new vector. 

In the case of no argument `env.reset()` will simply allocate the drones from left to right each in the next adjacent cell. Once there are no more available cells in the row it will go to the next row and do the same from left to right. The vector will also remain the same as before, when there is no argument in the reset function.

The method will also return a observation dictionary with the observations of all drones. 

### `env.step`:

The `env.step()` method defines the drone's next movement. When called upon, the method receives  a dictionary with all the drones names as keys and the action as values. For example, in an environment initialized with 10 drones: `env.step({'drone0': 2, 'drone1': 3, 'drone2': 2, 'drone3': 5:, 'drone4’: 1, 'drone5': 0, 'drone6': 2, 'drone7': 5, 'drone8': 0, 'drone9': 1})`. All drones must be in the dictionary and have an action value associated with it, every drone receives an action in every step, otherwise an error will be raised.

The method returns the **observation**, the **reward**, the **termination** state, the **truncation** state and **info**, in the respectful order.

### Person movement:

The person's movement is done using the probability matrix and the vector. The vector essentially dislocates the probabilities, which in turn defines the position of the person. The chances of a person being in a cell is determined by the probability of each cell. Moreover, the person can only move one cell at a time. This means that in every step, the person can only move to one of the cells adjacent to the one he is currently at. This was done in order to create a more realistic movement for the shipwrecked person.

#### Observation:

The observation is a dictionary with all the drones as keys. Each drone has a value of another dictionary with “observation” as key and a tuple as its value. The tuple follows the following pattern, `((x_position, y_position), probability_matrix)`. An output example can be seen below.

```bash
{
    'drone0': 
        {'observation': ((5, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]]))
        }, 
    'drone1': 
        {'observation': ((25, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]]))
        }, 
    'drone2': 
        {'observation': ((45, 5), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]]))
       }, 
       
       .................................
       
    'drone9': 
        {'observation': ((33, 45), array([[0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        ...,
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.],
                                        [0., 0., 0., ..., 0., 0., 0.]]))
        }
}
```

#### Reward:

The reward returns a dictionary with the drones names as keys and their respectful rewards as values, as well as a total reward which is the sum of all agents rewards. For example `{'drone0': 1, 'drone1': 89.0, 'drone2': 1, 'total_reward': 91.0}`

The rewards values goes as follows:

- **1** for every action by default
- **-100000** if the drone leaves the grid 
- **(*sum_of_rewards* * -1) -100000** if the drone does not find the person after timestep exceeds timestep_limit
- **-100000** if the drones collide 
- ***(probability of cell * 10000) if (probability of cell * 100 > 1) else -100*** for searching a cell
- ***10000 + 10000 * (1 - timestep / timestep_limit)*** if the drone searches the cell in which the person is located

#### Termination & Truncation:

The termination and truncation variables return a dictionary with all drones as keys and boolean as values. For example `{'drone0': False, 'drone1': False, 'drone2': False}`. The booleans will be False by default and will turn True in the event of the conditions below:

- If two or more drones collide
- If one of the drones leave the grid 
- If timestep exceeds timestep_limit
- If a drone searches the cell in which the person is located

#### Info:

Info is a dictionary that contains a key called "Found" that contains a boolean value. The value begins as `False`, and is only changed to `True` once any drone finds the shipwrecked person. The info section is to be used as an indicator to see if the person was found. For example, before finding the shipwrecked person, the dictionary will be `{"Found": False}`. Once the person is found, the dictionary will be `{"Found": True}`.

### `env.get_agents`:

`env.get_agents()` will return a list of all the possible agents initialized in the scene, you can use it to confirm that all the drones exist in the environment. For example `['drone0', 'drone1', 'drone2', 'drone3', 'drone4', 'drone5', 'drone6', 'drone7', 'drone8', 'drone9']` in an environment with 10 drones.  

### `env.close`:

`env.close()` will simply close the render window. Not a necessary function but may be used.

## How to cite this work

If you use this package, please consider citing it with this piece of BibTeX:

```
@misc{castanares2023dsse,
      title={DSSE: a drone swarm search environment}, 
      author={Manuel Castanares and Luis F. S. Carrete and Enrico F. Damiani and Leonardo D. M. de Abreu and José Fernando B. Brancalion and Fabrício J. Barth},
      year={2023},
      eprint={2307.06240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={https://doi.org/10.48550/arXiv.2307.06240}
}
```
