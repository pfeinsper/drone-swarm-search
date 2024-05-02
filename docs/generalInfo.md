
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
