
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
