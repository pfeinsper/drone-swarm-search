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