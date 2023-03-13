import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv


class CustomEnvironment(ParallelEnv):
    def __init__(self, grid_size = 7):
        self.grid_size = grid_size
        self.person_y = None
        self.person_x = None
        self.drone_y = None
        self.drone_x = None
        self.timestep = None
        self.possible_agents = ["drone"]
        self.probability_matrix = None

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.drone_x = 2
        self.drone_y = 1

        self.person_x = random.randint(2, 5)
        self.person_y = random.randint(2, 5)

        self.probability_matrix = [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3],
        ]

        observation = (
            self.drone_x + self.grid_size * self.drone_y,
            self.probability_matrix
        )
        observations = {
            "drone": {"observation": observation, "action_mask": [0, 1, 1, 0, 1]},
        }
        return observations

    def step(self, actions):
        # Execute actions
        drone_action = actions["drone"]

        isSearching = False

        if drone_action == 0 and self.drone_x > 0:
            self.drone_x -= 1
        elif drone_action == 1 and self.drone_x < self.grid_size -1:
            self.drone_x += 1
        elif drone_action == 2 and self.drone_y > 0:
            self.drone_y -= 1
        elif drone_action == 3 and self.drone_y < self.grid_size -1:
            self.drone_y += 1
        elif drone_action == 4:
            isSearching = True

        # Generate action masks
        drone_action_mask = np.ones(5)
        if self.drone_x == 0:
            drone_action_mask[0] = 0  # Block left movement
        elif self.drone_x == self.grid_size -1:
            drone_action_mask[1] = 0  # Block right movement
        if self.drone_y == 0:
            drone_action_mask[2] = 0  # Block down movement
        elif self.drone_y == self.grid_size -1:
            drone_action_mask[3] = 0  # Block up movement

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: -1 for a in self.agents}
        if self.drone_x == self.person_x and self.drone_y == self.person_y and isSearching:
            rewards = {"drone": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif isSearching and self.probability_matrix[self.drone_y][self.drone_x] < 0.2:
            rewards = {"drone": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {"drone": False}
        if self.timestep > 100:
            rewards = {"drone": -1}
            truncations = {"drone": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
            self.drone_x + self.grid_size * self.drone_y,
            self.probability_matrix
        )
        observations = {
            "drone": {
                "observation": observation,
                "action_mask": drone_action_mask,
            },
        }

        # Get dummy infos (not used in this example)
        infos = {"drone": {}}

        self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        print("drone_position: ({0}, {1})".format(self.drone_y, self.drone_x))
        grid[self.drone_y][self.drone_x] = "D"
        grid[self.person_y][self.person_x] = "X"

        print("----"*self.grid_size)
        for i in grid:
            string = "| "
            for e in i:
                string += "{0} | ".format(e)
            print(string)
        print("----"*self.grid_size)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
                return MultiDiscrete([self.grid_size * self.grid_size - 1] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(CustomEnvironment(10), num_cycles=1000)
