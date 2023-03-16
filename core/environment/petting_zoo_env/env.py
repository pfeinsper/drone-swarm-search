import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
import sys

sys.path.append("../")
from generator.probability import generate_probability_matrix
from generator.map import generate_map


class CustomEnvironment(ParallelEnv):
    def __init__(self, grid_size=7):
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

        self.drone_x = 0
        self.drone_y = 0

        self.probability_matrix = generate_probability_matrix(
            self.grid_size, self.grid_size
        )

        _, self.person_x, self.person_y = generate_map(self.probability_matrix)

        observation = (
            self.drone_x + self.grid_size * self.drone_y,
            self.probability_matrix,
        )
        observations = {
            "drone": {"observation": observation, "action_mask": [0, 1, 1, 0, 1]},
        }

        self.render_probability_matrix()
        return observations

    def step(self, actions):
        # Execute actions
        drone_action = actions["drone"]

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: -1 for a in self.agents}
        truncations = {"drone": False}

        isSearching = False

        if drone_action == 0:
            if self.drone_x > 0:
                self.drone_x -= 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 1:
            if self.drone_x < self.grid_size - 1:
                self.drone_x += 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 2:
            if self.drone_y > 0:
                self.drone_y -= 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 3:
            if self.drone_y < self.grid_size - 1:
                self.drone_y += 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}

        elif drone_action == 4:
            isSearching = True

        # Generate action masks
        drone_action_mask = np.ones(5)
        if self.drone_x == 0:
            drone_action_mask[0] = 0  # Block left movement
        elif self.drone_x == self.grid_size - 1:
            drone_action_mask[1] = 0  # Block right movement
        if self.drone_y == 0:
            drone_action_mask[2] = 0  # Block down movement
        elif self.drone_y == self.grid_size - 1:
            drone_action_mask[3] = 0  # Block up movement

        if (
            self.drone_x == self.person_x
            and self.drone_y == self.person_y
            and isSearching
        ):
            rewards = {"drone": 0}
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []

        elif isSearching:
            rewards = {
                "drone": self.probability_matrix[self.drone_y][self.drone_x] - 100
            }

        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > 500:
            rewards = {"drone": -1000}
            truncations = {"drone": True}
            terminations = {"drone": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
            self.drone_x + self.grid_size * self.drone_y,
            self.probability_matrix,
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
        print("drone_position: ({0}, {1})".format(self.drone_x, self.drone_y))
        grid[self.drone_y][self.drone_x] = "D"
        grid[self.person_y][self.person_x] = "X"

        print("----" * self.grid_size)
        for i in grid:
            string = "| "
            for e in i:
                string += "{0} | ".format(e)
            print(string)
        print("----" * self.grid_size)

    def render_probability_matrix(self):
        grid = self.probability_matrix
        print("PROBABILITY MATRIX:")

        print("----" * self.grid_size)
        for i in grid:
            string = "| "
            for e in i:
                if e >= 10:
                    string += "{0} | ".format(e)
                else:
                    string += "{0}  | ".format(e)
            print(string)
        print("----" * self.grid_size)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.grid_size * self.grid_size - 1] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return [0, 1, 2, 3, 4]


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(CustomEnvironment(7), num_cycles=1000)
