import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import MultiDiscrete

from pettingzoo.utils.env import ParallelEnv


class CustomEnvironment(ParallelEnv):
    def __init__(self, grid_size=7, render_mode="ansi"):
        self.grid_size = grid_size
        self.person_y = None
        self.person_x = None
        self.drone_y = None
        self.drone_x = None
        self.timestep = None
        self.possible_agents = ["drone"]
        self.probability_matrix = None

        self.render_mode = render_mode

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.drone_x = 0
        self.drone_y = 0

        # self.person_x = random.randint(2, 5)
        # self.person_y = random.randint(2, 5)

        self.person_x = 5
        self.person_y = 5

        self.probability_matrix = [
            [10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 20, 10, 10],
            [10, 10, 10, 10, 30, 50, 10],
            [10, 10, 10, 10, 10, 30, 30],
        ]

        observation = (
            (self.drone_x, self.drone_y),
            self.probability_matrix,
        )
        observations = {
            "drone": {"observation": observation, "action_mask": [0, 1, 1, 0, 1]},
        }
        return observations

    def step(self, actions):
        # Execute actions
        drone_action = actions["drone"]

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: -1 for a in self.agents}
        truncations = {"drone": False}

        isSearching = False

        if drone_action == 0:  # left
            if self.drone_x > 0:
                self.drone_x -= 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 1:  # right
            if self.drone_x < self.grid_size - 1:
                self.drone_x += 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 2:  # down
            if self.drone_y > 0:
                self.drone_y -= 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}
        elif drone_action == 3:  # up
            if self.drone_y < self.grid_size - 1:
                self.drone_y += 1
            else:
                rewards = {"drone": -1000}
                truncations = {"drone": True}
                terminations = {"drone": True}

        elif drone_action == 4:  # search
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
            (self.drone_x, self.drone_y),
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

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        print("drone_position: ({0}, {1})".format(self.drone_y, self.drone_x))
        grid[self.drone_y][self.drone_x] = "D"
        grid[self.person_y][self.person_x] = "X"

        print("----" * self.grid_size)
        for i in grid:
            string = "| "
            for e in i:
                string += "{0} | ".format(e)
            print(string)
        print("----" * self.grid_size)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # TODO: If x and y are the observation, then this should the observation space
        return MultiDiscrete([self.grid_size] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return [0, 1, 2, 3, 4]


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(CustomEnvironment(7), num_cycles=1000)
