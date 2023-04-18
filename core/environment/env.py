import functools
import random
from copy import copy, deepcopy

import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete

from pettingzoo.utils.env import ParallelEnv
import sys

# from generator.probability import generate_probability_matrix
# from generator.map import generate_map

# from core.environment.generator.probability import generate_probability_matrix
from core.environment.generator.map import generate_map, generate_matrix
from core.environment.generator.dynamic_probability import dynamic_probability


class CustomEnvironment(ParallelEnv):
    def __init__(self, grid_size=7, render_mode="ansi", n_drones=1):
        self.grid_size = grid_size
        self.person_y = None
        self.person_x = None
        self.timestep = None
        self.vector_x = 0
        self.vector_y = 0
        self.vector = (-0.1, 0.3)
        self.possible_agents = []
        self.agents_positions = {}
        for i in range(n_drones):
            self.possible_agents.append("drone" + str(i))
            self.agents_positions["drone" + str(i)] = [None, None]

        self.probability_matrix = None

        self.render_mode = render_mode
        self.probability_matrix = generate_matrix(self.grid_size)
        self.map, self.person_x, self.person_y = generate_map(self.probability_matrix)
        self.probability_matrix = self.probability_matrix.tolist()

    def default_drones_positions(self):
        counter_x = 0
        counter_y = 0
        for i in self.agents:
            self.agents_positions[i] = [counter_x, counter_y]
            counter_y += 1
            counter_x += 1

    def required_drone_positions(self, drones_positions: list):
        for i in range(len(drones_positions)):
            x, y = drones_positions[i]
            self.agents_positions[self.possible_agents[i]] = [y, x]

    def reset(self, seed=None, return_info=False, options=None, drones_positions=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.default_drones_positions() if drones_positions is None else self.required_drone_positions(
            drones_positions
        )

        observations = self.create_observations()
        return observations

    def create_observations(self):
        observations = {}
        new_map, new_x, new_y = dynamic_probability(
            self.probability_matrix, self.vector, self.vector_x, self.vector_y
        )
        self.probability_matrix = deepcopy(new_map)
        self.vector_x = deepcopy(new_x)
        self.vector_y = deepcopy(new_y)
        self.map, self.person_x, self.person_y = generate_map(self.probability_matrix)
        for i in self.possible_agents:
            observation = (
                (self.agents_positions[i][0], self.agents_positions[i][1]),
                self.probability_matrix,
            )

            observations[i] = {"observation": observation}

        return observations

    def step(self, actions):
        terminations = {a: False for a in self.agents}
        rewards = {a: -1 for a in self.agents}
        truncations = {a: False for a in self.agents}

        for i in self.agents:
            drone_action = actions[i]
            # Check termination conditions
            drone_x = self.agents_positions[i][0]
            drone_y = self.agents_positions[i][1]

            isSearching = False

            if drone_action == 0:  # left
                if drone_x > 0:
                    self.agents_positions[i][0] -= 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True

            elif drone_action == 1:
                if drone_x < self.grid_size - 1:
                    self.agents_positions[i][0] += 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True
            elif drone_action == 2:  # UP
                if drone_y > 0:
                    self.agents_positions[i][1] -= 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True
            elif drone_action == 3:  # DOWN
                if drone_y < self.grid_size - 1:
                    self.agents_positions[i][1] += 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True

            elif drone_action == 4:  # search
                isSearching = True
            elif drone_action == 5:  # idle
                pass

            if drone_x == self.person_x and drone_y == self.person_y and isSearching:
                rewards = {a: 0 for a in self.agents}
                terminations = {a: True for a in self.agents}
                truncations = {a: True for a in self.agents}
                self.agents = []

            elif isSearching:
                rewards[i] = self.probability_matrix[drone_y][drone_x] - 100

            # Check truncation conditions (overwrites termination conditions)
            if self.timestep > 500:
                rewards[i] = -1000
                truncations[i] = True
                terminations[i] = True
                self.agents = []
        self.timestep += 1

        # Get observations
        observations = self.create_observations()

        # Get dummy infos (not used in this example)
        infos = {e: {} for e in self.agents}

        # CHECK COLISION
        for ki, i in self.agents_positions.items():
            for ke, e in self.agents_positions.items():
                if ki is not ke:
                    if i[0] == e[0] and i[1] == e[1]:
                        truncations[ki] = True
                        terminations[ki] = True
                        rewards[ki] = -2000
        rewards["total_reward"] = sum([e for e in rewards.values()])

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)

        grid[self.person_y][self.person_x] = "X"
        for ki, i in self.agents_positions.items():
            grid[i[1], i[0]] = "D" + ki[-1]
        print("----" * self.grid_size)
        for i in grid:
            string = "| "
            for e in i:
                if len(str(e)) <= 1:
                    string += "{0} | ".format(e)
                else:
                    string += "{0}| ".format(e)
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
        # TODO: If x and y are the observation, then this should the observation space
        return MultiDiscrete([self.grid_size] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6)


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(
        CustomEnvironment(7, "human", n_drones=5),
        num_cycles=1000,
    )
