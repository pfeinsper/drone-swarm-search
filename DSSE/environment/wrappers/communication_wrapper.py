from pettingzoo.utils.wrappers import BaseParallelWrapper
from DSSE import DroneSwarmSearch
from numba import njit
import numpy as np


class CommunicationWrapper(BaseParallelWrapper):
    """
    Ads tracking of seen cells to the observation space
    """

    def __init__(self, env: DroneSwarmSearch, n_steps: int = 20):
        super().__init__(env)
        self.n_steps = n_steps
        self.passed_map = None

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.passed_map = np.zeros((self.env.grid_size, self.env.grid_size))
        return obs, infos

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        for pos in self.env.agents_positions:
            # (x, y) to (row, col)
            self.passed_map[pos[1], pos[0]] = self.env.timestep
        obs = self.add_communication(obs)
        return obs, reward, terminated, truncated, infos

    def add_communication(self, obs):
        # All observations have the same matrix, so we can just calculate it once
        if len(obs) > 0:
            new_matrix = self.modify_matrix(
                obs["drone0"][1], self.n_steps, self.passed_map, self.env.timestep
            )
        for idx, agent in enumerate(obs.keys()):
            obs[agent] = (obs[agent][0], new_matrix)
        return obs

    @staticmethod
    @njit(cache=True, fastmath=True)
    def modify_matrix(matrix, n_steps, passed_map, curr_ts):
        new_matrix = matrix.copy()
        height, width = matrix.shape
        for i in range(height):
            for j in range(width):
                if matrix[i, j] == 0:
                    continue
                multiplier = min((curr_ts - passed_map[i, j]) / n_steps, 1.0)
                new_matrix[i, j] = new_matrix[i, j] * multiplier
        return new_matrix
