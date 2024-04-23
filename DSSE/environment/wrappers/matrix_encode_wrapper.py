import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Box
from DSSE import DroneSwarmSearch


class MatrixEncodeWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the observation space to include the positions of all agents encoded in the probability matrix.
    """

    def __init__(self, env: DroneSwarmSearch):
        super().__init__(env)

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.env.possible_agents
        }

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.encode_matrix_obs(obs)
        return obs, infos

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        self.encode_matrix_obs(obs)
        return obs, reward, terminated, truncated, infos

    def encode_matrix_obs(self, obs):
        for idx, agent in enumerate(obs.keys()):
            prob_matrix = obs[agent][1].copy()
            for i, pos in enumerate(self.env.agents_positions):
                x, y = pos
                if i == idx:
                    prob_matrix[y, x] += 2
                else:
                    prob_matrix[y, x] = -1
            obs[agent] = prob_matrix

    def observation_space(self, agent):
        return Box(
            low=-1,
            high=3,
            shape=(self.env.grid_size, self.env.grid_size),
            dtype=np.float32,
        )
