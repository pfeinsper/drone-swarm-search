import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Box
from DSSE import DroneSwarmSearch


class AllFlattenWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the observation space to include the positions of all agents + the flatten matrix.
    """

    def __init__(self, env: DroneSwarmSearch):
        super().__init__(env)

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.env.possible_agents
        }

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        self.flatten_obs(obs)
        return obs, reward, terminated, truncated, infos

    def flatten_obs(self, obs):
        for idx, agent in enumerate(obs.keys()):
            agents_positions = np.array(self.env.agents_positions) / (
                self.env.grid_size - 1
            )
            agents_positions[[0, idx]] = agents_positions[[idx, 0]]
            obs[agent] = np.concatenate(
                (agents_positions.flatten(), obs[agent][1].flatten())
            )

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.flatten_obs(obs)
        return obs, infos

    def observation_space(self, agent):
        return Box(
            low=0,
            high=1,
            shape=(
                len(self.env.possible_agents) * 2
                + self.env.grid_size * self.env.grid_size,
            ),
            dtype=np.float64,
        )
