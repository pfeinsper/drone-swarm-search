import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Box
from DSSE import DroneSwarmSearch


class TopNProbsWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the observation space to include the positions of all agents, and the top n positions with highest probability in the observation
    """

    def __init__(self, env: DroneSwarmSearch, n_positions: int = 10):
        super().__init__(env)
        self.n_positions = n_positions

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.env.possible_agents
        }

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        self.modify_obs(obs)
        return obs, reward, terminated, truncated, infos

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.modify_obs(obs)
        return obs, infos

    def modify_obs(self, obs):
        for idx, agent in enumerate(obs.keys()):
            agents_positions = np.array(self.env.agents_positions, dtype=np.int64)
            agents_positions[[0, idx]] = agents_positions[[idx, 0]]
            obs[agent] = np.concatenate(
                (agents_positions.flatten(), self.get_top_prob_positions(obs[agent][1]))
            )

    def get_top_prob_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-self.n_positions :][::-1]
        positions = np.unravel_index(indices, probability_matrix.shape)
        positions = np.stack((positions[1], positions[0]), axis=-1)
        return positions.flatten()

    def observation_space(self, agent):
        agents_pos_len = len(self.env.possible_agents) * 2
        return Box(
            low=0,
            high=self.env.grid_size,
            shape=(agents_pos_len + self.n_positions * 2,),
            dtype=np.int64,
        )
