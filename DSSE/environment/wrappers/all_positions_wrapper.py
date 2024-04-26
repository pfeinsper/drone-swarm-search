import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Tuple, Box
from DSSE import DroneSwarmSearch


class AllPositionsWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the observation space to include the positions of all agents in all observations
    """
    def __init__(self, env: DroneSwarmSearch):
        super().__init__(env)
    
    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        self.add_other_positions_obs(obs)
        return obs, reward, terminated, truncated, infos
    
    def add_other_positions_obs(self, obs):
        prob_matrix = obs["drone0"][1]
        for idx, agent in enumerate(self.env.agents):
            agents_positions = np.array(self.env.agents_positions, dtype=np.int64)
            agents_positions[[0, idx]] = agents_positions[[idx, 0]]            
            obs[agent] = (
                agents_positions,
                prob_matrix
            )

    
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.add_other_positions_obs(obs)
        return obs, infos

    def observation_space(self, agent):
        return Tuple(
            (
                Box(0, self.env.grid_size, shape=(len(self.env.possible_agents), 2), dtype=np.int64),
                Box(
                    low=0,
                    high=1,
                    shape=(self.env.grid_size, self.env.grid_size),
                    dtype=np.float32,
                )
            )
        )
            



