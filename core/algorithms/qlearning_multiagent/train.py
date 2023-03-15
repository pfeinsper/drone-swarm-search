from pettingzoo.mpe import simple_v2
from qlearning_box import QLearningBoxMultiAgent


parallel_env = simple_v2.parallel_env(
    max_cycles=100000, continuous_actions=False, render_mode="human"
)
parallel_env.reset()

qlearn = QLearningBoxMultiAgent(
    env=parallel_env,
    agents=parallel_env.agents,
    alpha=0.2,
    gamma=0.4,
    epsilon=0.7,
    epsilon_min=0.05,
    epsilon_dec=0.99,
    episodes=10000,
)

qlearn.train()
