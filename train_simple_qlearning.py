from core.algorithms.qlearning_single.qlearning_box import QLearningBox
from core.environment.env import CustomEnvironment
from config import Config


env = CustomEnvironment(grid_size=14, render_mode="human")

observations = env.reset()
env.reset()
agent = env.agents[0]

qlearn = QLearningBox(
    env=env,
    agent=agent,
    alpha=0.2,
    gamma=0.4,
    epsilon=0.7,
    epsilon_min=0.05,
    epsilon_dec=0.99,
    episodes=10000,
    qtable_path=Config.qtable_path,
    plot_file_path=Config.plot_file_path,
)
q_table = qlearn.train()
