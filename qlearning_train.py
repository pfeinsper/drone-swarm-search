from core.algorithms.qlearning_single.qlearning_box import QLearningBox
from core.environment.env import CustomEnvironment
from config import Config


def train_qlearning(grid_size: int):
    env = CustomEnvironment(grid_size=grid_size, render_mode="ansi")
    env.reset()
    agent = env.agents[0]

    qlearn = QLearningBox(
        env=env,
        agent=agent,
        alpha=Config.alpha,
        gamma=Config.gamma,
        epsilon=Config.epsilon,
        epsilon_min=Config.epsilon_min,
        epsilon_dec=Config.epsilon_dec,
        episodes=Config.episodes,
        qtable_path=Config.qtable_path,
        plot_file_path=Config.plot_file_path,
    )
    qlearn.train()


if __name__ == "__main__":
    train_qlearning(grid_size=Config.grid_size)
