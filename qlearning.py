from pickle import load
import numpy as np

from core.environment.env import CustomEnvironment
from config import Config


def qlearning(grid_size: int, render_mode: str = "ansi"):
    env = CustomEnvironment(grid_size=grid_size, render_mode=render_mode)

    try:
        qtable = load(open(Config.qtable_path, "rb"))
    except FileNotFoundError:
        print("Q-table file not found. Please run train_simple_qlearning.py first.")
        exit()

    observations = env.reset()
    env.reset()
    agent = env.agents[0]

    done = False
    tot_reward = 0
    actions_taken = []
    (x, y), _ = env.reset()[agent]["observation"]

    while not done:
        action = np.argmax(qtable[x, y])
        actions_taken.append(action)
        observations, reward, _, done, _ = env.step({agent: action})
        tot_reward += reward[agent]
        done = done[agent]
        (x, y), _ = observations[agent]["observation"]

    return actions_taken


if __name__ == "__main__":
    actions_taken = qlearning(grid_size=Config.grid_size, render_mode="ansi")
    print(actions_taken)
