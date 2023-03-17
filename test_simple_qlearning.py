from pickle import load
import numpy as np

from core.environment.env import CustomEnvironment
from config import Config

env = CustomEnvironment(7)

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
(x, y), _ = env.reset()[agent]["observation"]
env.render()


while not done:
    action = np.argmax(qtable[x, y])
    observations, reward, _, done, info = env.step({agent: action})
    tot_reward += reward[agent]
    done = done[agent]
    (x, y), _ = observations[agent]["observation"]
    env.render()

print("Total reward: ", tot_reward)
