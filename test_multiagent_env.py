from pickle import load
import numpy as np

from core.environment.env import CustomEnvironment
from config import Config

env = CustomEnvironment(50, "human", 5)

try:
    qtable = load(open(Config.qtable_path, "rb"))
except FileNotFoundError:
    print("Q-table file not found. Please run train_simple_qlearning.py first.")
    exit()

observations = env.reset()
agent = env.agents[0]

done = False
tot_reward = 0
(x, y), _ = env.reset(drones_positions=[[4, 4], [8, 8], [10, 7], [15, 10], [4, 30]])[agent]["observation"]
env.render()


while not done:
    action = np.argmax(qtable[x, y])
    action1 = np.random.randint(0,5)
    action2 = np.random.randint(0,5)
    action3 = np.random.randint(0,5)
    action4 = np.random.randint(0,5)
    action5 = np.random.randint(0,5)
    observations, reward, _, done, info = env.step({"drone0": action1, "drone1" : action2, "drone2" : action3, "drone3" : action4, "drone4" : action5})
    tot_reward += reward[agent]
    done = done[agent]
    (x, y), _ = observations[agent]["observation"]
    env.render()

print("Total reward: ", tot_reward)
