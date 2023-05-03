from pickle import load
import numpy as np

from core.environment.env import DroneSwarmSearch
from config import Config

env = DroneSwarmSearch(
    grid_size=50, 
    render_mode="human", 
    render_grid = True,
    render_gradient = True,
    n_drones=11, 
    vector=[0.5, 0.5],
    person_initial_position = [5, 10],
    disperse_constant = 3)

try:
    qtable = load(open(Config.qtable_path, "rb"))
except FileNotFoundError:
    print("Q-table file not found. Please run train_simple_qlearning.py first.")
    exit()

observations = env.reset()
agent = env.agents[0]

done = False
tot_reward = 0
#(x, y), _ = env.reset(drones_positions=[[0, 1], [0, 8], [10, 7], [15, 10], [4, 30]])[agent]["observation"]
(x, y), _ = env.reset()[agent]["observation"]
env.render()


while not done:
    actions = {}
    for i in range(11):
        actions["drone{}".format(i)] = 5
    observations, reward, _, done, info = env.step(actions)
    tot_reward += reward[agent]
    done = done[agent]
    (x, y), _ = observations[agent]["observation"]
    env.render()

print("Total reward: ", tot_reward)
