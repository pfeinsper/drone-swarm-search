from core.environment.env import DroneSwarmSearch
import numpy as np
env = DroneSwarmSearch(
    grid_size=50, 
    render_mode="human", 
    render_grid = True,
    render_gradient = True,
    n_drones=10, 
    vector=[0.5, 0.5],
    person_initial_position = [5, 10],
    disperse_constant = 5)

def policy(obs, agent):
    actions = {}
    for i in range(11):
        actions["drone{}".format(i)] = np.random.randint(5)
    return actions


observations = env.reset(drones_positions=[[5, 5], [25, 5], [45, 5], [5, 15], [25, 15], [45, 15], [10, 35], [30, 35], [45, 25], [33, 45]])

rewards = 0
done = False

while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False
print(rewards)