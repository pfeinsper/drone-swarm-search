from core.environment.env import DroneSwarmSearch
import numpy as np
env = DroneSwarmSearch(
    grid_size=100, 
    render_mode="human", 
    render_grid = True,
    render_gradient = True,
    n_drones=3, 
    vector=[0.2, 0.2],
    person_initial_position = [0, 5],
    disperse_constant = 5)

def policy(obs, agent):
    actions = {}
    for i in range(11):
        actions["drone{}".format(i)] = np.random.randint(5)
    return actions


observations = env.reset(drones_positions=[[5, 5], [25, 5], [45, 5]])

rewards = 0
done = False

while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False
print(rewards)