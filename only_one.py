from core.environment.env import DroneSwarmSearch
import numpy as np
import torch

env = DroneSwarmSearch(
    grid_size=50, 
    render_mode="human", 
    render_grid = False,
    render_gradient = True,
    n_drones=1, 
    vector=[0.5, 0.5],
    person_initial_position = [5, 10],
    disperse_constant = 5)

def policy(obs, agent):
    actions = {}
    for i in range(1):
        actions["drone{}".format(i)] = np.random.randint(5)
    return actions


observations = env.reset(drones_positions=[[5, 5]])

rewards = 0
done = False

while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)

    drone_position = torch.tensor(observations['drone0']['observation'][0])
    flatten_obs = torch.flatten(torch.tensor(observations['drone0']['observation'][1]))
    all_obs = torch.cat((drone_position,flatten_obs), dim=-1)

    rewards += reward["total_reward"]
    #
    # TODO nao entendi
    #
    done = True if True in [e for e in done.values()] else False
print(rewards)