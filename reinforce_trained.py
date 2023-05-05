import gymnasium as gym
import torch
from core.environment.env import DroneSwarmSearch

#
# eu um estado 50x50 irah retornar um vetor de 2502 posicoes
#
def flatten_state(observations):
    drone_position = torch.tensor(observations['drone0']['observation'][0])
    flatten_obs = torch.flatten(torch.tensor(observations['drone0']['observation'][1]))
    all_obs = torch.cat((drone_position,flatten_obs), dim=-1)
    return all_obs

#
# Depois de treinado
#
print('##### Modelo treinado #####')

nn = torch.load('data/nn_10_10.pt')
nn = nn.float()

env = DroneSwarmSearch(
    grid_size=10, 
    render_mode="human", 
    render_grid = False,
    render_gradient = True,
    n_drones=1, 
    vector=[0.5, 0.5],
    person_initial_position = [1, 1],
    disperse_constant = 5)

state = env.reset(drones_positions=[[5, 5]])
obs = flatten_state(state)
#obs = torch.tensor(state, dtype=torch.float)
done = False

rewards = 0
done = False

while not done:
    probs = nn(obs.float())
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    print(action)
    obs_, reward, _, done, info = env.step({'drone0': action})
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False
    
    obs = flatten_state(obs_)
    #obs = torch.tensor(obs_, dtype=torch.float)
print(rewards)
    

