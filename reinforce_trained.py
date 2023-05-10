import torch
from config import get_config
from core.environment.env import DroneSwarmSearch

config = get_config(1)


def flatten_state(observations):
    drone_position = torch.tensor(observations["drone0"]["observation"][0])
    flatten_obs = torch.flatten(torch.tensor(observations["drone0"]["observation"][1]))
    all_obs = torch.cat((drone_position, flatten_obs), dim=-1)
    return all_obs


nn = torch.load("data/nn_10_10.pt")
nn = nn.float()

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="human",
    render_grid=False,
    render_gradient=False,
    n_drones=config.n_drones,
    vector=config.vector,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
)

state = env.reset(drones_positions=config.drones_initial_positions)
obs = flatten_state(state)
done = False

rewards = 0
done = False

while not done:
    probs = nn(obs.float())
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    obs_, reward, _, done, info = env.step({"drone0": action})
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False
    obs = flatten_state(obs_)

print(rewards)
