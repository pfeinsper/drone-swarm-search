import torch
from config import get_config
from core.environment.env import DroneSwarmSearch

config = get_config(3)


def flatten_state(observations, num_agents):

    flatten_all = []

    for drone_index in range(num_agents):
        drone_position = torch.tensor(observations["drone" + str(drone_index)]["observation"][0])
        flatten_obs = torch.flatten(
            torch.tensor(observations["drone" + str(drone_index)]["observation"][1])
        )
        others_position = torch.flatten(torch.tensor(
            [observations["drone" + str(index)]["observation"][0] for index in range(num_agents) if
             index != drone_index]))

        flatten_all.append(torch.cat((drone_position, others_position, flatten_obs), dim=-1))

    return flatten_all


nn = torch.load(f"data/nn_{config.grid_size}_{config.grid_size}.pt")
nn = nn.float()

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="human",
    render_grid=True,
    render_gradient=False,
    n_drones=config.n_drones,
    vector=config.vector,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
)

state = env.reset(drones_positions=config.drones_initial_positions)
obs_list = flatten_state(state, len(env.possible_agents))
done = False
rewards = 0


while not done:
    episode_actions = {}

    for drone_index in range(len(env.possible_agents)):
        probs = nn(obs_list[drone_index].float())
        dist = torch.distributions.Categorical(probs)
        episode_actions[f"drone{drone_index}"] = dist.sample().item()

    obs_list_, reward_dict, _, done, _ = env.step(episode_actions)

    rewards += reward_dict["total_reward"]
    done = True if True in [e for e in done.values()] else False
    obs_list = flatten_state(obs_list_, len(env.possible_agents))

print(rewards)
