import torch
import numpy as np
from config import get_config
from core.environment.env import DroneSwarmSearch

config = get_config(1)


def flatten_positions(positions):
    flattened = [pos for sublist in positions for pos in sublist]
    return flattened


def get_flatten_top_probabilities_positions(probability_matrix):
    flattened_probs = probability_matrix.flatten()
    indices = flattened_probs.argsort()[-10:][::-1]
    positions = [
        (idx // len(probability_matrix), idx % len(probability_matrix))
        for idx in indices
    ]

    return flatten_positions(positions)


def flatten_state(observations):
    flatten_all = []

    for drone_index in range(config.n_drones):
        drone_position = torch.tensor(
            observations["drone" + str(drone_index)]["observation"][0]
        )
        others_position = torch.flatten(
            torch.tensor(
                [
                    observations["drone" + str(index)]["observation"][0]
                    for index in range(config.n_drones)
                    if index != drone_index
                ]
            )
        )
        flatten_top_probabilities = torch.tensor(
            get_flatten_top_probabilities_positions(
                observations["drone" + str(drone_index)]["observation"][1]
            )
        )
        flatten_all.append(
            torch.cat(
                (drone_position, others_position, flatten_top_probabilities), dim=-1
            )
        )

    return flatten_all


def get_random_speed_vector():
    return [
        round(np.random.uniform(-0.1, 0.1), 1),
        round(np.random.uniform(-0.1, 0.1), 1),
    ]


def test_100_times():
    rewards = 0
    steps_count = 0
    total_success = 0

    for i in range(1000):
        episode_actions = {}
        state = env.reset(
            drones_positions=config.drones_initial_positions,
            vector=get_random_speed_vector(),
        )
        obs_list = flatten_state(state)
        done = False

        while not done:
            for drone_index in range(len(env.possible_agents)):
                probs = nn(obs_list[drone_index].float())
                dist = torch.distributions.Categorical(probs)
                episode_actions[f"drone{drone_index}"] = dist.sample().item()

            obs_list_, reward_dict, _, done, info = env.step(episode_actions)
            steps_count += 1

            rewards += reward_dict["total_reward"]
            done = True in [e for e in done.values()]
            obs_list = flatten_state(obs_list_)

        if info["Found"]:
            # print(f"Epsiode {i} found person")
            total_success += 1
        # else:
        # print(f"Epsiode {i} did not find person")

    print(
        f"Drones found person {total_success} times, {total_success / 1000 * 100} % of the time"
    )
    return total_success


nn = torch.load(f"data/nn_10_10_{config.n_drones}.pt")
nn = nn.float()

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="ansi",
    render_grid=True,
    render_gradient=False,
    n_drones=config.n_drones,
    vector=config.vector,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
)


test_100_times()
