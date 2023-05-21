import torch
from config import get_config
from core.environment.env import DroneSwarmSearch
from reinforce_mult import Reinforce

config = get_config(2)

nn = torch.load(f"data/nn_{config.grid_size}_{config.grid_size}_{config.n_drones}.pt")
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

reinforce = Reinforce(env=env)


def test_100_times():
    rewards = 0
    steps_count = 0
    total_success = 0

    for i in range(100):
        episode_actions = {}
        state = env.reset(drones_positions=config.drones_initial_positions)
        obs_list = reinforce.flatten_state(state)
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
            obs_list = reinforce.flatten_state(obs_list_)

        if info["Found"]:
            print(f"Epsiode {i} found person")
            total_success += 1
        else:
            print(f"Epsiode {i} did not find person")

    print(
        f"Drones found person {total_success} times, {total_success / 100 * 100} % of the time"
    )
    return total_success


if __name__ == "__main__":
    test_100_times()
