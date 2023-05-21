import torch
import numpy as np
import pandas as pd
from config import get_config
from core.environment.env import DroneSwarmSearch


class RLAgent:
    def __init__(self, env, y, lr, episodes, drones_initial_positions):
        self.env = env
        self.y = y
        self.lr = lr
        self.episodes = episodes
        self.drones_initial_positions = drones_initial_positions

        self.num_agents = len(env.possible_agents)
        self.num_entries = (self.num_agents + self.env.grid_size) * 2
        self.num_actions = len(env.action_space("drone0"))

        self.nn = self.create_neural_network()
        self.optimizer = self.create_optimizer(self.nn.parameters())

    def create_neural_network(self):
        nn = torch.nn.Sequential(
            torch.nn.Linear(self.num_entries, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_actions),
            torch.nn.Softmax(dim=-1),
        )
        return nn.float()

    def create_optimizer(self, parameters):
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        return optimizer

    def flatten_positions(self, positions):
        flattened = [pos for sublist in positions for pos in sublist]
        return flattened

    def get_flatten_top_probabilities_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-self.env.grid_size :][::-1]
        positions = [
            (idx // len(probability_matrix), idx % len(probability_matrix))
            for idx in indices
        ]

        return self.flatten_positions(positions)

    def get_random_speed_vector(self):
        return [
            round(np.random.uniform(-0.1, 0.1), 1),
            round(np.random.uniform(-0.1, 0.1), 1),
        ]

    def select_actions(self, obs_list):
        episode_actions = {}
        for drone_index in range(self.num_agents):
            probs = self.nn(obs_list[drone_index].float())
            distribution = torch.distributions.Categorical(probs)
            episode_actions[f"drone{drone_index}"] = distribution.sample().item()

        return episode_actions

    def flatten_state(self, observations):
        flatten_all = []

        for drone_index in range(self.num_agents):
            drone_position = torch.tensor(
                observations["drone" + str(drone_index)]["observation"][0]
            )
            others_position = torch.flatten(
                torch.tensor(
                    [
                        observations["drone" + str(index)]["observation"][0]
                        for index in range(self.num_agents)
                        if index != drone_index
                    ]
                )
            )
            flatten_top_probabilities = torch.tensor(
                self.get_flatten_top_probabilities_positions(
                    observations["drone" + str(drone_index)]["observation"][1]
                )
            )
            flatten_all.append(
                torch.cat(
                    (drone_position, others_position, flatten_top_probabilities), dim=-1
                )
            )

        return flatten_all

    def extract_rewards(self, reward_dict):
        rewards = [
            drone_reward for key, drone_reward in reward_dict.items() if "drone" in key
        ]
        return rewards

    def print_episode_stats(self, episode_num, show_actions, show_rewards):
        print(
            f"Up to episode = {episode_num}, Actions (mean) = {sum(show_actions) / len(show_actions)}, "
            f"Reward (mean) = {sum(show_rewards) / len(show_rewards)}"
        )

    def calculate_discounted_returns(self, rewards):
        discounted_returns = []
        for t in range(len(rewards)):
            G_list = []
            for drone_index in range(self.num_agents):
                agent_rewards = [r[drone_index] for r in rewards]
                G_list.append(
                    sum((self.y**k) * r for k, r in enumerate(agent_rewards[t:]))
                )
            discounted_returns.append(G_list)

        return discounted_returns

    def update_neural_network(self, states, actions, discounted_returns):
        for state_list, action_list, G_list in zip(states, actions, discounted_returns):
            for drone_index in range(self.num_agents):
                probs = self.nn(state_list[drone_index].float())
                distribution = torch.distributions.Categorical(probs=probs)
                log_prob = distribution.log_prob(action_list[drone_index])

                loss = -log_prob * G_list[drone_index]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self):
        statistics, show_rewards, show_actions, all_rewards = [], [], [], []
        stop = False

        for i in range(self.episodes + 1):
            if stop:
                break

            vector = self.get_random_speed_vector()
            state = self.env.reset(
                drones_positions=self.drones_initial_positions, vector=vector
            )
            obs_list = self.flatten_state(state)
            done = False
            actions, states, rewards = [], [], []
            count_actions, total_reward = 0, 0

            while not done:
                episode_actions = self.select_actions(obs_list)
                obs_list_, reward_dict, _, done, _ = self.env.step(episode_actions)

                actions.append(
                    torch.tensor(list(episode_actions.values()), dtype=torch.int)
                )
                states.append(obs_list)
                rewards.append(self.extract_rewards(reward_dict))
                obs_list = self.flatten_state(obs_list_)
                count_actions += self.num_agents
                total_reward += reward_dict["total_reward"]
                done = any(done.values())

            show_rewards.append(total_reward)
            all_rewards.append(total_reward)
            show_actions.append(count_actions)

            if len(all_rewards) > 100:
                if all([r >= 100000 for r in all_rewards[-50:]]):
                    stop = True
                    print("Acabou mais cedo")

            if i % 100 == 0:
                self.print_episode_stats(i, show_actions, show_rewards)
                show_rewards, show_actions = [], []

            statistics.append([i, count_actions, total_reward])
            discounted_returns = self.calculate_discounted_returns(rewards)

            self.update_neural_network(states, actions, discounted_returns)

        return self.nn, statistics


config = get_config(1)

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="ansi",
    render_grid=False,
    render_gradient=False,
    n_drones=config.n_drones,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
    timestep_limit=100,
)

rl_agent = RLAgent(
    env,
    y=0.999999,
    lr=0.000001,
    episodes=50_000,
    drones_initial_positions=config.drones_initial_positions,
)
nn, statistics = rl_agent.train()

torch.save(nn, f"data/nn_{config.grid_size}_{config.grid_size}_{config.n_drones}.pt")
df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
df.to_csv(
    f"data/statistics_{config.grid_size}_{config.grid_size}_{config.n_drones}.csv",
    index=False,
)
