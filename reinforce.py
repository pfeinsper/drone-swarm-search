import torch
import numpy as np
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
        self.num_obs = (
            2 * self.num_agents
            + env.observation_space("drone0").nvec[0]
            * env.observation_space("drone0").nvec[1]
        )
        self.num_actions = sum(
            [len(env.action_space(agent)) for agent in env.possible_agents]
        )

    def flatten_state(self, observations):
        drone_position = torch.tensor(observations["drone0"]["observation"][0])
        flatten_obs = torch.flatten(
            torch.tensor(observations["drone0"]["observation"][1])
        )
        all_obs = torch.cat((drone_position, flatten_obs), dim=-1)
        return all_obs

    def enhance_reward(self, drone_position, probability_matrix, action):
        if action in {0, 1, 2, 3}:
            match action:
                case 0:  # LEFT
                    previous_position = (drone_position[0] + 1, drone_position[1])
                case 1:  # RIGHT
                    previous_position = (drone_position[0] - 1, drone_position[1])
                case 2:  # UP
                    previous_position = (drone_position[0], drone_position[1] + 1)
                case 3:  # DOWN
                    previous_position = (drone_position[0], drone_position[1] - 1)

            max_value_index = np.argmax(probability_matrix)
            max_row_index, max_col_index = np.unravel_index(
                max_value_index, probability_matrix.shape
            )

            current_distance = np.sqrt(
                (drone_position[0] - max_row_index) ** 2
                + (drone_position[1] - max_col_index) ** 2
            )
            previous_distance = np.sqrt(
                (previous_position[0] - max_row_index) ** 2
                + (previous_position[1] - max_col_index) ** 2
            )

            if previous_distance < current_distance:
                return -100

            return 100

        return 0

    def get_reward_enhanced(self, observations, action, current_reward):
        drone_position = observations["drone0"]["observation"][0]
        probability_matrix = observations["drone0"]["observation"][1]
        return current_reward + self.enhance_reward(
            drone_position, probability_matrix, action
        )

    def train(self):
        nn = torch.nn.Sequential(
            torch.nn.Linear(self.num_obs, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_actions),
            torch.nn.Softmax(dim=-1),
        )
        optim = torch.optim.Adam(nn.parameters(), lr=self.lr)
        statistics = []

        nn = nn.float()

        for i in range(self.episodes + 1):
            state = self.env.reset(drones_positions=self.drones_initial_positions)
            obs = self.flatten_state(state)
            done = False
            actions, states, rewards = [], [], []
            count_actions = 0
            total_reward = 0

            while not done:
                probs = nn(obs.float())
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                obs_, reward, _, done, _ = self.env.step({"drone0": action})

                # TODO: Check if we'll keep this strategy
                reward = self.get_reward_enhanced(obs_, action, reward["total_reward"])

                actions.append(torch.tensor(action, dtype=torch.int))
                states.append(obs)
                rewards.append(reward)

                obs = self.flatten_state(obs_)
                count_actions += 1
                total_reward += reward

                done = any(done.values())

            if i % 100 == 0:
                print(
                    f"Episode = {i}, Actions = {count_actions}, Rewards = {total_reward}"
                )

            statistics.append([i, count_actions, total_reward])

            discounted_returns = []
            for t in range(len(rewards)):
                G = sum((self.y**k) * r for k, r in enumerate(rewards[t:]))
                discounted_returns.append(G)

            for state, action, G in zip(states, actions, discounted_returns):
                probs = nn(state.float())
                dist = torch.distributions.Categorical(probs=probs)
                log_prob = dist.log_prob(action)

                loss = -log_prob * G

                optim.zero_grad()
                loss.backward()
                optim.step()

        return nn, statistics


config = get_config(2)

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="ansi",
    render_grid=False,
    render_gradient=False,
    n_drones=config.n_drones,
    vector=config.vector,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
)

rl_agent = RLAgent(
    env,
    y=0.99999,
    lr=0.00001,
    episodes=10_000,
    drones_initial_positions=config.drones_initial_positions,
)
nn, statistics = rl_agent.train()

torch.save(nn, f"data/nn_{config.grid_size}_{config.grid_size}.pt")
# df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
# df.to_csv("results/statistics_10_10.csv")
