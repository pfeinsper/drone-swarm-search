import torch
import numpy as np
import pandas as pd
from core.environment.env import DroneSwarmSearch


class Model:
    def __init__(
        self,
        env,
        load_weights=False,
        nn_filename="data/nn_10_10.pt",
        n_episodes=1000,
        y=0.99999,
        learning_rate=0.001,
        drone_position=[0, 0],
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.nn_filename = nn_filename
        self.y = y
        self.learning_rate = learning_rate
        self.drone_position = drone_position

        self.num_agents = len(env.possible_agents)
        self.num_obs = (
            2 * self.num_agents
            + env.observation_space("drone0").nvec[0]
            * env.observation_space("drone0").nvec[1]
        )
        self.num_actions = sum(
            [len(env.action_space(agent)) for agent in env.possible_agents]
        )

        if load_weights:
            self.model = self._load_weights()
        else:
            self.model = self._build_model()
            self._optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.train()

    def _build_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.num_obs, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.num_actions),
            torch.nn.Softmax(dim=-1),
        )
        model = model.float()
        return model

    def _load_weights(self):
        return torch.load(self.nn_filename)

    def _save_weights(self):
        torch.save(self.model, self.nn_filename)

    def _export_statistics(self, statistics):
        df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
        df.to_csv("results/statistics_10_10.csv")

    def _flatten_state(self, observations):
        # transformed_state = np.array([])
        # for agent in self.env.possible_agents:
        #     position_x, position_y = state[agent]["observation"][0]
        #     transformed_state = np.concatenate(
        #         [transformed_state, [position_x, position_y]]
        #     )
        # obs = state["drone0"]["observation"][1].flatten()
        # transformed_state = np.concatenate([transformed_state, obs])

        # return torch.tensor(transformed_state, dtype=torch.float)
        drone_position = torch.tensor(observations["drone0"]["observation"][0])
        flatten_obs = torch.flatten(
            torch.tensor(observations["drone0"]["observation"][1])
        )
        all_obs = torch.cat((drone_position, flatten_obs), dim=-1)
        return all_obs

    def _get_actions(self, state_flatten):
        probs = self.model(state_flatten.float())
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return {"drone0": action}

    def train(self):
        statistics = []

        for i in range(self.n_episodes + 1):
            state = self.env.reset(drones_positions=[self.drone_position])
            obs = self._flatten_state(state)
            done = False
            actions, states, rewards = [], [], []
            count_actions, episode_reward = 0, 0

            while not done:
                action = self._get_actions(obs)
                obs_, reward, _, done, _ = self.env.step(action)

                actions.append(torch.tensor(action["drone0"], dtype=torch.int))
                states.append(obs)
                rewards.append(reward["total_reward"])

                obs = self._flatten_state(obs_)
                count_actions += 1
                episode_reward = episode_reward + reward["total_reward"]
                done = all(done.values())

            if i % 100 == 0:
                print(
                    f"Episode = {i}, Actions = {count_actions}, Rewards = {episode_reward}"
                )

            statistics.append([i, count_actions, episode_reward])

            discounted_returns = []
            for t in range(len(rewards)):
                G = 0.0
                for k, r in enumerate(rewards[t:]):
                    G += (self.y**k) * r
                discounted_returns.append(G)

            for state, action, G in zip(states, actions, discounted_returns):
                probs = self.model(state.float())
                dist = torch.distributions.Categorical(probs=probs)
                log_prob = dist.log_prob(action)
                loss = -log_prob * G

                self._optim.zero_grad()
                loss.backward()
                self._optim.step()

        self._save_weights()
        self._export_statistics(statistics)

    def play(self):
        obs = self.env.reset()
        obs = self._flatten_state(obs)
        done = False
        rewards = 0

        while not done:
            action = self._get_actions(obs)
            obs_, reward, _, done, _ = self.env.step(action)
            rewards += reward["total_reward"]
            done = all(done.values())
            obs = self._flatten_state(obs_)
        print(rewards)


env = DroneSwarmSearch(
    grid_size=5,
    render_mode="ansi",
    render_grid=True,
    render_gradient=True,
    n_drones=1,
    vector=[0.5, 0.5],
    person_initial_position=[1, 1],
    disperse_constant=2,
    timestep_limit=100,
)
env.reset()
model = Model(env, load_weights=False, n_episodes=10_000, y=0.99999, learning_rate=0.0001)
model.play()
