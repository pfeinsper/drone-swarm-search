import torch
import pandas as pd
from core.environment.env import DroneSwarmSearch


class RLAgent:
    def __init__(self, env, y, lr, episodes):
        self.env = env
        self.y = y
        self.lr = lr
        self.episodes = episodes

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
            state = self.env.reset(drones_positions=[[5, 5]])
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

                actions.append(torch.tensor(action, dtype=torch.int))
                states.append(obs)
                rewards.append(reward["total_reward"])

                obs = self.flatten_state(obs_)
                count_actions += 1
                total_reward += reward["total_reward"]

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


env = DroneSwarmSearch(
    grid_size=10,
    render_mode="ansi",
    render_grid=False,
    render_gradient=True,
    n_drones=1,
    vector=[0.5, 0.5],
    person_initial_position=[1, 1],
    disperse_constant=5,
)

rl_agent = RLAgent(env, y=0.99999, lr=0.00001, episodes=5_000)
nn, statistics = rl_agent.train()

torch.save(nn, "data/nn_10_10.pt")
df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
df.to_csv("results/statistics_10_10.csv")
