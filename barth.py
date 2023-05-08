import gymnasium as gym
import torch
import pandas as pd
from core.environment.env import DroneSwarmSearch


#
# eu um estado 50x50 irah retornar um vetor de 2502 posicoes
#
def flatten_state(observations):
    drone_position = torch.tensor(observations["drone0"]["observation"][0])
    flatten_obs = torch.flatten(torch.tensor(observations["drone0"]["observation"][1]))
    all_obs = torch.cat((drone_position, flatten_obs), dim=-1)
    return all_obs


def train(env, y, lr, episodes):
    nn = torch.nn.Sequential(
        torch.nn.Linear(102, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 6),
        torch.nn.Softmax(dim=-1),
    )
    # usa o Adam algorithm para otimização
    optim = torch.optim.Adam(nn.parameters(), lr=lr)

    statistics = []

    nn = nn.float()

    for i in range(episodes + 1):
        state = env.reset(drones_positions=[[5, 5]])
        obs = flatten_state(state)
        # obs = torch.tensor(state, dtype=torch.float)
        done = False
        Actions, States, Rewards = [], [], []
        count_actions = 0
        rewards = 0
        time_penality = 1

        while not done:
            probs = nn(obs.float())
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            obs_, reward, _, done, info = env.step({"drone0": action})

            Actions.append(torch.tensor(action, dtype=torch.int))
            States.append(obs)
            Rewards.append(reward["total_reward"] - time_penality)
            time_penality *= 1.1

            obs = flatten_state(obs_)
            # obs = torch.tensor(obs_, dtype=torch.float)
            count_actions += 1
            rewards = rewards + reward["total_reward"]

            done = True if True in [e for e in done.values()] else False

        if i % 100 == 0:
            print(f"Episode = {i}, Actions = {count_actions}, Rewards = {rewards}")

        statistics.append([i, count_actions, rewards])

        DiscountedReturns = []
        for t in range(len(Rewards)):
            G = 0.0
            for k, r in enumerate(Rewards[t:]):
                G += (y**k) * r
            DiscountedReturns.append(G)

        for State, Action, G in zip(States, Actions, DiscountedReturns):
            probs = nn(State.float())
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(Action)

            # importante: aqui deve ser negativo pq eh um gradient ascendent
            loss = -log_prob * G

            optim.zero_grad()
            loss.backward()
            optim.step()

    return nn, statistics


#
# iniciando o treinamento
#
print("##### Treinando o modelo #####")

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

observations = env.reset(drones_positions=[[5, 5]])

lr = 0.00003
y = 0.99999
nn, statistics = train(env, y, lr, 10_000)
torch.save(nn, "data/nn_10_10.pt")
df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
df.to_csv("results/statistics_10_10.csv")
