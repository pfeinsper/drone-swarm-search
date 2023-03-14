import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt


class QLearningBoxMultiAgent:
    def __init__(
        self,
        env,
        agents,
        alpha,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_dec,
        episodes,
    ):
        self.env = env
        self.agents = agents
        self.num_agents = len(self.agents)

        self.action_spaces = [self.env.action_spaces[a] for a in self.agents]
        self.observation_spaces = [self.env.observation_spaces[a] for a in self.agents]

        self.full_range = 102
        self.range = int((self.full_range - 2) / 2)
        self.max_range_x_y = self.full_range
        self.max_velocity = self.full_range

        self.observation_space.high = np.array(
            [
                self.max_velocity,
                self.max_velocity,
                self.max_range_x_y,
                self.max_range_x_y,
            ]
        )
        self.observation_space.low = np.array(
            [
                -self.max_velocity,
                -self.max_velocity,
                -self.max_range_x_y,
                -self.max_range_x_y,
            ]
        )

        # inicializando uma q-table com 5 dimensoes: x, y, vx, vy e acao
        self.Q = np.zeros(
            (
                self.max_range_x_y,
                self.max_range_x_y,
                self.max_velocity,
                self.max_velocity,
                self.action_space.n,
            )
        )

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state_adj):
        if np.random.random() < 1 - self.epsilon:
            x, y, vx, vy = state_adj + np.array(
                [self.range, self.range, self.range, self.range]
            )

            return np.argmax(self.Q[x, y, vx, vy])

        return np.random.randint(0, self.action_space.n)

    def transform_state(self, state):
        state_adj = state * np.array([10, 10, 10, 10])
        state_adj = np.round(state_adj, 0).astype(int)

        return state_adj

    def train(self, filename, plotFile=None):
        # Initialize variables to track rewards
        reward_lists = [[] for _ in range(self.num_agents)]
        ave_reward_lists = [[] for _ in range(self.num_agents)]
        actions_per_episode = [[] for _ in range(self.num_agents)]

        for i in range(self.episodes):
            for agent_idx, agent in enumerate(self.agents):
                # Initialize parameters
                done = False
                tot_reward, reward = 0, 0
                state = self.env.reset()[agent]

                # discretizando o estado
                state_adj = self.transform_state(state)
                qtd_actions = 0

                while done != True:
                    action = self.select_action(state_adj)
                    actions[agent] = action
                    states, rewards, dones, infos = self.env.step(actions)

                    state2 = state2[self.agent]
                    reward = reward[self.agent]
                    done = done[self.agent]

                    # Discretize state2
                    state2_adj = self.transform_state(state2)
                    x, y, vx, vy = state2_adj + np.array(
                        [self.range, self.range, self.range, self.range]
                    )

                    delta = self.alpha * (
                        reward
                        + self.gamma * np.max(self.Q[x, y, vx, vy])
                        - self.Q[x, y, vx, vy, action]
                    )
                    self.Q[x, y, vx, vy, action] += delta

                    # Update variables
                    tot_reward += reward
                    state_adj = state2_adj
                    qtd_actions = qtd_actions + 1

                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon * self.epsilon_dec

            # Track rewards
            reward_list.append(tot_reward)

            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                actions_per_episode.append(qtd_actions)
                reward_list = []

            if (i + 1) % 100 == 0:
                print(
                    "Episode {} Average Reward: {}  Actions in this episode {} ".format(
                        i + 1,
                        ave_reward,
                        actions_per_episode[len(actions_per_episode) - 1],
                    )
                )

            self.env.render()

        # TODO Q eh um vetor tridimensional. savetxt nao trabalha com arrays com mais de 2
        # dimensoes. Eh necessario fazer ajustes para armazenar a Q-table neste caso.
        # savetxt(filename, self.Q, delimiter=',')

        if plotFile is not None:
            self.plotactions(plotFile, ave_reward_list)
        return self.Q

    def plotactions(self, plotFile, rewards):
        plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Episodes")
        plt.savefig(plotFile + ".jpg")
        plt.close()


from pettingzoo.mpe import simple_spread_v2


parallel_env = simple_spread_v2.parallel_env(max_cycles=30, continuous_actions=False)
parallel_env.reset()

qlearn = QLearningBoxMultiAgent(
    env=parallel_env,
    agents=parallel_env.agents,
    alpha=0.2,
    gamma=0.4,
    epsilon=0.7,
    epsilon_min=0.05,
    epsilon_dec=0.99,
    episodes=10000,
)
