import numpy as np
import matplotlib.pyplot as plt
from pickle import dump


class QLearningBox:
    def __init__(
        self,
        env,
        agent,
        alpha,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_dec,
        episodes,
        qtable_path,
        plot_file_path,
    ):
        self.env = env
        self.agent = agent
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.qtable_path = qtable_path
        self.plot_file_path = plot_file_path

        self.action_space = self.env.action_space(self.agent)
        self.observation_space = self.env.observation_space(self.agent)

        # inicializando uma q-table com 3 dimensoes: x, y e acao
        self.Q = np.zeros(
            (
                self.observation_space.nvec[0],
                self.observation_space.nvec[1],
                len(self.action_space),
            )
        )

    def select_action(self, state):
        if np.random.random() < 1 - self.epsilon:
            position, _ = state
            x, y = position
            return np.argmax(self.Q[x, y])

        return np.random.choice(self.action_space)

    def train(self):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []
        actions_per_episode = []

        # Run Q-learning algorithm
        for i in range(self.episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            initial_state = self.env.reset()[self.agent]["observation"]

            qtd_actions = 0
            while done != True:
                action = self.select_action(initial_state)
                state, reward, _, done, _ = self.env.step({self.agent: action})

                state = state[self.agent]["observation"]
                reward = reward[self.agent]
                done = done[self.agent]

                (x, y), _ = state
                (x_, y_), _ = initial_state

                delta = self.alpha * (
                    reward + self.gamma * np.max(self.Q[x, y]) - self.Q[x_, y_, action]
                )
                self.Q[x_, y_, action] += delta

                # Update variables
                tot_reward += reward
                initial_state = state
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
                print(
                    "Episode {} Average Reward: {}  Actions in this episode {} ".format(
                        i + 1,
                        ave_reward,
                        actions_per_episode[len(actions_per_episode) - 1],
                    )
                )

        self.plotactions(ave_reward_list)
        self.save_q_table(self.qtable_path)

    def plotactions(self, rewards):
        plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Episodes")
        plt.savefig(self.plot_file_path)
        plt.close()

    def save_q_table(self, filename):
        dump(self.Q, open(filename, "wb"))
