import numpy as np
import matplotlib.pyplot as plt
import os
from pickle import dump


class BasicQLearningBox:
    def __init__(self, agents):
        self.agents = agents

        self.full_range = 102
        self.range = int((self.full_range - 2) / 2)
        self.max_range_x_y = self.full_range
        self.max_velocity = self.full_range

        self.high = np.array(
            [
                self.max_velocity,
                self.max_velocity,
                self.max_range_x_y,
                self.max_range_x_y,
            ]
        )
        self.low = np.array(
            [
                -self.max_velocity,
                -self.max_velocity,
                -self.max_range_x_y,
                -self.max_range_x_y,
            ]
        )

    def transform_state(self, states: dict) -> dict:
        state_adj = {}
        for agent in self.agents:
            state_adj[agent] = np.round(
                ((states[agent] - self.low) / (self.high - self.low) * self.range), 0
            ).astype(int)

        return state_adj


class QLearningBoxMultiAgent(BasicQLearningBox):
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
        super().__init__(agents)
        self.env = env
        self.num_agents = len(self.agents)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

        # Initialize action and observation spaces for each agent
        self.action_spaces = {
            agent: self.env.action_spaces[agent] for agent in self.agents
        }
        self.observation_spaces = {
            agent: self.env.observation_spaces[agent] for agent in self.agents
        }

        # Initialize Q-table for each agent
        self.Q: dict = {
            agent: np.zeros(
                (
                    self.max_range_x_y,
                    self.max_range_x_y,
                    self.max_velocity,
                    self.max_velocity,
                    self.action_spaces[agent].n,
                )
            )
            for agent in self.agents
        }

    def select_action(self, state: np.ndarray, agent: str) -> int:
        if np.random.random() < self.epsilon:
            x, y, vx, vy = state + np.array(
                [self.range, self.range, self.range, self.range]
            )

            action = np.argmax(self.Q[agent][x, y, vx, vy])
        else:
            action = self.action_spaces[agent].sample()

        return action

    def select_actions(self, state_adj: dict) -> dict:
        actions = {}
        for agent in self.agents:
            actions[agent] = self.select_action(state_adj[agent], agent)

        return actions

    def update_q_table(
        self,
        state: dict,
        action: int,
        reward: int,
    ):
        for agent in self.agents:
            state = state[agent]
            action = action[agent]
            reward = reward[agent]

            x, y, vx, vy = state + np.array(
                [self.range, self.range, self.range, self.range]
            )

            self.Q[agent][x, y, vx, vy, action] += self.alpha * (
                reward
                + self.gamma * np.max(self.Q[agent][x, y, vx, vy])
                - self.Q[agent][x, y, vx, vy, action]
            )

    def train(self):
        # Initialize variables to track rewards
        reward_list = {agent: [] for agent in self.agents}
        ave_reward_list = {agent: [] for agent in self.agents}
        actions_per_episode = {agent: [] for agent in self.agents}

        # Run Q-learning algorithm
        for i in range(self.episodes):
            # Initialize parameters
            done = False
            tot_rewards = {agent: 0 for agent in self.agents}
            qtd_actions = {agent: 0 for agent in self.agents}
            states = self.env.reset()

            state_adj = self.transform_state(states)

            while not done:
                # Select action
                actions = self.select_actions(state_adj)

                # Take action
                new_state, new_reward, new_terminated, new_done, _ = self.env.step(
                    actions
                )
                done = all(new_terminated.values())

                if done:
                    continue

                # Transform state
                new_state_adj = self.transform_state(new_state)

                # Update Q-table
                self.update_q_table(new_state_adj, actions, new_reward)

                # Update parameters
                state_adj = new_state_adj

                for agent in self.agents:
                    tot_rewards[agent] += new_reward[agent]
                    qtd_actions[agent] += 1

                # self.env.render()

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

            # Update reward list
            for agent in self.agents:
                reward_list[agent].append(tot_rewards[agent])

            if i % 100 == 0:
                for agent in self.agents:
                    ave_reward = np.mean(reward_list[agent])
                    ave_reward_list[agent].append(ave_reward)
                    actions_per_episode[agent].append(qtd_actions[agent])

                    # Reset reward list
                    reward_list[agent] = []

                print(
                    f"Episode: {i} | Average Reward: {ave_reward:.2f} | Epsilon: {self.epsilon:.2f}"
                    f" | Actions: {qtd_actions}"
                )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(base_dir, "results/")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.plot_rewards(base_dir, ave_reward_list)
        self.save_model(base_dir)

    def plot_rewards(self, path, ave_reward_list: dict):
        for agent in self.agents:
            plt.plot(ave_reward_list[agent])
            plt.title(f"Average Reward per 100 Episodes - {agent}")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.savefig(path + f"qlearning_{agent}.png")
            plt.close()

    def save_model(self, path: str):
        with open(path + "qlearning.pkl", "wb") as f:
            dump(self.Q, f)
