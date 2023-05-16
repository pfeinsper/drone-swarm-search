from core.environment.env import DroneSwarmSearch
from config import get_config

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


class DQN:
    def __init__(
        self,
        env,
        drones_initial_positions,
        n_episodes=10_000,
        batch_size=32,
    ):
        self.env = env
        self.drone_initial_positions = drones_initial_positions
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.num_drones = len(env.possible_agents)
        self.num_obs = (
            2 * self.num_drones
            + env.observation_space("drone0").nvec[0]
            * env.observation_space("drone0").nvec[1]
        )
        self.num_actions = len(env.action_space("drone0")) * self.num_drones
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(64, activation="relu", input_dim=self.num_obs))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(self.num_actions, activation="softmax"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0001))
        return model

    def transform_state(self, state):
        transformed_state = np.array([])
        for agent in self.env.possible_agents:
            position_x, position_y = state[agent]["observation"][0]
            transformed_state = np.concatenate(
                [transformed_state, [position_x, position_y]]
            )
        obs = state["drone0"]["observation"][1]
        transformed_state = np.concatenate([transformed_state, obs.flatten()])

        return transformed_state

    def get_random_actions(self):
        return {
            agent: np.random.randint(len(self.env.action_space(agent)))
            for agent in self.env.possible_agents
        }

    def get_q_actions(self, state):
        transformed_state = self.transform_state(state)
        q_values = self.model.predict(transformed_state[np.newaxis])
        actions = q_values.reshape(self.num_drones, self.num_actions)

        return {
            agent: action
            for agent, action in zip(
                self.env.possible_agents, np.argmax(actions, axis=1)
            )
        }

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            actions = self.get_random_actions()
        else:
            actions = self.get_q_actions(state)

        return actions

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self.transform_state(state)
            next_state = self.transform_state(next_state)
            actions = list(action.values())
            total_reward = reward["total_reward"]
            done = all(done.values())

            target = total_reward

            # TODO: como fazer isso para mais de um agente?
            if not done:
                target = total_reward + self.gamma * np.amax(
                    self.model.predict(tf.convert_to_tensor(next_state[np.newaxis]))[0]
                )

            state_tensor = tf.convert_to_tensor(state[np.newaxis])
            target_f = self.model.predict(state_tensor)

            # TODO: como fazer isso para mais de um agente?
            target_f[0][actions[0]] = target
            self.model.fit(state_tensor, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_random_speed_vector(self):
        """Returns a random speed vector for the environment, from -0.5 to 0.5, 0.1 step"""
        return [
            round(np.random.uniform(-0.2, 0.2), 1),
            round(np.random.uniform(0.1, 0.2), 1),
        ]

    def train(self):
        rewards = []
        for episode in range(self.n_episodes):
            state = self.env.reset(
                drones_positions=self.drone_initial_positions,
                vector=self.get_random_speed_vector(),
            )
            done = False
            total_reward = 0

            while not done:
                action = self.get_action(state, self.epsilon)
                next_state, reward, _, done, _ = self.env.step(action)
                total_reward += sum(reward.values())
                self.remember(state, action, reward, next_state, done)
                state = next_state

                done = all(done.values())

            rewards.append(total_reward)

            if episode % 100 == 0 and episode != 0:
                # TODO: isso é muito custoso, tem que ver como fazer
                self.replay()
                print(
                    "Episode: {}, Reward: {}, Epsilon: {}".format(
                        episode, total_reward, self.epsilon
                    )
                )

        self.save_weights()
        self.plot_learning_curves(rewards)

    def save_fig(self):
        plt.savefig(
            f"results/dqn_{self.env.grid_size}_{self.env.grid_size}_{self.env.n_drones}.png",
            format="png",
            dpi=300,
        )

    def plot_learning_curves(self, rewards):
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        self.save_fig("dqn_rewards_plot")

    def save_weights(self):
        self.model.save(
            f"data/dqn_{self.env.grid_size}_{self.env.grid_size}_{self.env.n_drones}.h5"
        )


config = get_config(1)
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
dqn = DQN(env, drones_initial_positions=config.drones_initial_positions)
dqn.train()
