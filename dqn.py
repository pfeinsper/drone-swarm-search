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
        load_weights=False,
        filename="dqn_weights_multi.h5",
        n_episodes=1000,
        batch_size=32,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.filename = filename
        self.batch_size = batch_size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.num_agents = len(env.possible_agents)
        self.num_obs = (
            2 * self.num_agents
            + env.observation_space("drone0").nvec[0]
            * env.observation_space("drone0").nvec[1]
        )
        self.num_actions = sum(
            [len(env.action_space(agent)) for agent in env.possible_agents]
        )
        self.memory = deque(maxlen=2000)

        if load_weights:
            self.model = self.load_weights()
        else:
            self.model = self.build_model()
            self.train()

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(64, activation="relu", input_dim=self.num_obs))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(self.num_actions, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
        return model

    def transform_state(self, state):
        transformed_state = np.array([])
        for agent in self.env.possible_agents:
            position_x, position_y = state[agent]["observation"][0]
            transformed_state = np.concatenate(
                [transformed_state, [position_x, position_y]]
            )

        # append observation
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
        Q_values = self.model.predict(transformed_state[np.newaxis])

        actions = {}
        actions_range = self.num_actions // self.num_agents
        current_action_range = (0, actions_range)

        for agent in self.env.possible_agents:
            actions[agent] = np.argmax(
                Q_values[0][current_action_range[0] : current_action_range[1]]
            )
            current_action_range = (
                current_action_range[1],
                current_action_range[1] + actions_range,
            )

        return actions

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return self.get_random_actions()

        return self.get_q_actions(state)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self.transform_state(state)
            action = list(action.values())
            reward = sum(reward.values())
            next_state = self.transform_state(next_state)
            done = all(done.values())
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(tf.convert_to_tensor(next_state[np.newaxis]))[0]
                )

            state_tensor = tf.convert_to_tensor(state[np.newaxis])
            target_f = self.model.predict(state_tensor)
            # TODO: como fazer isso para mais de um agente?
            target_f[0][action[0]] = target
            self.model.fit(state_tensor, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        rewards = []
        for episode in range(self.n_episodes):
            state = self.env.reset()
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

            if episode % 10 == 0:
                if len(self.memory) > self.batch_size:
                    self.replay()
                print(
                    "Episode: {}, Reward: {}, Epsilon: {}".format(
                        episode, total_reward, self.epsilon
                    )
                )

        self.model.save(self.filename)
        self.plot_learning_curves(rewards)

    def save_fig(self, filename, fig_extension="png", resolution=300):
        plt.savefig(filename + "." + fig_extension, dpi=resolution)

    def plot_learning_curves(self, rewards):
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        self.save_fig("dqn_rewards_plot")
        plt.show()

    def load_weights(self):
        model = keras.models.load_model(self.filename)
        print(model.summary())
        keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
        return model

    def play(self, n_episodes=10, n_max_steps=50):
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.get_action(obs)
                print(action)
                obs, _, done, _, _ = self.env.step(action)
                done = all(done.values())


from core.environment.env import DroneSwarmSearch

n_episodes = 1000
batch_size = 32
env = DroneSwarmSearch(
    grid_size=5,
    render_mode="ansi",
    render_grid=True,
    render_gradient=True,
    n_drones=1,
    vector=[0.5, 0.5],
    person_initial_position=[3, 3],
    disperse_constant=3,
    timestep_limit=30,
)
dqn = DQN(env, load_weights=False, n_episodes=n_episodes, batch_size=batch_size)
dqn.play()
