import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class DQN:
    def __init__(
        self,
        env,
        load_weights=False,
        n_episodes=500,
        n_steps=200,
        filename="dqn_weights_multi.h5",
        batch_size=32,
        discount_rate=0.95,
    ):
        self.env = env
        self.num_agents = len(env.possible_agents)
        self.num_obs = [
            sum(
                [
                    2  # position x and y
                    + env.observation_space(agent).nvec[0]  # columns
                    * env.observation_space(agent).nvec[1]  # rows
                    for agent in env.possible_agents
                ]
            )
        ]
        self.num_actions = sum(
            [len(env.action_space(agent)) for agent in env.possible_agents]
        )

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.filename = filename
        self.batch_size = batch_size
        self.discount_rate = discount_rate

        self.replay_memory = deque(maxlen=2000)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        self.loss_fn = keras.losses.mean_squared_error

        if load_weights:
            self.model = self.load_weights()
        else:
            self.model = self.build_model()
            self.fit()

    def build_model(self):
        model = keras.models.Sequential(
            [
                keras.layers.Dense(
                    32,
                    activation="elu",
                    input_shape=self.num_obs,
                ),
                keras.layers.Dense(32, activation="elu"),
                keras.layers.Dense(self.num_actions),
            ]
        )
        return model

    def transform_state(self, state):
        transformed_state = np.array([])
        for agent in self.env.possible_agents:
            (position_x, position_y), obs = state[agent]["observation"]
            transformed_state = np.concatenate(
                [transformed_state, np.append([position_x, position_y], obs).flatten()]
            )

        return transformed_state

    def transform_all_states(self, states):
        return np.array([self.transform_state(state) for state in states])

    def transform_dones(self, dones):
        return np.array([all(done.values()) for done in dones]).flatten()

    def transform_rewards(self, rewards):
        return np.array([sum(reward.values()) for reward in rewards]).flatten()

    def one_hot_encode_actions(self, action):
        one_hot_actions = np.zeros(self.num_actions)
        actions_range = self.num_actions // self.num_agents
        current_action_range = (0, actions_range)

        for agent in self.env.possible_agents:
            one_hot_actions[action[agent] + current_action_range[0]] = 1
            current_action_range = (
                current_action_range[1],
                current_action_range[1] + actions_range,
            )

        return one_hot_actions

    def one_hot_encode_all_actions(self, actions):
        return np.array([self.one_hot_encode_actions(action) for action in actions])

    def get_random_action(self):
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

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return self.get_random_action()

        return self.get_q_actions(state)

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated, info = self.env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, truncated, info

    def training_step(self):
        experiences = self.sample_experiences(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(self.transform_all_states(next_states))
        max_next_Q_values = np.max(next_Q_values, axis=1)

        # TODO: Check if this is correct
        target_Q_values = (
            self.transform_rewards(rewards)
            + (1 - self.transform_dones(dones)) * self.discount_rate * max_next_Q_values
        )
        target_Q_values = target_Q_values.reshape(-1, 1)
        # END TODO

        mask = self.one_hot_encode_all_actions(actions)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(self.transform_all_states(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def fit(self):
        rewards = []
        best_score = -np.inf
        best_weights = None

        for episode in range(self.n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                epsilon = max(1 - episode / 500, 0.01)
                obs, reward, done, _, _ = self.play_one_step(obs, epsilon)
                episode_reward += reward["total_reward"]
                done = all(done.values())

            rewards.append(episode_reward)

            if episode_reward >= best_score:
                best_score = episode_reward
                best_weights = self.model.get_weights()

            print(
                "\rEpisode: {}, Reward: {}, eps: {:.3f}".format(
                    episode, episode_reward + 1, epsilon
                ),
                end="",
            )  # Not shown

            if episode > 50:
                self.training_step()

        self.model.set_weights(best_weights)
        self.model.save(self.filename)
        self.plot_learning_curves(rewards)

    def save_fig(self, filename, fig_extension="png", resolution=300):
        plt.tight_layout()
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
        return model

    def play(self, n_episodes=10, n_max_steps=50):
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.epsilon_greedy_policy(obs)
                obs, _, done, _, _ = self.env.step(action)
                done = all(done.values())
