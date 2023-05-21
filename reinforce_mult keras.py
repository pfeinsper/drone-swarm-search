import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from config import get_config
from core.environment.env import DroneSwarmSearch


class Reinforce:
    def __init__(self, env):
        self.env = env

        self.num_top_positions = 10
        self.num_agents = len(env.possible_agents)

    def flatten_positions(self, positions):
        flattened = [pos for sublist in positions for pos in sublist]
        return flattened

    def get_flatten_top_probabilities_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-self.num_top_positions :][::-1]
        positions = [
            (idx // len(probability_matrix), idx % len(probability_matrix))
            for idx in indices
        ]

        return self.flatten_positions(positions)

    def flatten_state(self, observations):
        flatten_all = []

        for drone_index in range(self.num_agents):
            drone_position = np.array(
                observations["drone" + str(drone_index)]["observation"][0]
            )

            others_postions_list = [
                observations["drone" + str(index)]["observation"][0]
                for index in range(self.num_agents)
                if index != drone_index
            ]

            flatten_top_probabilities = np.array(
                self.get_flatten_top_probabilities_positions(
                    observations["drone" + str(drone_index)]["observation"][1]
                )
            )

            if len(others_postions_list) > 0:
                others_positions = np.concatenate(others_postions_list)
                flatten_all.append(
                    np.concatenate(
                        (drone_position, others_positions, flatten_top_probabilities)
                    )
                )
            else:
                flatten_all.append(
                    np.concatenate((drone_position, flatten_top_probabilities))
                )

        return flatten_all

    def get_random_speed_vector(self):
        return [
            round(np.random.uniform(-0.1, 0.1), 1),
            round(np.random.uniform(-0.1, 0.1), 1),
        ]


class ReinforceAgent(Reinforce):
    def __init__(self, env, y, lr, episodes, drones_initial_positions):
        super().__init__(env)
        self.y = y
        self.lr = lr
        self.episodes = episodes
        self.drones_initial_positions = drones_initial_positions

        self.num_agents = len(env.possible_agents)
        self.num_entries = (self.num_agents + self.num_top_positions) * 2
        self.num_actions = len(env.action_space("drone0"))

        self.nn = self.create_neural_network()
        self.optimizer = keras.optimizers.Adam(lr=self.lr)

    def create_neural_network(self):
        nn = keras.models.Sequential(
            [
                keras.layers.Dense(
                    512, activation="relu", input_shape=(self.num_entries,)
                ),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(self.num_actions, activation="softmax"),
            ]
        )
        return nn

    def select_actions(self, obs_list):
        episode_actions = {}

        for drone_index in range(self.num_agents):
            obs = np.expand_dims(obs_list[drone_index], axis=0)
            probs = self.nn.predict(obs)[0]
            action = np.random.choice(self.num_actions, p=probs)
            episode_actions[f"drone{drone_index}"] = action

        return episode_actions

    def extract_rewards(self, reward_dict):
        rewards = [
            drone_reward for key, drone_reward in reward_dict.items() if "drone" in key
        ]
        return rewards

    def print_episode_stats(self, episode_num, show_actions, show_rewards):
        print(
            f"Up to episode = {episode_num}, Actions (mean) = {sum(show_actions) / len(show_actions)}, "
            f"Reward (mean) = {sum(show_rewards) / len(show_rewards)}"
        )

    def calculate_discounted_returns(self, rewards):
        discounted_returns = []
        for t in range(len(rewards)):
            G_list = []
            for drone_index in range(self.num_agents):
                agent_rewards = [r[drone_index] for r in rewards]
                G_list.append(
                    sum((self.y**k) * r for k, r in enumerate(agent_rewards[t:]))
                )
            discounted_returns.append(G_list)

        return discounted_returns

    def update_neural_network(self, states, actions, discounted_returns):
        for state_list, action_list, G_list in zip(states, actions, discounted_returns):
            for drone_index in range(self.num_agents):
                obs = np.expand_dims(state_list[drone_index], axis=0)

                with tf.GradientTape() as tape:
                    logits = self.nn(obs)
                    action = action_list[drone_index]
                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=[action]
                    )
                    loss = -log_prob * G_list[drone_index]

                gradients = tape.gradient(loss, self.nn.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.nn.trainable_variables)
                )

    def train(self):
        statistics, show_rewards, show_actions, all_rewards = [], [], [], []
        stop = False

        for i in range(self.episodes + 1):
            if stop:
                break

            vector = self.get_random_speed_vector()
            state = self.env.reset(
                drones_positions=self.drones_initial_positions, vector=vector
            )
            obs_list = self.flatten_state(state)
            done = False
            actions, states, rewards = [], [], []
            count_actions, total_reward = 0, 0

            while not done:
                episode_actions = self.select_actions(obs_list)
                obs_list_, reward_dict, _, done, infos = self.env.step(episode_actions)

                if infos["Found"]:
                    pass

                actions.append(np.array(list(episode_actions.values())))
                states.append(obs_list)
                rewards.append(self.extract_rewards(reward_dict))
                obs_list = self.flatten_state(obs_list_)
                count_actions += self.num_agents
                total_reward += reward_dict["total_reward"]
                done = any(done.values())

            show_rewards.append(total_reward)
            all_rewards.append(total_reward)
            show_actions.append(count_actions)

            if len(all_rewards) > 100:
                if all([r >= 100000 for r in all_rewards[-80:]]):
                    stop = True
                    print("Acabou mais cedo")

            if i % 100 == 0:
                self.print_episode_stats(i, show_actions, show_rewards)
                show_rewards, show_actions = [], []

            statistics.append([i, count_actions, total_reward])
            discounted_returns = self.calculate_discounted_returns(rewards)
            self.update_neural_network(states, actions, discounted_returns)

        return self.nn, statistics


if __name__ == "__main__":
    config = get_config(2)

    env = DroneSwarmSearch(
        grid_size=config.grid_size,
        render_mode="ansi",
        render_grid=False,
        render_gradient=False,
        n_drones=config.n_drones,
        person_initial_position=config.person_initial_position,
        disperse_constant=config.disperse_constant,
        timestep_limit=100,
    )

    rl_agent = ReinforceAgent(
        env,
        y=0.999999,
        lr=0.000001,
        episodes=100_000,
        drones_initial_positions=config.drones_initial_positions,
    )
    nn, statistics = rl_agent.train()

    nn.save(f"data/nn_{config.grid_size}_{config.grid_size}_{config.n_drones}.h5")
    df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
    df.to_csv(
        f"data/statistics_{config.grid_size}_{config.grid_size}_{config.n_drones}.csv",
        index=False,
    )
