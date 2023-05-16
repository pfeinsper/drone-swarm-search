from core.environment.env import DroneSwarmSearch
from config import get_config
from tensorflow import keras

config = get_config(3)


def get_actions(self, state):
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


env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="human",
    render_grid=True,
    render_gradient=False,
    n_drones=config.n_drones,
    vector=config.vector,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
)
model = keras.models.load_model(
    f"results/dqn_{env.grid_size}_{env.grid_size}_{env.n_drones}.png"
)
print(model.summary())
keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


obs = env.reset()
done = False
while not done:
    env.render()
    action = get_actions(obs)
    print(action)
    obs, _, done, _, _ = env.step(action)
    done = all(done.values())
