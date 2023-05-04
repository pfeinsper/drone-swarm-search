from core.algorithms.deep_qlearning.deep_qlearning import DQN
from core.environment.env import DroneSwarmSearch


n_episodes = 300
n_steps = 50
env = DroneSwarmSearch(
    grid_size=5,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    n_drones=1,
    vector=[0.5, 0.5],
    person_initial_position=[1, 1],
    disperse_constant=5,
)
dqn = DQN(env, load_weights=True, n_episodes=n_episodes, n_steps=n_steps)
dqn.play()
