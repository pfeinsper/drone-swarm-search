from core.environment.env import DroneSwarmSearch
import numpy as np

env = DroneSwarmSearch(
    grid_size=50,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    n_drones=1,
    vector=[0.3, 0.3],
    person_initial_position=[2, 2],
    disperse_constant=5,
)


def policy(obs, agent):
    actions = {}
    for i in range(11):
        # actions["drone{}".format(i)] = np.random.randint(5)
        actions["drone{}".format(i)] = 4
    return actions


observations = env.reset(drones_positions=[[25, 25]])

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False
    # print(reward["total_reward"])
