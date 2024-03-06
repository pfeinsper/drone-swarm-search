from DSSE import DroneSwarmSearch
from DSSE import Actions
# from DSSE import DroneData

env = DroneSwarmSearch(
    grid_size=20,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    n_drones=2,
    vector=[-0.2, 0],
    person_initial_position=(19, 19),
    disperse_constant=1,
    # drone_data=DroneData(
    #     speed=10,
    #     sweep_width=5,
    #     track_spacing=5
    # ),
)


def policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = Actions.SEARCH.value # value: int = 8
    return actions


observations = env.reset(drones_positions=[(0, 10), (0, 11)])

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = any(done.values())
