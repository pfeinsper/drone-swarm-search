from DSSE import DroneSwarmSearch
from DSSE import Actions
from DSSE import DroneData
from DSSE import PersonData

drone_data = DroneData(
    number_of_drones=2,
    speed=10,
    sweep_width=5,
    track_spacing=5
)

person_data = PersonData(
    number_of_persons=1,
    initial_position=(19, 19)
)

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=[-0.2, 0],
    disperse_constant=7,
    person=person_data,
    drone=drone_data,
)


def policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = Actions.SEARCH.value # value: int = 8
        # actions[agent] = Actions.RIGHT.value
    return actions


observations = env.reset(drones_positions=[(0, 10), (0, 11)])

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = any(done.values())
