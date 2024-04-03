from DSSE import DroneSwarmSearch
from DSSE import Actions

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(3.2, 3.1),
    disperse_constant=5,
    timestep_limit=200,
    person_amount=5,
    person_initial_position=(10, 10),
    drone_amount=2,
    drone_speed=10,
    drone_probability_of_detection=0.9,
    pre_render_time = 0,
)

def policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = Actions.SEARCH.value # value: int = 8
        # actions[agent] = Actions.RIGHT.value
    return actions

opt = {
    "drones_positions": [(10, 10), (0, 11)],
}
observations = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += sum(reward.values())
    done = any(done.values())
