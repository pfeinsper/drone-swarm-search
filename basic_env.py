from DSSE import DroneSwarmSearch
from DSSE import Actions

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(3.2, 3.1),
    timestep_limit=200,
    person_amount=2,
    dispersion_inc=0.1,
    person_initial_position=(10, 10),
    drone_amount=1,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time = 0,
)

def policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = Actions.SEARCH.value # value: int = 8
    return actions

opt = {
    "drones_positions": [(0, 10)],
    "individual_pods": [1, 0.5]
}
observations, info = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    done = any(done.values())
