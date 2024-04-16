from DSSE import DroneSwarmSearch
from DSSE import Actions
from DSSE.tests.drone_policy import policy

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(1, 1),
    timestep_limit=300,
    person_amount=4,
    dispersion_inc=0.05,
    person_initial_position=(10, 10),
    drone_amount=2,
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
    "drones_positions": [(10, 0), (10, 10)],
    "individual_multiplication": [0.1, 0.4, 0.5, 1.2],
}
observations, info = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    done = any(done.values())
