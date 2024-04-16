from DSSE import CoverageDroneSwarmSearch

env = CoverageDroneSwarmSearch(
    grid_size=40,
    drone_amount=3,
    dispersion_inc=0.1,
    vector=(1, 1),
    render_mode="human",
)

opt = {
    "drones_positions": [(0, 10), (10, 10), (20, 10)],
}
obs, info = env.reset(options=opt)

step = 0
while env.agents:
    step += 1
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print(infos["drone0"])
