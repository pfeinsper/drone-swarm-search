from DSSE import CoverageDroneSwarmSearch

env = CoverageDroneSwarmSearch(
    drone_amount=3,
    render_mode="human",
    disaster_position=(-24.04, -46.17),  # (lat, long)
    pre_render_time=10, # hours to simulate
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
