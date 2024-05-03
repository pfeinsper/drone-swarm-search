from DSSE import CoverageDroneSwarmSearch

env = CoverageDroneSwarmSearch(
    drone_amount=1,
    render_mode="human",
    disaster_position=(-24.04, -46.17),  # (lat, long)
    pre_render_time=10, # hours to simulate
    prob_matrix_path="DSSE/tests/matrix.npy",
)

opt = {
    "drones_positions": [(0, 10)],
}
obs, info = env.reset(options=opt)

print(info)

step = 0
while env.agents:
    step += 1
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(infos)

print(infos["drone0"])
