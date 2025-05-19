from DSSE import DroneSwarmSearch

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(1, 1),
    timestep_limit=300,
    person_amount=4,
    dispersion_inc=0.05,
    person_initial_position=(15, 15),
    drone_amount=2,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time=0,
)


def random_policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = env.action_space(agent).sample()
    return actions


opt = {
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
    "vector": (0.3, 0.3),
}
observations, info = env.reset(options=opt)

episode_rewards = []

for episode in range(1000):
    # Reset the environment before starting each new episode
    observations, info = env.reset(options=opt)
    done = False
    total_reward = 0

    # Run one episode
    while not done:
        actions = random_policy(observations, env.get_agents())
        observations, rewards, terminations, truncations, infos = env.step(actions)

        total_reward += sum(rewards.values())

        done = any(terminations.values()) or any(truncations.values())

    # Store the total reward of the episode
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

sorted_rewards = sorted(episode_rewards, reverse=True)

print("\nSorted Rewards (from largest to smallest):")
for idx, reward in enumerate(sorted_rewards, 1):
    print(f"Episode {idx}: Reward = {reward}")
