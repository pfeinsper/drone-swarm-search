from DSSE import DroneSwarmSearch
<<<<<<< HEAD

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
=======
import pygame as pg

env = DroneSwarmSearch(
    grid_size=20,
    render_mode="human",
    render_grid=False,
    render_gradient=False,
>>>>>>> 17c652b (Finished everything)
    vector=(1, 1),
    timestep_limit=300,
    person_amount=4,
    dispersion_inc=0.05,
<<<<<<< HEAD
    person_initial_position=(15, 15),
    drone_amount=2,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time=0,
=======
    person_initial_position=(1, 1),
    drone_amount=4,
    drone_speed=10,
    probability_of_detection=1,
    pre_render_time=0,
    fps=5
>>>>>>> 17c652b (Finished everything)
)


def random_policy(obs, agents):
    actions = {}
    for agent in agents:
        actions[agent] = env.action_space(agent).sample()
    return actions


opt = {
<<<<<<< HEAD
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
    "vector": (0.3, 0.3),
}
observations, info = env.reset(options=opt)

rewards = 0
done = False
while not done:
    actions = random_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
=======
    "vector": (0.1, 0.5),
}
observations, info = env.reset(options=opt)

clock = pg.time.Clock()

rewards = 0
done = False
while not done:
    clock.tick()
    actions = random_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
    print(round(clock.get_fps()), end="\r")
>>>>>>> 17c652b (Finished everything)
