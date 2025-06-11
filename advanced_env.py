from DSSE import AdvancedCoverageDroneSwarmSearch, AdvancedActions
import pygame as pg
from datetime import timedelta
import numpy as np

# Here we create an instance of the AdvancedCoverageDroneSwarmSearch environment
# We use the H5 dataset that we created in example_creating_dataset.py
env = AdvancedCoverageDroneSwarmSearch(
    dataset_pth="DSSE/environment/TrajectoryDatasets/sample_dataset.h5",
    drone_amount=5, 
    drone_speed=10, 
    drone_height=70, 
    survival_time=timedelta(hours=24),
    drone_fov=90,
    grid_cell_size=1000, 
    pre_render_time=timedelta(milliseconds=0),
    render_gradient=True,
    render_grid=False,
    render_mode="human",
    render_fps=3)

# This policy is a greedy policy that chooses the action that gets the agent closer to the cell with the highest probability
# It is a simple heuristic that can be used to test the environment
def greedy_policy(obs, agents):
    actions = {}
    for agent in agents:        
        # If the agents are on top of each other, it will choose a random action to avoid clumping
        if obs[agent][0] in [obs[a][0] for a in agents if a != agent]:
            actions[agent] = env.action_space(agent).sample()
            continue

        # This policy choses the action that gets the agent closer to the cell with the highest probability
        prob_matrix = obs[agent][1]
        max_prob = prob_matrix.max()
        if max_prob == 0:
            actions[agent] = env.action_space(agent).sample()
            continue
        max_pos = tuple(zip(*np.where(prob_matrix == max_prob)))[0]
        agent_pos = obs[agent][0]
        if max_pos[0] < agent_pos[0]:
            if max_pos[1] < agent_pos[1]:
                actions[agent] = AdvancedActions.UP_LEFT.value
            elif max_pos[1] > agent_pos[1]:
                actions[agent] = AdvancedActions.UP_RIGHT.value
            else:
                actions[agent] = AdvancedActions.UP.value
        elif max_pos[0] > agent_pos[0]:
            if max_pos[1] < agent_pos[1]:
                actions[agent] = AdvancedActions.DOWN_LEFT.value
            elif max_pos[1] > agent_pos[1]:
                actions[agent] = AdvancedActions.DOWN_RIGHT.value
            else:
                actions[agent] = AdvancedActions.DOWN.value
        else:
            if max_pos[1] < agent_pos[1]:
                actions[agent] = AdvancedActions.LEFT.value
            elif max_pos[1] > agent_pos[1]:
                actions[agent] = AdvancedActions.RIGHT.value
            else:
                actions[agent] = env.action_space(agent).sample()
    return actions


opt = {
    "drones_positions": [(10,10), (9,9), (8, 8), (12, 11), (13, 13)]
}
observations, info = env.reset(options=opt)

clock = pg.time.Clock()

rewards = 0
done = False
while not done:
    clock.tick()
    actions = greedy_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
    print(f"FPS is: {round(clock.get_fps())}")
