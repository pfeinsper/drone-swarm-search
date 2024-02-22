"""
Example of how to use the environment with a simple policy
"""

import sys
import numpy as np
from core.environment.env import DroneSwarmSearch

ACTION_SPACE = {
            (-1, 0): 0,  # Move Left
            (1, 0): 1,   # Move Right
            (0, -1): 2,  # Move Up
            (0, 1): 3,   # Move Down
            (-1, -1): 4, # Diagonal Up Left
            (1, -1): 5,  # Diagonal Up Right
            (-1, 1): 6,  # Diagonal Down Left
            (1, 1): 7,   # Diagonal Down Right
            (0, 0): 8    # Search Cell
        }

def calculate_direction(movement_vector):
    """
    Calculate the direction based on the movement vector
    """
    if np.array_equal(movement_vector, [0, 0]):
        return 0, 0
    # Convert movement vector to direction
    angle = np.arctan2(movement_vector[1], movement_vector[0])
    # Calculate cosine and sine values
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    # Determine direction based on the sign of cosine and sine
    x_direction = np.sign(cos_val) if abs(cos_val) > 0.0001 else 0
    y_direction = np.sign(sin_val) if abs(sin_val) > 0.0001 else 0
    return x_direction, y_direction

def policy(obs, agents):
    """
    Simple policy that moves the drones to the right
    """
    actions = {}
    for agent in agents:
        actions[agent] = np.random.randint(5)
        # actions[f"drone{i}"] = 4
    return actions

def policy2(obs, agents):
    """
    Simple policy that follows the max probability
    """
    actions = {}
    for agent in agents:
        agent_observation = obs[agent]["observation"]
        agent_position = agent_observation[0]
        probability_matrix = agent_observation[1]
        max_probability_index = np.unravel_index(
            indices=probability_matrix.argmax(),
            shape=probability_matrix.shape,
            order="F"
        )
        movement_vector = np.array(max_probability_index) - agent_position
        x_direction, y_direction = calculate_direction(movement_vector)

        # Convert direction to action based on Action Space
        actions[agent] = ACTION_SPACE[(x_direction, y_direction)]
    return actions

def main():
    """
    Main function
    """

    env = DroneSwarmSearch(
        grid_size=50,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        n_drones=1,
        vector=[-0.2, 0],
        person_initial_position=[19, 19],
        disperse_constant=7,
    )

    observations = env.reset(drones_positions=[[0, 10]])

    rewards = 0
    done = False
    while not done:
        actions = policy2(observations, env.get_agents())
        observations, reward, termination, done, info = env.step(actions)
        rewards += reward["total_reward"]
        done = True in list(done.values())

    if done:
        env.close()
        print(f"Total reward: {rewards}")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
