from pettingzoo.mpe import simple_v2
import os
from pickle import load

from qlearning_box import BasicQLearningBox


if __name__ == "__main__":
    parallel_env = simple_v2.parallel_env(
        max_cycles=30, continuous_actions=False, render_mode="human"
    )
    parallel_env.reset()

    basic_qlearn = BasicQLearningBox(parallel_env.agents)

    # Test using pickle file
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "results/")
    with open(path + "qlearning.pkl", "rb") as f:
        Q = load(f)

    # Test using trained model
    for i in range(10):
        done = False
        tot_rewards = {agent: 0 for agent in parallel_env.agents}
        states = parallel_env.reset()

        while not done:
            # Select action
            state = basic_qlearn.transform_state(states)

            actions = {}
            for agent in parallel_env.agents:
                action = Q[agent][state[agent]].argmax()
                actions[agent] = action

            # Take action
            new_state, new_reward, _, new_done, _ = parallel_env.step(actions)
            done = all(new_done.values())

            # Update parameters
            states = new_state

            for agent in parallel_env.agents:
                tot_rewards[agent] += new_reward[agent]

            parallel_env.render()
