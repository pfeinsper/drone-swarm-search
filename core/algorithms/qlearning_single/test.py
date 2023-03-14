from pettingzoo.mpe import simple_v2
import numpy as np


def get_state(state):
    range = 50
    state_adj = state * np.array([10, 10, 10, 10])
    state_adj = np.round(state_adj, 0).astype(int)
    x, y, vx, vy = state_adj + np.array([range, range, range, range])
    return x, y, vx, vy


def test_n_times(n_times, q_table, agent):
    env = simple_v2.parallel_env(
        max_cycles=30, continuous_actions=False, render_mode="human"
    )

    rewards = 0

    for _ in range(0, n_times):
        state = env.reset()[agent]
        state = get_state(state)
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, _, done, _ = env.step({agent: action})

            state = state[agent]
            state = get_state(state)
            reward = reward[agent]
            done = done[agent]

            env.render()

        rewards += reward
    return rewards / 1000
