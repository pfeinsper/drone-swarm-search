from pettingzoo.mpe import simple_v2
from qlearning_box import QLearningBox

from test import test_n_times


parallel_env = simple_v2.parallel_env(max_cycles=30, continuous_actions=False)
observations = parallel_env.reset()

for agent in parallel_env.agents:
    parallel_env.reset()
    qlearn = QLearningBox(
        env=parallel_env,
        agent=agent,
        alpha=0.2,
        gamma=0.4,
        epsilon=0.7,
        epsilon_min=0.05,
        epsilon_dec=0.99,
        episodes=10000,
    )
    train_dir = "core/algorithms/QLeaning/"
    q_table = qlearn.train(
        filename=f"{train_dir}/qtable_{agent}.csv",
        plotFile=f"{train_dir}/qtable_{agent}",
    )

    mean_reward = test_n_times(1, q_table, agent)
    print(f"Final reward:{mean_reward}")
