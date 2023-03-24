import os

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")


class Config(object):
    qtable_path = "results/qtable.pkl"
    plot_file_path = "results/qtable.png"
    plot_comparison_file_path = "results/comparison.png"
    grid_size = 16
    alpha = 0.2
    gamma = 0.4
    epsilon = 0.7
    epsilon_min = 0.05
    epsilon_dec = 0.99
    episodes = 10000
