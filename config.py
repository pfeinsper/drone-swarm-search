import os

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")


class Config(object):
    qtable_path = "results/qtable.pkl"
    plot_file_path = "results/qtable.png"
