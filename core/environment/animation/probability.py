import matplotlib.pyplot as plt
from numpy import array


def animate_probability(matrix: array):
    fig, ax = plt.subplots(figsize=(10, 10))
    mat = ax.matshow(matrix, vmin=0, vmax=0.05)
    fig.colorbar(mat)
    plt.show()
