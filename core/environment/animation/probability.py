import matplotlib.pyplot as plt
from numpy import array


def animate_probability(matrix: array):
    length = len(matrix)
    fig, ax = plt.subplots(figsize=(10, 10))

    # for i in range(length):
    #     for j in range(length):
    #         _ = ax.text(
    #             j, i, matrix[i, j], ha="center", va="center", color="w"
    #         )
    ax.matshow(matrix)
    plt.show()
