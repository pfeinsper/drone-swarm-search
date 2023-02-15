import matplotlib.pyplot as plt
from numpy import array


def animate_probability(matrix: array):
    length = len(matrix)
    _, ax = plt.subplots()

    for i in range(length):
        for j in range(length):
            _ = ax.text(
                j, i, matrix[i, j], ha="center", va="center", color="w"
            )

    ax.matshow(matrix)
    plt.show()
