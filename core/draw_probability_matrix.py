import matplotlib.pyplot as plt
import numpy as np


def draw_matrix(matrix):
    length = len(matrix)
    fig, ax = plt.subplots()
    for i in range(length):
        for j in range(length):
            text = ax.text(
                j, i, matrix[i][j], ha="center", va="center", color="w"
            )
    # ax.matshow(matrix, cmap=plt.cm.Spectral)
    ax.matshow(matrix)
    plt.show()
