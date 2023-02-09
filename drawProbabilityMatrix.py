import matplotlib.pyplot as plt
import numpy as np

def drawMatrix(matrix):
    length = len(matrix)
    fig, ax = plt.subplots()
    for i in range(length):
        for j in range(length):
            text = ax.text(j, i, matrix[i][j],
                        ha="center", va="center", color="w")
    #fig.tight_layout()
    #plt.matshow(matrix, cmap=plt.cm.Spectral)
    ax.matshow(matrix)
    plt.show()


test = [[ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ],
        [ 0.1, 0.4, 0.1, 0.1, 0.1, 2.5, 0.1, 0.1, 0.1, 0.1 ],
        [ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1 ],
        [ 0.1, 7.7, 9.5, 0.1, 33.4, 0.1, 1.2, 0.1, 0.1, 0.3],
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.0,  0.1, 0.1], 
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 6.9, 0.1 ],
        [ 0.1, 13.8, 0.1, 0.1, 0.1, 0.2, 1.3, 0.1, 0.1, 0.1],
        [ 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.6, 0.1, 0.4, 0.1 ],
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 2.0, 0.1 ],
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]]

drawMatrix(test)