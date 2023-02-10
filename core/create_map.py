import numpy as np
from random import randint


def create_map(matrix: np.array) -> np.array:

    probabilities: list = [
        probability for line in matrix for probability in line
    ]
    probabilities_times_random_factor: list = [
        (randint(1, 100) * probability) / 100 for probability in probabilities
    ]

    person_position = probabilities_times_random_factor.index(
        max(probabilities_times_random_factor)
    )

    position_matrix: np.array = [
        [0 for _ in range(0, matrix[0].size)] for _ in matrix
    ]

    position_matrix[person_position // matrix[0].size][
        (person_position % len(matrix))
    ] = "P"

    position_matrix[0][0] = "X"

    return position_matrix
