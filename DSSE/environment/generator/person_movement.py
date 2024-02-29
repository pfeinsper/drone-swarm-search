from typing import Tuple
from random import randint, random
from numpy import array, unravel_index, argmax
import numpy as np


def noise_person_movement(
    current_movement: Tuple[int], drift_vector: list[int], epsilon=1.0
) -> Tuple[int]:
    chance = random()
    if chance < epsilon:
        randomized_movement = array([randint(-1, 1), randint(-1, 1)])
        angle = angle_between(randomized_movement, drift_vector)
        # Only noises the movement if the new movement isnt against the vector.
        if angle < 120 or angle > 240:
            return randomized_movement
    return current_movement


def angle_between(movement: np.array, drift_vector: list[int]) -> float:
    direction_movement = get_unit_vector(movement)
    direction_vector = get_unit_vector(array(drift_vector))
    dot_product = np.dot(direction_movement, direction_vector)
    return np.degrees(np.arccos(dot_product))

def get_unit_vector(original_vector: np.array) -> float:
    vector_norm = np.linalg.norm(original_vector)
    if vector_norm == 0.0:
        vector_norm = 1.0
    return original_vector / vector_norm


def update_shipwrecked_position(probability_matrix: np.array) -> Tuple[int]:
    """
    Function that takes a 3x3 cut of the DynamicProbability matrix, multiplies it by a random numbers matrix [0, 1],
    and returns the column and line of the highest probability on the resulting matrix.

    Output:
        (movement_x, movement_y): tuple[int]
    """
    random_numbers_matrix = np.random.rand(*probability_matrix.shape)
    probabilities_mult_random_factor = random_numbers_matrix * probability_matrix

    # Using a numpy function to find the line and column of the greatest probability in the random factor multiplied matrix.
    max_probabilities = unravel_index(
        argmax(probabilities_mult_random_factor, axis=None), probability_matrix.shape
    )
    max_line = max_probabilities[0]
    max_column = max_probabilities[1]

    return movement_to_cartesian(max_column, max_line)


def movement_to_cartesian(mov_x: int, mov_y: int) -> Tuple[int]:
    """
    The movement of the shipwrecked person on the input follows the scheme (for the value of line and column):
        - if 0 -> Move to the left (x - 1) or to the top (y - 1).
        - if 1 -> No movement.
        - if 2 -> Move to the right (x + 1) or to the bottom (y + 1).

    So this function converts from this matrix movement notation to cartesian, as the matrix that creates this indexes is only 3x3,
    just removing 1 converts it back to cartesian movement.
    """
    x_component = mov_x - 1
    y_component = mov_y - 1
    return x_component, y_component
