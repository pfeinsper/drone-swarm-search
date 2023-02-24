import sys

from core.environment.generator.probability import generate_probability_matrix
from core.environment.generator.map import generate_map

from core.environment.animation.probability import animate_probability
from core.environment.animation.draw_path import create_search_animation

from core.algorithms.greedy.path import generate_path
from core.algorithms.greedy.movements import calculate_movements
from core.algorithms.greedy.positions import create_movement_matrix


def greedy_search(MATRIX_SIZE: int):
    probability_matrix = generate_probability_matrix(MATRIX_SIZE, MATRIX_SIZE)

    environment_map = generate_map(probability_matrix)

    path = generate_path(probability_matrix)
    path = [(0, 0)] + path

    movements = calculate_movements(path)

    movement_matrix, total_cost = create_movement_matrix(environment_map, movements)

    animate_probability(probability_matrix)
    create_search_animation(movement_matrix)
    print(f"Total cost: {total_cost}")


if __name__ == "__main__":
    matrix_size = None

    try:
        matrix_size = int(sys.argv[1])
    except IndexError:
        matrix_size = 4

    greedy_search(matrix_size)
