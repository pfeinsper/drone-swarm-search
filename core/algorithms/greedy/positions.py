from numpy import array
from typing import List, Tuple
import sys

sys.path.append("../../..")
from core.algorithms.greedy.movements import Movements


def create_movement_matrix(
    matrix: array, instructions: List[Movements]
) -> Tuple[array, int]:
    total_cost = 0
    movement_cost = 5
    search_cost = 4 * movement_cost
    current_position = [0, 0]
    all_moviments = [array(matrix.copy())]
    for instruction in instructions:
        current_value = matrix[current_position[0]][current_position[1]]
        matrix[current_position[0]][current_position[1]] = (
            "0" if current_value == "X" else "P"
        )
        if instruction == Movements.DOWN:
            total_cost += movement_cost
            current_position[0] += 1
        elif instruction == Movements.UP:
            total_cost += movement_cost
            current_position[0] -= 1
        elif instruction == Movements.RIGHT:
            total_cost += movement_cost
            current_position[1] += 1
        elif instruction == Movements.LEFT:
            total_cost += movement_cost
            current_position[1] -= 1
        elif instruction == Movements.DIAGONAL_DOWN_RIGHT:
            total_cost += movement_cost
            current_position[0] += 1
            current_position[1] += 1
        elif instruction == Movements.DIAGONAL_DOWN_LEFT:
            total_cost += movement_cost
            current_position[0] += 1
            current_position[1] -= 1
        elif instruction == Movements.DIAGONAL_UP_RIGHT:
            total_cost += movement_cost
            current_position[0] -= 1
            current_position[1] += 1
        elif instruction == Movements.DIAGONAL_UP_LEFT:
            total_cost += movement_cost
            current_position[0] -= 1
            current_position[1] -= 1
        elif instruction == Movements.SEARCH:
            total_cost += search_cost
            if matrix[current_position[0]][current_position[1]] == "P":
                matrix[current_position[0]][current_position[1]] = "F"
                all_moviments.append(array(matrix.copy()))
                break

        updated_value = matrix[current_position[0]][current_position[1]]
        matrix[current_position[0]][current_position[1]] = (
            "X" if updated_value == "0" else "PX"
        )
        all_moviments.append(array(matrix.copy()))
    array(all_moviments)
    return all_moviments, total_cost