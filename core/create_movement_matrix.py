from numpy import array
from enum import Enum
import sys
sys.path.append("..")
from core.calculate_array_movements import Movements

def create_movement_matrix(matrix: array, instructions: list) -> "tuple[array, int]":
    time = 0
    current_position = [0, 0]
    all_moviments = [array(matrix.copy())]
    for instruction in instructions:
        current_value = matrix[current_position[0]][current_position[1]]
        matrix[current_position[0]][current_position[1]] = (
            "0" if current_value == "X" else "P"
        )
        if instruction == Movements.DOWN:
            time += 5
            current_position[0] += 1
        elif instruction == Movements.UP:
            time += 5
            current_position[0] -= 1
        elif instruction == Movements.RIGHT:
            time += 5
            current_position[1] += 1
        elif instruction == Movements.LEFT:
            time += 5
            current_position[1] -= 1
        elif instruction == Movements.DIAGONAL_DOWN_RIGHT:
            time += 5
            current_position[0] += 1
            current_position[1] += 1
        elif instruction == Movements.DIAGONAL_DOWN_LEFT:
            time += 5
            current_position[0] += 1
            current_position[1] -= 1
        elif instruction == Movements.DIAGONAL_UP_RIGHT:
            time += 5
            current_position[0] -= 1
            current_position[1] += 1
        elif instruction == Movements.DIAGONAL_UP_LEFT:
            time += 5
            current_position[0] -= 1
            current_position[1] -= 1
        elif instruction == Movements.SEARCH:
            time += 20
            if matrix[current_position[0]][current_position[1]] == "P":
                matrix[current_position[0]][current_position[1]] = "F"
                all_moviments.append(array(matrix.copy()))
                break

        updated_value = matrix[current_position[0]][current_position[1]]
        matrix[current_position[0]][current_position[1]] = (
            'X' if updated_value == '0' else 'PX'
        )
        all_moviments.append(array(matrix.copy()))
    array(all_moviments)
    return all_moviments, time
