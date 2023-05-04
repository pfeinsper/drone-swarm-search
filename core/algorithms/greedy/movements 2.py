from typing import List, Tuple
from enum import Enum

POINT_TYPE = Tuple[int, int]


class Movements(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DIAGONAL_UP_LEFT = "DIAGONAL UP LEFT"
    DIAGONAL_UP_RIGHT = "DIAGONAL UP RIGHT"
    DIAGONAL_DOWN_LEFT = "DIAGONAL DOWN LEFT"
    DIAGONAL_DOWN_RIGHT = "DIAGONAL DOWN RIGHT"
    SEARCH = "SEARCH"


diagonal_translator = {
    (Movements.UP, Movements.LEFT): (Movements.DIAGONAL_UP_LEFT, (-1, -1)),
    (Movements.UP, Movements.RIGHT): (Movements.DIAGONAL_UP_RIGHT, (-1, 1)),
    (Movements.DOWN, Movements.LEFT): (Movements.DIAGONAL_DOWN_LEFT, (1, -1)),
    (Movements.DOWN, Movements.RIGHT): (Movements.DIAGONAL_DOWN_RIGHT, (1, 1)),
}


parallel_translator = {
    Movements.UP: (Movements.UP, (-1, 0)),
    Movements.DOWN: (Movements.DOWN, (1, 0)),
    Movements.LEFT: (Movements.LEFT, (0, -1)),
    Movements.RIGHT: (Movements.RIGHT, (0, 1)),
}


def calculate_diagonal_movements(
    current_point: POINT_TYPE, next_point: POINT_TYPE
) -> Tuple[List[Movements], POINT_TYPE]:
    """Calculate and return the DIAGONAL movements and current position."""

    distance_line = next_point[0] - current_point[0]
    distance_column = next_point[1] - current_point[1]

    movement_line = Movements.DOWN if distance_line > 0 else Movements.UP
    movement_column = Movements.RIGHT if distance_column > 0 else Movements.LEFT
    diagonal_movement, points_to_move = diagonal_translator[
        (movement_line, movement_column)
    ]

    number_of_movements = min(abs(distance_line), abs(distance_column))
    movements = [diagonal_movement] * number_of_movements
    position_after_movements = (
        current_point[0] + points_to_move[0] * number_of_movements,
        current_point[1] + points_to_move[1] * number_of_movements,
    )

    return (movements, position_after_movements)


def calculate_path(
    current_point: POINT_TYPE, next_point: POINT_TYPE
) -> List[Movements]:
    """Calculate the path between two points."""
    path: List[Movements] = []
    distance_line = next_point[0] - current_point[0]
    distance_column = next_point[1] - current_point[1]

    # Check if the points are not in the same line or column before calculating the diagonal movements
    if current_point[0] != next_point[0] or current_point[1] != next_point[1]:
        (
            diagonal_movements,
            point_after_diagonal,
        ) = calculate_diagonal_movements(current_point, next_point)
        path.extend(diagonal_movements)

        if point_after_diagonal == next_point:
            return path

        distance_line = next_point[0] - point_after_diagonal[0]
        distance_column = next_point[1] - point_after_diagonal[1]

    if distance_line != 0:
        movement_line = Movements.DOWN if distance_line > 0 else Movements.UP
        number_of_movements = abs(distance_line)
        path.extend([movement_line] * number_of_movements)
    else:
        movement_column = Movements.RIGHT if distance_column > 0 else Movements.LEFT
        number_of_movements = abs(distance_column)
        path.extend([movement_column] * number_of_movements)

    return path


def calculate_movements(
    movements_sequence: List[POINT_TYPE],
) -> List[Movements]:
    """Calculate movements from a sequence of coordinates."""

    movements: List[Movements] = []

    for current_point, next_point in zip(movements_sequence, movements_sequence[1:]):
        path_to_next_point = calculate_path(current_point, next_point)
        movements.extend(path_to_next_point)
        movements.append(Movements.SEARCH)

    return movements
