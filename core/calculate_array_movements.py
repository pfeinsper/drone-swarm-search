from typing import List, Tuple
from enum import Enum

POINT_TYPE = Tuple[int, int]


class Movements(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DIOGONAL_UP_LEFT = "DIOGONAL UP LEFT"
    DIOGONAL_UP_RIGHT = "DIOGONAL UP RIGHT"
    DIOGONAL_DOWN_LEFT = "DIOGONAL DOWN LEFT"
    DIOGONAL_DOWN_RIGHT = "DIOGONAL DOWN RIGHT"
    SEARCH = "SEARCH"


diagonal_translator = {
    (Movements.UP, Movements.LEFT): (Movements.DIOGONAL_UP_LEFT, (-1, -1)),
    (Movements.UP, Movements.RIGHT): (Movements.DIOGONAL_UP_RIGHT, (-1, 1)),
    (Movements.DOWN, Movements.LEFT): (Movements.DIOGONAL_DOWN_LEFT, (1, -1)),
    (Movements.DOWN, Movements.RIGHT): (Movements.DIOGONAL_DOWN_RIGHT, (1, 1)),
}


def calculate_diagonal_movements(
    current_point: POINT_TYPE, next_point: POINT_TYPE
) -> Tuple[List[Movements], POINT_TYPE]:
    """Calculate and return the diogonal movements and current position."""

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
    calculate_diagonal_movements(current_point, next_point)

    return path


def calculate_movements(movements_sequence: List[POINT_TYPE]) -> List[Movements]:
    """Calculate movements from a sequence of coordinates."""
    movements: List[Movements] = []

    for current_point, next_point in zip(movements_sequence, movements_sequence[1:]):
        path_to_next_point = calculate_path(current_point, next_point)

    return movements
