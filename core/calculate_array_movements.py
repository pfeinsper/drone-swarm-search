from typing import Tuple, List, Enum


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


def calculate_movements(movements_sequence: List[Tuple[int, int]]) -> List[Movements]:
    """Calculate movements from a sequence of coordinates."""
    movements: List[Movements] = []

    return movements
