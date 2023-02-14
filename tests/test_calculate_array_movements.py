import pytest

from core.calculate_array_movements import (
    calculate_movements,
    calculate_diagonal_movements,
    Movements,
)


@pytest.mark.parametrize(
    "current_point, next_point, expected_movements, expected_position",
    [
        (
            (4, 4),
            (0, 1),
            [
                Movements.DIOGONAL_UP_LEFT,
                Movements.DIOGONAL_UP_LEFT,
                Movements.DIOGONAL_UP_LEFT,
            ],
            (1, 1),
        ),
        (
            (4, 4),
            (0, 5),
            [
                Movements.DIOGONAL_UP_RIGHT,
            ],
            (3, 5),
        ),
    ],
)
def test_calculate_diagonal_movements(
    current_point, next_point, expected_movements, expected_position
):
    """Test the calculate_diagonal_movements function."""
    movements, position = calculate_diagonal_movements(current_point, next_point)
    assert movements == expected_movements
    assert position == expected_position


# Mock the movements sequence entries
@pytest.mark.parametrize(
    "movements_sequence, expected_movements",
    [
        (
            [(0, 0), (3, 1), (2, 2), (3, 3)],
            [
                Movements.SEARCH,
                Movements.DIOGONAL_DOWN_RIGHT,
                Movements.DOWN,
                Movements.DOWN,
                Movements.SEARCH,
                Movements.DIOGONAL_UP_RIGHT,
                Movements.SEARCH,
                Movements.DIOGONAL_UP_RIGHT,
                Movements.SEARCH,
            ],
        ),
    ],
)
def test_calculate_movements(movements_sequence, expected_movements):
    """Test the calculate_movements function."""
    assert calculate_movements(movements_sequence) == expected_movements
