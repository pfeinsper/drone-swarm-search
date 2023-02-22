import pytest

from core.calculate_array_movements import (
    calculate_movements,
    calculate_path,
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
                Movements.DIAGONAL_UP_LEFT,
                Movements.DIAGONAL_UP_LEFT,
                Movements.DIAGONAL_UP_LEFT,
            ],
            (1, 1),
        ),
        (
            (4, 4),
            (0, 5),
            [
                Movements.DIAGONAL_UP_RIGHT,
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


@pytest.mark.parametrize(
    "current_point, next_point, expected_movements",
    [
        (
            (4, 4),
            (4, 5),
            [
                Movements.RIGHT,
            ],
        ),
        (
            (4, 4),
            (4, 3),
            [
                Movements.LEFT,
            ],
        ),
        (
            (4, 4),
            (5, 4),
            [
                Movements.DOWN,
            ],
        ),
        (
            (4, 4),
            (3, 4),
            [
                Movements.UP,
            ],
        ),
        (
            (4, 4),
            (0, 1),
            [
                Movements.DIAGONAL_UP_LEFT,
                Movements.DIAGONAL_UP_LEFT,
                Movements.DIAGONAL_UP_LEFT,
                Movements.UP,
            ],
        ),
        (
            (4, 4),
            (0, 5),
            [
                Movements.DIAGONAL_UP_RIGHT,
                Movements.UP,
                Movements.UP,
                Movements.UP,
            ],
        ),
    ],
)
def test_calculate_path(current_point, next_point, expected_movements):
    """Test the calculate_path function."""
    assert calculate_path(current_point, next_point) == expected_movements


# Mock the movements sequence entries
@pytest.mark.parametrize(
    "movements_sequence, expected_movements",
    [
        (
            [(0, 0), (3, 1), (2, 2), (0, 3)],
            [
                Movements.DIAGONAL_DOWN_RIGHT,
                Movements.DOWN,
                Movements.DOWN,
                Movements.SEARCH,
                Movements.DIAGONAL_UP_RIGHT,
                Movements.SEARCH,
                Movements.DIAGONAL_UP_RIGHT,
                Movements.UP,
                Movements.SEARCH,
            ],
        ),
    ],
)
def test_calculate_movements(movements_sequence, expected_movements):
    """Test the calculate_movements function."""
    assert calculate_movements(movements_sequence) == expected_movements
