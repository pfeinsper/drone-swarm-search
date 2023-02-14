import pytest

from core.calculate_array_movements import calculate_movements, Movements


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
