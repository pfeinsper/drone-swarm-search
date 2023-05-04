from core.algorithms.greedy.path import generate_path
from numpy import array


def test_order():
    assert generate_path(array([[10, 22, 13, 15], [18, 15, 10, 12]])) == [
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 3),
        (0, 2),
        (1, 3),
        (1, 2),
        (0, 0),
    ]
    assert generate_path(
        array([[7, 22, 15, 15], [18, 15, 10, 44], [7, 22, 35, 28]])
    ) == [
        (1, 3),
        (2, 2),
        (2, 3),
        (2, 1),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 0),
        (0, 0),
    ]
    assert generate_path(
        array([[10, 22, 13, 15, 7, 9], [18, 15, 10, 12, 1, 2]])
    ) == [
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 3),
        (0, 2),
        (1, 3),
        (1, 2),
        (0, 0),
        (0, 5),
        (0, 4),
        (1, 5),
        (1, 4),
    ]
