from core.algorithms.greedy.positions import create_movement_matrix
from core.algorithms.greedy.movements import Movements
from numpy import array

matrix = [
    ["X", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "P", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
]

matrix_small = [["X", "0", "0"], ["0", "P", "0"], ["0", "0", "0"]]


def test_one_movement():
    matrix_0, time_0 = create_movement_matrix(array(matrix), [Movements.DOWN])
    matrix_1, time_1 = create_movement_matrix(array(matrix), [Movements.RIGHT])
    matrix_2, time_2 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT]
    )
    matrix_3, time_3 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT, Movements.UP]
    )
    matrix_4, time_4 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT, Movements.LEFT]
    )
    matrix_5, time_5 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT, Movements.DIAGONAL_UP_LEFT]
    )
    matrix_6, time_6 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT, Movements.DIAGONAL_UP_RIGHT]
    )
    matrix_7, time_7 = create_movement_matrix(
        array(matrix), [Movements.DIAGONAL_DOWN_RIGHT, Movements.DIAGONAL_DOWN_LEFT]
    )
    matrix_8, time_8 = create_movement_matrix(array(matrix), [Movements.SEARCH])
    assert matrix_0[1][0][0] == "0" and matrix_0[1][1][0] == "X"
    assert time_0 == 5
    assert matrix_1[1][0][0] == "0" and matrix_1[1][0][1] == "X"
    assert time_1 == 5
    assert matrix_2[1][0][0] == "0" and matrix_2[1][1][1] == "X"
    assert time_2 == 5
    assert matrix_3[2][0][0] == "0" and matrix_3[2][0][1] == "X"
    assert time_3 == 10
    assert matrix_4[2][0][0] == "0" and matrix_4[2][1][0] == "X"
    assert time_4 == 10
    assert matrix_5[1][0][0] == "0" and matrix_5[2][0][0] == "X"
    assert time_5 == 10
    assert matrix_6[2][0][0] == "0" and matrix_6[2][0][2] == "X"
    assert time_6 == 10
    assert matrix_7[2][0][0] == "0" and matrix_7[2][2][0] == "X"
    assert time_7 == 10
    assert matrix_8[0][0][0] == "X" and matrix_8[1][0][0] == "X"
    assert time_8 == 20


def test_movement_over_person():
    matrix_0, time_0 = create_movement_matrix(
        array(matrix_small, dtype="object"),
        [Movements.DIAGONAL_DOWN_RIGHT, Movements.DIAGONAL_DOWN_RIGHT],
    )
    assert matrix_0[0][1][1] == "P"
    assert matrix_0[1][1][1] == "PX"
    assert matrix_0[2][1][1] == "P"
    assert matrix_0[2][2][2] == "X"


def test_search():
    matrix_0, time_0 = create_movement_matrix(
        array(matrix_small, dtype="object"),
        [
            Movements.DIAGONAL_DOWN_RIGHT,
            Movements.SEARCH,
            Movements.DIAGONAL_DOWN_RIGHT,
        ],
    )
    matrix_1, time_1 = create_movement_matrix(
        array(matrix_small, dtype="object"),
        [Movements.DOWN, Movements.SEARCH, Movements.DOWN, Movements.SEARCH],
    )
    assert (
        matrix_0[0][1][1] == "P"
        and matrix_0[1][1][1] == "PX"
        and matrix_0[2][1][1] == "F"
    )
    assert len(matrix_0) == 3 and time_0 == 25
    assert (
        matrix_1[0][1][1] == "P"
        and matrix_1[1][1][0] == "X"
        and matrix_1[2][1][0] == "X"
        and matrix_1[3][2][0] == "X"
        and matrix_1[4][2][0] == "X"
    )
    assert len(matrix_1) == 5 and time_1 == 50
