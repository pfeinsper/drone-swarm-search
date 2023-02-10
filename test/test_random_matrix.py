import pytest
import core.create_random_matrix as create_random_matrix


def test_matrix_sum():
    assert round(create_random_matrix.create_random_matrix(10, 2).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(10, 10).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(30, 100).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(35, 20).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(53, 61).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(13, 77).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(32, 2346).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(1, 2).sum()) == 100
    assert round(create_random_matrix.create_random_matrix(2, 4).sum()) == 100


def test_matrix_size():
    assert create_random_matrix.create_random_matrix(10, 2).size == 20
    assert create_random_matrix.create_random_matrix(10, 10).size == 100
    assert create_random_matrix.create_random_matrix(30, 100).size == 3000
    assert create_random_matrix.create_random_matrix(35, 20).size == 700
    assert create_random_matrix.create_random_matrix(53, 61).size == 3233
    assert create_random_matrix.create_random_matrix(13, 77).size == 1001
    assert create_random_matrix.create_random_matrix(32, 2346).size == 75072
    assert create_random_matrix.create_random_matrix(1, 2).size == 2
    assert create_random_matrix.create_random_matrix(2, 4).size == 8


def test_matrix_dimensions():
    matrix_0 = create_random_matrix.create_random_matrix(10, 2)
    matrix_1 = create_random_matrix.create_random_matrix(10, 10)
    matrix_2 = create_random_matrix.create_random_matrix(30, 100)
    matrix_3 = create_random_matrix.create_random_matrix(35, 20)
    matrix_4 = create_random_matrix.create_random_matrix(53, 61)
    matrix_5 = create_random_matrix.create_random_matrix(13, 77)
    matrix_6 = create_random_matrix.create_random_matrix(32, 2346)
    matrix_7 = create_random_matrix.create_random_matrix(1, 2)
    matrix_8 = create_random_matrix.create_random_matrix(2, 4)
    assert len(matrix_0) == 10
    assert matrix_0[0].size == 2
    assert len(matrix_1) == 10
    assert matrix_1[0].size == 10
    assert len(matrix_2) == 30
    assert matrix_2[0].size == 100
    assert len(matrix_3) == 35
    assert matrix_3[0].size == 20
    assert len(matrix_4) == 53
    assert matrix_4[0].size == 61
    assert len(matrix_5) == 13
    assert matrix_5[0].size == 77
    assert len(matrix_6) == 32
    assert matrix_6[0].size == 2346
    assert len(matrix_7) == 1
    assert matrix_7[0].size == 2
    assert len(matrix_8) == 2
    assert matrix_8[0].size == 4
