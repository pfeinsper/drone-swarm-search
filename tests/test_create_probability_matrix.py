from core.create_probability_matrix import create_probability_matrix


def test_matrix_sum():
    assert round(create_probability_matrix(10, 2).sum()) == 100
    assert round(create_probability_matrix(10, 10).sum()) == 100
    assert round(create_probability_matrix(30, 100).sum()) == 100
    assert round(create_probability_matrix(35, 20).sum()) == 100
    assert round(create_probability_matrix(53, 61).sum()) == 100
    assert round(create_probability_matrix(13, 77).sum()) == 100
    assert round(create_probability_matrix(32, 2346).sum()) == 100
    assert round(create_probability_matrix(1, 2).sum()) == 100
    assert round(create_probability_matrix(2, 4).sum()) == 100


def test_matrix_size():
    assert create_probability_matrix(10, 2).size == 20
    assert create_probability_matrix(10, 10).size == 100
    assert create_probability_matrix(30, 100).size == 3000
    assert create_probability_matrix(35, 20).size == 700
    assert create_probability_matrix(53, 61).size == 3233
    assert create_probability_matrix(13, 77).size == 1001
    assert create_probability_matrix(32, 2346).size == 75072
    assert create_probability_matrix(1, 2).size == 2
    assert create_probability_matrix(2, 4).size == 8


def test_matrix_dimensions():
    assert create_probability_matrix(10, 2).shape == (10, 2)
    assert create_probability_matrix(10, 10).shape == (10, 10)
    assert create_probability_matrix(30, 100).shape == (30, 100)
    assert create_probability_matrix(35, 20).shape == (35, 20)
    assert create_probability_matrix(53, 61).shape == (53, 61)
    assert create_probability_matrix(13, 77).shape == (13, 77)
    assert create_probability_matrix(32, 2346).shape == (32, 2346)
    assert create_probability_matrix(1, 2).shape == (1, 2)
    assert create_probability_matrix(2, 4).shape == (2, 4)

